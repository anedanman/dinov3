# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Unit tests for the register-research features:

* cross-crop register consistency loss (Hungarian / fixed matching),
* slot competition restricted to later layers (student.slot_start_layer),
* learnable register-budget gate (student.register_budget_gate),
* register diagnostics metrics,
* register-attention visualization panel layout.

Run:  python register_project/tools/test_register_features.py
"""

import numpy as np
import torch

from dinov3.layers.attention import PatchClsSeparateRegisterBudgetAttention, RegisterSlotAttention, SelfAttention
from dinov3.loss.register_consistency_loss import register_consistency_loss
from dinov3.models.vision_transformer import DinoVisionTransformer

PASSED = []


def check(name, cond):
    status = "OK " if cond else "FAIL"
    print(f"[{status}] {name}")
    PASSED.append(bool(cond))


def test_consistency_loss():
    torch.manual_seed(0)
    B, R, D = 4, 7, 16
    reg = torch.randn(2, B, R, D)

    # Identical student/teacher crops -> zero loss under both matchings.
    loss = register_consistency_loss(reg, reg, matching="hungarian", pairs=[(0, 0), (1, 1)])
    check("consistency: identical crops -> 0", loss.item() < 1e-6)

    # Teacher registers permuted per image: hungarian recovers the permutation, fixed does not.
    perm = torch.stack([torch.randperm(R) for _ in range(B)])
    teacher = reg.clone()
    for b in range(B):
        teacher[:, b] = reg[:, b, perm[b]]
    loss_h = register_consistency_loss(reg, teacher, matching="hungarian", pairs=[(0, 0), (1, 1)])
    loss_f = register_consistency_loss(reg, teacher, matching="fixed", pairs=[(0, 0), (1, 1)])
    check("consistency: hungarian invariant to register permutation", loss_h.item() < 1e-6)
    check("consistency: fixed matching is NOT permutation invariant", loss_f.item() > 0.1)

    # Default pairs are cross-view; gradients flow to the student only.
    student = torch.randn(2, B, R, D, requires_grad=True)
    teacher = torch.randn(2, B, R, D)
    loss = register_consistency_loss(student, teacher)
    loss.backward()
    check("consistency: in [0,2] and differentiable", 0.0 <= loss.item() <= 2.0 and student.grad is not None)

    # subtract_mean: collapsed registers (shared direction + small residuals)
    # trivially saturate the raw-cosine loss but not the residual loss.
    shared = torch.randn(1, B, 1, D) * 10.0
    resid_s = 0.01 * torch.randn(1, B, R, D)
    resid_t = 0.01 * torch.randn(1, B, R, D)
    collapsed_s, collapsed_t = shared + resid_s, shared + resid_t
    loss_raw = register_consistency_loss(collapsed_s, collapsed_t, pairs=[(0, 0)])
    loss_resid = register_consistency_loss(collapsed_s, collapsed_t, pairs=[(0, 0)], subtract_mean=True)
    check("consistency: raw cosine saturates under collapse", loss_raw.item() < 1e-3)
    check("consistency: subtract_mean stays informative under collapse", loss_resid.item() > 0.1)

    # subtract_mean is invariant to any shared offset and still zero for identical crops.
    loss_id = register_consistency_loss(reg + 5.0, reg + 5.0, pairs=[(0, 0), (1, 1)], subtract_mean=True)
    check("consistency: subtract_mean, identical crops -> 0", loss_id.item() < 1e-6)
    student2 = torch.randn(2, B, R, D, requires_grad=True)
    loss2 = register_consistency_loss(student2, teacher, subtract_mean=True)
    loss2.backward()
    check("consistency: subtract_mean differentiable", student2.grad is not None and torch.isfinite(student2.grad).all())


def _tiny_vit(**kwargs):
    base = dict(
        img_size=32,
        patch_size=16,
        embed_dim=64,
        depth=4,
        num_heads=4,
        n_storage_tokens=3,
        norm_layer="layernorm",
        ffn_layer="mlp",
    )
    base.update(kwargs)
    model = DinoVisionTransformer(**base)
    model.init_weights()
    model.eval()
    return model


def test_slot_start_layer():
    model = _tiny_vit(
        register_attn_type="slot",
        patch_cls_attn_type="separate_register_budget",
        slot_start_layer=2,
    )
    types = [type(blk.attn) for blk in model.blocks]
    check(
        "slot_start_layer: early blocks use separate-budget attention",
        types[0] is PatchClsSeparateRegisterBudgetAttention and types[1] is PatchClsSeparateRegisterBudgetAttention,
    )
    check(
        "slot_start_layer: late blocks use slot attention",
        types[2] is RegisterSlotAttention and types[3] is RegisterSlotAttention,
    )
    check(
        "slot_start_layer: per-layer attn type for extraction",
        model.register_attn_type_at_layer(0) == "standard"
        and model.register_attn_type_at_layer(2) == "slot"
        and model.register_attn_type_at_layer(-1) == "slot",
    )
    # standard patch_cls + restriction: early blocks are plain SelfAttention
    model2 = _tiny_vit(register_attn_type="slot", slot_start_layer=3)
    check(
        "slot_start_layer: plain SelfAttention before start when budget off",
        type(model2.blocks[0].attn) is SelfAttention and type(model2.blocks[3].attn) is RegisterSlotAttention,
    )
    # Forward + attention extraction across the layer-type boundary.
    x = torch.randn(2, 3, 32, 32)
    out = model.forward_features(x)
    maps = model.get_register_attention_layers(x, layers=None)
    check(
        "slot_start_layer: forward + extraction work",
        out["x_storage_tokens"].shape == (2, 3, 64)
        and maps["register_to_patch"]["masks"].shape[0] == 4
        and torch.isfinite(maps["register_to_patch"]["masks"]).all(),
    )


def test_register_budget_gate():
    torch.manual_seed(0)
    dim, heads, R, N, B = 64, 4, 3, 1 + 3 + 4, 2

    gated = PatchClsSeparateRegisterBudgetAttention(dim, num_heads=heads, qkv_bias=True, n_storage_tokens=R, register_budget_gate=True)
    plain = PatchClsSeparateRegisterBudgetAttention(dim, num_heads=heads, qkv_bias=True, n_storage_tokens=R)
    plain.load_state_dict({k: v for k, v in gated.state_dict().items() if k != "reg_budget_gate"})
    torch.nn.init.ones_(gated.reg_budget_gate)

    x = torch.randn(B, N, dim)
    out_gated = gated(x)
    out_plain = plain(x)
    check("gate: init=1 reproduces ungated output", torch.allclose(out_gated, out_plain, atol=1e-6))

    with torch.no_grad():
        gated.reg_budget_gate.zero_()
    out_zero = gated(x)
    check("gate: gate=0 changes CLS/patch rows", not torch.allclose(out_zero, out_plain, atol=1e-4))

    # Slot attention variant with gate, plus gradient flow into the gate.
    torch.nn.init.ones_(gated.reg_budget_gate)
    slot = RegisterSlotAttention(
        dim,
        num_heads=heads,
        qkv_bias=True,
        n_storage_tokens=R,
        patch_cls_attn_type="separate_register_budget",
        register_budget_gate=True,
    )
    torch.nn.init.ones_(slot.reg_budget_gate)
    out = slot(x).sum()
    out.backward()
    check("gate: slot variant has gate gradients", slot.reg_budget_gate.grad is not None and torch.isfinite(slot.reg_budget_gate.grad).all())

    # Model-level wiring + init_weights initializes gates to 1.
    model = _tiny_vit(
        register_attn_type="slot",
        patch_cls_attn_type="separate_register_budget",
        register_budget_gate=True,
    )
    gates = [blk.attn.reg_budget_gate for blk in model.blocks]
    check(
        "gate: model init_weights sets all gates to 1",
        all(g is not None and torch.allclose(g, torch.ones_like(g)) for g in gates),
    )


def test_diagnostics():
    from dinov3.eval.register_tokens.diagnostics import compute_register_diagnostics, make_fixed_views

    model = _tiny_vit(
        register_attn_type="slot",
        patch_cls_attn_type="separate_register_budget",
        register_budget_gate=True,
    )
    images = torch.randn(4, 3, 32, 32)
    views = make_fixed_views(images)
    check("diagnostics: fixed views deterministic", torch.allclose(views[0], make_fixed_views(images)[0]))
    metrics = compute_register_diagnostics(model, images, views=views, device="cpu")
    expected = [
        "register_diag/slot_share_max",
        "register_diag/slot_usage_entropy",
        "register_diag/active_slots",
        "register_diag/patch_assign_entropy",
        "register_diag/reg_norm_mean",
        "register_diag/reg_pairwise_cos",
        "register_diag/reg_resid_pairwise_cos",
        "register_diag/reg_resid_norm_frac",
        "register_diag/patch_norm_outlier_frac",
        "register_diag/patch_norm_max_over_median",
        "register_diag/register_norm_over_patch_median",
        "register_diag/xcrop_matched_cos",
        "register_diag/xcrop_identity_match_frac",
        "register_diag/xcrop_resid_matched_cos",
        "register_diag/xcrop_resid_identity_match_frac",
        "register_gate/mean",
        "register_gate/layer_00",
    ]
    missing = [k for k in expected if k not in metrics]
    check(f"diagnostics: all expected keys present (missing={missing})", not missing)
    check("diagnostics: all values finite", all(np.isfinite(v) for v in metrics.values()))
    check(
        "diagnostics: ranges sane",
        0.0 <= metrics["register_diag/slot_usage_entropy"] <= 1.0
        and 0.0 < metrics["register_diag/active_slots"] <= 3.0
        and -1.0 <= metrics["register_diag/xcrop_matched_cos"] <= 1.0
        and abs(metrics["register_gate/mean"] - 1.0) < 1e-6,
    )


def test_viz_panels():
    from dinov3.eval.register_tokens.attention_viz import render_register_attention

    model = _tiny_vit(register_attn_type="slot", patch_cls_attn_type="separate_register_budget")
    N, S = 2, 32
    images = torch.randn(N, 3, S, S)
    display = [np.random.randint(0, 255, (S, S, 3), dtype=np.uint8) for _ in range(N)]
    panels = render_register_attention(model, images, display, device="cpu")
    R = model.n_storage_tokens
    expected_width = S + 2 * (4 + (R + 2) * S)  # input + 2 directions x (sep + R+2 columns)
    check("viz: header + one panel per image", len(panels) == N + 1)
    check(
        "viz: consistent panel widths",
        all(p.shape[1] == expected_width for p in panels) and panels[0].dtype == np.uint8,
    )
    per_head = render_register_attention(model, images, display, device="cpu", head_reduce="none")
    check("viz: per-head layout renders", len(per_head) == N + 1 and per_head[0].shape[1] == per_head[1].shape[1])


def main():
    test_consistency_loss()
    test_slot_start_layer()
    test_register_budget_gate()
    test_diagnostics()
    test_viz_panels()
    if not all(PASSED):
        raise SystemExit(1)
    print(f"all {len(PASSED)} register-feature checks passed")


if __name__ == "__main__":
    main()
