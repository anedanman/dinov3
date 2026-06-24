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
from dinov3.models.vision_transformer import DinoVisionTransformer, _regularized_lowdin

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

    bipartite = _tiny_vit(
        register_attn_type="slot",
        patch_cls_attn_type="separate_register_budget",
        patch_to_patch_attention=False,
        register_attn_exclude_cls=False,
    )
    bipartite_out = bipartite.forward_features(x)
    bipartite_maps = bipartite.get_register_attention_layers(x, layers=[-1])
    patch_prefix_mass = (
        bipartite_maps["patch_to_register"]["masks"].sum(dim=-2)
        + bipartite_maps["patch_to_register"]["cls"]
    )
    check(
        "no patch-to-patch: forward and attention extraction work",
        torch.isfinite(bipartite_out["x_prenorm"]).all()
        and torch.isfinite(bipartite_maps["patch_to_register"]["masks"]).all()
        and torch.allclose(patch_prefix_mass, torch.ones_like(patch_prefix_mass), atol=1e-6),
    )


def test_predicted_gaussian_register_insertion():
    torch.manual_seed(0)
    model = _tiny_vit(
        register_attn_type="slot",
        patch_cls_attn_type="separate_register_budget",
        register_init="predicted_gaussian",
        register_insert_layer=2,
        slot_start_layer=2,
    )
    images = torch.randn(2, 3, 32, 32)
    prepared, _ = model.prepare_tokens_with_masks(images)
    types = [type(block.attn) for block in model.blocks]
    check(
        "predicted registers: absent initially and standard attention before insertion",
        prepared.shape[1] == 1 + 4
        and types[:2] == [SelfAttention, SelfAttention]
        and types[2:] == [RegisterSlotAttention, RegisterSlotAttention],
    )

    first = model.forward_features(images)
    second = model.forward_features(images)
    check(
        "predicted registers: inserted at mid-depth and sampled per forward",
        first["x_prenorm"].shape == (2, 1 + 3 + 4, 64)
        and not torch.equal(first["x_storage_tokens"], second["x_storage_tokens"]),
    )

    maps = model.get_register_attention_layers(images)
    check(
        "predicted registers: attention extraction starts after insertion",
        maps["layers"] == [2, 3]
        and maps["register_to_patch"]["masks"].shape[:4] == (2, 2, 4, 3),
    )

    model.zero_grad(set_to_none=True)
    model.classifier_pooling = "global_avg_registers"
    loss = (model(images) * torch.randn(2, 64)).sum()
    loss.backward()
    predictor_grads = [parameter.grad for parameter in model.register_predictor.parameters()]
    check(
        "predicted registers: predictor receives finite nonzero gradients",
        all(grad is not None and torch.isfinite(grad).all() for grad in predictor_grads)
        and sum(grad.abs().sum().item() for grad in predictor_grads) > 0.0,
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


def test_register_orthogonalization():
    torch.manual_seed(0)
    model = _tiny_vit(
        register_attn_type="slot",
        patch_cls_attn_type="separate_register_budget",
        register_orthogonalize=True,
    )
    R, D = model.n_storage_tokens, model.embed_dim
    B, P = 4, 4

    def off_diag_cos(reg):
        z = torch.nn.functional.normalize(reg.float(), dim=-1)
        cos = torch.einsum("brd,bsd->brs", z, z)
        return (cos - torch.eye(R).expand_as(cos)).abs().max().item()

    # Random registers become (numerically) orthogonal, norms preserved.
    x = torch.randn(B, 1 + R + P, D)
    y = model._orthogonalize_registers(x)
    reg_in, reg_out = x[:, 1 : R + 1], y[:, 1 : R + 1]
    check("orth: register rows orthogonal after projection", off_diag_cos(reg_out) < 1e-2)
    check(
        "orth: norms preserved",
        torch.allclose(reg_out.norm(dim=-1), reg_in.norm(dim=-1), rtol=1e-3),
    )
    check(
        "orth: cls/patch rows untouched",
        torch.equal(y[:, 0], x[:, 0]) and torch.equal(y[:, R + 1 :], x[:, R + 1 :]),
    )

    # Nearly collapsed registers (shared direction + tiny residuals): stays
    # finite, differentiable, and much more orthogonal than the input. A random
    # directional loss exercises the whitening derivative; a squared-norm loss
    # mostly exercises the optional norm-restoration path instead.
    shared = torch.randn(B, 1, D) * 5.0
    x2 = torch.randn(B, 1 + R + P, D)
    x2[:, 1 : R + 1] = shared + 0.01 * torch.randn(B, R, D)
    x2.requires_grad_(True)
    y2 = model._orthogonalize_registers(x2)
    (y2[:, 1 : R + 1] * torch.randn_like(y2[:, 1 : R + 1])).sum().backward()
    cos_in = off_diag_cos(x2[:, 1 : R + 1].detach())
    cos_out = off_diag_cos(y2[:, 1 : R + 1].detach())
    check(
        "orth: collapsed registers de-correlated without NaNs",
        torch.isfinite(y2).all() and torch.isfinite(x2.grad).all() and cos_out < 0.5 * cos_in,
    )

    # The custom VJP must agree with finite differences. This also covers an
    # exactly repeated Gram spectrum, where generic eigh/SVD backward produces
    # undefined eigenvector/singular-vector gradients despite the polar map
    # itself having a well-defined derivative.
    gradcheck_random = torch.randn(1, 3, 7, dtype=torch.float64, requires_grad=True)
    gradcheck_repeated = torch.eye(3, 7, dtype=torch.float64).unsqueeze(0).requires_grad_()
    check(
        "orth: custom backward passes gradcheck",
        torch.autograd.gradcheck(
            lambda z: _regularized_lowdin(z, 1e-4),
            (gradcheck_random,),
            eps=1e-6,
            atol=2e-5,
            rtol=2e-4,
        )
        and torch.autograd.gradcheck(
            lambda z: _regularized_lowdin(z, 1e-4),
            (gradcheck_repeated,),
            eps=1e-6,
            atol=2e-5,
            rtol=2e-4,
        ),
    )

    x_bf16 = torch.randn(2, 1 + R + P, D, dtype=torch.bfloat16, requires_grad=True)
    y_bf16 = model._orthogonalize_registers(x_bf16)
    y_bf16.float().mul(torch.randn_like(y_bf16.float())).sum().backward()
    check(
        "orth: bf16 boundary preserves dtype and finite gradients",
        y_bf16.dtype == torch.bfloat16 and x_bf16.grad is not None and torch.isfinite(x_bf16.grad).all(),
    )

    try:
        DinoVisionTransformer(
            img_size=16,
            patch_size=16,
            embed_dim=4,
            depth=1,
            num_heads=1,
            n_storage_tokens=5,
            register_orthogonalize=True,
        )
        feasible_shape_rejected = False
    except AssertionError as error:
        feasible_shape_rejected = "n_storage_tokens <= embed_dim" in str(error)
    check("orth: rejects more registers than embedding dimensions", feasible_shape_rejected)

    # Full forward path applies it: prenorm registers of the last block output
    # are mutually orthogonal.
    out = model.forward_features(torch.randn(2, 3, 32, 32))
    prenorm_reg = out["x_prenorm"][:, 1 : R + 1]
    check("orth: forward_features output registers orthogonal", off_diag_cos(prenorm_reg) < 1e-2)
    model.zero_grad(set_to_none=True)
    (out["x_storage_tokens"] * torch.randn_like(out["x_storage_tokens"])).sum().backward()
    parameter_grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    check(
        "orth: full model backward gradients finite",
        len(parameter_grads) > 0 and all(torch.isfinite(grad).all() for grad in parameter_grads),
    )


def test_classifier_pooling():
    model = _tiny_vit(register_init="learned")
    model.classifier_pooling = "global_avg_registers"
    images = torch.randn(2, 3, 32, 32)
    features = model.forward_features(images)
    pooled = model(images)
    check(
        "classifier pooling: register-only mean excludes CLS and patches",
        torch.allclose(pooled, features["x_storage_tokens"].mean(dim=1), atol=1e-6),
    )


def test_diagnostics():
    from dinov3.eval.register_tokens.diagnostics import (
        attention_score_metrics,
        compute_register_diagnostics,
        make_fixed_views,
    )

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
    attention_metrics = attention_score_metrics(model, images)
    check(
        "diagnostics: layerwise attention scores present",
        len(attention_metrics) == model.n_blocks * 4
        and "register_attention/layer_00/register_to_patch_entropy" in attention_metrics,
    )
    check("diagnostics: attention scores finite", all(np.isfinite(v) for v in attention_metrics.values()))


def test_viz_panels():
    from dinov3.eval.register_tokens.attention_viz import (
        render_register_attention,
        render_register_embedding_similarity,
    )

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

    similarity = model.get_register_patch_embedding_similarity(images, layer=-1)
    register_similarity, cls_similarity = model.get_prefix_patch_embedding_similarity(images, layer=-1)
    check(
        "embedding similarity: expected shape and cosine range",
        similarity.shape == (N, R, S // model.patch_size, S // model.patch_size)
        and register_similarity.shape == similarity.shape
        and cls_similarity.shape == (N, S // model.patch_size, S // model.patch_size)
        and similarity.min() >= -1.0001
        and similarity.max() <= 1.0001,
    )
    sim_panels = render_register_embedding_similarity(model, images, display, device="cpu")
    check(
        "embedding similarity: input plus one panel per register",
        len(sim_panels) == N + 1 and all(p.shape[1] == (R + 2) * S for p in sim_panels),
    )


def main():
    test_consistency_loss()
    test_slot_start_layer()
    test_predicted_gaussian_register_insertion()
    test_register_budget_gate()
    test_register_orthogonalization()
    test_classifier_pooling()
    test_diagnostics()
    test_viz_panels()
    if not all(PASSED):
        raise SystemExit(1)
    print(f"all {len(PASSED)} register-feature checks passed")


if __name__ == "__main__":
    main()
