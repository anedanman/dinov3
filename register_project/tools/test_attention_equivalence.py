# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Equivalence test for the SDPA-based register attention fast paths.

Compares RegisterSlotAttention / PatchClsSeparateRegisterBudgetAttention against
a straightforward reference implementation (explicit fp32 softmax attention with
row overwrites, matching the original implementation). Run on CPU or GPU:

    python register_project/tools/test_attention_equivalence.py
"""

import torch
import torch.nn.functional as F

from dinov3.layers.attention import (
    PatchClsSeparateRegisterBudgetAttention,
    RegisterSlotAttention,
    compute_register_competition,
)


def _reference_separate_budget_rows(q, k, v, R, scale):
    """Original manual implementation: fp32 two-softmax CLS/patch rows."""
    q_nonreg = torch.cat([q[:, :, :1, :], q[:, :, 1 + R :, :]], dim=-2).float()
    k_nonreg = torch.cat([k[:, :, :1, :], k[:, :, 1 + R :, :]], dim=-2).float()
    v_nonreg = torch.cat([v[:, :, :1, :], v[:, :, 1 + R :, :]], dim=-2)
    k_reg = k[:, :, 1 : 1 + R, :].float()
    v_reg = v[:, :, 1 : 1 + R, :]
    nonreg_attn = torch.softmax(torch.matmul(q_nonreg, k_nonreg.transpose(-2, -1)) * scale, dim=-1)
    reg_attn = torch.softmax(torch.matmul(q_nonreg, k_reg.transpose(-2, -1)) * scale, dim=-1)
    return torch.matmul(nonreg_attn.to(v.dtype), v_nonreg) + torch.matmul(reg_attn.to(v.dtype), v_reg)


def _reference_full_attention(q, k, v, scale):
    attn = torch.softmax(torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale, dim=-1)
    return torch.matmul(attn.to(v.dtype), v)


def _qkv_to_heads(qkv, num_heads):
    B, N, C3 = qkv.shape
    C = C3 // 3
    q, k, v = torch.unbind(qkv.reshape(B, N, 3, num_heads, C // num_heads), 2)
    return [t.transpose(1, 2) for t in (q, k, v)]


def _check(name, actual, expected, tol):
    err = (actual - expected).abs().max().item()
    status = "OK " if err <= tol else "FAIL"
    print(f"[{status}] {name}: max abs err = {err:.3e} (tol {tol:.1e})")
    return err <= tol


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    B, N, R, dim, heads = 3, 1 + 7 + 64, 7, 96, 6
    scale = (dim // heads) ** -0.5
    tol = 1e-4

    ok = True
    for slot_mode in ("slot", "literal"):
        for patch_cls_attn_type in ("standard", "separate_register_budget"):
            attn = RegisterSlotAttention(
                dim,
                num_heads=heads,
                qkv_bias=True,
                n_storage_tokens=R,
                slot_mode=slot_mode,
                exclude_cls=True,
                patch_cls_attn_type=patch_cls_attn_type,
            ).to(device=device, dtype=dtype)
            x = torch.randn(B, N, dim, device=device, dtype=dtype)
            qkv = attn.qkv(x)
            out = attn.compute_attention(qkv)

            q, k, v = _qkv_to_heads(qkv, heads)
            if patch_cls_attn_type == "separate_register_budget":
                rows = _reference_separate_budget_rows(q, k, v, R, scale)
            else:
                q_nonreg = torch.cat([q[:, :, :1, :], q[:, :, 1 + R :, :]], dim=-2)
                rows = _reference_full_attention(q_nonreg, k, v, scale)
            reg = compute_register_competition(q, k, v, R, scale, renorm=slot_mode == "slot", exclude_cls=True)
            ref = torch.cat([rows[:, :, :1, :], reg, rows[:, :, 1:, :]], dim=-2)
            ref = ref.transpose(1, 2).reshape(B, N, dim)
            ok &= _check(f"RegisterSlotAttention(slot_mode={slot_mode}, patch_cls={patch_cls_attn_type})", out, ref, tol)

    attn = PatchClsSeparateRegisterBudgetAttention(
        dim, num_heads=heads, qkv_bias=True, n_storage_tokens=R
    ).to(device=device, dtype=dtype)
    x = torch.randn(B, N, dim, device=device, dtype=dtype)
    qkv = attn.qkv(x)
    out = attn.compute_attention(qkv)
    q, k, v = _qkv_to_heads(qkv, heads)
    rows = _reference_separate_budget_rows(q, k, v, R, scale)
    reg = _reference_full_attention(q[:, :, 1 : 1 + R, :], k, v, scale)
    ref = torch.cat([rows[:, :, :1, :], reg, rows[:, :, 1:, :]], dim=-2)
    ref = ref.transpose(1, 2).reshape(B, N, dim)
    ok &= _check("PatchClsSeparateRegisterBudgetAttention", out, ref, tol)

    if not ok:
        raise SystemExit(1)
    print("all attention equivalence checks passed")


if __name__ == "__main__":
    main()
