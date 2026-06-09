# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from dinov3.utils import cat_keep_shapes, uncat_with_shapes
from torch import Tensor, nn


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


def _split_register_keys(t: Tensor, n_storage_tokens: int, exclude_cls: bool = False) -> Tensor:
    """Drop the register key/value columns, keeping patches and optionally cls.

    Token layout along dim=-2 is [cls(1), storage/registers(R), patches(P)].
    Returns a tensor over the non-register keys: [B, heads, P or 1 + P, d].
    """
    patches = t[:, :, 1 + n_storage_tokens :, :]
    if exclude_cls:
        return patches
    cls = t[:, :, :1, :]
    return torch.cat([cls, patches], dim=-2)


def compute_register_competition(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    n_storage_tokens: int,
    scale: float,
    renorm: bool,
    exclude_cls: bool,
    eps: float = 1e-8,
) -> Tensor:
    """Slot-attention-style output for the register query rows.

    Registers attend only to non-register tokens (patches, optionally cls) and the
    attention logits are softmaxed across the *query* (register) dimension so the
    registers compete for each input token, as in Locatello et al. slot
    attention.

    Args:
        q, k, v: [B, heads, N, d] with token layout [cls, registers, patches].
        renorm: if True, renormalize the competition weights over the keys so
            each register output is a weighted mean (slot-attention style); if
            False, use the weights directly (``out = A @ v``, "the rest is the same").

    Returns:
        Register-row outputs [B, heads, R, d] in the dtype of ``v``.
    """
    R = n_storage_tokens
    q_reg = q[:, :, 1 : 1 + R, :].float()  # [B, h, R, d]
    k_nr = _split_register_keys(k, R, exclude_cls=exclude_cls).float()  # [B, h, P or 1+P, d]
    v_nr = _split_register_keys(v, R, exclude_cls=exclude_cls)  # [B, h, P or 1+P, d]

    logits = torch.matmul(q_reg, k_nr.transpose(-2, -1)) * scale  # [B, h, R, P or 1+P]
    # Competition: softmax across the register (query) dimension, per key.
    attn = torch.softmax(logits, dim=-2)  # [B, h, R, P or 1+P]
    if renorm:
        attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
    out_reg = torch.matmul(attn.to(v_nr.dtype), v_nr)  # [B, h, R, d]
    return out_reg


def compute_patch_cls_separate_register_budget(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    n_storage_tokens: int,
    scale: float,
) -> Tensor:
    """Attention output for CLS/patch rows with a separate register budget.

    Token layout is [cls, registers, patches]. The CLS and patch queries attend
    to non-register keys with one key-softmax and to register keys with a second
    key-softmax. The two value averages are then added, so registers no longer
    compete with CLS/patch tokens for the same attention probability mass.
    """
    R = n_storage_tokens
    q_nonreg = torch.cat([q[:, :, :1, :], q[:, :, 1 + R :, :]], dim=-2)  # [B,h,1+P,d]
    k_nonreg = torch.cat([k[:, :, :1, :], k[:, :, 1 + R :, :]], dim=-2)  # [B,h,1+P,d]
    v_nonreg = torch.cat([v[:, :, :1, :], v[:, :, 1 + R :, :]], dim=-2)  # [B,h,1+P,d]
    out_nonreg = F.scaled_dot_product_attention(q_nonreg, k_nonreg, v_nonreg, scale=scale)
    out_reg = F.scaled_dot_product_attention(
        q_nonreg, k[:, :, 1 : 1 + R, :], v[:, :, 1 : 1 + R, :], scale=scale
    )
    return out_nonreg + out_reg


@torch.no_grad()
def extract_register_attention_maps(
    qkv: Tensor,
    num_heads: int,
    n_storage_tokens: int,
    scale: float,
    rope=None,
    apply_rope_fn=None,
    attn_type: str = "standard",
    slot_renorm: bool = True,
    slot_exclude_cls: bool = True,
    patch_cls_attn_type: str = "standard",
    direction: str = "register_to_patch",
) -> Tuple[Tensor, Tensor]:
    """Compute register masks and the matching CLS map for viz / MBO.

    Returns:
        masks: [B, heads, R, P], where R is the number of registers and P is
            the number of patch tokens.
        cls_map: [B, heads, P], using the analogous CLS direction:
            cls->patch for register_to_patch, patch->cls for patch_to_register.

    For register_to_patch, masks use the same softmax convention the model was
    trained with:
      * "standard": softmax over all keys, then read off the patch columns.
      * "slot": competition softmax over the register dimension (registers vs
        patches, optionally cls), optionally key-renormalized, then read off
        patch columns.

    For patch_to_register, patch rows behave as standard attention for both the
    baseline and slot-register model; non-register rows are unchanged by
    RegisterSlotAttention unless ``patch_cls_attn_type`` gives them a separate
    register budget.
    """
    assert direction in ("register_to_patch", "patch_to_register"), f"unknown direction={direction}"
    assert patch_cls_attn_type in ("standard", "separate_register_budget"), (
        f"unknown patch_cls_attn_type={patch_cls_attn_type}"
    )
    B, N, _ = qkv.shape
    C = qkv.shape[-1] // 3
    qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads)
    q, k, _ = torch.unbind(qkv, 2)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    if rope is not None and apply_rope_fn is not None:
        q, k = apply_rope_fn(q, k, rope)
    R = n_storage_tokens
    P = N - 1 - R

    if direction == "register_to_patch":
        q_reg = q[:, :, 1 : 1 + R, :].float()
        if attn_type == "slot":
            k_nr = _split_register_keys(k, R, exclude_cls=slot_exclude_cls).float()  # [B,h,P or 1+P,d]
            logits = torch.matmul(q_reg, k_nr.transpose(-2, -1)) * scale  # [B,h,R,P or 1+P]
            masks = torch.softmax(logits, dim=-2)  # competition across registers
            if slot_renorm:
                masks = masks / (masks.sum(dim=-1, keepdim=True) + 1e-8)
            masks = masks if slot_exclude_cls else masks[:, :, :, 1:]
        else:
            logits = torch.matmul(q_reg, k.float().transpose(-2, -1)) * scale  # [B,h,R,N]
            attn = torch.softmax(logits, dim=-1)  # over all keys
            masks = attn[:, :, :, 1 + R :]  # patch columns -> [B,h,R,P]

        q_cls = q[:, :, :1, :].float()
        if patch_cls_attn_type == "separate_register_budget":
            k_nonreg = torch.cat([k[:, :, :1, :], k[:, :, 1 + R :, :]], dim=-2).float()
            cls_logits = torch.matmul(q_cls, k_nonreg.transpose(-2, -1)) * scale  # [B,h,1,1+P]
            cls_map = torch.softmax(cls_logits, dim=-1)[:, :, 0, 1:]  # [B,h,P]
        else:
            cls_logits = torch.matmul(q_cls, k.float().transpose(-2, -1)) * scale  # [B,h,1,N]
            cls_map = torch.softmax(cls_logits, dim=-1)[:, :, 0, 1 + R :]  # [B,h,P]
        return masks, cls_map

    q_patch = q[:, :, 1 + R :, :].float()  # [B,h,P,d]
    if patch_cls_attn_type == "separate_register_budget":
        k_reg = k[:, :, 1 : 1 + R, :].float()
        reg_logits = torch.matmul(q_patch, k_reg.transpose(-2, -1)) * scale  # [B,h,P,R]
        reg_attn = torch.softmax(reg_logits, dim=-1)
        masks = reg_attn.transpose(-2, -1)  # [B,h,R,P]

        k_nonreg = torch.cat([k[:, :, :1, :], k[:, :, 1 + R :, :]], dim=-2).float()
        nonreg_logits = torch.matmul(q_patch, k_nonreg.transpose(-2, -1)) * scale  # [B,h,P,1+P]
        cls_map = torch.softmax(nonreg_logits, dim=-1)[:, :, :, 0]  # patch->cls, [B,h,P]
    else:
        logits = torch.matmul(q_patch, k.float().transpose(-2, -1)) * scale  # [B,h,P,N]
        attn = torch.softmax(logits, dim=-1)
        masks = attn[:, :, :, 1 : 1 + R].transpose(-2, -1)  # [B,h,R,P]
        cls_map = attn[:, :, :, 0]  # patch->cls, [B,h,P]
    assert masks.shape[-1] == P
    return masks, cls_map


@torch.no_grad()
def extract_register_patch_attention(
    qkv: Tensor,
    num_heads: int,
    n_storage_tokens: int,
    scale: float,
    rope=None,
    apply_rope_fn=None,
    attn_type: str = "standard",
    slot_renorm: bool = True,
    slot_exclude_cls: bool = True,
    patch_cls_attn_type: str = "standard",
) -> Tensor:
    """Compute register->patch attention weights for visualization / MBO masks."""
    masks, _ = extract_register_attention_maps(
        qkv,
        num_heads=num_heads,
        n_storage_tokens=n_storage_tokens,
        scale=scale,
        rope=rope,
        apply_rope_fn=apply_rope_fn,
        attn_type=attn_type,
        slot_renorm=slot_renorm,
        slot_exclude_cls=slot_exclude_cls,
        patch_cls_attn_type=patch_cls_attn_type,
        direction="register_to_patch",
    )
    return masks


class PatchClsSeparateRegisterBudgetAttention(SelfAttention):
    """Self-attention with separate register budget for CLS/patch query rows.

    Register query rows use standard self-attention. CLS and patch rows use one
    softmax over CLS+patch keys and one softmax over register keys; the two
    outputs are added.
    """

    def __init__(self, *args, n_storage_tokens: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert n_storage_tokens > 0, "separate register budget requires n_storage_tokens > 0"
        self.n_storage_tokens = n_storage_tokens

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        R = self.n_storage_tokens
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        # Register query rows keep standard attention over all keys.
        out_reg = F.scaled_dot_product_attention(q[:, :, 1 : 1 + R, :], k, v)
        patch_cls_rows = compute_patch_cls_separate_register_budget(q, k, v, R, self.scale)
        x = torch.cat([patch_cls_rows[:, :, :1, :], out_reg, patch_cls_rows[:, :, 1:, :]], dim=-2)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class RegisterSlotAttention(SelfAttention):
    """Self-attention where register tokens use slot-attention-style competition.

    Non-register tokens (cls + patches) behave exactly as in :class:`SelfAttention`
    (standard softmax over all keys, including registers). Register query rows
    instead:
      * cannot attend to any register token (incl. themselves),
      * exclude the cls token by default,
      * softmax their logits across the register/query dimension (competition).

    ``slot_mode`` controls aggregation after the competition softmax:
      * "slot" (default): renormalize over keys -> weighted mean (slot attention).
      * "literal": ``out = A @ v`` with no second normalization.

    ``patch_cls_attn_type`` optionally changes CLS/patch rows to give registers
    their own independent key-softmax budget.
    """

    def __init__(
        self,
        *args,
        n_storage_tokens: int = 0,
        slot_mode: str = "slot",
        exclude_cls: bool = True,
        patch_cls_attn_type: str = "standard",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert n_storage_tokens > 0, "RegisterSlotAttention requires n_storage_tokens > 0"
        assert slot_mode in ("slot", "literal"), f"unknown slot_mode={slot_mode}"
        assert patch_cls_attn_type in ("standard", "separate_register_budget"), (
            f"unknown patch_cls_attn_type={patch_cls_attn_type}"
        )
        self.n_storage_tokens = n_storage_tokens
        self.slot_mode = slot_mode
        self.slot_renorm = slot_mode == "slot"
        self.exclude_cls = exclude_cls
        self.patch_cls_attn_type = patch_cls_attn_type

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        R = self.n_storage_tokens
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        if self.patch_cls_attn_type == "separate_register_budget":
            patch_cls_rows = compute_patch_cls_separate_register_budget(q, k, v, R, self.scale)
        else:
            # CLS/patch rows use standard attention over all keys (incl. registers).
            q_nonreg = torch.cat([q[:, :, :1, :], q[:, :, 1 + R :, :]], dim=-2)
            patch_cls_rows = F.scaled_dot_product_attention(q_nonreg, k, v, scale=self.scale)
        out_reg = compute_register_competition(q, k, v, R, self.scale, self.slot_renorm, self.exclude_cls)
        x = torch.cat([patch_cls_rows[:, :, :1, :], out_reg, patch_cls_rows[:, :, 1:, :]], dim=-2)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
