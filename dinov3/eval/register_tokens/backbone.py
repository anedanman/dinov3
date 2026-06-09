# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Build/maintain a plain (non-FSDP, non-compiled) eval backbone.

The training teacher (EMA) is wrapped by FSDP + torch.compile, which makes the
custom attention-extraction methods awkward to call. Instead we keep a plain
:class:`DinoVisionTransformer` on cuda and copy the current EMA weights into it
before each evaluation. This is cheap relative to the eval period.
"""

import logging

import torch
from torch import nn

from dinov3.models import vision_transformer as vits

logger = logging.getLogger("dinov3")


def _vit_kwargs_from_cfg(cfg):
    s = cfg.student
    img_size = cfg.crops.global_crops_size
    if not isinstance(img_size, int):
        img_size = max(img_size)
    return dict(
        img_size=img_size,
        patch_size=s.patch_size,
        pos_embed_rope_base=s.pos_embed_rope_base,
        pos_embed_rope_min_period=s.pos_embed_rope_min_period,
        pos_embed_rope_max_period=s.pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=s.pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=s.pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=s.pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=s.pos_embed_rope_rescale_coords,
        qkv_bias=s.qkv_bias,
        layerscale_init=s.layerscale,
        norm_layer=s.norm_layer,
        ffn_layer=s.ffn_layer,
        ffn_bias=s.ffn_bias,
        proj_bias=s.proj_bias,
        n_storage_tokens=s.n_storage_tokens,
        mask_k_bias=s.mask_k_bias,
        untie_cls_and_patch_norms=s.untie_cls_and_patch_norms,
        untie_global_and_local_cls_norm=s.untie_global_and_local_cls_norm,
        register_attn_type=s.get("register_attn_type", "standard"),
        slot_mode=s.get("slot_mode", "slot"),
        register_attn_exclude_cls=s.get("register_attn_exclude_cls", True),
        patch_cls_attn_type=s.get("patch_cls_attn_type", "standard"),
        slot_start_layer=s.get("slot_start_layer", 0),
        register_budget_gate=s.get("register_budget_gate", False),
        register_init=s.get("register_init", "learned"),
        register_gaussian_std_init=s.get("register_gaussian_std_init", 0.02),
    )


def build_eval_backbone(cfg, device="cuda") -> nn.Module:
    """Construct a plain ViT matching cfg.student, on ``device``."""
    kwargs = _vit_kwargs_from_cfg(cfg)
    model = vits.__dict__[cfg.student.arch](**kwargs)
    model = model.to(device)
    model.init_weights()  # materialize all params/buffers (incl. RoPE periods) before EMA copy
    model.eval()
    return model


@torch.no_grad()
def sync_eval_backbone(eval_backbone: nn.Module, ema_backbone: nn.Module) -> None:
    """Copy EMA-teacher backbone weights into the plain eval backbone.

    Handles DTensor parameters (FSDP) by materializing full tensors.
    """
    src = ema_backbone.state_dict()
    dst = eval_backbone.state_dict()
    converted = {}
    for k, v in src.items():
        if hasattr(v, "full_tensor"):  # DTensor
            v = v.full_tensor()
        converted[k] = v
    missing, unexpected = [], []
    for k in dst:
        if k in converted:
            dst[k].copy_(converted[k].to(dst[k].device, dtype=dst[k].dtype))
        else:
            missing.append(k)
    for k in converted:
        if k not in dst:
            unexpected.append(k)
    if missing:
        logger.warning(f"[eval backbone sync] missing keys (kept init): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        logger.warning(f"[eval backbone sync] unexpected src keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    eval_backbone.eval()
