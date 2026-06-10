# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Cheap quantitative register-token diagnostics.

Runs every few hundred steps on a small fixed image batch and produces scalar
indicators of how the register/slot mechanism behaves, long before the slow
MBO/probe evals move:

* slot usage: per-register patch share, usage entropy, active-slot count and
  per-patch competition softness (collapse indicators);
* register feature stats: norms and pairwise register cosine (redundancy);
* patch-token outliers: high-norm patch fraction and norm ratios, monitoring
  whether registers keep absorbing outlier tokens (the original register
  motivation from "Vision Transformers Need Registers");
* cross-crop agreement: Hungarian-matched register cosine between two fixed
  views of the same images (what the register-consistency loss optimizes);
* register-budget gates: per-layer learnable gate values, when enabled.

All metrics are computed on the EMA-teacher eval backbone.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from dinov3.loss.register_consistency_loss import hungarian_match

logger = logging.getLogger("dinov3")

_EPS = 1e-8


def _entropy(p: Tensor, dim: int) -> Tensor:
    """Shannon entropy along `dim` of a (re)normalized distribution."""
    p = p / (p.sum(dim=dim, keepdim=True) + _EPS)
    return -(p * (p + _EPS).log()).sum(dim=dim)


@torch.no_grad()
def slot_usage_metrics(masks: Tensor) -> Dict[str, float]:
    """Slot-usage statistics from register->patch attention masks.

    Args:
        masks: [B, R, P] head-averaged register->patch masks (any positive
            weighting; columns are renormalized over registers internally).
    """
    B, R, P = masks.shape
    p_reg = masks / (masks.sum(dim=1, keepdim=True) + _EPS)  # per-patch distribution over registers
    assign = p_reg.argmax(dim=1)  # [B, P]
    shares = torch.zeros(B, R, device=masks.device)
    shares.scatter_add_(1, assign, torch.full_like(p_reg[:, 0], 1.0 / P))
    log_r = math.log(R) if R > 1 else 1.0
    return {
        "slot_share_max": shares.max(dim=1).values.mean().item(),
        "slot_share_min": shares.min(dim=1).values.mean().item(),
        "slot_usage_entropy": (_entropy(shares, dim=1) / log_r).mean().item(),
        "active_slots": (shares > 0.5 / R).float().sum(dim=1).mean().item(),
        "patch_assign_entropy": (_entropy(p_reg, dim=1) / log_r).mean().item(),
        "slot_spatial_entropy": (_entropy(masks, dim=2) / math.log(P)).mean().item(),
    }


@torch.no_grad()
def outlier_metrics(prenorm: Tensor, n_storage_tokens: int, outlier_ratio: float = 2.0) -> Dict[str, float]:
    """High-norm patch-token statistics from pre-norm last-layer tokens.

    Args:
        prenorm: [B, 1 + R + P, D] pre-norm output tokens.
    """
    R = n_storage_tokens
    cls_norm = prenorm[:, 0].norm(dim=-1)  # [B]
    reg_norm = prenorm[:, 1 : 1 + R].norm(dim=-1)  # [B, R]
    patch_norm = prenorm[:, 1 + R :].norm(dim=-1)  # [B, P]
    med = patch_norm.median(dim=1, keepdim=True).values  # [B, 1]
    rel = patch_norm / (med + _EPS)
    return {
        "patch_norm_outlier_frac": (rel > outlier_ratio).float().mean().item(),
        "patch_norm_outlier_frac_3x": (rel > 3.0).float().mean().item(),
        "patch_norm_max_over_median": rel.max(dim=1).values.mean().item(),
        "patch_norm_p99_over_median": rel.quantile(0.99, dim=1).mean().item(),
        "register_norm_over_patch_median": (reg_norm / (med + _EPS)).mean().item(),
        "cls_norm_over_patch_median": (cls_norm / (med.squeeze(1) + _EPS)).mean().item(),
    }


@torch.no_grad()
def register_feature_metrics(reg: Tensor) -> Dict[str, float]:
    """Register feature statistics. reg: [B, R, D] (post-norm register tokens)."""
    B, R, _ = reg.shape
    norms = reg.norm(dim=-1)  # [B, R]
    out = {"reg_norm_mean": norms.mean().item()}
    if R > 1:
        z = F.normalize(reg.float(), dim=-1)
        cos = torch.einsum("brd,bsd->brs", z, z)  # [B, R, R]
        off_diag = cos.sum(dim=(1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(dim=1)
        out["reg_pairwise_cos"] = (off_diag / (R * (R - 1))).mean().item()
        # Same statistic on mean-subtracted residuals: registers share one
        # dominant direction, so the raw cosine saturates near 1 while the
        # residuals carry the actual slot differentiation.
        resid = reg.float() - reg.float().mean(dim=1, keepdim=True)
        zr = F.normalize(resid, dim=-1)
        cos_r = torch.einsum("brd,bsd->brs", zr, zr)
        off_diag_r = cos_r.sum(dim=(1, 2)) - cos_r.diagonal(dim1=1, dim2=2).sum(dim=1)
        out["reg_resid_pairwise_cos"] = (off_diag_r / (R * (R - 1))).mean().item()
        out["reg_resid_norm_frac"] = (resid.norm(dim=-1) / (norms + 1e-6)).mean().item()
    return out


@torch.no_grad()
def cross_crop_agreement(reg_a: Tensor, reg_b: Tensor) -> Dict[str, float]:
    """Hungarian-matched register cosine across two views. reg_*: [B, R, D].

    Reported both on raw registers and on mean-subtracted residuals; the raw
    cosine saturates once the registers collapse onto a shared direction.
    """
    out = {}
    for prefix, xa, xb in (
        ("xcrop", reg_a.float(), reg_b.float()),
        (
            "xcrop_resid",
            reg_a.float() - reg_a.float().mean(dim=1, keepdim=True),
            reg_b.float() - reg_b.float().mean(dim=1, keepdim=True),
        ),
    ):
        a = F.normalize(xa, dim=-1)
        b = F.normalize(xb, dim=-1)
        sim = torch.einsum("brd,bsd->brs", a, b)  # [B, R, R]
        cols = hungarian_match(sim)  # [B, R]
        matched = sim.gather(2, cols.unsqueeze(-1)).squeeze(-1)  # [B, R]
        identity = torch.arange(sim.shape[1], device=cols.device).expand_as(cols)
        out[f"{prefix}_matched_cos"] = matched.mean().item()
        out[f"{prefix}_identity_match_frac"] = (cols == identity).float().mean().item()
    return out


@torch.no_grad()
def gate_metrics(backbone) -> Dict[str, float]:
    """Per-layer register-budget gate values (only when the gate is enabled)."""
    out = {}
    values = []
    for i, blk in enumerate(getattr(backbone, "blocks", [])):
        gate = getattr(getattr(blk, "attn", None), "reg_budget_gate", None)
        if gate is None:
            continue
        g = gate.detach().float()
        out[f"register_gate/layer_{i:02d}"] = g.mean().item()
        out[f"register_gate/layer_{i:02d}_min"] = g.min().item()
        out[f"register_gate/layer_{i:02d}_max"] = g.max().item()
        values.append(g.mean())
    if values:
        out["register_gate/mean"] = torch.stack(values).mean().item()
    return out


@torch.no_grad()
def compute_register_diagnostics(
    backbone,
    images: Tensor,
    views: Optional[Tuple[Tensor, Tensor]] = None,
    outlier_ratio: float = 2.0,
    device: str = "cuda",
) -> Dict[str, float]:
    """All register diagnostics for one fixed batch. Returns wandb-ready keys."""
    images = images.to(device, non_blocking=True)
    out = backbone.forward_features(images)
    masks, _ = backbone.get_register_attention_maps(
        images, layer=-1, direction="register_to_patch", layer_reduce="select", head_reduce="mean"
    )  # [B, R, H, W]

    metrics: Dict[str, float] = {}
    metrics.update(slot_usage_metrics(masks.flatten(2)))
    metrics.update(register_feature_metrics(out["x_storage_tokens"]))
    metrics.update(outlier_metrics(out["x_prenorm"], backbone.n_storage_tokens, outlier_ratio))

    if views is not None:
        reg_a = backbone.forward_features(views[0].to(device, non_blocking=True))["x_storage_tokens"]
        reg_b = backbone.forward_features(views[1].to(device, non_blocking=True))["x_storage_tokens"]
        metrics.update(cross_crop_agreement(reg_a, reg_b))

    metrics = {f"register_diag/{k}": v for k, v in metrics.items()}
    metrics.update(gate_metrics(backbone))
    return metrics


def make_fixed_views(images: Tensor, seed: int = 0, scale=(0.4, 1.0)) -> Tuple[Tensor, Tensor]:
    """Two deterministic random-resized-crop views per image (geometric only)."""
    from torchvision.transforms.v2 import RandomResizedCrop
    from torchvision.transforms.v2 import functional as TF

    S = images.shape[-1]
    views = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(2):
            crops = []
            for img in images:
                i, j, h, w = RandomResizedCrop.get_params(img, scale=scale, ratio=(3.0 / 4.0, 4.0 / 3.0))
                crops.append(TF.resized_crop(img, i, j, h, w, [S, S], antialias=True))
            views.append(torch.stack(crops, dim=0))
    return views[0], views[1]


class RegisterDiagnostics:
    """Holds the fixed diagnostic batch; called from RegisterEvaluator."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._images = None
        self._views = None

    def _ensure_images(self):
        if self._images is not None:
            return
        from .attention_viz import load_viz_images

        dcfg = self.cfg.register_diagnostics
        self._images, _ = load_viz_images(
            self.cfg,
            num_images=int(dcfg.num_images),
            image_size=int(dcfg.get("image_size", self.cfg.register_viz.image_size)),
        )
        if dcfg.get("crop_pairs", True):
            self._views = make_fixed_views(self._images)

    @torch.no_grad()
    def run(self, backbone) -> Dict[str, float]:
        self._ensure_images()
        return compute_register_diagnostics(
            backbone,
            self._images,
            views=self._views,
            outlier_ratio=float(self.cfg.register_diagnostics.get("outlier_ratio", 2.0)),
        )
