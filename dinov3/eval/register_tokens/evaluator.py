# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Stateful in-training evaluator for register-token attention.

Owns a plain eval backbone (synced from the EMA teacher before each run), a fixed
set of visualization images, and runs the register-attention visualization and
COCO MBO evaluation on a step schedule. sync/run_mbo/run_diffcut are collective
(images sharded across ranks; all ranks must call); viz/diagnostics are
main-process-only.
"""

import logging

import torch

from .attention_viz import load_coco_viz_images, load_viz_images, render_patch_pca, render_register_attention
from .backbone import build_eval_backbone, sync_eval_backbone
from .diagnostics import RegisterDiagnostics
from .diffcut import compute_coco_diffcut, render_diffcut_viz
from .mbo import compute_coco_mbo

logger = logging.getLogger("dinov3")


class RegisterEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.viz_enabled = cfg.register_viz.enabled and cfg.student.n_storage_tokens > 0
        self.mbo_enabled = cfg.mbo.enabled and cfg.student.n_storage_tokens > 0
        # DiffCut runs on patch features only, so it does not require registers.
        diffcut_cfg = cfg.get("diffcut", None)
        self.diffcut_enabled = bool(diffcut_cfg and diffcut_cfg.enabled)
        diag_cfg = cfg.get("register_diagnostics", None)
        self.diag_enabled = bool(diag_cfg and diag_cfg.enabled and cfg.student.n_storage_tokens > 0)
        self._diagnostics = RegisterDiagnostics(cfg) if self.diag_enabled else None
        self._eval_backbone = None
        self._viz_images = None
        self._viz_display = None
        self._pca_images = None
        self._pca_display = None
        self._pca_hires_images = None
        self._pca_hires_display = None
        self._coco_viz_images = None
        self._coco_viz_display = None
        self._coco_viz_attempted = False

    def _backbone(self):
        if self._eval_backbone is None:
            self._eval_backbone = build_eval_backbone(self.cfg)
        return self._eval_backbone

    def _ensure_viz_images(self):
        if self._viz_images is None:
            self._viz_images, self._viz_display = load_viz_images(self.cfg, self.cfg.register_viz.num_images)

    def _ensure_pca_viz_images(self):
        if self._pca_images is not None:
            return
        n = int(self.cfg.register_viz.get("patch_pca_num_images", 16) or 0)
        n = n or self.cfg.register_viz.num_images
        self._pca_images, self._pca_display = load_viz_images(self.cfg, n)
        hires_size = int(self.cfg.register_viz.get("patch_pca_hires_size", 1024) or 0)
        if hires_size > 0:
            # Same deterministic image set, higher resolution.
            self._pca_hires_images, self._pca_hires_display = load_viz_images(self.cfg, n, image_size=hires_size)

    def _ensure_coco_viz_images(self):
        if self._coco_viz_attempted:
            return
        self._coco_viz_attempted = True
        num_images = self.cfg.register_viz.get("coco_num_images", self.cfg.register_viz.num_images)
        if num_images is None:
            num_images = self.cfg.register_viz.num_images
        self._coco_viz_images, self._coco_viz_display = load_coco_viz_images(self.cfg, int(num_images))

    def should_run_viz(self, step: int) -> bool:
        return self.viz_enabled and (step + 1) % self.cfg.register_viz.every_n_steps == 0

    def should_run_mbo(self, step: int) -> bool:
        return self.mbo_enabled and (step + 1) % self.cfg.mbo.every_n_steps == 0

    def should_run_diffcut(self, step: int) -> bool:
        return self.diffcut_enabled and (step + 1) % self.cfg.diffcut.every_n_steps == 0

    def should_run_diagnostics(self, step: int) -> bool:
        return self.diag_enabled and (step + 1) % self.cfg.register_diagnostics.every_n_steps == 0

    @torch.no_grad()
    def sync(self, model):
        """Refresh eval backbone weights from the EMA teacher backbone."""
        ema_backbone = model.model_ema["backbone"]
        sync_eval_backbone(self._backbone(), ema_backbone)

    @torch.no_grad()
    def run_viz(self):
        self._ensure_viz_images()
        out = {}
        out["register_attention"] = render_register_attention(
            self._backbone(),
            self._viz_images,
            self._viz_display,
            layer=self.cfg.register_viz.layer,
        )
        if self.cfg.register_viz.get("patch_pca", True):
            self._ensure_pca_viz_images()
            out["patch_pca"] = render_patch_pca(self._backbone(), self._pca_images, self._pca_display)
            if self._pca_hires_images is not None:
                out["patch_pca_hires"] = render_patch_pca(
                    self._backbone(),
                    self._pca_hires_images,
                    self._pca_hires_display,
                    max_forward_batch=4,
                )
        if self.cfg.register_viz.get("coco", True):
            self._ensure_coco_viz_images()
            if self._coco_viz_images is not None:
                if self.cfg.register_viz.get("patch_pca", True):
                    out["patch_pca_coco"] = render_patch_pca(
                        self._backbone(), self._coco_viz_images, self._coco_viz_display
                    )
                specs = [
                    ("last", int(self.cfg.register_viz.get("layer", -1)), "select", "mean"),
                    ("penultimate", -2, "select", "mean"),
                    ("all_layers", -1, "mean", "mean"),
                    ("second_half_layers", -1, "second_half", "mean"),
                    ("last_per_head", int(self.cfg.register_viz.get("layer", -1)), "select", "none"),
                ]
                for name, layer, layer_reduce, head_reduce in specs:
                    out[f"register_attention_coco_{name}"] = render_register_attention(
                        self._backbone(),
                        self._coco_viz_images,
                        self._coco_viz_display,
                        layer=layer,
                        layer_reduce=layer_reduce,
                        head_reduce=head_reduce,
                    )
        return out  # dict of key -> list of [H,W,3] uint8 arrays

    @torch.no_grad()
    def run_mbo(self):
        return compute_coco_mbo(self._backbone(), self.cfg)

    @torch.no_grad()
    def run_diffcut(self):
        return compute_coco_diffcut(self._backbone(), self.cfg)

    @torch.no_grad()
    def run_diffcut_viz(self):
        return render_diffcut_viz(self._backbone(), self.cfg)

    @torch.no_grad()
    def run_diagnostics(self):
        return self._diagnostics.run(self._backbone())
