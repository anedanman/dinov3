# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Stateful in-training evaluator for register-token attention.

Owns a plain eval backbone (synced from the EMA teacher before each run), a fixed
set of visualization images, and runs the register-attention visualization and
COCO MBO evaluation on a step schedule. Designed to be called from the main
training loop on the main process only.
"""

import logging

import torch

from .attention_viz import load_viz_images, render_register_attention
from .backbone import build_eval_backbone, sync_eval_backbone
from .mbo import compute_coco_mbo

logger = logging.getLogger("dinov3")


class RegisterEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.viz_enabled = cfg.register_viz.enabled and cfg.student.n_storage_tokens > 0
        self.mbo_enabled = cfg.mbo.enabled and cfg.student.n_storage_tokens > 0
        self._eval_backbone = None
        self._viz_images = None
        self._viz_display = None

    def _backbone(self):
        if self._eval_backbone is None:
            self._eval_backbone = build_eval_backbone(self.cfg)
        return self._eval_backbone

    def _ensure_viz_images(self):
        if self._viz_images is None:
            self._viz_images, self._viz_display = load_viz_images(self.cfg, self.cfg.register_viz.num_images)

    def should_run_viz(self, step: int) -> bool:
        return self.viz_enabled and (step + 1) % self.cfg.register_viz.every_n_steps == 0

    def should_run_mbo(self, step: int) -> bool:
        return self.mbo_enabled and (step + 1) % self.cfg.mbo.every_n_steps == 0

    @torch.no_grad()
    def sync(self, model):
        """Refresh eval backbone weights from the EMA teacher backbone."""
        ema_backbone = model.model_ema["backbone"]
        sync_eval_backbone(self._backbone(), ema_backbone)

    @torch.no_grad()
    def run_viz(self):
        self._ensure_viz_images()
        panels = render_register_attention(
            self._backbone(),
            self._viz_images,
            self._viz_display,
            layer=self.cfg.register_viz.layer,
        )
        return panels  # list of [H,W,3] uint8 arrays

    @torch.no_grad()
    def run_mbo(self):
        return compute_coco_mbo(self._backbone(), self.cfg)
