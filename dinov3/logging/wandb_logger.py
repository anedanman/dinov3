# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Thin Weights & Biases wrapper (main-process only, no-op otherwise)."""

import logging
import os
import math

from omegaconf import OmegaConf

import dinov3.distributed as distributed

logger = logging.getLogger("dinov3")


def init_wandb(cfg):
    """Initialize a wandb run on the main process. Returns the run or None."""
    wcfg = cfg.train.get("wandb", None)
    if wcfg is None or not wcfg.enabled:
        return None
    if not distributed.is_main_process():
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; disabling wandb logging.")
        return None

    name = wcfg.name or os.path.basename(os.path.normpath(cfg.train.output_dir))
    run = wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        name=name,
        group=wcfg.group,
        tags=list(wcfg.tags) if wcfg.tags else None,
        mode=wcfg.mode,
        dir=cfg.train.output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info(f"wandb initialized: project={wcfg.project} name={name} mode={wcfg.mode}")
    return run


def log_scalars(run, metrics: dict, step: int):
    if run is None:
        return
    run.log({k: v for k, v in metrics.items()}, step=step)


def log_images(run, panels, step: int, key: str = "register_attention", caption=None):
    """Stack a list of HxWx3 uint8 numpy arrays vertically and log as one image."""
    if run is None or not panels:
        return
    import numpy as np
    from PIL import Image
    import wandb

    grid = np.concatenate(panels, axis=0)  # stack panels vertically -> single image
    max_pixels = int(os.environ.get("DINOV3_WANDB_MAX_IMAGE_PIXELS", "4000000"))
    image = Image.fromarray(grid)
    if max_pixels > 0 and grid.shape[0] * grid.shape[1] > max_pixels:
        scale = math.sqrt(max_pixels / float(grid.shape[0] * grid.shape[1]))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(
            (max(1, int(grid.shape[1] * scale)), max(1, int(grid.shape[0] * scale))),
            resampling,
        )
    if caption is None:
        caption = f"{len(panels)} images"
    run.log({key: wandb.Image(image, caption=caption)}, step=step)
    image.close()


def finish(run):
    if run is not None:
        run.finish()
