# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Thin Weights & Biases wrapper (main-process only, no-op otherwise)."""

import logging
import os

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


def log_images(run, panels, step: int, key: str = "register_attention", captions=None):
    """Log a list of HxWx3 uint8 numpy arrays as a wandb image gallery."""
    if run is None or not panels:
        return
    import wandb

    if captions is None:
        captions = [f"img_{i}" for i in range(len(panels))]
    run.log({key: [wandb.Image(p, caption=c) for p, c in zip(panels, captions)]}, step=step)


def finish(run):
    if run is not None:
        run.finish()
