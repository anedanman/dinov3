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


def _to_capped_image(arr):
    """HxWx3 uint8 array -> PIL image, downscaled to the wandb pixel cap."""
    from PIL import Image

    max_pixels = int(os.environ.get("DINOV3_WANDB_MAX_IMAGE_PIXELS", "4000000"))
    image = Image.fromarray(arr)
    if max_pixels > 0 and arr.shape[0] * arr.shape[1] > max_pixels:
        scale = math.sqrt(max_pixels / float(arr.shape[0] * arr.shape[1]))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(
            (max(1, int(arr.shape[1] * scale)), max(1, int(arr.shape[0] * scale))),
            resampling,
        )
    return image


def log_images(run, panels, step: int, key: str = "register_attention", caption=None, stack: bool = True):
    """Log a list of HxWx3 uint8 numpy arrays.

    ``stack=True`` concatenates panels vertically into one image (then applies
    the pixel cap to the whole grid). ``stack=False`` logs each panel as its
    own image under the same key, so the cap applies per panel — use this for
    high-resolution panels that would be crushed by downscaling a stacked grid.
    """
    if run is None or not panels:
        return
    import numpy as np
    import wandb

    if stack:
        grid = np.concatenate(panels, axis=0)
        images = [wandb.Image(_to_capped_image(grid), caption=caption or f"{len(panels)} images")]
    else:
        images = [wandb.Image(_to_capped_image(p), caption=f"{caption or key} {i}") for i, p in enumerate(panels)]
    run.log({key: images if len(images) > 1 else images[0]}, step=step)


def log_checkpoint_artifact(run, ckpt_path, step: int):
    """Upload a milestone checkpoint directory as a wandb artifact (async upload)."""
    if run is None:
        return
    import wandb

    name = f"{run.name}-ckpt".replace("/", "-")
    artifact = wandb.Artifact(name=name, type="checkpoint", metadata={"step": step})
    artifact.add_dir(str(ckpt_path))
    run.log_artifact(artifact, aliases=["latest", f"step-{step}"])
    logger.info(f"wandb: uploading checkpoint artifact {name} step={step} from {ckpt_path}")


def finish(run):
    if run is not None:
        run.finish()
