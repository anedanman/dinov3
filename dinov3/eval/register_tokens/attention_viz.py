# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Visualization of register-token attention as object-centric maps."""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("dinov3")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalize(img: torch.Tensor, mean, std) -> np.ndarray:
    """[3,H,W] normalized tensor -> [H,W,3] uint8."""
    mean = torch.tensor(mean, device=img.device).view(3, 1, 1)
    std = torch.tensor(std, device=img.device).view(3, 1, 1)
    img = (img * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _colorize(attn: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """[H,W] in [0,1] -> [H,W,3] uint8 heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)
    return (cmap(attn)[:, :, :3] * 255).astype(np.uint8)


def _distinct_colors(n: int) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    cmap = cm.get_cmap("tab20" if n <= 20 else "gist_ncar")
    return (np.stack([cmap(i / max(n, 1))[:3] for i in range(n)]) * 255).astype(np.uint8)


@torch.no_grad()
def load_viz_images(cfg, num_images: int) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Load a fixed set of images for visualization.

    Returns (normalized [N,3,S,S] tensor, list of display uint8 images).
    """
    from dinov3.data import make_dataset
    from dinov3.data.transforms import make_eval_transform

    size = cfg.register_viz.image_size
    mean, std = cfg.crops.rgb_mean, cfg.crops.rgb_std
    transform = make_eval_transform(
        resize_size=size, crop_size=size, resize_square=True, mean=mean, std=std
    )
    dataset_str = _expand(cfg.register_viz.dataset)
    dataset = make_dataset(dataset_str=dataset_str, transform=transform, target_transform=lambda _: ())
    # Deterministic, evenly spaced indices for stable tracking across training.
    n = min(num_images, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, n).astype(int)
    imgs = []
    for i in idxs:
        img, _ = dataset[int(i)]
        imgs.append(img)
    batch = torch.stack(imgs, dim=0)
    display = [_denormalize(img, mean, std) for img in batch]
    return batch, display


def _expand(s: str) -> str:
    import os

    # Expand ~ inside the root=... token of a dataset string.
    parts = s.split(":")
    out = []
    for p in parts:
        if p.startswith("root="):
            out.append("root=" + os.path.expanduser(p[len("root="):]))
        else:
            out.append(p)
    return ":".join(out)


@torch.no_grad()
def render_register_attention(
    eval_backbone,
    images: torch.Tensor,
    display: List[np.ndarray],
    layer: int = -1,
    device: str = "cuda",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> List[np.ndarray]:
    """Produce one composite panel per image.

    Each panel is a horizontal strip: [original | reg_0 overlay | ... | reg_R-1 |
    argmax assignment]. The argmax panel colors each pixel by the register that
    attends to it most (the object-centric grouping).
    """
    images = images.to(device)
    attn = eval_backbone.get_register_patch_attention(images, layer=layer)  # [N,R,h,w]
    N, R, h, w = attn.shape
    S = images.shape[-1]
    # Upsample attention to image resolution.
    attn_up = F.interpolate(attn, size=(S, S), mode="bilinear", align_corners=False)  # [N,R,S,S]
    colors = _distinct_colors(R)

    panels = []
    for i in range(N):
        base = display[i]
        a = attn_up[i]  # [R,S,S]
        # Per-register normalized heatmaps overlaid on the image.
        reg_imgs = []
        for r in range(R):
            m = a[r]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            heat = _colorize(m.cpu().numpy())
            overlay = (0.5 * base + 0.5 * heat).astype(np.uint8)
            reg_imgs.append(overlay)
        # Argmax assignment over registers -> object-centric segmentation.
        assign = a.argmax(dim=0).cpu().numpy()  # [S,S]
        seg = colors[assign]
        seg_overlay = (0.45 * base + 0.55 * seg).astype(np.uint8)
        strip = np.concatenate([base] + reg_imgs + [seg_overlay], axis=1)
        panels.append(strip)
    return panels
