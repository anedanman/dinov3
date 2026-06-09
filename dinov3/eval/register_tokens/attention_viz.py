# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Visualization of register-token attention as object-centric maps."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("dinov3")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _empty_target_transform(_):
    return ()


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


def _normalize_map(m: torch.Tensor) -> np.ndarray:
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.cpu().numpy()


def _heat_overlay(base: np.ndarray, m: torch.Tensor, alpha: float = 0.5, normalized: bool = False) -> np.ndarray:
    heat = _colorize(m.clamp(0, 1).cpu().numpy() if normalized else _normalize_map(m))
    return ((1.0 - alpha) * base + alpha * heat).astype(np.uint8)


def _label_strip(labels, widths, height: int = 18) -> np.ndarray:
    """One-row header of column labels, matching a list of column widths."""
    from PIL import Image, ImageDraw

    total = int(sum(widths))
    img = Image.new("RGB", (total, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = 0
    for label, w in zip(labels, widths):
        if label:
            draw.text((x + 3, 2), label, fill=(0, 0, 0))
        x += int(w)
    return np.asarray(img, dtype=np.uint8)


def _distinct_colors(n: int) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    cmap = cm.get_cmap("tab20" if n <= 20 else "gist_ncar")
    return (np.stack([cmap(i / max(n, 1))[:3] for i in range(n)]) * 255).astype(np.uint8)


def _separator(height: int, width: int = 4) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


@torch.no_grad()
def load_viz_images(cfg, num_images: int, image_size: Optional[int] = None) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Load a fixed set of images for visualization.

    Returns (normalized [N,3,S,S] tensor, list of display uint8 images).
    """
    from dinov3.data import make_dataset
    from dinov3.data.transforms import make_eval_transform

    size = int(image_size) if image_size is not None else cfg.register_viz.image_size
    mean, std = cfg.crops.rgb_mean, cfg.crops.rgb_std
    transform = make_eval_transform(
        resize_size=size, crop_size=size, resize_square=True, mean=mean, std=std
    )
    dataset_str = _expand(cfg.register_viz.dataset)
    dataset = make_dataset(dataset_str=dataset_str, transform=transform, target_transform=_empty_target_transform)
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


@torch.no_grad()
def load_coco_viz_images(cfg, num_images: int) -> Tuple[Optional[torch.Tensor], List[np.ndarray]]:
    """Load a deterministic set of COCO images for validation visualizations."""
    import glob
    import os

    from dinov3.data.transforms import make_eval_transform

    if num_images <= 0:
        return None, []

    coco_root = os.path.expanduser(cfg.mbo.coco_root)
    split = cfg.mbo.split
    img_dir = os.path.join(coco_root, split)
    if not os.path.isdir(img_dir):
        logger.warning(f"COCO image directory not found at {img_dir}; skipping COCO register viz.")
        return None, []

    paths = []
    ann_path = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    if os.path.exists(ann_path):
        try:
            from pycocotools.coco import COCO

            coco = COCO(ann_path)
            img_ids = sorted(coco.getImgIds())
            paths = [os.path.join(img_dir, info["file_name"]) for info in coco.loadImgs(img_ids)]
        except Exception as e:
            logger.warning(f"Could not load COCO annotations for viz ({e}); falling back to image files.")
    if not paths:
        paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        logger.warning(f"No COCO images found under {img_dir}; skipping COCO register viz.")
        return None, []

    size = int(cfg.mbo.get("image_size", cfg.register_viz.image_size))
    mean, std = cfg.crops.rgb_mean, cfg.crops.rgb_std
    transform = make_eval_transform(resize_size=size, crop_size=size, resize_square=True, mean=mean, std=std)
    n = min(num_images, len(paths))
    idxs = np.linspace(0, len(paths) - 1, n).astype(int)
    imgs = []
    for idx in idxs:
        try:
            imgs.append(transform(Image.open(paths[int(idx)]).convert("RGB")))
        except Exception:
            continue
    if not imgs:
        return None, []
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


def _assignment_overlay(base: np.ndarray, attn: torch.Tensor, colors: np.ndarray) -> np.ndarray:
    """Argmax-register segmentation, alpha-modulated by assignment confidence."""
    p = attn / (attn.sum(dim=0, keepdim=True) + 1e-8)  # per-pixel distribution over registers
    conf, assign = p.max(dim=0)
    seg = colors[assign.cpu().numpy()]
    # Confident pixels show the register color strongly; ambiguous ones stay closer to the input.
    alpha = (0.25 + 0.5 * conf).cpu().numpy()[..., None]
    return ((1.0 - alpha) * base + alpha * seg).astype(np.uint8)


def _render_mean_direction(base: np.ndarray, attn: torch.Tensor, cls_map: torch.Tensor, colors: np.ndarray) -> List[np.ndarray]:
    """Render per-register heatmaps, one joint mask, and one CLS map."""
    # Normalize register heatmaps jointly (shared max) so relative register
    # magnitudes stay comparable within the image.
    shared = attn / (attn.max() + 1e-8)
    reg_imgs = [_heat_overlay(base, shared[r], normalized=True) for r in range(attn.shape[0])]
    seg_overlay = _assignment_overlay(base, attn, colors)
    cls_overlay = _heat_overlay(base, cls_map)
    return reg_imgs + [seg_overlay, cls_overlay]


def _render_per_head_direction(
    base: np.ndarray,
    attn: torch.Tensor,
    cls_map: torch.Tensor,
    colors: np.ndarray,
) -> List[np.ndarray]:
    """Render one register-assignment mask and one CLS map per head."""
    head_masks = []
    cls_maps = []
    for h in range(attn.shape[0]):
        head_masks.append(_assignment_overlay(base, attn[h], colors))
        cls_maps.append(_heat_overlay(base, cls_map[h]))
    return head_masks + cls_maps


@torch.no_grad()
def render_register_attention(
    eval_backbone,
    images: torch.Tensor,
    display: List[np.ndarray],
    layer: int = -1,
    layer_reduce: str = "select",
    head_reduce: str = "mean",
    device: str = "cuda",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> List[np.ndarray]:
    """Produce one composite panel per image.

    Mean-head panels are horizontal strips:
    [original | r2p reg_0..reg_R-1 | r2p assignment | r2p cls |
     p2r reg_0..reg_R-1 | p2r assignment | p2r cls].

    Per-head panels replace individual register heatmaps with one assignment
    mask per head, followed by one CLS map per head.
    """
    images = images.to(device)
    S = images.shape[-1]
    directions = ("register_to_patch", "patch_to_register")
    attn_up = {}
    cls_up = {}
    R = eval_backbone.n_storage_tokens
    for direction in directions:
        attn, cls_map = eval_backbone.get_register_attention_maps(
            images,
            layer=layer,
            direction=direction,
            layer_reduce=layer_reduce,
            head_reduce=head_reduce,
        )
        if head_reduce == "mean":
            attn_up[direction] = F.interpolate(attn, size=(S, S), mode="bilinear", align_corners=False)
            cls_up[direction] = F.interpolate(
                cls_map[:, None], size=(S, S), mode="bilinear", align_corners=False
            )[:, 0]
        else:
            N, heads, R, h, w = attn.shape
            attn_up[direction] = F.interpolate(
                attn.reshape(N * heads, R, h, w), size=(S, S), mode="bilinear", align_corners=False
            ).reshape(N, heads, R, S, S)
            cls_up[direction] = F.interpolate(
                cls_map.reshape(N * heads, 1, h, w), size=(S, S), mode="bilinear", align_corners=False
            ).reshape(N, heads, S, S)

    N = images.shape[0]
    colors = _distinct_colors(R)

    # Column labels matching the panel layout, rendered once as a header strip.
    labels = ["input"]
    widths = [S]
    for direction in directions:
        tag = "r2p" if direction == "register_to_patch" else "p2r"
        labels.append("")
        widths.append(4)  # separator
        if head_reduce == "mean":
            labels += [f"{tag} reg{r}" for r in range(R)] + [f"{tag} assign", f"{tag} cls"]
            widths += [S] * (R + 2)
        else:
            heads = attn_up[direction].shape[1]
            labels += [f"{tag} head{h}" for h in range(heads)] + [f"{tag} cls h{h}" for h in range(heads)]
            widths += [S] * (2 * heads)

    panels = [_label_strip(labels, widths)]
    for i in range(N):
        base = display[i]
        columns = [base]
        for direction in directions:
            columns.append(_separator(base.shape[0]))
            if head_reduce == "mean":
                columns.extend(_render_mean_direction(base, attn_up[direction][i], cls_up[direction][i], colors))
            else:
                columns.extend(_render_per_head_direction(base, attn_up[direction][i], cls_up[direction][i], colors))
        strip = np.concatenate(columns, axis=1)
        panels.append(strip)
    return panels
