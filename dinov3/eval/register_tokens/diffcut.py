# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""DiffCut zero-shot segmentation eval on COCO.

Reimplementation of the recursive normalized-cut segmentation from
"DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features
through Recursive Normalized Cut" (Couairon et al., NeurIPS 2024,
https://github.com/PaulCouairon/DiffCut), applied to DINOv3 patch features:

  1. affinity = min-max-normalized patch cosine similarity, raised to ``alpha``;
  2. recursive Shi-Malik NCut bipartition (second-smallest generalized
     eigenvector, best of 100 thresholds), recursing while ncut < ``tau``;
  3. pixels assigned to the nearest cluster mean embedding on bilinearly
     upsampled features.

The NCut graph is average-pooled down to at most ``max_graph_nodes`` patches
(DiffCut's own graph is 32x32 = 1024 nodes at 1024px input); the final pixel
assignment always uses the full-resolution patch features.

Metrics follow the repo's MBO conventions (per-GT-mask best IoU over predicted
segments, instance and semantic variants) so they are directly comparable with
the register-attention MBO numbers.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .mbo import _best_overlaps, _eval_transform, _rank_world, _resize_gt

logger = logging.getLogger("dinov3")


def _second_smallest_eigenvector(A: torch.Tensor, d: torch.Tensor) -> torch.Tensor | None:
    """Second-smallest generalized eigenvector of (D - A) x = lambda D x."""
    d_inv_sqrt = d.clamp_min(1e-12) ** -0.5
    L_sym = d_inv_sqrt[:, None] * (torch.diag(d) - A) * d_inv_sqrt[None, :]
    try:
        _, eigenvectors = torch.linalg.eigh(L_sym)
    except torch.linalg.LinAlgError:
        return None
    vec = d_inv_sqrt * eigenvectors[:, 1]
    # Deterministic sign (eigenvector sign is arbitrary).
    return vec * torch.sign(vec[torch.argmax(torch.abs(vec))])


def _best_ncut_bipartition(y: torch.Tensor, A: torch.Tensor, d: torch.Tensor, k: int = 100):
    """Search k thresholds on the eigenvector, return (bipartition bool [n], ncut value)."""
    thresholds = torch.linspace(float(y.min()) * 0.99, float(y.max()) * 0.99, k, device=y.device)
    parts = y[None, :] > thresholds[:, None]  # [k, n]
    x = 2.0 * parts.float() - 1.0
    frac = (d[None, :] * parts.float()).sum(1) / d.sum()  # [k]
    b = frac / (1.0 - frac).clamp_min(1e-12)
    yv = (1.0 + x) - b[:, None] * (1.0 - x)  # [k, n]
    L = torch.diag(d) - A
    num = ((yv @ L) * yv).sum(1)
    den = (yv * yv * d[None, :]).sum(1)
    ncuts = num / den.clamp_min(1e-12)
    # Degenerate partitions (one side empty) are not valid cuts.
    n_pos = parts.sum(1)
    valid = (n_pos > 0) & (n_pos < parts.shape[1]) & torch.isfinite(ncuts)
    if not valid.any():
        return None, float("inf")
    ncuts = torch.where(valid, ncuts, torch.full_like(ncuts, float("inf")))
    best = int(torch.argmin(ncuts))
    return parts[best], float(ncuts[best])


def _recursive_ncut(affinity: torch.Tensor, keep: torch.Tensor, tau: float, out_masks: List[torch.Tensor]) -> None:
    """Recursively bipartition the subgraph of ``keep`` nodes, appending each
    accepted bipartition (as a full-size bool mask) to ``out_masks``."""
    idx = keep.nonzero(as_tuple=True)[0]
    if idx.numel() <= 1:
        return
    A = affinity[idx][:, idx]
    d = A.sum(1)
    vec = _second_smallest_eigenvector(A, d)
    if vec is None:
        return
    part, ncut = _best_ncut_bipartition(vec, A, d)
    if part is None or ncut >= tau:
        return
    mask = torch.zeros_like(keep)
    mask[idx[part]] = True
    out_masks.append(mask)
    _recursive_ncut(affinity, mask, tau, out_masks)
    _recursive_ncut(affinity, keep & ~mask, tau, out_masks)


def _assemble_clusters(masks: List[torch.Tensor], n: int, device) -> torch.Tensor:
    """Overlay accepted bipartitions into consecutive cluster labels [n]."""
    acc = torch.zeros(n, device=device)
    max_value = 1.0
    for m in masks:
        acc += max_value * m.float()
        max_value = float(acc.max()) + 1.0
    return torch.unique(acc, return_inverse=True)[1]


@torch.no_grad()
def diffcut_labels(
    patch_feats: torch.Tensor,
    grid_hw: tuple,
    out_size: int,
    tau: float,
    alpha: float,
    max_graph_nodes: int,
) -> torch.Tensor:
    """Patch features [P, D] on device -> [out_size, out_size] int64 segment labels."""
    h, w = grid_hw
    device = patch_feats.device
    feats = F.normalize(patch_feats.float(), p=2, dim=1).T.reshape(1, -1, h, w)  # [1, D, h, w]

    pool = 1
    while (h // pool) * (w // pool) > max_graph_nodes:
        pool *= 2
    graph_feats = F.normalize(F.avg_pool2d(feats, pool), p=2, dim=1) if pool > 1 else feats
    gh, gw = graph_feats.shape[-2:]
    x = graph_feats[0].reshape(-1, gh * gw)  # [D, n]

    affinity = x.T @ x
    affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min() + 1e-12)
    affinity = affinity**alpha

    masks: List[torch.Tensor] = []
    keep = torch.ones(gh * gw, dtype=torch.bool, device=device)
    _recursive_ncut(affinity, keep, tau, masks)
    labels = _assemble_clusters(masks, gh * gw, device)  # [n]

    # Cluster mean embeddings on the graph grid, pixel assignment at out_size.
    onehot = F.one_hot(labels).float()  # [n, K]
    embeds = (x @ onehot) / onehot.sum(0).clamp_min(1.0)  # [D, K]
    up = F.interpolate(feats, size=(out_size, out_size), mode="bilinear", align_corners=False)
    sim = up[0].reshape(feats.shape[1], -1).T.half() @ embeds.half()  # [out*out, K]
    return sim.argmax(dim=1).reshape(out_size, out_size)


@torch.no_grad()
def compute_coco_diffcut(eval_backbone, cfg, device: str = "cuda") -> Dict[str, float]:
    """DiffCut zero-shot segmentation MBO on COCO at each configured resolution."""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        logger.warning("pycocotools not installed; skipping DiffCut eval.")
        return {}

    dcfg = cfg.diffcut
    coco_root = os.path.expanduser(dcfg.coco_root)
    split = dcfg.split
    ann_path = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    img_dir = os.path.join(coco_root, split)
    if not os.path.exists(ann_path):
        logger.warning(f"COCO annotations not found at {ann_path}; skipping DiffCut eval.")
        return {}

    coco = COCO(ann_path)
    img_ids = sorted(coco.getImgIds())[: int(dcfg.max_images)]
    # Collective: shard images across ranks; per-resolution overlap lists are
    # gathered after each resolution pass (all ranks share the resolution loop).
    rank, world = _rank_world()
    img_ids = img_ids[rank::world]
    tau = float(dcfg.tau)
    alpha = float(dcfg.alpha)
    max_nodes = int(dcfg.get("max_graph_nodes", 1024))
    patch_size = eval_backbone.patch_size

    out: Dict[str, float] = {}
    for size in [int(s) for s in dcfg.resolutions]:
        transform = _eval_transform(size, cfg.crops.rgb_mean, cfg.crops.rgb_std)
        batch_size = max(1, int(dcfg.get("batch_size", 16)) * (256 // size) ** 2)
        grid = size // patch_size
        inst_overlaps: List[float] = []
        sem_overlaps: List[float] = []
        n_segments: List[float] = []
        n_used = 0

        batch_imgs, batch_meta = [], []

        def flush():
            nonlocal n_used
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs, dim=0).to(device)
            feats = eval_backbone.forward_features(x)["x_norm_patchtokens"]  # [B, P, D]
            for b, meta in enumerate(batch_meta):
                labels = diffcut_labels(feats[b], (grid, grid), size, tau, alpha, max_nodes)
                n_masks = int(labels.max()) + 1
                pred_onehot = F.one_hot(labels, num_classes=n_masks).permute(2, 0, 1).bool()
                inst_overlaps.extend(_best_overlaps(meta["instances"], pred_onehot))
                sem_overlaps.extend(_best_overlaps(meta["semantic"], pred_onehot))
                n_segments.append(float(n_masks))
                n_used += 1
            batch_imgs.clear()
            batch_meta.clear()

        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            path = os.path.join(img_dir, info["file_name"])
            if not os.path.exists(path):
                continue
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
            instances, by_cat = [], defaultdict(list)
            for a in anns:
                m = coco.annToMask(a)
                if m.sum() == 0:
                    continue
                rm = _resize_gt(m, size)
                instances.append(rm)
                by_cat[a["category_id"]].append(rm)
            if not instances:
                continue
            semantic = [torch.stack(v).any(0) for v in by_cat.values()]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            batch_imgs.append(transform(img))
            batch_meta.append({"instances": instances, "semantic": semantic})
            if len(batch_imgs) >= batch_size:
                flush()
        flush()

        if world > 1:
            payload = {"inst": inst_overlaps, "sem": sem_overlaps, "nseg": n_segments, "n": n_used}
            shards: List[dict | None] = [None] * world
            torch.distributed.all_gather_object(shards, payload)
            inst_overlaps = [v for s in shards for v in s["inst"]]
            sem_overlaps = [v for s in shards for v in s["sem"]]
            n_segments = [v for s in shards for v in s["nseg"]]
            n_used = sum(s["n"] for s in shards)

        if inst_overlaps:
            out[f"r{size}_mbo_instance"] = float(np.mean(inst_overlaps))
        if sem_overlaps:
            out[f"r{size}_mbo_semantic"] = float(np.mean(sem_overlaps))
        if n_segments:
            out[f"r{size}_num_segments"] = float(np.mean(n_segments))
        out[f"r{size}_n_images"] = float(n_used)

    logger.info(f"[DiffCut] {out}")
    return out


@torch.no_grad()
def render_diffcut_viz(eval_backbone, cfg, device: str = "cuda") -> List[np.ndarray]:
    """Panels of DiffCut segmentations on a fixed COCO image set.

    Images run left to right; rows are [input | one segmentation row per
    resolution], each segment drawn in a distinct color blended over the input.
    """
    import glob

    from .attention_viz import _distinct_colors, _label_strip, _separator

    dcfg = cfg.diffcut
    n = int(dcfg.get("viz_num_images", 10))
    disp = int(dcfg.get("viz_display_size", 512))
    img_dir = os.path.join(os.path.expanduser(dcfg.coco_root), dcfg.split)
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n]
    if not paths:
        logger.warning(f"No COCO images found under {img_dir}; skipping DiffCut viz.")
        return []
    pil_imgs = [Image.open(p).convert("RGB") for p in paths]
    display = [np.asarray(im.resize((disp, disp), Image.BILINEAR), dtype=np.uint8) for im in pil_imgs]

    resolutions = [int(s) for s in dcfg.resolutions]
    tau = float(dcfg.tau)
    alpha = float(dcfg.alpha)
    max_nodes = int(dcfg.get("max_graph_nodes", 1024))
    patch_size = eval_backbone.patch_size

    seg_rows = {}
    for size in resolutions:
        transform = _eval_transform(size, cfg.crops.rgb_mean, cfg.crops.rgb_std)
        batch = torch.stack([transform(im) for im in pil_imgs])
        bs = max(1, int(dcfg.get("batch_size", 16)) * (256 // size) ** 2)
        feats = torch.cat(
            [
                eval_backbone.forward_features(batch[i : i + bs].to(device))["x_norm_patchtokens"]
                for i in range(0, len(pil_imgs), bs)
            ]
        )
        grid = size // patch_size
        rows = []
        for i in range(len(pil_imgs)):
            labels = diffcut_labels(feats[i], (grid, grid), disp, tau, alpha, max_nodes)
            colors = _distinct_colors(int(labels.max()) + 1)
            seg = colors[labels.cpu().numpy()]
            rows.append((0.45 * display[i] + 0.55 * seg).astype(np.uint8))
        seg_rows[size] = rows

    row_labels = ["input"] + [f"ncut {size}px" for size in resolutions]
    label_col = np.rot90(_label_strip(row_labels, [disp] * len(row_labels)), k=3).copy()
    grid_cols = [label_col]
    for i in range(len(pil_imgs)):
        col = np.concatenate([display[i]] + [seg_rows[size][i] for size in resolutions], axis=0)
        grid_cols.extend([_separator(col.shape[0]), col])
    return [np.concatenate(grid_cols, axis=1)]
