# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""COCO MBO (Mean Best Overlap) for register-attention masks.

We turn the register-token attention maps into a set of predicted segments via
per-pixel argmax over registers (each pixel is assigned to the register that
attends to it most), then compute MBO against COCO ground truth:

  * instance MBO (mBO^i): GT = individual object instances.
  * semantic MBO (mBO^c): GT = per-category masks (instances of the same class
    merged).

MBO = average over GT masks of the best IoU achievable with any predicted mask.

Ground truth comes from ``instances_<split>.json`` (the "things" categories) via
pycocotools, which keeps dependencies light. Evaluation is done at a fixed
``image_size`` square resolution (GT masks are resized with nearest-neighbor to
match the attention grid).
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("dinov3")


def _eval_transform(size, mean, std):
    from dinov3.data.transforms import make_eval_transform

    return make_eval_transform(resize_size=size, crop_size=size, resize_square=True, mean=mean, std=std)


def _resize_gt(mask: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="nearest")[0, 0]
    return t > 0.5


def _best_overlaps(gt_masks: List[torch.Tensor], pred_onehot: torch.Tensor) -> List[float]:
    """For each GT mask return the best IoU over predicted masks.

    pred_onehot: [R, S, S] boolean. gt_masks: list of [S, S] boolean.
    """
    if len(gt_masks) == 0:
        return []
    pred = pred_onehot.float().flatten(1)  # [R, S*S]
    pred_area = pred.sum(1)  # [R]
    out = []
    for gt in gt_masks:
        g = gt.float().flatten().to(pred.device)  # [S*S]
        inter = (pred * g[None]).sum(1)  # [R]
        union = pred_area + g.sum() - inter
        iou = inter / union.clamp(min=1e-8)
        out.append(float(iou.max().item()))
    return out


@torch.no_grad()
def compute_coco_mbo(
    eval_backbone,
    cfg,
    device: str = "cuda",
    batch_size: int = 16,
) -> Dict[str, float]:
    """Compute instance and/or semantic MBO over COCO val.

    Returns a dict with keys among {mbo_instance, mbo_semantic, n_images}.
    """
    try:
        from pycocotools.coco import COCO
    except ImportError:
        logger.warning("pycocotools not installed; skipping MBO eval.")
        return {}

    mcfg = cfg.mbo
    coco_root = os.path.expanduser(mcfg.coco_root)
    split = mcfg.split
    ann_path = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    img_dir = os.path.join(coco_root, split)
    if not os.path.exists(ann_path):
        logger.warning(f"COCO annotations not found at {ann_path}; skipping MBO eval.")
        return {}

    size = mcfg.image_size
    transform = _eval_transform(size, cfg.crops.rgb_mean, cfg.crops.rgb_std)
    coco = COCO(ann_path)
    img_ids = sorted(coco.getImgIds())
    if mcfg.max_images is not None:
        img_ids = img_ids[: mcfg.max_images]

    inst_overlaps: List[float] = []
    sem_overlaps: List[float] = []
    n_used = 0

    # Process in batches for backbone efficiency.
    batch_imgs, batch_meta = [], []

    def flush():
        nonlocal n_used
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        attn = eval_backbone.get_register_patch_attention(x, layer=mcfg.layer)  # [B,R,h,w]
        attn = F.interpolate(attn, size=(size, size), mode=mcfg.upsample, align_corners=False)
        assign = attn.argmax(dim=1)  # [B,S,S]
        R = attn.shape[1]
        for b, meta in enumerate(batch_meta):
            pred_onehot = F.one_hot(assign[b], num_classes=R).permute(2, 0, 1).bool()  # [R,S,S]
            if mcfg.instance and meta["instances"]:
                inst_overlaps.extend(_best_overlaps(meta["instances"], pred_onehot))
            if mcfg.semantic and meta["semantic"]:
                sem_overlaps.extend(_best_overlaps(meta["semantic"], pred_onehot))
            n_used += 1
        batch_imgs.clear()
        batch_meta.clear()

    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(img_dir, info["file_name"])
        if not os.path.exists(path):
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
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

    out: Dict[str, float] = {"mbo_n_images": float(n_used)}
    if mcfg.instance and inst_overlaps:
        out["mbo_instance"] = float(np.mean(inst_overlaps))
    if mcfg.semantic and sem_overlaps:
        out["mbo_semantic"] = float(np.mean(sem_overlaps))
    logger.info(f"[MBO] {out}")
    return out
