# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Lightweight periodic probes for in-training validation.

These probes intentionally trade benchmark completeness for low operational
friction during pretraining: a capped ImageNet KNN probe, a frozen-feature
linear classifier, and a small COCO patch-level linear segmentation probe.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import dinov3.distributed as distributed
from dinov3.data import make_dataset
from dinov3.data.transforms import make_eval_transform
from dinov3.eval.register_tokens.backbone import build_eval_backbone, sync_eval_backbone

logger = logging.getLogger("dinov3")


def _empty_target_transform(_):
    return _


def _expand_dataset_path(dataset_path: str) -> str:
    parts = []
    for part in str(dataset_path).split(":"):
        if part.startswith("root="):
            parts.append("root=" + os.path.expanduser(part[len("root=") :]))
        else:
            parts.append(part)
    return ":".join(parts)


def _default_val_dataset(dataset_path: str) -> str:
    path = str(dataset_path)
    if "split=TRAIN" in path:
        return path.replace("split=TRAIN", "split=VAL")
    if "split=train" in path:
        return path.replace("split=train", "split=val")
    return path


def _subset(dataset: Dataset, max_images: int | None, seed: int) -> Dataset:
    if max_images is None or max_images <= 0 or max_images >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: int(max_images)].tolist()
    return Subset(dataset, indices)


def _make_classification_dataset(dataset_path: str, image_size: int, max_images: int | None, seed: int) -> Dataset:
    transform = make_eval_transform(
        resize_size=image_size,
        crop_size=image_size,
        resize_square=True,
    )
    dataset = make_dataset(
        dataset_str=_expand_dataset_path(dataset_path),
        transform=transform,
        target_transform=_empty_target_transform,
    )
    return _subset(dataset, max_images, seed)


@torch.no_grad()
def _extract_cls_features(
    backbone: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    features, labels = [], []
    backbone.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        out = backbone(images, is_training=True)
        features.append(out["x_norm_clstoken"].float().cpu())
        labels.append(torch.as_tensor(targets, dtype=torch.long).cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def _extract_cls_and_avg_register_features(
    backbone: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    cls_features, avg_register_features, labels = [], [], []
    has_register_features = True
    backbone.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        out = backbone(images, is_training=True)
        cls_features.append(out["x_norm_clstoken"].float().cpu())
        storage_tokens = out.get("x_storage_tokens", None)
        if storage_tokens is None or storage_tokens.shape[1] == 0:
            has_register_features = False
        elif has_register_features:
            avg_register_features.append(storage_tokens.float().mean(dim=1).cpu())
        labels.append(torch.as_tensor(targets, dtype=torch.long).cpu())
    avg_register = torch.cat(avg_register_features, dim=0) if has_register_features else None
    return torch.cat(cls_features, dim=0), avg_register, torch.cat(labels, dim=0)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 5)) -> dict[str, float]:
    maxk = min(max(topk), logits.shape[1])
    pred = logits.topk(maxk, dim=1).indices
    out = {}
    for k in topk:
        kk = min(k, logits.shape[1])
        correct = pred[:, :kk].eq(labels[:, None]).any(dim=1).float().mean()
        out[f"top{k}"] = float(correct.item() * 100.0)
    return out


def _run_knn_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    k: int,
    temperature: float,
    num_classes: int,
    device: str,
    chunk_size: int = 512,
) -> dict[str, float]:
    train_features = F.normalize(train_features, dim=1).to(device)
    val_features = F.normalize(val_features, dim=1).to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    k = min(int(k), train_features.shape[0])
    logits_all = []
    train_features_t = train_features.T.contiguous()
    for start in range(0, val_features.shape[0], chunk_size):
        sims = val_features[start : start + chunk_size] @ train_features_t
        top_sims, top_idx = sims.topk(k, dim=1)
        neighbor_labels = train_labels[top_idx]
        weights = torch.softmax(top_sims / float(temperature), dim=1)
        logits = torch.zeros(top_idx.shape[0], num_classes, device=device)
        logits.scatter_add_(1, neighbor_labels, weights)
        logits_all.append(logits.cpu())
    logits = torch.cat(logits_all, dim=0)
    return _accuracy(logits, val_labels.cpu(), topk=(1, 5))


def _run_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> dict[str, float]:
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    head = nn.Linear(train_features.shape[1], num_classes).to(device)
    nn.init.normal_(head.weight, mean=0.0, std=0.01)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    for _ in range(int(epochs)):
        perm = torch.randperm(train_features.shape[0], device=device, generator=generator)
        for start in range(0, train_features.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = head(train_features[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = head(val_features).cpu()
    return _accuracy(logits, val_labels.cpu(), topk=(1, 5))


class CocoSemanticPatchDataset(Dataset):
    def __init__(self, *, coco_root: str, split: str, image_size: int, max_images: int | None, seed: int) -> None:
        try:
            from pycocotools.coco import COCO
        except ImportError as exc:
            raise RuntimeError("pycocotools is required for COCO linear segmentation eval") from exc

        self.coco_root = os.path.expanduser(coco_root)
        self.split = split
        self.image_size = int(image_size)
        self.image_dir = os.path.join(self.coco_root, split)
        ann_path = os.path.join(self.coco_root, "annotations", f"instances_{split}.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"COCO annotations not found: {ann_path}")
        self.coco = COCO(ann_path)
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        image_ids = [img_id for img_id in sorted(self.coco.getImgIds()) if self.coco.getAnnIds(imgIds=img_id)]
        if max_images is not None and max_images > 0 and max_images < len(image_ids):
            rng = np.random.default_rng(seed)
            image_ids = rng.choice(image_ids, size=int(max_images), replace=False).tolist()
        self.image_ids = image_ids
        self.transform = make_eval_transform(
            resize_size=self.image_size,
            crop_size=self.image_size,
            resize_square=True,
        )

    @property
    def num_classes(self) -> int:
        return len(self.cat_ids) + 1

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_id = int(self.image_ids[index])
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.image_dir, info["file_name"])
        image = Image.open(path).convert("RGB")
        label = np.zeros((int(info["height"]), int(info["width"])), dtype=np.int64)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        for ann in self.coco.loadAnns(ann_ids):
            mask = self.coco.annToMask(ann).astype(bool)
            if mask.any():
                label[mask] = self.cat_to_label[int(ann["category_id"])]
        label_t = torch.from_numpy(label)[None, None].float()
        label_t = F.interpolate(label_t, size=(self.image_size, self.image_size), mode="nearest")[0, 0].long()
        return self.transform(image), label_t


def _segmentation_metrics(confusion: torch.Tensor) -> dict[str, float]:
    confusion = confusion.float()
    tp = confusion.diag()
    total = confusion.sum()
    pixel_acc = tp.sum() / total.clamp(min=1)
    union = confusion.sum(0) + confusion.sum(1) - tp
    valid = union > 0
    fg_valid = valid.clone()
    if fg_valid.numel() > 0:
        fg_valid[0] = False
    iou = tp / union.clamp(min=1)
    return {
        "pixel_acc": float(pixel_acc.item() * 100.0),
        "miou": float(iou[valid].mean().item() * 100.0) if valid.any() else 0.0,
        "miou_fg": float(iou[fg_valid].mean().item() * 100.0) if fg_valid.any() else 0.0,
    }


def _run_coco_linear_segmentation(
    backbone: nn.Module,
    cfg: Any,
    *,
    device: str,
) -> dict[str, float]:
    scfg = cfg.evaluation.simple.coco_linear_segmentation
    train_dataset = CocoSemanticPatchDataset(
        coco_root=scfg.coco_root,
        split=scfg.train_split,
        image_size=int(scfg.image_size),
        max_images=scfg.max_train_images,
        seed=int(scfg.seed),
    )
    val_dataset = CocoSemanticPatchDataset(
        coco_root=scfg.coco_root,
        split=scfg.val_split,
        image_size=int(scfg.image_size),
        max_images=scfg.max_val_images,
        seed=int(scfg.seed) + 1,
    )
    num_classes = train_dataset.num_classes
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(scfg.batch_size),
        shuffle=True,
        num_workers=int(scfg.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(scfg.batch_size),
        shuffle=False,
        num_workers=int(scfg.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    head = None
    optimizer = None
    backbone.eval()
    for _ in range(int(scfg.epochs)):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                out = backbone(images, is_training=True)
                patches = out["x_norm_patchtokens"].float()
                grid = int(round(patches.shape[1] ** 0.5))
                patch_labels = F.interpolate(labels[:, None].float(), size=(grid, grid), mode="nearest")[:, 0].long()
            if head is None:
                head = nn.Linear(patches.shape[-1], num_classes).to(device)
                optimizer = torch.optim.AdamW(head.parameters(), lr=float(scfg.lr), weight_decay=float(scfg.weight_decay))
            logits = head(patches).reshape(images.shape[0], grid, grid, num_classes).permute(0, 3, 1, 2)
            loss = F.cross_entropy(logits, patch_labels)
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    if head is None:
        return {}
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = backbone(images, is_training=True)
            patches = out["x_norm_patchtokens"].float()
            grid = int(round(patches.shape[1] ** 0.5))
            patch_labels = F.interpolate(labels[:, None].float(), size=(grid, grid), mode="nearest")[:, 0].long()
            pred = head(patches).argmax(dim=-1).reshape(images.shape[0], grid, grid)
            idx = patch_labels.reshape(-1) * num_classes + pred.reshape(-1)
            confusion += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return _segmentation_metrics(confusion.cpu())


@dataclass
class _Due:
    knn: bool
    knn_full: bool
    linear: bool
    coco_seg: bool

    @property
    def any(self) -> bool:
        return self.knn or self.knn_full or self.linear or self.coco_seg


class SimplePeriodicEvaluator:
    def __init__(self, cfg: Any, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._eval_backbone = None
        ecfg = cfg.evaluation.get("simple", {})
        self.enabled = bool(ecfg.get("enabled", False))

    def _backbone(self):
        if self._eval_backbone is None:
            self._eval_backbone = build_eval_backbone(self.cfg, device=self.device)
        return self._eval_backbone

    def due(self, step: int, *, final_step: int | None = None) -> _Due:
        if not self.enabled:
            return _Due(False, False, False, False)
        ecfg = self.cfg.evaluation.simple

        def is_due(section: str) -> bool:
            pcfg = ecfg.get(section, {})
            period = int(pcfg.get("period_steps", 0))
            return bool(pcfg.get("enabled", False)) and period > 0 and (step + 1) % period == 0

        knn_cfg = ecfg.get("knn", {})
        knn_full = (
            bool(knn_cfg.get("enabled", False))
            and bool(knn_cfg.get("full_at_end", False))
            and final_step is not None
            and step == int(final_step)
        )
        return _Due(
            knn=is_due("knn") and not knn_full,
            knn_full=knn_full,
            linear=is_due("linear"),
            coco_seg=is_due("coco_linear_segmentation"),
        )

    @torch.no_grad()
    def sync(self, model) -> None:
        sync_eval_backbone(self._backbone(), model.model_ema["backbone"])

    def run(self, step: int, due: _Due) -> dict[str, float]:
        if not distributed.is_main_process() or not due.any:
            return {}
        metrics: dict[str, float] = {}
        backbone = self._backbone()
        ecfg = self.cfg.evaluation.simple
        if due.knn:
            try:
                metrics.update({f"eval_knn/{k}": v for k, v in self._run_knn(backbone, ecfg.knn).items()})
            except Exception as exc:
                logger.warning("Simple KNN eval failed at step %d: %s", step, exc)
        if due.knn_full:
            try:
                metrics.update({f"eval_knn_full/{k}": v for k, v in self._run_knn(backbone, ecfg.knn, full=True).items()})
            except Exception as exc:
                logger.warning("Full KNN eval failed at step %d: %s", step, exc)
        if due.linear:
            try:
                metrics.update({f"eval_linear/{k}": v for k, v in self._run_linear(backbone, ecfg.linear).items()})
            except Exception as exc:
                logger.warning("Simple linear eval failed at step %d: %s", step, exc)
        if due.coco_seg:
            try:
                seg_metrics = _run_coco_linear_segmentation(backbone, self.cfg, device=self.device)
                metrics.update({f"eval_coco_linear_seg/{k}": v for k, v in seg_metrics.items()})
            except Exception as exc:
                logger.warning("COCO linear segmentation eval failed at step %d: %s", step, exc)
        logger.info("[simple periodic eval] step=%d metrics=%s", step, metrics)
        return metrics

    def _classification_features(self, backbone: nn.Module, pcfg: Any, *, max_train: int | None, max_val: int | None):
        train_dataset_path = pcfg.get("train_dataset", None) or self.cfg.train.dataset_path
        val_dataset_path = pcfg.get("val_dataset", None) or _default_val_dataset(train_dataset_path)
        image_size = int(pcfg.get("image_size", self.cfg.crops.global_crops_size))
        batch_size = int(pcfg.get("batch_size", 128))
        num_workers = int(pcfg.get("num_workers", 4))
        train_dataset = _make_classification_dataset(train_dataset_path, image_size, max_train, int(pcfg.get("seed", 0)))
        val_dataset = _make_classification_dataset(val_dataset_path, image_size, max_val, int(pcfg.get("seed", 0)) + 1)
        train_features, train_labels = _extract_cls_features(
            backbone, train_dataset, batch_size=batch_size, num_workers=num_workers, device=self.device
        )
        val_features, val_labels = _extract_cls_features(
            backbone, val_dataset, batch_size=batch_size, num_workers=num_workers, device=self.device
        )
        num_classes = int(torch.maximum(train_labels.max(), val_labels.max()).item()) + 1
        return train_features, train_labels, val_features, val_labels, num_classes

    def _run_knn(self, backbone: nn.Module, pcfg: Any, *, full: bool = False) -> dict[str, float]:
        train_dataset_path = pcfg.get("train_dataset", None) or self.cfg.train.dataset_path
        val_dataset_path = pcfg.get("val_dataset", None) or _default_val_dataset(train_dataset_path)
        image_size = int(pcfg.get("image_size", self.cfg.crops.global_crops_size))
        batch_size = int(pcfg.get("batch_size", 128))
        num_workers = int(pcfg.get("num_workers", 4))
        max_train_images = None if full else pcfg.get("max_train_images", 20000)
        max_val_images = None if full else pcfg.get("max_val_images", 5000)
        if full:
            max_train_images = pcfg.get("full_max_train_images", None)
            max_val_images = pcfg.get("full_max_val_images", None)
        train_dataset = _make_classification_dataset(
            train_dataset_path,
            image_size,
            max_train_images,
            int(pcfg.get("seed", 0)),
        )
        val_dataset = _make_classification_dataset(
            val_dataset_path,
            image_size,
            max_val_images,
            int(pcfg.get("seed", 0)) + 1,
        )
        train_cls, train_avg_register, train_labels = _extract_cls_and_avg_register_features(
            backbone,
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=self.device,
        )
        val_cls, val_avg_register, val_labels = _extract_cls_and_avg_register_features(
            backbone,
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=self.device,
        )
        num_classes = int(torch.maximum(train_labels.max(), val_labels.max()).item()) + 1
        metrics = _run_knn_probe(
            train_cls,
            train_labels,
            val_cls,
            val_labels,
            k=int(pcfg.get("k", 20)),
            temperature=float(pcfg.get("temperature", 0.07)),
            num_classes=num_classes,
            device=self.device,
        )
        if bool(pcfg.get("avg_register", True)) and train_avg_register is not None and val_avg_register is not None:
            avg_register_metrics = _run_knn_probe(
                train_avg_register,
                train_labels,
                val_avg_register,
                val_labels,
                k=int(pcfg.get("k", 20)),
                temperature=float(pcfg.get("temperature", 0.07)),
                num_classes=num_classes,
                device=self.device,
            )
            metrics.update({f"avg_register_{k}": v for k, v in avg_register_metrics.items()})
        return metrics

    def _run_linear(self, backbone: nn.Module, pcfg: Any) -> dict[str, float]:
        feats = self._classification_features(
            backbone,
            pcfg,
            max_train=pcfg.get("max_train_images", 50000),
            max_val=pcfg.get("max_val_images", 10000),
        )
        train_features, train_labels, val_features, val_labels, num_classes = feats
        return _run_linear_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_classes=num_classes,
            epochs=int(pcfg.get("epochs", 10)),
            batch_size=int(pcfg.get("probe_batch_size", 4096)),
            lr=float(pcfg.get("lr", 0.01)),
            weight_decay=float(pcfg.get("weight_decay", 0.0)),
            device=self.device,
        )
