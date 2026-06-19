# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Single-GPU ImageNet-1k training for regular and slot-like ViT registers."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import tarfile
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

from dinov3.data import make_dataset
from dinov3.data.loaders import SamplerType, make_data_loader
from dinov3.eval.register_tokens.attention_viz import (
    load_viz_images,
    render_register_attention,
    render_register_embedding_similarity,
)
from dinov3.eval.register_tokens.diagnostics import (
    attention_score_metrics,
    compute_register_diagnostics,
    make_fixed_views,
)
from dinov3.logging.wandb_logger import log_images
from dinov3.models import vision_transformer as vits

logger = logging.getLogger("register_classifier")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("opts", nargs="*")
    return parser.parse_args()


def _expand_dataset_path(dataset: str) -> str:
    parts = dataset.split(":")
    return ":".join("root=" + os.path.expanduser(p[5:]) if p.startswith("root=") else p for p in parts)


def _load_config(args: argparse.Namespace) -> DictConfig:
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(args.opts))
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    cfg.train.output_dir = os.path.abspath(os.path.expanduser(cfg.train.output_dir))
    cfg.train.dataset = _expand_dataset_path(cfg.train.dataset)
    cfg.validation.dataset = _expand_dataset_path(cfg.validation.dataset)
    cfg.register_viz.dataset = _expand_dataset_path(cfg.register_viz.dataset)
    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(cfg: DictConfig):
    precision = str(cfg.train.precision).lower()
    if precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_gpu_normalizer(cfg: DictConfig, device: torch.device):
    """Cast a uint8 NCHW batch to a normalized float tensor on the GPU.

    The dataloaders return uint8 (4x less host RAM and H2D traffic, and the
    per-pixel float-cast + normalize stay off the CPU workers); normalization is
    applied here in the train/val loops instead of in the CPU transform.
    """
    mean = torch.tensor(cfg.crops.rgb_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.crops.rgb_std, device=device).view(1, 3, 1, 1)

    def normalize(images: torch.Tensor) -> torch.Tensor:
        images = images.to(device, non_blocking=True).float().div_(255.0)
        return images.sub_(mean).div_(std)

    return normalize


def _build_model(cfg: DictConfig) -> nn.Module:
    mcfg = cfg.model
    model = vits.__dict__[mcfg.arch](
        img_size=int(mcfg.image_size),
        patch_size=int(mcfg.patch_size),
        qkv_bias=bool(mcfg.qkv_bias),
        drop_path_rate=float(mcfg.drop_path_rate),
        layerscale_init=float(mcfg.layerscale),
        n_storage_tokens=int(mcfg.n_storage_tokens),
        register_attn_type=str(mcfg.register_attn_type),
        slot_mode=str(mcfg.slot_mode),
        register_attn_exclude_cls=bool(mcfg.register_attn_exclude_cls),
        patch_cls_attn_type=str(mcfg.patch_cls_attn_type),
        register_init=str(mcfg.register_init),
    )
    model.classifier_pooling = str(mcfg.get("pooling", "cls"))
    model.head = nn.Linear(model.embed_dim, int(mcfg.num_classes))
    model.init_weights()
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    token_names = {"cls_token", "mask_token", "storage_tokens", "storage_tokens_log_sigma"}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.split(".")[-1] in token_names:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _lr_at_step(cfg: DictConfig, step: int) -> float:
    warmup = int(cfg.optim.warmup_steps)
    base_lr = float(cfg.optim.lr)
    min_lr = float(cfg.optim.min_lr)
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if str(cfg.optim.get("scheduler", "cosine")) == "constant":
        return base_lr
    total = int(cfg.train.total_steps)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _classification_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    if targets.ndim == 1:
        return F.cross_entropy(logits, targets, label_smoothing=float(cfg.optim.label_smoothing))
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def _mixup_targets(targets: torch.Tensor, perm: torch.Tensor, lam: float, cfg: DictConfig) -> torch.Tensor:
    num_classes = int(cfg.model.num_classes)
    smoothing = float(cfg.optim.label_smoothing)
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = torch.full((targets.shape[0], num_classes), off_value, device=targets.device, dtype=torch.float32)
    y2 = torch.full_like(y1, off_value)
    y1.scatter_(1, targets[:, None], on_value)
    y2.scatter_(1, targets[perm, None], on_value)
    return lam * y1 + (1.0 - lam) * y2


def _maybe_mixup(images: torch.Tensor, targets: torch.Tensor, cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    mix_cfg = cfg.train.get("mixup", None)
    if mix_cfg is None or not bool(mix_cfg.get("enabled", False)):
        return images, targets
    prob = float(mix_cfg.get("prob", 0.0))
    # Gate on the CPU RNG; a GPU torch.rand(...).item() here would sync the stream
    # every step and serialize the non_blocking H2D copy above against compute.
    if prob <= 0.0 or (prob < 1.0 and random.random() >= prob):
        return images, targets
    alpha = float(mix_cfg.get("alpha", 0.8))
    lam = 1.0 if alpha <= 0.0 else float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(images.shape[0], device=images.device)
    return lam * images + (1.0 - lam) * images[perm], _mixup_targets(targets, perm, lam, cfg)


def _probe_microbatch(model: nn.Module, cfg: DictConfig, device: torch.device) -> tuple[int, int]:
    target = int(cfg.train.batch_size)
    candidates = [target]
    if bool(cfg.train.get("auto_microbatch", True)):
        while candidates[-1] > int(cfg.train.get("min_microbatch", 32)):
            candidates.append(candidates[-1] // 2)

    for microbatch in candidates:
        if target % microbatch != 0:
            continue
        try:
            model.train()
            images = torch.randn(
                microbatch,
                3,
                int(cfg.model.image_size),
                int(cfg.model.image_size),
                device=device,
            )
            targets = torch.randint(int(cfg.model.num_classes), (microbatch,), device=device)
            with _autocast(cfg):
                images, loss_targets = _maybe_mixup(images, targets, cfg)
                loss = _classification_loss(model(images), loss_targets, cfg)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del images, targets, loss
            torch.cuda.empty_cache()
            accumulation = target // microbatch
            logger.info("VRAM probe passed: microbatch=%d accumulation=%d", microbatch, accumulation)
            return microbatch, accumulation
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            logger.warning("VRAM probe failed for microbatch=%d", microbatch)
    raise RuntimeError(f"No microbatch candidate could realize target batch size {target}")


def _build_train_transform(cfg: DictConfig):
    aug_cfg = cfg.train.get("augmentations", None) or {}
    transform_list = [
        v2.ToImage(),
        v2.RandomResizedCrop(
            int(cfg.model.image_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True
        ),
    ]
    hflip_prob = float(aug_cfg.get("hflip_prob", 0.5))
    if hflip_prob > 0.0:
        transform_list.append(v2.RandomHorizontalFlip(hflip_prob))
    randaugment = aug_cfg.get("randaugment", None)
    if randaugment is not None and bool(randaugment.get("enabled", False)):
        transform_list.append(
            v2.RandAugment(
                num_ops=int(randaugment.get("num_ops", 2)),
                magnitude=int(randaugment.get("magnitude", 10)),
            )
        )
    # Returns uint8 CHW; float-cast + normalize run on the GPU (see _make_gpu_normalizer).
    transform = v2.Compose(transform_list)
    logger.info("Built classification train transform (uint8 out)\n%s", transform)
    return transform


def _build_loaders(cfg: DictConfig, microbatch: int, start_step: int, accumulation: int):
    gpu_decode = bool(cfg.train.get("gpu_decode", False))
    # uint8 eval transform (resize shorter side -> center crop); normalize happens on the GPU.
    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(
                int(cfg.validation.resize_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True
            ),
            v2.CenterCrop(int(cfg.model.image_size)),
        ]
    )
    # In gpu_decode mode the train loader yields raw JPEG bytes (workers only do
    # I/O); decode + augmentation run on the GPU (see gpu_aug.py). Otherwise the
    # CPU transform produces uint8 NCHW.
    train_base = make_dataset(dataset_str=cfg.train.dataset, transform=None)
    train_dataset = train_base if gpu_decode else make_dataset(
        dataset_str=cfg.train.dataset, transform=_build_train_transform(cfg)
    )
    val_dataset = make_dataset(dataset_str=cfg.validation.dataset, transform=val_transform)
    effective_batch = microbatch * accumulation
    steps_per_epoch = len(train_dataset) // effective_batch
    cfg.train.steps_per_epoch = steps_per_epoch
    if int(cfg.train.get("epochs", 0)) > 0:
        cfg.train.total_steps = int(cfg.train.epochs) * steps_per_epoch
    logger.info(
        "Epoch schedule: len(train)=%d effective_batch=%d steps_per_epoch=%d total_steps=%d gpu_decode=%s",
        len(train_dataset),
        effective_batch,
        steps_per_epoch,
        int(cfg.train.total_steps),
        gpu_decode,
    )
    advance = start_step * microbatch * accumulation
    if gpu_decode:
        from register_project.classification.gpu_aug import build_gpu_decode_loader

        train_loader = build_gpu_decode_loader(cfg, train_base, microbatch, advance, make_data_loader, SamplerType)
    else:
        train_loader = make_data_loader(
            dataset=train_dataset,
            batch_size=microbatch,
            num_workers=int(cfg.train.num_workers),
            shuffle=True,
            seed=int(cfg.seed),
            sampler_type=SamplerType.SHARDED_INFINITE_NEW,
            sampler_advance=advance,
            drop_last=True,
            persistent_workers=bool(cfg.train.persistent_workers),
            pin_memory=bool(cfg.train.pin_memory),
            prefetch_factor=int(cfg.train.prefetch_factor),
            multiprocessing_context=str(cfg.train.multiprocessing_context),
        )
    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=int(cfg.validation.batch_size),
        num_workers=int(cfg.validation.num_workers),
        shuffle=False,
        sampler_type=None,
        drop_last=False,
        persistent_workers=bool(cfg.validation.num_workers > 0),
        pin_memory=True,
        prefetch_factor=2,
    )
    extra_val_loaders = _build_extra_val_loaders(cfg, val_transform)
    return train_loader, val_loader, extra_val_loaders


@torch.inference_mode()
def _validate(model: nn.Module, loader, cfg: DictConfig, device: torch.device, normalize) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    count = 0
    for images, targets in loader:
        images = normalize(images)
        targets = targets.to(device, non_blocking=True)
        with _autocast(cfg):
            logits = model(images)
            loss = _classification_loss(logits, targets, cfg)
        top5 = logits.topk(5, dim=1).indices
        count += targets.numel()
        loss_sum += loss.item() * targets.numel()
        correct1 += (top5[:, 0] == targets).sum().item()
        correct5 += (top5 == targets[:, None]).any(dim=1).sum().item()
    model.train()
    return {
        "validation/loss": loss_sum / count,
        "validation/top1": 100.0 * correct1 / count,
        "validation/top5": 100.0 * correct5 / count,
    }


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    logger.info("Downloading %s -> %s", url, path)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def _imagenet_real_labels(path: Path) -> list[list[int]]:
    _download(
        "https://raw.githubusercontent.com/google-research/reassessed-imagenet/master/real.json",
        path,
    )
    with path.open() as handle:
        return json.load(handle)


@torch.inference_mode()
def _validate_imagenet_real(model: nn.Module, loader, cfg: DictConfig, device: torch.device, normalize, labels_path: Path) -> dict[str, float]:
    real_labels = _imagenet_real_labels(labels_path)
    model.eval()
    correct = 0
    count = 0
    offset = 0
    for images, _ in loader:
        images = normalize(images)
        with _autocast(cfg):
            preds = model(images).argmax(dim=1).cpu().tolist()
        for pred in preds:
            labels = real_labels[offset]
            offset += 1
            if labels:
                correct += int(pred in labels)
                count += 1
    model.train()
    return {"validation_imagenet_real/top1": 100.0 * correct / max(1, count), "validation_imagenet_real/count": count}


def _prepare_imagenet_v2(root: Path) -> Path:
    dataset_dir = root / "imagenetv2-matched-frequency-format-val"
    if dataset_dir.is_dir():
        return dataset_dir
    archive = root / "imagenetv2-matched-frequency.tar.gz"
    if archive.exists() and not tarfile.is_tarfile(archive):
        logger.warning("Removing incomplete or invalid ImageNetV2 archive: %s", archive)
        archive.unlink()
    _download(
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
        archive,
    )
    logger.info("Extracting %s -> %s", archive, root)
    try:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(root)
    except (tarfile.TarError, EOFError) as exc:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to extract ImageNetV2 archive {archive}; removed it for retry") from exc
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"ImageNetV2 extraction did not create {dataset_dir}")
    return dataset_dir


class NumericImageFolder(Dataset):
    """Image folder whose class label is the integer parent directory name."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.samples = []
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            label = int(class_dir.name)
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in exts:
                    self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _build_extra_val_loaders(cfg: DictConfig, val_transform):
    loaders = {}
    extra = cfg.get("extra_validation", None)
    if extra is None:
        return loaders
    if bool(extra.get("imagenet_real", {}).get("enabled", False)):
        loaders["imagenet_real"] = Path(os.path.expanduser(extra.imagenet_real.labels_path))
    if bool(extra.get("imagenet_v2", {}).get("enabled", False)):
        root = Path(os.path.expanduser(extra.imagenet_v2.root))
        dataset = NumericImageFolder(_prepare_imagenet_v2(root), transform=val_transform)
        loaders["imagenet_v2"] = make_data_loader(
            dataset=dataset,
            batch_size=int(extra.imagenet_v2.get("batch_size", cfg.validation.batch_size)),
            num_workers=int(extra.imagenet_v2.get("num_workers", cfg.validation.num_workers)),
            shuffle=False,
            sampler_type=None,
            drop_last=False,
            persistent_workers=bool(int(extra.imagenet_v2.get("num_workers", cfg.validation.num_workers)) > 0),
            pin_memory=True,
            prefetch_factor=2,
        )
    return loaders


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint_latest.pt"


def _load_checkpoint(model: nn.Module, optimizer, output_dir: Path, device: torch.device) -> tuple[int, float]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return 0, -1.0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    if "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device=device)
    step = int(checkpoint["step"])
    logger.info("Resumed %s at completed step %d", path, step)
    return step, float(checkpoint.get("best_top1", -1.0))


def _save_checkpoint(
    model: nn.Module,
    optimizer,
    output_dir: Path,
    step: int,
    best_top1: float,
    cfg: DictConfig,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_top1": best_top1,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    latest = _checkpoint_path(output_dir)
    tmp = latest.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, latest)

    period = int(cfg.checkpointing.keep_every_steps)
    if period > 0 and step % period == 0:
        milestone = output_dir / f"checkpoint_step_{step:07d}.pt"
        if not milestone.exists():
            shutil.copy2(latest, milestone)


def _init_wandb(cfg: DictConfig, output_dir: Path):
    if not bool(cfg.wandb.enabled):
        return None
    import wandb

    id_path = output_dir / ".wandb_run_id"
    if id_path.exists():
        run_id = id_path.read_text().strip()
    else:
        run_id = wandb.util.generate_id()
        id_path.write_text(run_id + "\n")
    return wandb.init(
        project=str(cfg.wandb.project),
        entity=cfg.wandb.get("entity", None),
        name=str(cfg.wandb.name),
        group=cfg.wandb.get("group", None),
        tags=list(cfg.wandb.tags),
        mode=str(cfg.wandb.mode),
        dir=str(output_dir),
        id=run_id,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _save_viz_grid(panels: list[np.ndarray], path: Path) -> None:
    grid = np.concatenate(panels, axis=0)
    Image.fromarray(grid).save(path)


def _log_json(path: Path, metrics: dict[str, float], step: int) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")],
    )
    OmegaConf.save(cfg, output_dir / "config.yaml")
    _set_seed(int(cfg.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    normalize = _make_gpu_normalizer(cfg, device)

    model = _build_model(cfg).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    logger.info("Built %s with %.2fM parameters", cfg.model.arch, total_parameters / 1e6)
    microbatch, accumulation = _probe_microbatch(model, cfg, device)
    if args.probe_only:
        print(json.dumps({"microbatch": microbatch, "grad_accum_steps": accumulation}))
        return

    optimizer = torch.optim.AdamW(
        _parameter_groups(model, float(cfg.optim.weight_decay)),
        lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.betas),
        eps=float(cfg.optim.eps),
    )
    start_step, best_top1 = _load_checkpoint(model, optimizer, output_dir, device)
    train_loader, val_loader, extra_val_loaders = _build_loaders(cfg, microbatch, start_step, accumulation)
    gpu_decode = bool(cfg.train.get("gpu_decode", False))
    if gpu_decode:
        from register_project.classification.gpu_aug import GpuAugmentor, GpuAugPrefetcher

        train_iterator = iter(GpuAugPrefetcher(train_loader, GpuAugmentor(cfg, device), device))
    else:
        train_iterator = iter(train_loader)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    train_model = torch.compile(model) if bool(cfg.train.compile) else model
    run = _init_wandb(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    viz_images = viz_display = viz_views = None

    try:
        train_model.train()
        window_start = time.perf_counter()
        window_samples = 0
        for step in range(start_step, int(cfg.train.total_steps)):
            lr = _lr_at_step(cfg, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            # Accumulate metrics on-device and only sync (.item()) at logging cadence;
            # a per-step .item() here would stall the stream and starve the GPU.
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            seen = 0
            for _ in range(accumulation):
                images, targets = next(train_iterator)
                if gpu_decode:
                    # Prefetcher already returns normalized float images + targets on-device.
                    pass
                else:
                    images = normalize(images)
                    targets = targets.to(device, non_blocking=True)
                with _autocast(cfg):
                    mixed_images, loss_targets = _maybe_mixup(images, targets, cfg)
                    logits = train_model(mixed_images)
                    loss = _classification_loss(logits, loss_targets, cfg)
                (loss / accumulation).backward()
                loss_sum += loss.detach() * targets.numel()
                correct += (logits.detach().argmax(dim=1) == targets).sum()
                seen += targets.numel()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.clip_grad_norm))
            optimizer.step()
            completed_step = step + 1
            window_samples += seen

            if completed_step % int(cfg.train.log_every_steps) == 0 or completed_step == 1:
                elapsed = time.perf_counter() - window_start
                epoch = completed_step / float(cfg.train.steps_per_epoch)
                metrics = {
                    "train/loss": loss_sum.item() / seen,
                    "train/top1": 100.0 * correct.item() / seen,
                    "train/epoch": epoch,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/images_per_second": window_samples / max(elapsed, 1e-6),
                    "train/max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "train/microbatch": microbatch,
                    "train/grad_accum_steps": accumulation,
                }
                logger.info(
                    "step=%d epoch=%.2f loss=%.4f top1=%.2f lr=%.3e img/s=%.1f mem=%.1fGiB",
                    completed_step,
                    epoch,
                    metrics["train/loss"],
                    metrics["train/top1"],
                    lr,
                    metrics["train/images_per_second"],
                    metrics["train/max_memory_gib"],
                )
                _log_json(metrics_path, metrics, completed_step)
                if run is not None:
                    run.log(metrics, step=completed_step)
                window_start = time.perf_counter()
                window_samples = 0

            has_registers = int(cfg.model.n_storage_tokens) > 0
            validation_due = completed_step % int(cfg.validation.every_n_steps) == 0
            diagnostics_due = has_registers and bool(cfg.register_diagnostics.get("enabled", True)) and (
                completed_step % int(cfg.register_diagnostics.every_n_steps) == 0
            )
            viz_due = has_registers and bool(cfg.register_viz.get("enabled", True)) and validation_due
            if diagnostics_due or viz_due:
                if viz_images is None:
                    viz_images, viz_display = load_viz_images(
                        cfg,
                        num_images=int(cfg.register_viz.num_images),
                        image_size=int(cfg.register_viz.image_size),
                    )
                    viz_views = make_fixed_views(viz_images)
                model.eval()
                if diagnostics_due:
                    diag_images = viz_images[: int(cfg.register_diagnostics.num_images)]
                    diag_views = tuple(v[: diag_images.shape[0]] for v in viz_views)
                    diag = compute_register_diagnostics(
                        model,
                        diag_images,
                        views=diag_views,
                        outlier_ratio=float(cfg.register_diagnostics.outlier_ratio),
                        device=str(device),
                    )
                    diag.update(attention_score_metrics(model, diag_images.to(device, non_blocking=True)))
                    _log_json(metrics_path, diag, completed_step)
                    if run is not None:
                        run.log(diag, step=completed_step)
                if viz_due:
                    similarity_specs = [("last", -1), ("penultimate", -2)]
                    for name, layer in similarity_specs:
                        panels = render_register_embedding_similarity(
                            model,
                            viz_images,
                            viz_display,
                            layer=layer,
                            device=str(device),
                        )
                        path = output_dir / f"embedding_similarity_{name}_step_{completed_step:07d}.png"
                        _save_viz_grid(panels, path)
                        log_images(
                            run,
                            panels,
                            completed_step,
                            key=f"register_viz/embedding_similarity_{name}",
                            caption=f"pre-QKV CLS/register to patch cosine similarity ({name})",
                        )

                    attention_specs = [
                        ("last", -1, "select", "mean"),
                        ("penultimate", -2, "select", "mean"),
                        ("all_layers", -1, "mean", "mean"),
                        ("second_half_layers", -1, "second_half", "mean"),
                        ("last_per_head", -1, "select", "none"),
                    ]
                    for name, layer, layer_reduce, head_reduce in attention_specs:
                        panels = render_register_attention(
                            model,
                            viz_images,
                            viz_display,
                            layer=layer,
                            layer_reduce=layer_reduce,
                            head_reduce=head_reduce,
                            device=str(device),
                        )
                        path = output_dir / f"attention_maps_{name}_step_{completed_step:07d}.png"
                        _save_viz_grid(panels, path)
                        log_images(
                            run,
                            panels,
                            completed_step,
                            key=f"register_viz/attention_maps_{name}",
                            caption=f"register and CLS attention maps ({name})",
                        )
                    logger.info("Saved and logged visualization panels at step %d", completed_step)
                train_model.train()

            if validation_due or completed_step == int(cfg.train.total_steps):
                val_metrics = _validate(model, val_loader, cfg, device, normalize)
                best_top1 = max(best_top1, val_metrics["validation/top1"])
                val_metrics["validation/best_top1"] = best_top1
                val_metrics["validation/epoch"] = completed_step / float(cfg.train.steps_per_epoch)
                if "imagenet_real" in extra_val_loaders:
                    val_metrics.update(
                        _validate_imagenet_real(
                            model,
                            val_loader,
                            cfg,
                            device,
                            normalize,
                            extra_val_loaders["imagenet_real"],
                        )
                    )
                if "imagenet_v2" in extra_val_loaders:
                    v2_metrics = _validate(model, extra_val_loaders["imagenet_v2"], cfg, device, normalize)
                    val_metrics.update(
                        {
                            "validation_imagenet_v2/loss": v2_metrics["validation/loss"],
                            "validation_imagenet_v2/top1": v2_metrics["validation/top1"],
                            "validation_imagenet_v2/top5": v2_metrics["validation/top5"],
                            "validation_imagenet_v2/epoch": val_metrics["validation/epoch"],
                        }
                    )
                logger.info(
                    "validation step=%d top1=%.3f top5=%.3f loss=%.4f",
                    completed_step,
                    val_metrics["validation/top1"],
                    val_metrics["validation/top5"],
                    val_metrics["validation/loss"],
                )
                _log_json(metrics_path, val_metrics, completed_step)
                if run is not None:
                    run.log(val_metrics, step=completed_step)
                train_model.train()

            if validation_due or completed_step == int(cfg.train.total_steps):
                _save_checkpoint(model, optimizer, output_dir, completed_step, best_top1, cfg)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
