#!/usr/bin/env python3
"""Validate that the register-token training stack is ready to launch."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]


def _ok(message: str) -> None:
    print(f"[ok] {message}")


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _expand_dataset_root(dataset: str) -> Path | None:
    for token in dataset.split(":"):
        if token.startswith("root="):
            return Path(os.path.expanduser(token[len("root=") :]))
    return None


def _split_name(dataset: str) -> str | None:
    for token in dataset.split(":"):
        if token.startswith("split="):
            value = token[len("split=") :].lower()
            return "val" if value in {"val", "validation"} else value
    return None


def _require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        _fail(f"{label} missing:\n  " + "\n  ".join(missing))
    _ok(label)


def check_imports() -> None:
    modules = [
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "omegaconf",
        "wandb",
        "huggingface_hub",
        "pyarrow",
        "pycocotools",
        "matplotlib",
        "scipy",
        "sklearn",
        "termcolor",
        "submitit",
        "iopath",
        "pandas",
        "torchmetrics",
        "fvcore",
        "ftfy",
        "regex",
    ]
    failures = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {exc}")
    if failures:
        _fail("training dependency imports failed:\n  " + "\n  ".join(failures))
    _ok("training dependency imports")


def load_cfg(config: Path, overrides: list[str]):
    default_cfg = OmegaConf.load(REPO / "dinov3/configs/ssl_default_config.yaml")
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config), OmegaConf.from_cli(overrides))
    from dinov3.train.step_schedule import normalize_step_schedule

    return normalize_step_schedule(cfg)


def check_data(cfg) -> None:
    train_root = _expand_dataset_root(cfg.train.dataset_path)
    train_split = _split_name(cfg.train.dataset_path)
    if cfg.train.dataset_path.startswith("ImageNetPacked:"):
        if train_root is None or train_split is None:
            _fail(f"cannot parse ImageNetPacked dataset path: {cfg.train.dataset_path}")
        _require_files(
            [train_root / f"{train_split}.bin", train_root / f"{train_split}_index.npy"],
            f"packed ImageNet {train_split}",
        )

    if cfg.register_viz.enabled and cfg.register_viz.dataset.startswith("ImageNetPacked:"):
        viz_root = _expand_dataset_root(cfg.register_viz.dataset)
        viz_split = _split_name(cfg.register_viz.dataset)
        if viz_root is None or viz_split is None:
            _fail(f"cannot parse register_viz dataset path: {cfg.register_viz.dataset}")
        _require_files(
            [viz_root / f"{viz_split}.bin", viz_root / f"{viz_split}_index.npy"],
            f"packed ImageNet {viz_split} for register_viz",
        )

    if cfg.mbo.enabled:
        coco_root = Path(os.path.expanduser(cfg.mbo.coco_root))
        _require_files(
            [coco_root / "annotations" / f"instances_{cfg.mbo.split}.json", coco_root / cfg.mbo.split],
            "COCO MBO data",
        )


def check_model(cfg, device: str) -> None:
    from dinov3.eval.register_tokens.backbone import build_eval_backbone

    if device == "cuda" and not torch.cuda.is_available():
        _fail("CUDA requested but torch.cuda.is_available() is false")

    if torch.cuda.is_available():
        _ok(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[warn] CUDA is not available; model check will run on CPU")

    model = build_eval_backbone(cfg, device=device)
    size = int(cfg.crops.global_crops_size)
    x = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        attn = model.get_register_patch_attention(x, layer=int(cfg.register_viz.layer))
    expected_regs = int(cfg.student.n_storage_tokens)
    if attn.ndim != 4 or attn.shape[0] != 1 or attn.shape[1] != expected_regs:
        _fail(f"unexpected register attention shape: {tuple(attn.shape)}")
    if not torch.isfinite(attn).all():
        _fail("register attention contains non-finite values")
    _ok(f"register attention forward pass: shape={tuple(attn.shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config to validate.")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset existence checks.")
    parser.add_argument("--skip-model", action="store_true", help="Skip model forward-pass check.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for the model forward-pass check.",
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf overrides, e.g. train.wandb.enabled=false")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.overrides and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]

    check_imports()
    cfg = load_cfg(args.config, args.overrides)
    _ok(f"config loaded: {args.config}")
    if int(cfg.student.n_storage_tokens) <= 0:
        _fail("student.n_storage_tokens must be > 0 for register-token training")
    if not args.skip_data:
        check_data(cfg)
    if not args.skip_model:
        check_model(cfg, args.device)
    _ok("train readiness checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        raise SystemExit(1)
