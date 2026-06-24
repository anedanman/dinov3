# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""GPU-side ImageNet train augmentation: nvJPEG decode + RandomResizedCrop +
HFlip + RandAugment + normalize, matching the CPU torchvision v2 pipeline.

The packed dataset already stores encoded JPEG bytes, so the DataLoader workers
only do I/O (read raw bytes) and decode/augmentation happens on the GPU in the
main process, overlapped with training via a side CUDA stream.

Exactness: every image is RandomResizedCrop'd to a fixed square before
RandAugment, so the only size-dependent RandAugment magnitudes (TranslateX/Y)
are constant across the batch. The per-image randomness is therefore just
(which op, sign sampled per image); we reuse torchvision's *own*
``RandAugment._apply_image_or_video_transform`` and ``_AUGMENTATION_SPACE`` and
apply each op to the sub-batch of images that share (op, sign). That is
identical to applying the op per image -- see tools/test_gpu_aug.py.
"""

from __future__ import annotations

import io
import logging
from typing import Callable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg
from torchvision.transforms import RandomResizedCrop as _RRCv1
from torchvision.transforms.v2 import RandAugment as _RandAugmentV2
from torchvision.transforms.v2 import functional as F2
from torchvision.transforms.v2.functional import pil_to_tensor

logger = logging.getLogger("register_classifier")


class RawBytesDataset(Dataset):
    """Yields (encoded-bytes uint8 tensor, int label) without decoding.

    Wraps a dataset exposing ``get_image_data(i) -> bytes`` and
    ``get_target(i) -> int`` (e.g. ImageNetPacked).
    """

    def __init__(self, base):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        data = self._base.get_image_data(index)
        # bytearray gives a writable buffer so torch.frombuffer does not warn.
        return torch.frombuffer(bytearray(data), dtype=torch.uint8), int(self._base.get_target(index))


def collate_raw(batch: List[Tuple[torch.Tensor, int]]):
    """Pack a batch's encoded bytes into ONE flat tensor + lengths.

    Returning a list of per-sample tensors would transfer one shared-memory file
    descriptor per image (worker->main), exhausting the fd limit at batch>1k.
    Concatenating makes it 2 tensors per batch regardless of batch size; the
    prefetcher splits them back with ``torch.split``.
    """
    byte_tensors = [b for b, _ in batch]
    lengths = torch.tensor([int(b.numel()) for b in byte_tensors], dtype=torch.long)
    blob = torch.cat(byte_tensors) if byte_tensors else torch.empty(0, dtype=torch.uint8)
    targets = torch.tensor([t for _, t in batch], dtype=torch.long)
    return (blob, lengths), targets


def _decode_one_cpu_fallback(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Decode an image PIL-side (handles PNG/CMYK/corrupt that nvJPEG rejects)."""
    img = Image.open(io.BytesIO(bytes(data.numpy().tobytes()))).convert("RGB")
    return pil_to_tensor(img).to(device, non_blocking=True)


class GpuAugmentor:
    """Callable: list of encoded-byte tensors -> normalized float NCHW on GPU.

    Mirrors the CPU transform order: decode -> RandomResizedCrop(BICUBIC) ->
    RandomHorizontalFlip -> RandAugment -> float/255 + normalize.
    """

    def __init__(self, cfg, device: torch.device):
        self.device = device
        self.size = int(cfg.model.image_size)
        self.scale = (0.08, 1.0)
        self.ratio = (3.0 / 4.0, 4.0 / 3.0)

        aug = cfg.train.get("augmentations", None) or {}
        self.hflip_p = float(aug.get("hflip_prob", 0.5))
        ra_cfg = aug.get("randaugment", None) or {}
        self.randaug_on = bool(ra_cfg.get("enabled", False))
        num_ops = int(ra_cfg.get("num_ops", 2))
        magnitude = int(ra_cfg.get("magnitude", 10))

        # torchvision RandAugment instance, reused purely for its exact op space
        # and op-application math (never called via its own forward()).
        self._ra = _RandAugmentV2(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=31)
        self.num_ops = num_ops
        self._keys = list(self._ra._AUGMENTATION_SPACE.keys())
        # Precompute (magnitude_at_idx, signed) per op for the fixed square size.
        self._op_meta = []
        for key in self._keys:
            mag_fn, signed = self._ra._AUGMENTATION_SPACE[key]
            mags = mag_fn(self._ra.num_magnitude_bins, self.size, self.size)
            mag = None if mags is None else float(mags[self._ra.magnitude])
            self._op_meta.append((key, mag, bool(signed)))

        mean = torch.tensor(cfg.crops.rgb_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg.crops.rgb_std, device=device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    # -- decode -------------------------------------------------------------
    def _decode(self, byte_list: List[torch.Tensor]) -> List[torch.Tensor]:
        try:
            return decode_jpeg(byte_list, mode=ImageReadMode.RGB, device=self.device)
        except Exception:
            out = []
            for data in byte_list:
                try:
                    out.append(decode_jpeg(data, mode=ImageReadMode.RGB, device=self.device))
                except Exception:
                    out.append(_decode_one_cpu_fallback(data, self.device))
            return out

    # -- random resized crop (per image, sizes differ) ----------------------
    def _rrc(self, images: List[torch.Tensor]) -> torch.Tensor:
        out = torch.empty((len(images), 3, self.size, self.size), dtype=torch.uint8, device=self.device)
        for idx, img in enumerate(images):
            i, j, h, w = _RRCv1.get_params(img, list(self.scale), list(self.ratio))
            out[idx] = F2.resized_crop(
                img, i, j, h, w, [self.size, self.size],
                interpolation=F2.InterpolationMode.BICUBIC, antialias=True,
            )
        return out

    # -- horizontal flip (vectorized) ---------------------------------------
    def _flip(self, batch: torch.Tensor) -> torch.Tensor:
        if self.hflip_p <= 0.0:
            return batch
        mask = torch.rand(batch.shape[0], device=self.device) < self.hflip_p
        if bool(mask.any()):
            batch[mask] = torch.flip(batch[mask], dims=[-1])
        return batch

    # -- randaugment (grouped per (op, sign), torchvision op math) ----------
    def _apply_op(self, sub: torch.Tensor, key: str, magnitude: float) -> torch.Tensor:
        return self._ra._apply_image_or_video_transform(
            sub, key, magnitude, interpolation=self._ra.interpolation, fill=self._ra._fill
        )

    def _apply_position(self, batch: torch.Tensor, op_idx: torch.Tensor, negate: torch.Tensor) -> torch.Tensor:
        """Apply one RandAugment op per image, grouped by (op, sign).

        ``op_idx[i]`` selects the op for image i; ``negate[i]`` flips the sign of
        the magnitude for signed ops. Grouping is exactly equivalent to applying
        the op to each image individually (proven in tools/test_gpu_aug.py).
        """
        for ki, (key, mag, signed) in enumerate(self._op_meta):
            if key == "Identity":
                continue
            op_mask = op_idx == ki
            if not bool(op_mask.any()):
                continue
            if mag is None:  # AutoContrast / Equalize: magnitude ignored
                rows = op_mask.nonzero(as_tuple=True)[0]
                batch[rows] = self._apply_op(batch[rows], key, 0.0)
            elif signed:
                for neg in (False, True):
                    rows = (op_mask & (negate == neg)).nonzero(as_tuple=True)[0]
                    if rows.numel():
                        batch[rows] = self._apply_op(batch[rows], key, -mag if neg else mag)
            else:
                rows = op_mask.nonzero(as_tuple=True)[0]
                batch[rows] = self._apply_op(batch[rows], key, mag)
        return batch

    def _randaugment(self, batch: torch.Tensor) -> torch.Tensor:
        n, n_ops = batch.shape[0], len(self._keys)
        for _ in range(self.num_ops):
            op_idx = torch.randint(n_ops, (n,), device=self.device)
            negate = torch.rand(n, device=self.device) <= 0.5
            batch = self._apply_position(batch, op_idx, negate)
        return batch

    # -- normalize ----------------------------------------------------------
    def _normalize(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.float().div_(255.0).sub_(self._mean).div_(self._std)

    def __call__(self, byte_list: List[torch.Tensor]) -> torch.Tensor:
        images = self._decode(byte_list)
        batch = self._rrc(images)
        batch = self._flip(batch)
        if self.randaug_on:
            batch = self._randaugment(batch)
        return self._normalize(batch)


class GpuAugPrefetcher:
    """Iterator that runs the augmentor on a side CUDA stream, so decode+aug of
    batch N+1 overlaps the forward/backward of batch N."""

    def __init__(self, loader, augmentor: GpuAugmentor, device: torch.device):
        self._it = iter(loader)
        self._aug = augmentor
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next = None
        self._preload()

    def _preload(self) -> None:
        try:
            (blob, lengths), targets = next(self._it)
        except StopIteration:
            self._next = None
            return
        byte_list = list(torch.split(blob, lengths.tolist()))  # views; no copy
        with torch.cuda.stream(self._stream):
            images = self._aug(byte_list)
            targets = targets.to(self._device, non_blocking=True)
        self._next = (images, targets)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self._stream)
        images, targets = self._next
        images.record_stream(torch.cuda.current_stream())
        targets.record_stream(torch.cuda.current_stream())
        self._preload()
        return images, targets


def build_gpu_decode_loader(cfg, base_dataset, microbatch: int, advance: int, make_data_loader, SamplerType):
    """Wrap a packed dataset as a raw-bytes loader for the GPU decode path."""
    dataset = RawBytesDataset(base_dataset)
    return make_data_loader(
        dataset=dataset,
        batch_size=microbatch,
        num_workers=int(cfg.train.num_workers),
        shuffle=True,
        seed=int(cfg.seed),
        sampler_type=SamplerType.SHARDED_INFINITE_NEW,
        sampler_advance=advance,
        drop_last=True,
        collate_fn=collate_raw,
        persistent_workers=bool(cfg.train.persistent_workers),
        pin_memory=False,  # decode_jpeg handles H2D; avoids pinning a ~100MB blob per in-flight batch
        prefetch_factor=int(cfg.train.prefetch_factor),
        multiprocessing_context=str(cfg.train.multiprocessing_context),
    )
