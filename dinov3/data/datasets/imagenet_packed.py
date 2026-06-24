# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Memory-mappable ImageNet dataset built from HuggingFace parquet shards.

The HuggingFace ``ILSVRC/imagenet-1k`` release stores images as ~100 MB parquet
row groups, which makes per-sample random access (as required by the SSL
``InfiniteSampler``) very expensive. To get cheap O(1) random access we repack
the encoded JPEG bytes once into a single flat blob per split plus a small
structured index. The index is loaded eagerly; the blob is read on demand with
positional reads (``os.pread``) so workers never map the whole multi-hundred-GB
file (which would leave touched pages resident and creep host RAM upward).

Layout produced by :func:`build_packed_imagenet` under ``root``::

    <root>/
        train.bin            # concatenated raw (encoded) image bytes
        train_index.npy      # structured array: offset(u64), length(u64), label(i32)
        val.bin
        val_index.npy

This conforms to :class:`ExtendedVisionDataset` (``get_image_data`` returns the
encoded bytes), so it plugs into the existing decode/transform pipeline.
"""

import logging
import os
import re
from enum import Enum
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")

_Target = int

# Structured dtype for the on-disk index.
INDEX_DTYPE = np.dtype([("offset", "<u8"), ("length", "<u8"), ("label", "<i4")])


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def length(self) -> int:
        return {_Split.TRAIN: 1_281_167, _Split.VAL: 50_000}[self]

    # HuggingFace names the validation shards "validation-*".
    @property
    def parquet_glob(self) -> str:
        return {"train": "train-*.parquet", "val": "validation-*.parquet"}[self.value]


class ImageNetPacked(ExtendedVisionDataset):
    """Packed ImageNet-1k dataset (random-access via mmap)."""

    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "ImageNetPacked.Split",
        root: str,
        extra: Optional[str] = None,  # accepted for API symmetry, unused
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        root = os.path.expanduser(root)  # allow "~" in dataset strings / configs
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._split = split
        self._index_path = os.path.join(root, f"{split.value}_index.npy")
        self._blob_path = os.path.join(root, f"{split.value}.bin")
        if not (os.path.exists(self._index_path) and os.path.exists(self._blob_path)):
            raise FileNotFoundError(
                f"Packed ImageNet split '{split.value}' not found under {root}. "
                f"Build it with tools/build_packed_imagenet.py "
                f"(expected {self._blob_path} and {self._index_path})."
            )
        # The index is mmap'd by default so forkserver/spawn workers do not each
        # own a private copy. The blob is read with positional reads
        # (os.pread) rather than mmap'd: mapping the whole multi-hundred-GB blob
        # leaves every touched page resident per worker, so host RAM creeps up
        # over a run. pread copies exactly the requested bytes; we also advise the
        # kernel to drop those pages because random SSL sampling has little reuse.
        mmap_index = os.environ.get("DINOV3_PACKED_MMAP_INDEX", "1").lower() not in {"0", "false", "no"}
        self._index = np.load(self._index_path, mmap_mode="r" if mmap_index else None)
        self._fd = None  # opened lazily so the descriptor is created per DataLoader worker
        self._drop_cache = os.environ.get("DINOV3_PACKED_DROP_CACHE", "1").lower() not in {"0", "false", "no"}
        self._page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

    @property
    def split(self) -> "ImageNetPacked.Split":
        return self._split

    def _get_fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self._blob_path, os.O_RDONLY)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
                try:
                    os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)
                except OSError:
                    pass
        return self._fd

    def get_image_data(self, index: int) -> bytes:
        entry = self._index[index]
        offset = int(entry["offset"])
        length = int(entry["length"])
        fd = self._get_fd()
        # os.pread does not use or mutate the fd offset, so it is safe even if the
        # descriptor is shared across forked workers. Regular-file reads return the
        # full count except on rare short reads; top up if that happens.
        data = os.pread(fd, length, offset)
        while len(data) < length:
            more = os.pread(fd, length - len(data), offset + len(data))
            if not more:
                break
            data += more
        if len(data) != length:
            raise OSError(f"short read from {self._blob_path}: got {len(data)} bytes, expected {length}")
        if self._drop_cache:
            self._drop_file_cache(fd, offset, length)
        return data

    def _drop_file_cache(self, fd: int, offset: int, length: int) -> None:
        if not (hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")):
            return
        page_offset = offset - (offset % self._page_size)
        page_end = ((offset + length + self._page_size - 1) // self._page_size) * self._page_size
        page_length = page_end - page_offset
        try:
            os.posix_fadvise(fd, page_offset, page_length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass

    def __del__(self):
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def get_target(self, index: int) -> Optional[Target]:
        return int(self._index[index]["label"])

    def get_targets(self) -> np.ndarray:
        return self._index["label"]

    def __len__(self) -> int:
        return len(self._index)


def _check_complete_hf_shards(shards: list[str], split: "ImageNetPacked.Split") -> None:
    """Fail early when a HuggingFace shard download is incomplete."""
    parsed = []
    for shard in shards:
        match = re.search(r"-(\d+)-of-(\d+)\.parquet$", os.path.basename(shard))
        if match is not None:
            parsed.append((int(match.group(1)), int(match.group(2))))
    if not parsed:
        return

    totals = {total for _, total in parsed}
    if len(totals) != 1:
        raise FileNotFoundError(f"Inconsistent parquet shard totals for split={split.value}: {sorted(totals)}")
    total = totals.pop()
    seen = {idx for idx, _ in parsed}
    missing = sorted(set(range(total)) - seen)
    if len(seen) != total or missing:
        preview = ", ".join(str(i) for i in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Incomplete ImageNet parquet for split={split.value}: found {len(seen)}/{total} shards in "
            f"{os.path.dirname(shards[0])}. Missing shard indices: {preview}{suffix}"
        )


def build_packed_imagenet(
    parquet_dir: str,
    out_dir: str,
    split: "ImageNetPacked.Split",
    image_column: str = "image",
    label_column: str = "label",
    log_every_rows: int = 50_000,
) -> Tuple[str, str]:
    """Repack a parquet ImageNet split into a flat blob + index.

    Reads parquet row groups sequentially (cheap) and streams the encoded image
    bytes into ``<out_dir>/<split>.bin`` while recording (offset, length, label)
    into ``<out_dir>/<split>_index.npy``.
    """
    import glob

    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(parquet_dir, split.parquet_glob)))
    if not shards:
        raise FileNotFoundError(f"No parquet shards matching {split.parquet_glob} in {parquet_dir}")
    _check_complete_hf_shards(shards, split)

    blob_path = os.path.join(out_dir, f"{split.value}.bin")
    index_path = os.path.join(out_dir, f"{split.value}_index.npy")
    tmp_blob = blob_path + ".tmp"
    tmp_index = index_path + ".tmp.npy"

    offsets, lengths, labels = [], [], []
    offset = 0
    n = 0
    logger.info(f"Packing split={split.value} from {len(shards)} shards -> {blob_path}")
    with open(tmp_blob, "wb") as bf:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for rg in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg, columns=[image_column, label_column])
                imgs = table.column(image_column).to_pylist()
                labs = table.column(label_column).to_pylist()
                for img, lab in zip(imgs, labs):
                    data = img["bytes"]
                    bf.write(data)
                    offsets.append(offset)
                    lengths.append(len(data))
                    labels.append(int(lab))
                    offset += len(data)
                    n += 1
                    if n % log_every_rows == 0:
                        logger.info(f"  packed {n:,} images ({offset / 1e9:.1f} GB)")

    index = np.empty(n, dtype=INDEX_DTYPE)
    index["offset"] = np.asarray(offsets, dtype=np.uint64)
    index["length"] = np.asarray(lengths, dtype=np.uint64)
    index["label"] = np.asarray(labels, dtype=np.int32)
    np.save(tmp_index, index)

    os.replace(tmp_blob, blob_path)
    os.replace(tmp_index, index_path)
    logger.info(f"Done: {n:,} images, {offset / 1e9:.1f} GB -> {blob_path}")
    return blob_path, index_path
