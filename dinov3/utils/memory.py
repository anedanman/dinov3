# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Small host-memory helpers for long-running training jobs."""

import ctypes
import gc
import os
import sys
from typing import Callable, Optional, Union

_MALLOC_TRIM: Optional[Union[Callable[[int], int], bool]] = None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def malloc_trim() -> bool:
    """Ask glibc to return free heap arenas to the OS when available."""
    global _MALLOC_TRIM

    if _MALLOC_TRIM is None:
        if not sys.platform.startswith("linux"):
            _MALLOC_TRIM = False
        else:
            try:
                trim = ctypes.CDLL("libc.so.6").malloc_trim
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                _MALLOC_TRIM = trim
            except (AttributeError, OSError):
                _MALLOC_TRIM = False

    if not _MALLOC_TRIM:
        return False
    return bool(_MALLOC_TRIM(0))


def release_memory(*, empty_cuda_cache: bool = False) -> None:
    """Collect Python cycles, trim glibc arenas, and optionally clear CUDA cache."""
    gc.collect()
    malloc_trim()
    if empty_cuda_cache:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PeriodicMemoryTrimmer:
    def __init__(self, env_name: str, default_every: int) -> None:
        self.every = env_int(env_name, default_every)
        self.counter = 0

    def maybe_trim(self) -> bool:
        if self.every <= 0:
            return False
        self.counter += 1
        if self.counter % self.every != 0:
            return False
        return malloc_trim()
