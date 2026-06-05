# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from dinov3.utils.memory import PeriodicMemoryTrimmer

from .decoders import Decoder, ImageDataDecoder, TargetDecoder


# Explicit per-item malloc_trim(0) is an expensive heap walk in the dataloader
# hot path. Default off (0) for throughput; glibc still returns freed memory
# lazily via MALLOC_TRIM_THRESHOLD_. Set DINOV3_DATASET_MALLOC_TRIM_EVERY>0 to
# re-enable periodic trimming on RAM-constrained machines.
_SAMPLE_TRIMMER = PeriodicMemoryTrimmer("DINOV3_DATASET_MALLOC_TRIM_EVERY", 0)


def _maybe_trim_worker_heap() -> None:
    """Periodically return freed decoder/augmentation arenas from workers to the OS."""
    _SAMPLE_TRIMMER.maybe_trim()


class ExtendedVisionDataset(VisionDataset):
    def __init__(
        self,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = TargetDecoder,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.image_decoder = image_decoder
        self.target_decoder = target_decoder

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self.image_decoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        del image_data
        _maybe_trim_worker_heap()
        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
