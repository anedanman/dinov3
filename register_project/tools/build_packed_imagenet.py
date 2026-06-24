#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Repack HuggingFace parquet ImageNet-1k into a memory-mappable blob + index.

Example:
    python register_project/tools/build_packed_imagenet.py \
        --parquet-dir ~/datasets/imagenet-1k/data \
        --out-dir ~/datasets/imagenet1k_packed \
        --splits train val
"""

import argparse
import logging
import os
import sys

# Make the dinov3 package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dinov3.data.datasets.imagenet_packed import ImageNetPacked, build_packed_imagenet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="~/datasets/imagenet-1k/data")
    p.add_argument("--out-dir", default="~/datasets/imagenet1k_packed")
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    args = p.parse_args()

    parquet_dir = os.path.expanduser(args.parquet_dir)
    out_dir = os.path.expanduser(args.out_dir)
    for split_name in args.splits:
        split = ImageNetPacked.Split[split_name.upper()]
        build_packed_imagenet(parquet_dir, out_dir, split)


if __name__ == "__main__":
    main()
