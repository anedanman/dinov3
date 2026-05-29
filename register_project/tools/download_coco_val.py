#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Download MS COCO val2017 images + annotations for MBO evaluation.

Downloads (into --out-dir):
    val2017.zip                      -> val2017/*.jpg
    annotations_trainval2017.zip     -> annotations/instances_val2017.json (+ others)

Example:
    python register_project/tools/download_coco_val.py --out-dir ~/datasets/coco
"""

import argparse
import os
import sys
import urllib.request
import zipfile

URLS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists")
        return

    def hook(blocks, bs, total):
        done = blocks * bs
        pct = (100 * done / total) if total > 0 else 0
        sys.stdout.write(f"\r  {os.path.basename(dest)}: {done / 1e6:.0f} MB ({pct:.1f}%)")
        sys.stdout.flush()

    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest + ".tmp", reporthook=hook)
    os.replace(dest + ".tmp", dest)
    print()


def _unzip(zip_path: str, out_dir: str, marker: str):
    if os.path.exists(os.path.join(out_dir, marker)):
        print(f"[skip] {marker} already extracted")
        return
    print(f"[unzip] {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="~/datasets/coco")
    p.add_argument("--keep-zips", action="store_true", help="keep the downloaded zip files")
    args = p.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for name, url in URLS.items():
        _download(url, os.path.join(out_dir, name))

    _unzip(os.path.join(out_dir, "val2017.zip"), out_dir, marker="val2017")
    _unzip(os.path.join(out_dir, "annotations_trainval2017.zip"), out_dir, marker="annotations")

    if not args.keep_zips:
        for name in URLS:
            zp = os.path.join(out_dir, name)
            if os.path.exists(zp):
                os.remove(zp)
                print(f"[clean] removed {zp}")

    print(f"\nDone. COCO val at: {out_dir}")
    print(f"  images: {os.path.join(out_dir, 'val2017')}")
    print(f"  annotations: {os.path.join(out_dir, 'annotations', 'instances_val2017.json')}")


if __name__ == "__main__":
    main()
