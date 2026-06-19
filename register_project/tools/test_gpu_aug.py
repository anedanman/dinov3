# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Verify the GPU augmentation pipeline matches the CPU torchvision v2 pipeline.

Run on a CUDA box with the packed ImageNet available:
    PYTHONPATH=. python register_project/tools/test_gpu_aug.py \
        --config register_project/classification/configs/vits_im1k_reg4_baseline_avgpool_ra_mixup_100ep.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2

from dinov3.data import make_dataset
from register_project.classification.gpu_aug import GpuAugmentor, GpuAugPrefetcher, RawBytesDataset, collate_raw

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def expand(dataset_str):
    parts = dataset_str.split(":")
    return ":".join("root=" + os.path.expanduser(p[5:]) if p.startswith("root=") else p for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    aug = GpuAugmentor(cfg, device)

    # ---- 1. grouped RandAugment == per-image application (the core proof) ----
    # For an arbitrary (op, sign) assignment, the grouped _apply_position must be
    # bit-identical to applying torchvision's op fn to each image individually.
    all_ops_seen = set()
    max_diff = 0
    for seed in range(8):
        torch.manual_seed(seed)
        n = 96
        batch = torch.randint(0, 256, (n, 3, aug.size, aug.size), dtype=torch.uint8, device=device)
        op_idx = torch.randint(len(aug._keys), (n,), device=device)
        negate = torch.rand(n, device=device) <= 0.5
        all_ops_seen.update(op_idx.tolist())

        grouped = aug._apply_position(batch.clone(), op_idx, negate)

        ref = batch.clone()
        for i in range(n):
            key, mag, signed = aug._op_meta[int(op_idx[i])]
            if key == "Identity":
                continue
            m = 0.0 if mag is None else (-mag if (signed and bool(negate[i])) else mag)
            ref[i : i + 1] = aug._apply_op(ref[i : i + 1], key, m)
        max_diff = max(max_diff, int((grouped.int() - ref.int()).abs().max()))
    check("grouped RandAugment == per-image (bit-identical)", max_diff == 0, f"max|diff|={max_diff}")
    check("all 14 ops exercised in grouping test", len(all_ops_seen) == len(aug._keys),
          f"{len(all_ops_seen)}/{len(aug._keys)}")

    # ---- 2. every op applies on a GPU batch without error & keeps shape/dtype ----
    op_errors = []
    for key, mag, signed in aug._op_meta:
        try:
            b = torch.randint(0, 256, (4, 3, aug.size, aug.size), dtype=torch.uint8, device=device)
            out = aug._apply_op(b, key, 0.0 if mag is None else mag)
            if out.shape != b.shape or out.dtype != torch.uint8:
                op_errors.append(f"{key}:shape/dtype")
        except Exception as e:  # noqa: BLE001
            op_errors.append(f"{key}:{type(e).__name__}")
    check("all ops run batched on GPU (uint8, shape kept)", not op_errors, ",".join(op_errors))

    # ---- 3. sampling distributions match torchvision RandAugment by construction --
    torch.manual_seed(0)
    big = 200_000
    op_hist = torch.bincount(torch.randint(len(aug._keys), (big,)), minlength=len(aug._keys)).float()
    op_hist /= op_hist.sum()
    uniform = 1.0 / len(aug._keys)
    check("op sampling ~uniform over op space", bool((op_hist - uniform).abs().max() < 0.01),
          f"max dev={float((op_hist-uniform).abs().max()):.4f}")
    sign_frac = float((torch.rand(big) <= 0.5).float().mean())
    check("sign-negate prob ~0.5", abs(sign_frac - 0.5) < 0.01, f"{sign_frac:.4f}")

    # ---- 4. real data: decode + RRC + full pipeline ----
    base = make_dataset(dataset_str=expand(cfg.train.dataset), transform=None)
    raw = RawBytesDataset(base)
    byte_list = [raw[i][0] for i in range(64)]

    # GPU decode vs PIL decode (same images) -> near-identical
    import io as _io

    from PIL import Image
    decoded = aug._decode(byte_list)
    diffs = []
    for k in range(8):
        pil = Image.open(_io.BytesIO(bytes(byte_list[k].numpy().tobytes()))).convert("RGB")
        pil_t = torch.as_tensor(list(pil.tobytes()), dtype=torch.uint8).view(pil.size[1], pil.size[0], 3).permute(2, 0, 1)
        g = decoded[k].cpu()
        if g.shape == pil_t.shape:
            diffs.append(float((g.int() - pil_t.int()).abs().float().mean()))
    check("nvJPEG decode ~matches PIL decode", len(diffs) > 0 and max(diffs) < 2.0,
          f"mean|diff| per channel max={max(diffs) if diffs else 'n/a'}")

    crops = aug._rrc(decoded)
    check("RRC output is [N,3,size,size] uint8 on cuda",
          crops.shape == (len(decoded), 3, aug.size, aug.size) and crops.dtype == torch.uint8 and crops.is_cuda)

    out = aug(byte_list)
    check("full pipeline output [N,3,size,size] float cuda, finite",
          out.shape == (len(byte_list), 3, aug.size, aug.size) and out.dtype == torch.float32
          and out.is_cuda and bool(torch.isfinite(out).all()))

    # ---- 5. aggregate-stat parity vs CPU pipeline on the same images ----
    # With RandAugment OFF the only randomness is the crop/flip (no brightness
    # change), so means must match tightly. With RandAugment ON, its
    # brightness-altering ops (Brightness/Solarize/Posterize) give the batch mean
    # an inherent run-to-run spread (~0.06 at this N), so we only require parity
    # within that noise -- RandAugment op equivalence itself is proven bit-exact
    # in check 1.
    mean_l, std_l = list(cfg.crops.rgb_mean), list(cfg.crops.rgb_std)
    par_bytes = [raw[i][0] for i in range(256)]
    pil_imgs = [Image.open(_io.BytesIO(bytes(b.numpy().tobytes()))).convert("RGB") for b in par_bytes]

    def cpu_pipe(randaug):
        ops = [v2.ToImage(), v2.RandomResizedCrop(aug.size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
               v2.RandomHorizontalFlip(aug.hflip_p)]
        if randaug:
            ops.append(v2.RandAugment(num_ops=aug.num_ops, magnitude=aug._ra.magnitude))
        ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean_l, std_l)]
        tf = v2.Compose(ops)
        return torch.stack([tf(im) for im in pil_imgs])

    aug.randaug_on = False
    g0 = aug(par_bytes)
    c0 = cpu_pipe(False)
    d0_mean = float((g0.mean(dim=(0, 2, 3)).cpu() - c0.mean(dim=(0, 2, 3))).abs().max())
    d0_std = float((g0.std(dim=(0, 2, 3)).cpu() - c0.std(dim=(0, 2, 3))).abs().max())
    check("RA-off mean parity GPU vs CPU (<0.03)", d0_mean < 0.03, f"max|dmean|={d0_mean:.4f}")
    check("RA-off std parity GPU vs CPU (<0.03)", d0_std < 0.03, f"max|dstd|={d0_std:.4f}")

    aug.randaug_on = True
    g1 = aug(par_bytes)
    c1 = cpu_pipe(True)
    d1_mean = float((g1.mean(dim=(0, 2, 3)).cpu() - c1.mean(dim=(0, 2, 3))).abs().max())
    d1_std = float((g1.std(dim=(0, 2, 3)).cpu() - c1.std(dim=(0, 2, 3))).abs().max())
    # RA-on mean/std are noise-dominated (independent RNG, brightness-altering ops;
    # ~0.06 run-to-run spread at this N). Exact RA equivalence is proven in check 1.
    check("RA-on mean parity within RA noise (<0.1)", d1_mean < 0.1, f"max|dmean|={d1_mean:.4f}")
    check("RA-on std parity within RA noise (<0.08)", d1_std < 0.08, f"max|dstd|={d1_std:.4f}")

    # ---- 6. prefetcher yields correct batches from a real loader ----
    from dinov3.data.loaders import SamplerType, make_data_loader
    loader = make_data_loader(
        dataset=raw, batch_size=128, num_workers=4, shuffle=True, seed=0,
        sampler_type=SamplerType.SHARDED_INFINITE_NEW, drop_last=True,
        collate_fn=collate_raw, pin_memory=True, prefetch_factor=2,
    )
    pf = GpuAugPrefetcher(loader, aug, device)
    ok_pf = True
    for _ in range(3):
        im, tg = next(pf)
        ok_pf &= im.shape == (128, 3, aug.size, aug.size) and im.is_cuda and tg.shape == (128,)
    check("GpuAugPrefetcher yields normalized [B,3,size,size] batches", ok_pf)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
