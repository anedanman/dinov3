# Register tokens as object-centric representations (DINOv3)

Research scaffolding around DINOv3 to study **register/storage tokens as
object-centric representations**, plus an alternative **slot-register attention**
mechanism.

Everything is driven by a single YAML config (merged on top of
`dinov3/configs/ssl_default_config.yaml`). Two ready-to-run experiments:

| Experiment | Config | Register attention |
|---|---|---|
| Baseline | `configs/vits_im1k_reg7_baseline.yaml` | standard self-attention |
| Slot-register | `configs/vits_im1k_reg7_slot.yaml` | slot-attention competition |

Both: **ViT-S/16 (~21.6M params)**, **7 register tokens**, **effective batch 512**
(128 micro-batch × 4 gradient-accumulation steps on a single GPU), **256/112 crops**,
step-based schedule, Weights & Biases logging, periodic register-attention
visualization, and **COCO MBO** (instance + semantic) at validation.

---

## Quick start on a new machine

```bash
git clone https://github.com/anedanman/dinov3 && cd dinov3
git checkout register-tokens
bash register_project/scripts/setup_all.sh          # creates env + checks/builds data
# (add ALLOW_IMAGENET_DOWNLOAD=1 with an HF token to also fetch ImageNet parquet)
conda activate dinov3
wandb login                                         # or set train.wandb.mode=offline
bash register_project/scripts/train_baseline.sh     # or train_slot.sh
```

`setup_all.sh` = `setup_env.sh` (conda env + torch/cu128 + deps) then
`prepare_data.sh` (idempotent dataset check/build). Both can be run individually.
Key overrides (env vars): `ENV_NAME`, `TORCH_INDEX_URL`/`TORCH_VERSION`/`TORCHVISION_VERSION`
(for non-Blackwell CUDA), `PACKED_DIR`/`PARQUET_DIR`/`COCO_DIR`.

---

## 0. Environment

A conda env `dinov3` is already created (cloned from a Blackwell-compatible
`torch 2.11.0+cu128` env + `torchvision 0.26.0+cu128`, plus `wandb pyarrow
pycocotools matplotlib scipy scikit-learn omegaconf submitit fvcore ...`).

```bash
conda activate dinov3        # python 3.11, torch 2.11+cu128
```

The launch scripts set `PYTHONPATH` to the repo root, so no `pip install -e`
is required.

## 1. Data preparation

### ImageNet-1k (repack parquet → memory-mappable blob)

The HF parquet shards have ~100 MB row groups, which are too coarse for the
random-access SSL sampler. We repack once into a flat blob + index:

```bash
python register_project/tools/build_packed_imagenet.py \
    --parquet-dir ~/datasets/imagenet-1k/data \
    --out-dir     ~/datasets/imagenet1k_packed \
    --splits train val
```

Produces `train.bin/train_index.npy` (~156 GB) and `val.bin/val_index.npy`.
The dataset is exposed as `ImageNetPacked:split=TRAIN:root=~/datasets/imagenet1k_packed`.

### COCO val (for MBO)

```bash
python register_project/tools/download_coco_val.py --out-dir ~/datasets/coco
# -> ~/datasets/coco/val2017/*.jpg
# -> ~/datasets/coco/annotations/instances_val2017.json
```

## 2. Weights & Biases

```bash
wandb login          # once
```
Configure under `train.wandb` in the YAML (`enabled`, `project`, `name`, `mode`).
Set `mode: offline` (or `train.wandb.enabled: false`) to run without a network.

## 3. Train

```bash
# baseline (standard register attention)
bash register_project/scripts/train_baseline.sh

# slot-register variant
bash register_project/scripts/train_slot.sh
```

Useful env vars: `OUTPUT_DIR`, `NGPUS`, `MASTER_PORT`, `ENV_PY`.
Extra config overrides can be appended, e.g.:

```bash
bash register_project/scripts/train_slot.sh schedule.total_steps=50000 train.batch_size_per_gpu=256
```

Outputs (checkpoints, logs, `config.yaml`) land in `runs/<name>/`.

---

## What was added / changed

**Slot-register attention** — `dinov3/layers/attention.py`
- `RegisterSlotAttention`: registers never attend to any register; their
  logits over patches (or cls+patches with `register_attn_exclude_cls=false`)
  are softmaxed across the **register/query** dimension (competition).
  `slot_mode="slot"` adds slot-attention key renormalization (weighted mean);
  `slot_mode="literal"` uses `out = A @ V`. All non-register tokens are
  unchanged and may attend to registers.
  Implemented as fast SDPA for the bulk + explicit competition for register rows.
- `extract_register_attention_maps(...)`: register→patch and patch→register
  weights for viz/MBO, plus matching CLS maps. `extract_register_patch_attention(...)`
  remains as the register→patch compatibility wrapper.

**Model wiring** — `dinov3/models/vision_transformer.py`, `dinov3/models/__init__.py`
- `register_attn_type` (`standard`|`slot`), `slot_mode`,
  `register_attn_exclude_cls`, and `register_init` flow from config.
- `register_init="gaussian"` samples per-image register tokens from trainable
  per-register mean/log-std parameters.
- `DinoVisionTransformer.get_register_attention_maps(...)` returns either
  head-averaged `[B, R, H, W]` masks or per-head `[B, heads, R, H, W]` masks.
  `get_register_patch_attention(x, layer)` remains as the register→patch wrapper.

**Config** — `dinov3/configs/ssl_default_config.yaml`
- `student.register_attn_type`, `student.slot_mode`,
  `student.register_attn_exclude_cls`, `student.register_init`
- `train.wandb.*`, `schedule.*` (step-based), `register_viz.*`, `mbo.*`

**Step-based scheduling** — `dinov3/train/step_schedule.py`
- `schedule.enabled` maps `total_steps / warmup_steps /
  freeze_last_layer_steps / teacher_temp_warmup_steps` onto the epoch machinery
  (by setting `OFFICIAL_EPOCH_LENGTH = 1`). Eval/ckpt/viz/MBO periods are already
  in steps.

**Training loop** — `dinov3/train/train.py`
- W&B init + per-step scalar logging.
- Periodic register-attention viz + COCO MBO (`RegisterEvaluator`), run on a
  plain eval backbone kept in sync with the EMA teacher.

**Dataset** — `dinov3/data/datasets/imagenet_packed.py` (+ loader registration)
- `ImageNetPacked`: mmap blob + index, random access for the SSL sampler.

**Register-token eval package** — `dinov3/eval/register_tokens/`
- `backbone.py` (plain eval backbone + EMA sync), `attention_viz.py`
  (per-register heatmaps + argmax object-centric segmentation),
  `mbo.py` (COCO instance/semantic MBO), `evaluator.py` (scheduling/orchestration).

**Tools** — `register_project/tools/` (parquet repack, COCO download).

---

## Notes & defaults

- **Registers attend to cls + patches only**, never to any register (incl. self).
- **Viz/MBO masks** include register→patch and patch→register views. COCO
  validation also reports last block, penultimate block, all-layer averaged,
  second-half-layer averaged, and last-block per-head masks. The legacy
  `mbo_instance` / `mbo_semantic` keys remain aliases for last-block
  register→patch.
- **MBO** turns register attention into segments via per-pixel argmax over
  registers, then reports mean-best-IoU vs COCO GT (instance = per-object,
  semantic = per-category). GT from `instances_val2017.json` (things classes).
- **LR**: `optim.scaling_rule: sqrt_wrt_1024` scales the base LR by batch size
  automatically (≈2.8e-3 at batch 512). Tune in the YAML if unstable.
- Single-GPU training uses FSDP (`SHARD_GRAD_OP`) + `torch.compile`. Set
  `train.compile: false` for faster startup while iterating.
- **Effective batch 512** is reached via **gradient accumulation**
  (`train.batch_size_per_gpu: 128` × `train.grad_accum_steps: 4`). This box has a
  95 GB GPU but only **31 GB host RAM**, so: a full 512 micro-batch OOMs the GPU,
  and 10 dataloader workers at large batch OOM the host. The 128×4 setup peaks at
  ~38 GB GPU / ~16 GB RAM with `num_workers: 8`. To change the effective batch,
  adjust either factor (e.g. `batch_size_per_gpu=256 grad_accum_steps=2` for fewer,
  larger steps if you have GPU headroom). LR scaling uses the *effective* batch.
- Dataset roots in configs may use `~` (expanded automatically).
- **`train.compile` and host RAM**: `torch.compile=true` grows host RAM (~0.3 GB/min
  via Inductor) and OOMs a 31 GB box in ~50 min (the OOM killer sends SIGTERM →
  `torchrun` reports `Process ... got signal: 15`). The configs ship `compile:
  false`, which is stable (~16 GB flat). Re-enable compile only on a high-RAM
  machine. To diagnose an OOM kill: `journalctl -k --since "10 min ago" | grep -i oom`.
