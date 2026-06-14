# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo contains two deep learning projects for face generation:

1. **Root project** — DCGAN trained on CelebA/LFW to generate 64×64 faces.
2. **`avatar_studio/`** — StyleGAN2-FFHQ + StyleCLIP Mapper for text-driven 1024×1024 avatar editing (no diffusion).

---

## Setup

```bash
pip install -r requirements.txt                       # root DCGAN project
pip install -r avatar_studio/requirements.txt         # avatar_studio sub-project
pip install git+https://github.com/openai/CLIP.git   # avatar_studio only
```

Checkpoints are excluded from git (`.gitignore` blocks `*.pt`, `*.pth`). Data is excluded too — placed under `./data/`.

---

## DCGAN — Common Commands

**Train:**
```bash
python train.py --data_root ./data/celeba/img_align_celeba --epochs 80
python train.py --data_root ./data/lfw --dataset lfw --epochs 25 --batch_size 128
python train.py --resume checkpoints/latest.pt   # resume from checkpoint
```

**Generate samples:**
```bash
python generate.py --ckpt checkpoints/latest.pt --num 64 --out samples/gen.png
python generate.py --ckpt checkpoints/latest.pt --num 1000 --out samples/fid_fake --as_dir
```

**Evaluate (FID + Inception Score):**
```bash
python evaluate.py --ckpt checkpoints/latest.pt --num 10000
python evaluate.py --ckpt checkpoints/latest.pt --skip_real   # reuse cached real images
```

**Latent interpolation:**
```bash
python interpolate.py --ckpt checkpoints/latest.pt --steps 10 --mode slerp --out samples/interp.png
python interpolate.py --ckpt checkpoints/latest.pt --rows 5 --steps 30 --gif samples/interp.gif
```

**TensorBoard:**
```bash
tensorboard --logdir logs/
```

---

## Avatar Studio — Common Commands

All commands run from the **`avatar_studio/`** directory.

**Train mappers:**
```bash
bash scripts/train_all_mappers.sh                          # all 10 presets (skips existing)
bash scripts/train_all_mappers.sh holographic              # single preset
ITERATIONS=20000 bash scripts/train_all_mappers.sh        # quick test run
```

**Evaluate mappers (visual before/after grids):**
```bash
python scripts/eval_mappers.py --out_dir eval_grids/ --n 8 --strength 0.1
```

**Generate an avatar:**
```python
from avatar_studio.pipeline import AvatarPipeline
pipe = AvatarPipeline.from_config()
img = pipe.generate(text="holographic", mapper_ckpt="checkpoints/mappers/holographic.pt", seed=42).image
```

**e4e inversion (photo → W+ latent):**
```bash
python scripts/invert.py --input photo.jpg --out latent.pt
```

---

## Architecture

### DCGAN (root)

- **`config.py`** — Single `Config` class; all scripts import it and patch it with `argparse` values. All paths and hyperparameters live here.
- **`models.py`** — `Generator` (5 ConvTranspose2d blocks, z→64×64) and `Discriminator` (5 Conv2d blocks with spectral norm, no BN). Spectral norm on D prevents BN/SN conflicts.
- **`train.py`** — Standard DCGAN loop: update D on real+fake, then update G. Optional EMA of G weights (`--ema_decay`), optional instance noise on D inputs (`--d_noise`), optional label smoothing (`--label_smooth`). Checkpoints saved as `epoch_NNN.pt` + `latest.pt`.
- **`dataset.py`** — `FlatImageDataset` recursively finds all images under a root. `get_dataloader` dispatches by `cfg.dataset` (`lfw`, `celeba`, `folder`). CelebA and LFW both use a 178-px center crop before resizing to `image_size`.
- **`utils.py`** — `denorm` converts from `[-1,1]` to `[0,1]`. `slerp`/`lerp` for latent interpolation.
- **`evaluate.py`** — Dumps real and fake images to flat directories, then calls `fidelity` CLI (falls back to `pytorch_fid`).

### Avatar Studio (`avatar_studio/`)

- **`pipeline.py`** — `AvatarPipeline`: top-level entry; loads StyleGAN2 + e4e encoder + mapper, runs end-to-end.
- **`edit/mapper.py`** — `MapperTrainer`: trains a small MLP that predicts Δw in W+ space. Only the mapper (~10M params) trains; StyleGAN2, CLIP, and ArcFace are frozen. Loss = λ_clip·CLIP + λ_id·ArcFace cosine + λ_l2·‖Δw‖².
- **`models/`** — `stylegan2.py` (StyleGAN2 wrapper), `e4e.py` (encoder4editing wrapper), `clip_loss.py`, `id_loss.py`.
- **`vendor/`** — Vendored third-party code: StyleGAN2 (rosinality port), psp encoders, InsightFace IR-SE-50. Do not modify.
- **`configs/default.yaml`** — Default paths for checkpoints (`checkpoints/`), mapper outputs, etc.

### Checkpoint format (DCGAN)

```python
{
  "G": G.state_dict(),
  "G_ema": G_ema.state_dict(),   # None if ema_decay == 0
  "D": D.state_dict(),
  "opt_G": opt_G.state_dict(),
  "opt_D": opt_D.state_dict(),
  "epoch": int,
  "z_dim": int,
}
```

All inference scripts (`generate.py`, `evaluate.py`, `interpolate.py`) prefer `G_ema` weights when available (`use_ema_for_eval=True`).
