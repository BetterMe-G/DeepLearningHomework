# StyleGAN2-ADA on CelebA 64×64

Trains StyleGAN2 (config F-style, no progressive growing) on CelebA at 64×64
resolution using NVIDIA's official `stylegan2-ada-pytorch` engine, then
produces sample grids, FID/IS numbers, and linear + spherical W+ interpolation.

## Hardware

Designed for **1× NVIDIA A800 (80GB)**. Training takes ~10–15h for 25,000 kimg.
For other GPU counts, override the `GPUS` env var (must be a power of 2: 1/2/4/8).

## Run on a remote A800 server

```bash
# (1) On the LOCAL Mac: rsync the project to the server.
#     Only stylegan2_ada/ needs uploading; data/celeba/ and stylegan2-ada-pytorch/
#     are already on the server.
rsync -avP --exclude='data/' --exclude='runs/' --exclude='samples/' \
    ./stylegan2_ada/ \
    <user>@<server>:/hpc_stor03/sjtu_home/siru.ge/DeepLearningHomework/stylegan2_ada/

# (2) SSH into the server and run the one-shot launcher:
ssh <user>@<server>
cd /hpc_stor03/sjtu_home/siru.ge/DeepLearningHomework/stylegan2_ada
bash launch_remote.sh
```

`launch_remote.sh` does it all: env check → CelebA path resolve → `prepare_data.sh`
(skip if `data/celeba64.zip` already exists) → `train.sh` in background with nohup
→ prints monitoring commands.

It auto-detects the default CelebA path `../data/celeba/img_align_celeba`. If your
path is different, pass it explicitly:
```bash
CELEBA_SRC=/path/to/img_align_celeba bash launch_remote.sh
```

## Quick start (if CelebA is already prepared)

```bash
cd stylegan2_ada
bash prepare_data.sh        # one-time: builds data/celeba64.zip (skip if exists)
bash train.sh               # launches torchrun; ~10-15h, can run in background
bash generate_samples.sh    # 8×8 grid of generated faces
bash eval_metrics.sh        # final FID-50k_full + IS-50k
bash interpolate.sh         # lerp + slerp grids + GIFs
```

`interpolate.sh` and `generate_samples.sh` pick the latest snapshot automatically
(latest mtime), so you can rerun them after intermediate snapshots.

## Layout

```
stylegan2_ada/
├── README.md                # this file
├── prepare_data.sh          # CelebA → 64×64 ZIP
├── train.sh                 # torchrun launcher (env: GPUS, BATCH, KIMG, RUN_NAME)
├── generate_samples.sh      # wraps upstream generate.py
├── eval_metrics.sh          # wraps upstream calc_metrics.py
├── interpolate.py           # custom lerp + slerp in W+ space
├── interpolate.sh           # runs interpolate.py on the latest snapshot
├── data/                    # celeba64.zip  (gitignored)
├── runs/                    # network-snapshot-*.pkl + metric-fid50k_full.jsonl  (gitignored)
└── samples/                 # grid/, interp_*.png, interp_*.gif  (gitignored)
```

## Training knobs (env vars on `train.sh`)

| Var        | Default       | Notes                                            |
|------------|---------------|--------------------------------------------------|
| `GPUS`     | `1`           | Must be a power of 2: 1 / 2 / 4 / 8.             |
| `BATCH`    | `32`          | Per-GPU batch size.                              |
| `KIMG`     | `25000`       | Total images seen (in thousands).                |
| `RUN_NAME` | `celeba64`    | Subdir under `runs/`.                            |

Quick presets:
```bash
KIMG=5000  bash train.sh   # quick 3-5h sanity run
KIMG=50000 BATCH=64 bash train.sh  # higher quality, longer
GPUS=4 KIMG=25000 bash train.sh   # DDP across 4 A800s, ~5-7h
```

## Monitoring

```bash
tail -f runs/*/log.txt                          # per-tick training log
tail -f runs/*/metric-fid50k_full.jsonl         # FID curve (decreasing is good)
nvidia-smi                                       # GPU utilization
```

The training loop writes `metric-fid50k_full.jsonl` and `metric-is50k.jsonl`
into the run dir. The first FID evaluation runs after a few kimg.

## Outputs

After training completes:

- `samples/grid/seed0000.png` … `seed0063.png` — 64 individual 64×64 face PNGs
- `samples/metrics/final.txt` — final FID-50k_full and IS-50k (via `tee`)
- `samples/interp_lerp.png`, `samples/interp_slerp.png` — 5×16 interpolation grids
- `samples/interp_lerp.gif`, `samples/interp_slerp.gif` — animated, first row only

## Configuration choices

- **`--cfg=stylegan2`** — no progressive growing; modern equivalent of original
  config F. Stable at 64×64.
- **`--aug=noaug`** — CelebA's 200k images don't need adaptive augmentation.
- **`--mirror=1`** — horizontal-flip augmentation, free quality win.
- **`--gamma=1`** — R1 weight, paper default.
- **`--metrics=fid50k_full,is50k`** — full FID (50k) and IS (50k), evaluated
  periodically during training. FID uses the training set as the reference
  (standard StyleGAN2 practice).
- **`--snap=10`** — keep enough snapshots to pick a best-by-FID if the final
  isn't optimal.

## Expected results

For 25,000 kimg on 1×A800 with the above config, expect:
- **FID-50k_full**: ~5–15 (lower is better)
- **IS-50k**: ~2.0–2.5 (higher is better)

Numbers vary with the seed and stochastic training order; if the first run
plateaus early, try a different `--seed` or a longer `KIMG`.

## Reusing the trained model

The `.pkl` snapshot in `runs/` is a complete pickle of the Generator, EMA
Generator, Discriminator, and training options. To use it elsewhere:

```python
import sys, torch
sys.path.insert(0, "/path/to/stylegan2-ada-pytorch")
import legacy, dnnlib
with dnnlib.util.open_url("runs/00000-celeba64*/network-snapshot-025000.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"].cuda().eval()
imgs = G.synthesis(G.mapping(torch.randn(8, G.z_dim, device="cuda"), None,
                             skip_w_avg=False),
                   noise_mode="const", force_fp32=True)
# imgs: (8, 3, 64, 64) in [-1, 1]
```

## Notes

- ADA disabled, mirror enabled.
- FID-50k_full is slow (~50k Inception forward passes); plan ~10–15 min for
  the final re-evaluation.
- Inception V3 weights are downloaded on first `eval_metrics.sh` run (~100MB,
  cached at `~/.cache/torch_hub`).
- A800 may spend 2–3 min JIT-compiling custom ops on first launch. Don't be
  alarmed by early "0%" progress.
