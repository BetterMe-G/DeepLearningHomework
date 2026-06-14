#!/usr/bin/env python3
"""compute_fid.py — compute FID + IS using cached torchvision Inception V3.

Uses the locally cached torchvision Inception V3 weights (avoids torch-fidelity /
pytorch-fid weight downloads, which fail on HPCs with restricted internet).

CUDA optimizations:
  - DataLoader with num_workers for parallel image decode (PIL is single-threaded)
  - pin_memory + non_blocking for faster CPU->GPU transfer
  - bfloat16 autocast on Ampere+ GPUs (A100/A800/...) for ~2x forward-pass speedup
  - Default batch size raised to 64 to keep GPU busy

Usage:
    python3 ablation/scripts/compute_fid.py --name none
    python3 ablation/scripts/compute_fid.py --name ema_only

Writes ablation/eval/<name>/fid.log with format:
    FID: 12.345
    IS: 2.345 ± 0.123
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ABLATION = Path(__file__).resolve().parent.parent


# ============================================================
# Models
# ============================================================
def get_models(device):
    """Two Inception V3 instances sharing the same cached weights:
       - feat_m: head removed, returns 2048-d features (for FID)
       - cls_m:  full, returns 1000-d logits (for IS)
    """
    feat_m = M.inception_v3(weights=M.Inception_V3_Weights.IMAGENET1K_V1)
    feat_m.fc = nn.Identity()
    feat_m.eval().to(device)

    cls_m = M.inception_v3(weights=M.Inception_V3_Weights.IMAGENET1K_V1)
    cls_m.eval().to(device)
    return feat_m, cls_m


# ============================================================
# Data loading (parallel)
# ============================================================
class ImgFolderDataset(Dataset):
    """Lazy PNG loader + Inception preprocess transform."""

    def __init__(self, files, img_size=299):
        self.files = files
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def encode_dir(d, model, device, bs=64, max_n=10000, label="feats",
               num_workers=4, use_amp=True):
    """Forward all images in d through `model`; return numpy (N, dim)."""
    files = sorted(Path(d).glob("*.png"))[:max_n]
    print(f"  {label}: {len(files)} images from {d}")

    ds = ImgFolderDataset(files)
    loader = DataLoader(
        ds,
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # bf16 autocast only on Ampere+ (sm_80+). Fall back to fp32 elsewhere.
    can_amp = (
        use_amp
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device)[0] >= 8
    )

    out = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {label}"):
            batch = batch.to(device, non_blocking=True)
            if can_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    feat = model(batch)
                feat = feat.float()  # back to fp32 for downstream numerics
            else:
                feat = model(batch)
            out.append(feat.cpu().numpy())
    dt = time.time() - t0
    print(f"  {label}: {dt:.1f}s ({len(files) / max(dt, 1e-6):.0f} img/s)")
    return np.concatenate(out, axis=0)


# ============================================================
# Metrics
# ============================================================
def calc_fid(r, f):
    """Standard FID: ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r Sigma_f))."""
    mu_r, mu_f = r.mean(0), f.mean(0)
    sg_r, sg_f = np.cov(r, rowvar=False), np.cov(f, rowvar=False)
    diff = mu_r - mu_f
    cm, _ = linalg.sqrtm(sg_r @ sg_f, disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(diff @ diff + np.trace(sg_r) + np.trace(sg_f) - 2 * np.trace(cm))


def calc_is(logits, splits=10):
    """Inception Score with `splits` partitions (standard = 10)."""
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    N = probs.shape[0]
    scores = []
    for k in range(splits):
        idx = np.arange(k * N // splits, (k + 1) * N // splits)
        part = probs[idx]
        p_y = part.mean(axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))
    return float(np.mean(scores)), float(np.std(scores))


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True,
                   help="run name; reads ablation/eval/<name>/fake, writes ablation/eval/<name>/fid.log")
    p.add_argument("--real_dir", default="./data/fid_real")
    p.add_argument("--max_n", type=int, default=10000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true", help="disable bf16 autocast even on Ampere+")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache", default=str(ABLATION / "results" / "real_feats.npz"))
    a = p.parse_args()

    dev = torch.device(a.device)
    feat_m, cls_m = get_models(dev)
    real_feats_cache = Path(a.cache)
    fake_dir = ABLATION / "eval" / a.name / "fake"
    log_file = ABLATION / "eval" / a.name / "fid.log"

    use_amp = not a.no_amp
    print(f"[{a.name}] device={dev}  bs={a.bs}  workers={a.num_workers}  amp={'bf16' if use_amp else 'off'}")

    # ----- Real features (cached across runs) -----
    if real_feats_cache.exists():
        r = np.load(real_feats_cache)["feats"]
        print(f"[{a.name}] loaded cached real features {r.shape}")
    else:
        print(f"[{a.name}] computing real features from {a.real_dir}")
        r = encode_dir(a.real_dir, feat_m, dev, a.bs, a.max_n, "real feats",
                       a.num_workers, use_amp)
        real_feats_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(real_feats_cache, feats=r)
        print(f"[{a.name}] cached real features")

    # ----- Fake features -----
    print(f"[{a.name}] computing fake features from {fake_dir}")
    f = encode_dir(fake_dir, feat_m, dev, a.bs, a.max_n, "fake feats",
                   a.num_workers, use_amp)

    fid = calc_fid(r, f)
    print(f"[{a.name}] FID = {fid:.3f}")

    # ----- IS on fake images -----
    print(f"[{a.name}] computing Inception Score")
    logits = encode_dir(fake_dir, cls_m, dev, a.bs, a.max_n, "fake logits",
                        a.num_workers, use_amp)
    is_mean, is_std = calc_is(logits)
    print(f"[{a.name}] IS = {is_mean:.3f} ± {is_std:.3f}")

    # ----- Write log file (compare.py parses these two lines) -----
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(f"FID: {fid:.3f}\nIS: {is_mean:.3f} ± {is_std:.3f}\n")
    print(f"[{a.name}] wrote {log_file}")


if __name__ == "__main__":
    main()
