"""
Latent-space linear and spherical interpolation in W+ for a trained StyleGAN2.

Loads a network snapshot (.pkl), samples random pairs of W+ vectors, and
produces both an interpolation grid (PNG) and an animated GIF per mode.

Usage:
    # Use the latest snapshot automatically:
    bash interpolate.sh

    # Or specify explicitly:
    python interpolate.py \\
        --network runs/00000-celeba64*/network-snapshot-025000.pkl \\
        --rows 5 --steps 16 --mode both
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import imageio

# Make upstream stylegan2-ada-pytorch modules importable
UPSTREAM = Path(__file__).resolve().parent.parent / "stylegan2-ada-pytorch"
sys.path.insert(0, str(UPSTREAM))

import dnnlib
import legacy


# ---------------------------------------------------------------------------
# Interpolation primitives (in W+ space, num_ws × w_dim tensors)
# ---------------------------------------------------------------------------

def lerp_w(val, w_a, w_b):
    """Linear interp on the full W+ tensor (num_ws, w_dim)."""
    return (1.0 - val) * w_a + val * w_b


def _slerp_layer(val, low, high):
    """Spherical interpolation between two (1, w_dim) layer vectors.

    Falls back to lerp when the two endpoints are near-collinear (sin(omega)
    close to zero) to avoid division-by-zero / NaN.
    """
    eps = 1e-6
    low_n = low / (low.norm(dim=-1, keepdim=True) + eps)
    high_n = high / (high.norm(dim=-1, keepdim=True) + eps)
    omega = torch.acos((low_n * high_n).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps))
    so = torch.sin(omega)
    near_collinear = so.abs() < 1e-4
    s = (torch.sin((1.0 - val) * omega) / (so + eps)) * low + \
        (torch.sin(val * omega) / (so + eps)) * high
    l = (1.0 - val) * low + val * high
    return torch.where(near_collinear.expand_as(s), l, s)


def slerp_w(val, w_a, w_b):
    """Per-layer spherical interp across the W+ tensor (num_ws, w_dim)."""
    val = torch.as_tensor(val, dtype=w_a.dtype, device=w_a.device)
    layers = [_slerp_layer(val, w_a[i], w_b[i]) for i in range(w_a.shape[0])]
    return torch.stack(layers)


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def _to_uint8(t):
    """[-1, 1] float tensor -> uint8 [0, 255] numpy array, shape (N, H, W, C)."""
    t = (t.clamp(-1, 1) + 1) * 127.5
    return t.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()


def save_png_grid(images, path, nrow):
    """Save a (N, 3, H, W) tensor in [-1, 1] as a grid PNG (uint8)."""
    from PIL import Image
    arr = _to_uint8(images)
    rows = [np.concatenate(list(arr[r:r + nrow]), axis=1) for r in range(0, arr.shape[0], nrow)]
    grid = np.concatenate(rows, axis=0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(grid).save(path)
    print(f"[INFO] Saved grid ({grid.shape[1]}x{grid.shape[0]}) to {path}")


def save_gif(images, path, duration=0.1):
    """Save a (N, 3, H, W) tensor in [-1, 1] as an animated GIF."""
    arr = _to_uint8(images)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, list(arr), duration=duration)
    print(f"[INFO] Saved GIF ({arr.shape[0]} frames) to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--network", type=str, default="",
                   help="Path to network-snapshot-XXXXXX.pkl; default = latest under runs/")
    p.add_argument("--rows", type=int, default=5, help="number of interpolation rows")
    p.add_argument("--steps", type=int, default=16, help="frames per row (incl. endpoints)")
    p.add_argument("--mode", type=str, default="both", choices=["lerp", "slerp", "both"])
    p.add_argument("--out_dir", type=str, default="samples/")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _latest_snapshot():
    p = Path("runs")
    pkls = sorted(p.glob("*/network-snapshot-*.pkl"), key=lambda x: x.stat().st_mtime)
    if not pkls:
        raise FileNotFoundError("No network-snapshot-*.pkl found under runs/")
    return str(pkls[-1])


def main():
    args = parse_args()
    network_path = args.network or _latest_snapshot()
    print(f"[INFO] Loading network from {network_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(network_path) as f:
        net = legacy.load_network_pkl(f)
    G = net["G_ema"].to(device).eval().requires_grad_(False)

    num_ws = G.num_ws
    w_dim = G.w_dim
    z_dim = G.z_dim
    print(f"[INFO] Generator: img_resolution={G.img_resolution}, num_ws={num_ws}, w_dim={w_dim}, z_dim={z_dim}")

    torch.manual_seed(args.seed)
    z = torch.randn(2 * args.rows, z_dim, device=device)
    w_all = G.mapping(z, None)  # (2R, num_ws, w_dim)
    pairs = [(w_all[2 * r], w_all[2 * r + 1]) for r in range(args.rows)]

    def render_row(w_a, w_b, mode):
        ws = []
        for s in range(args.steps):
            t = s / max(args.steps - 1, 1)
            if mode == "lerp":
                ws.append(lerp_w(t, w_a, w_b))
            else:
                ws.append(slerp_w(t, w_a, w_b))
        ws = torch.stack(ws)  # (steps, num_ws, w_dim)
        with torch.no_grad():
            imgs = G.synthesis(ws, noise_mode="const", force_fp32=True)
        return imgs  # (steps, 3, H, W) in [-1, 1]

    modes = ["lerp", "slerp"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"[INFO] Generating {mode} interpolation ({args.rows} rows × {args.steps} steps)...")
        row_imgs = [render_row(a, b, mode) for a, b in pairs]
        all_imgs = torch.cat(row_imgs, dim=0)
        save_png_grid(all_imgs, os.path.join(args.out_dir, f"interp_{mode}.png"), nrow=args.steps)
        # GIF animates only the first row (so the file size is reasonable)
        save_gif(row_imgs[0], os.path.join(args.out_dir, f"interp_{mode}.gif"))


if __name__ == "__main__":
    main()
