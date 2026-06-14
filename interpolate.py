"""
Latent-space interpolation between two random faces.

Usage:
    # Single row of 10 steps between two random points
    python interpolate.py --ckpt checkpoints/latest.pt --steps 10 --out samples/interp.png

    # Multiple rows, slerp (smoother for normal latents)
    python interpolate.py --ckpt checkpoints/latest.pt --rows 5 --steps 10 \
        --mode slerp --out samples/interp_slerp.png

    # Save an animated GIF
    python interpolate.py --ckpt checkpoints/latest.pt --steps 30 --gif samples/interp.gif
"""
import argparse
import os
import torch
import imageio
import numpy as np

from config import Config
from models import Generator
from utils import set_seed, save_image_grid, denorm, lerp, slerp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--rows", type=int, default=1, help="number of interpolation rows")
    p.add_argument("--steps", type=int, default=10, help="frames per row (inclusive of endpoints)")
    p.add_argument("--mode", type=str, default="slerp", choices=["slerp", "lerp"])
    p.add_argument("--out", type=str, default="samples/interp.png")
    p.add_argument("--gif", type=str, default="")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--z_dim", type=int, default=Config.z_dim)
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--use_ema", action="store_true", default=Config.use_ema_for_eval)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    G = Generator(z_dim=args.z_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and args.use_ema and ckpt.get("G_ema") is not None:
        G.load_state_dict(ckpt["G_ema"])
    else:
        G.load_state_dict(ckpt["G"] if "G" in ckpt else ckpt)
    G.eval()

    interp_fn = slerp if args.mode == "slerp" else lerp

    all_imgs = []
    for r in range(args.rows):
        z0 = torch.randn(args.z_dim, device=device)
        z1 = torch.randn(args.z_dim, device=device)
        zs = []
        for s in range(args.steps):
            t = s / max(args.steps - 1, 1)
            zs.append(interp_fn(t, z0, z1))
        zs = torch.stack(zs).view(args.steps, args.z_dim, 1, 1)
        with torch.no_grad():
            imgs = G(zs)
        all_imgs.append(imgs)

    grid = torch.cat(all_imgs, dim=0)
    save_image_grid(grid, args.out, nrow=args.steps)
    print(f"[INFO] Saved interpolation grid to {args.out}")

    if args.gif:
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        # Build a per-frame mosaic of the rows (rows stacked vertically)
        # Easiest: just animate the first row.
        first_row = all_imgs[0]
        frames = (denorm(first_row).mul(255).byte()
                  .permute(0, 2, 3, 1).cpu().numpy())
        imageio.mimsave(args.gif, [f for f in frames], duration=0.1)
        print(f"[INFO] Saved animated GIF to {args.gif}")


if __name__ == "__main__":
    main()
