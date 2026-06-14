"""
Generate face samples from a trained DCGAN checkpoint.

Usage:
    python generate.py --ckpt checkpoints/latest.pt --num 64 --out samples/gen.png
    python generate.py --ckpt checkpoints/latest.pt --num 1000 --out samples/fid_fake --as_dir
"""
import argparse
import os
import torch
from tqdm import tqdm

from config import Config
from models import Generator
from utils import set_seed, save_image_grid, save_tensor_as_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num", type=int, default=64, help="number of samples")
    p.add_argument("--out", type=str, default="samples/gen.png",
                   help="output path. If --as_dir, treated as directory.")
    p.add_argument("--as_dir", action="store_true",
                   help="write each sample as its own PNG (for FID).")
    p.add_argument("--batch", type=int, default=64)
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

    if args.as_dir:
        os.makedirs(args.out, exist_ok=True)
        n_done = 0
        with torch.no_grad():
            pbar = tqdm(total=args.num, desc="Generating")
            while n_done < args.num:
                b = min(args.batch, args.num - n_done)
                z = torch.randn(b, args.z_dim, 1, 1, device=device)
                fake = G(z)
                for i in range(b):
                    save_tensor_as_png(
                        fake[i], os.path.join(args.out, f"{n_done+i:06d}.png")
                    )
                n_done += b
                pbar.update(b)
            pbar.close()
        print(f"[INFO] Wrote {args.num} images to {args.out}")
    else:
        with torch.no_grad():
            z = torch.randn(args.num, args.z_dim, 1, 1, device=device)
            fake = G(z)
        nrow = int(args.num ** 0.5) or 1
        save_image_grid(fake, args.out, nrow=nrow)
        print(f"[INFO] Wrote grid to {args.out}")


if __name__ == "__main__":
    main()
