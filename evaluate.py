"""
Evaluate FID and IS for generated faces.

Steps:
  1. Dump N real images (resized to image_size) into a flat folder.
  2. Generate N fake images from the checkpoint into a flat folder.
  3. Use torch-fidelity to compute FID and Inception Score.

Usage:
    python evaluate.py --ckpt checkpoints/latest.pt --num 10000

Requires:
    pip install torch-fidelity pytorch-fid
"""
import argparse
import os
import subprocess
import torch
from tqdm import tqdm
from PIL import Image

from config import Config
from models import Generator
from dataset import get_dataloader
from utils import set_seed, save_tensor_as_png, denorm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--dataset", type=str, default=Config.dataset)
    p.add_argument("--data_root", type=str, default=Config.data_root)
    p.add_argument("--image_size", type=int, default=Config.image_size)
    p.add_argument("--real_dir", type=str, default=Config.fid_real_dir)
    p.add_argument("--fake_dir", type=str, default=Config.fid_fake_dir)
    p.add_argument("--z_dim", type=int, default=Config.z_dim)
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--use_ema", action="store_true", default=Config.use_ema_for_eval)
    p.add_argument("--skip_real", action="store_true",
                   help="reuse existing real_dir without re-dumping.")
    return p.parse_args()


def dump_real_images(cfg, real_dir, n):
    os.makedirs(real_dir, exist_ok=True)
    ds, loader = get_dataloader(cfg)
    n = min(n, len(ds))
    print(f"[INFO] Dumping {n} real images to {real_dir} ...")
    written = 0
    for batch in tqdm(loader):
        for i in range(batch.size(0)):
            if written >= n:
                return
            save_tensor_as_png(batch[i],
                               os.path.join(real_dir, f"{written:06d}.png"))
            written += 1


def dump_fake_images(G, fake_dir, n, z_dim, batch, device):
    os.makedirs(fake_dir, exist_ok=True)
    print(f"[INFO] Generating {n} fake images to {fake_dir} ...")
    G.eval()
    written = 0
    with torch.no_grad():
        pbar = tqdm(total=n)
        while written < n:
            b = min(batch, n - written)
            z = torch.randn(b, z_dim, 1, 1, device=device)
            fake = G(z)
            for i in range(b):
                save_tensor_as_png(fake[i],
                                   os.path.join(fake_dir, f"{written+i:06d}.png"))
            written += b
            pbar.update(b)
        pbar.close()


def main():
    args = parse_args()
    # apply args to Config so dataset.get_dataloader sees them
    for k, v in vars(args).items():
        setattr(Config, k, v)
    set_seed(Config.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. real images
    if not args.skip_real:
        # Force batch_size 1 for cleaner dumping? We can keep cfg as-is.
        dump_real_images(Config, args.real_dir, args.num)
    else:
        print(f"[INFO] Skipping real-image dump (reusing {args.real_dir}).")

    # 2. fake images
    G = Generator(z_dim=args.z_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and args.use_ema and ckpt.get("G_ema") is not None:
        G.load_state_dict(ckpt["G_ema"])
    else:
        G.load_state_dict(ckpt["G"] if "G" in ckpt else ckpt)
    dump_fake_images(G, args.fake_dir, args.num, args.z_dim, args.batch, device)

    # 3. metrics via torch-fidelity (FID + Inception Score)
    print("[INFO] Computing FID and IS via torch-fidelity ...")
    cmd = [
        "fidelity",
        "--gpu", "0",
        "--fid",
        "--isc",
        "--input1", args.fake_dir,
        "--input2", args.real_dir,
    ]
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[WARN] `fidelity` CLI failed ({e}).")
        print("       Falling back to pytorch-fid for FID only.")
        subprocess.run(["python", "-m", "pytorch_fid",
                        args.fake_dir, args.real_dir], check=False)


if __name__ == "__main__":
    main()
