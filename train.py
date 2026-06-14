"""
DCGAN training script.

Usage:
    python train.py --data_root ./data/lfw --epochs 25 --batch_size 128

Outputs:
    checkpoints/G_epoch_*.pt, D_epoch_*.pt, latest.pt
    samples/iter_*.png
    logs/   (TensorBoard scalars)
"""
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from config import Config
from models import Generator, Discriminator, weights_init
from dataset import get_dataloader
from utils import set_seed, save_image_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=Config.dataset)
    p.add_argument("--data_root", type=str, default=Config.data_root)
    p.add_argument("--image_size", type=int, default=Config.image_size)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--lr_d", type=float, default=Config.lr_d)
    p.add_argument("--beta1", type=float, default=Config.beta1)
    p.add_argument("--beta2", type=float, default=Config.beta2)
    p.add_argument("--label_smooth", type=float, default=Config.label_smooth)
    p.add_argument("--d_noise", type=float, default=Config.d_noise)
    p.add_argument("--hflip_p", type=float, default=Config.hflip_p)
    p.add_argument("--ema_decay", type=float, default=Config.ema_decay)
    p.add_argument("--use_ema_for_eval", action="store_true", default=Config.use_ema_for_eval)
    p.add_argument("--z_dim", type=int, default=Config.z_dim)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--out_dir", type=str, default=Config.out_dir)
    p.add_argument("--sample_dir", type=str, default=Config.sample_dir)
    p.add_argument("--log_dir", type=str, default=Config.log_dir)
    p.add_argument("--save_every", type=int, default=Config.save_every)
    p.add_argument("--sample_every", type=int, default=Config.sample_every)
    p.add_argument("--log_every", type=int, default=Config.log_every)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--resume", type=str, default="",
                   help="path to checkpoint to resume from")
    p.add_argument("--loss", type=str, default="bce",
                   choices=["bce", "hinge", "wgan_gp"],
                   help="loss type: bce (default), hinge, or wgan_gp")
    p.add_argument("--no_sn", action="store_true",
                   help="disable spectral norm on D (required for wgan_gp)")
    p.add_argument("--n_critic", type=int, default=5,
                   help="D updates per G update, used only for wgan_gp")
    p.add_argument("--gp_lambda", type=float, default=10.0,
                   help="gradient penalty weight for wgan_gp")
    return p.parse_args()


def gradient_penalty(D, real, fake, device):
    """WGAN-GP gradient penalty (Gulrajani et al., 2017)."""
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    d_out = D(interp)
    grad = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grad.view(b, -1).norm(2, dim=1) - 1) ** 2).mean()


def main():
    args = parse_args()
    # apply args back to Config so other modules see same values
    for k, v in vars(args).items():
        setattr(Config, k, v)
    Config.ensure_dirs()
    set_seed(args.seed)

    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- Data ----
    ds, loader = get_dataloader(Config)
    print(f"[INFO] Dataset size: {len(ds)} images "
          f"({Config.dataset} @ {Config.data_root})")

    # ---- Models ----
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator(use_sn=not args.no_sn).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    G_ema = deepcopy(G).eval() if args.ema_decay > 0 else None
    if G_ema is not None:
        for p in G_ema.parameters():
            p.requires_grad_(False)

    # ---- Loss & Optim ----
    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=args.lr,   betas=(args.beta1, args.beta2))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Fixed noise for visualizing training progress
    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)
    real_label, fake_label = args.label_smooth, 0.0

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        if G_ema is not None and "G_ema" in ckpt:
            G_ema.load_state_dict(ckpt["G_ema"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[INFO] Resumed from {args.resume} (epoch {start_epoch})")

    writer = SummaryWriter(args.log_dir)
    global_step = 0

    # ---- Train ----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        # Linearly anneal discriminator input noise to 0 by the last epoch.
        noise_std = args.d_noise * max(0.0, 1.0 - epoch / max(args.epochs - 1, 1))
        for i, real in enumerate(pbar):
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            def _noisy(x):
                return x + noise_std * torch.randn_like(x) if noise_std > 0 else x

            # generate fake (shared between D and G updates)
            noise = torch.randn(b, args.z_dim, 1, 1, device=device)
            fake = G(noise)

            # ---------------- (1) Update D ----------------
            D.zero_grad()
            real_in, fake_in = _noisy(real), _noisy(fake.detach())

            if args.loss == "bce":
                out_real = D(real_in)
                out_fake_d = D(fake_in)
                label_r = torch.full((b,), real_label, device=device)
                label_f = torch.full((b,), fake_label, device=device)
                loss_D = criterion(out_real, label_r) + criterion(out_fake_d, label_f)
                D_x    = torch.sigmoid(out_real).mean().item()
                D_G_z1 = torch.sigmoid(out_fake_d).mean().item()
            elif args.loss == "hinge":
                out_real = D(real_in)
                out_fake_d = D(fake_in)
                loss_D = F.relu(1.0 - out_real).mean() + F.relu(1.0 + out_fake_d).mean()
                D_x    = out_real.mean().item()
                D_G_z1 = out_fake_d.mean().item()
            else:  # wgan_gp
                out_real = D(real_in)
                out_fake_d = D(fake_in)
                gp = gradient_penalty(D, real, fake, device)
                loss_D = out_fake_d.mean() - out_real.mean() + args.gp_lambda * gp
                D_x    = out_real.mean().item()
                D_G_z1 = out_fake_d.mean().item()

            loss_D.backward()
            opt_D.step()

            # ---------------- (2) Update G ----------------
            # For wgan_gp update G every n_critic D steps; for others every step.
            do_g = (args.loss != "wgan_gp") or (global_step % args.n_critic == 0)
            loss_G = torch.tensor(0.0, device=device)
            D_G_z2 = 0.0

            if do_g:
                G.zero_grad()
                if args.loss == "bce":
                    out_fake2 = D(fake)
                    loss_G = criterion(out_fake2, torch.full((b,), 1.0, device=device))
                    D_G_z2 = torch.sigmoid(out_fake2).mean().item()
                else:  # hinge or wgan_gp
                    out_fake2 = D(fake)
                    loss_G = -out_fake2.mean()
                    D_G_z2 = out_fake2.mean().item()
                loss_G.backward()
                opt_G.step()
                if G_ema is not None:
                    with torch.no_grad():
                        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                            p_ema.mul_(args.ema_decay).add_(p, alpha=1.0 - args.ema_decay)

            # ---------------- Logging ----------------
            if global_step % args.log_every == 0:
                pbar.set_postfix({
                    "loss_D": f"{loss_D.item():.3f}",
                    "loss_G": f"{loss_G.item():.3f}",
                    "D(x)": f"{D_x:.3f}",
                    "D(G(z))": f"{D_G_z1:.3f}/{D_G_z2:.3f}",
                })
                writer.add_scalar("loss/D", loss_D.item(), global_step)
                writer.add_scalar("loss/G", loss_G.item(), global_step)
                writer.add_scalar("D/real", D_x, global_step)
                writer.add_scalar("D/fake_before", D_G_z1, global_step)
                writer.add_scalar("D/fake_after", D_G_z2, global_step)

            if global_step % args.sample_every == 0:
                G_eval = G_ema if (args.use_ema_for_eval and G_ema is not None) else G
                G_eval.eval()
                with torch.no_grad():
                    fake_grid = G_eval(fixed_noise)
                save_image_grid(
                    fake_grid,
                    os.path.join(args.sample_dir, f"iter_{global_step:07d}.png"),
                    nrow=8,
                )
                G.train()

            global_step += 1

        # ---- end-of-epoch ----
        dt = time.time() - t0
        print(f"[INFO] Epoch {epoch+1} done in {dt:.1f}s")
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "G": G.state_dict(),
                "G_ema": G_ema.state_dict() if G_ema is not None else None,
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "epoch": epoch + 1,
                "z_dim": args.z_dim,
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"epoch_{epoch+1:03d}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))
            print(f"[INFO] Saved checkpoint epoch_{epoch+1:03d}.pt")

    writer.close()
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
