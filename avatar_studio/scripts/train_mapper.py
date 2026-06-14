"""Train a StyleCLIP Latent Mapper for one text description.

Example:
    python scripts/train_mapper.py \
        --description "a person with curly red hair" \
        --output checkpoints/mappers/red_curly.pt \
        --iterations 50000
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import torch

# allow `python scripts/train_mapper.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_studio.config import load_config, resolve_path
from avatar_studio.models.stylegan2 import StyleGAN2Generator
from avatar_studio.models.clip_loss import GlobalCLIPLoss
from avatar_studio.models.id_loss import IDLoss
from avatar_studio.edit.mapper import MapperConfig, TrainConfig, MapperTrainer
from avatar_studio.utils.image import save_image
from avatar_studio.utils.logger import get_logger


_log = get_logger("train_mapper")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--description", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda_clip", type=float, default=None)
    p.add_argument("--lambda_id",   type=float, default=None)
    p.add_argument("--lambda_l2",   type=float, default=None)
    p.add_argument("--no_id", action="store_true",
                   help="disable ArcFace ID loss (use if you don't have the checkpoint)")
    p.add_argument("--no_coarse", action="store_true")
    p.add_argument("--no_medium", action="store_true")
    p.add_argument("--no_fine",   action="store_true")
    p.add_argument("--sample_dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    if device.startswith("cuda"):
        # Respect CUDA_VISIBLE_DEVICES per process. Inside each process,
        # the selected GPU is exposed as cuda:0.
        device = "cuda:0"

    mcfg = cfg.mapper
    tcfg = TrainConfig(
        description=args.description,
        iterations  = args.iterations or mcfg.iterations,
        batch_size  = args.batch_size or mcfg.batch_size,
        lr          = args.lr         or mcfg.lr,
        lambda_clip = args.lambda_clip if args.lambda_clip is not None else mcfg.lambda_clip,
        lambda_id   = args.lambda_id   if args.lambda_id   is not None else mcfg.lambda_id,
        lambda_l2   = args.lambda_l2   if args.lambda_l2   is not None else mcfg.lambda_l2,
        log_every   = mcfg.log_every,
        save_every  = mcfg.save_every,
        truncation  = cfg.truncation,
    )

    G = StyleGAN2Generator(
        ckpt_path=resolve_path(cfg, cfg.checkpoints.stylegan2),
        image_size=cfg.image_size,
        latent_dim=cfg.latent_dim,
        n_mlp=cfg.n_mlp,
        channel_multiplier=cfg.channel_multiplier,
        truncation=cfg.truncation,
        truncation_mean_samples=cfg.truncation_mean_samples,
        device=device,
    )
    clip_loss = GlobalCLIPLoss(device)
    id_loss = None
    if not args.no_id:
        try:
            id_loss = IDLoss(resolve_path(cfg, cfg.checkpoints.arcface), device)
        except FileNotFoundError:
            _log.warning("ArcFace ckpt missing — disabling id loss")
            tcfg.lambda_id = 0.0

    mapper_cfg = MapperConfig(
        n_latent=G.n_latent,
        no_coarse=args.no_coarse or mcfg.no_coarse,
        no_medium=args.no_medium or mcfg.no_medium,
        no_fine=args.no_fine or mcfg.no_fine,
        n_layers=mcfg.n_layers,
        hidden_dim=mcfg.hidden_dim,
    )

    trainer = MapperTrainer(G, clip_loss, id_loss, mapper_cfg, tcfg, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir) if args.sample_dir else out.parent / f"{out.stem}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    _log.info("training mapper for: %r", args.description)
    t0 = time.time()
    for it in range(1, tcfg.iterations + 1):
        m = trainer.step()
        if it % tcfg.log_every == 0:
            dt = time.time() - t0
            _log.info("[%6d/%d] loss=%.4f clip=%.4f l2=%.4f id=%.4f  (%.1fs)",
                      it, tcfg.iterations, m["loss"], m["clip"], m["l2"], m["id"], dt)
        if it % tcfg.save_every == 0 or it == tcfg.iterations:
            torch.save(trainer.state_dict(), out)
            # render a quick before/after sample for visual sanity
            with torch.no_grad():
                w = G.sample_w(1, truncation=cfg.truncation)
                wp = G.w_to_wplus(w)
                src = G.synthesize(wp)
                edited = G.synthesize(wp + trainer.mapper(wp) * 0.1)
            save_image(src,    sample_dir / f"iter_{it:06d}_src.png")
            save_image(edited, sample_dir / f"iter_{it:06d}_edit.png")
            _log.info("saved checkpoint -> %s", out)

    _log.info("done.")


if __name__ == "__main__":
    main()
