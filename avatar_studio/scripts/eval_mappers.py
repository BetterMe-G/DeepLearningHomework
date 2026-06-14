"""Visual sanity-check for trained mappers.

For each `checkpoints/mappers/*.pt`, renders the same N fixed seeds as
before/after pairs and writes one PNG grid per mapper to `out_dir/`.

Run after `train_all_mappers.sh` to eyeball which mappers actually work
before exposing them in the product.

    python scripts/eval_mappers.py --out_dir eval_grids/ --n 8
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
import torchvision.utils as vutils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_studio.config import load_config, resolve_path
from avatar_studio.models.stylegan2 import StyleGAN2Generator
from avatar_studio.edit.mapper import MapperInferer
from avatar_studio.utils.logger import get_logger


_log = get_logger("eval_mappers")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--mappers_dir", default=None,
                   help="defaults to cfg.checkpoints.mappers_dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=8, help="seeds per grid")
    p.add_argument("--strength", type=float, default=0.1)
    p.add_argument("--seed_base", type=int, default=20240901)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    mappers_dir = Path(args.mappers_dir or resolve_path(cfg, cfg.checkpoints.mappers_dir))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    G = StyleGAN2Generator(
        ckpt_path=resolve_path(cfg, cfg.checkpoints.stylegan2),
        image_size=cfg.image_size, latent_dim=cfg.latent_dim,
        n_mlp=cfg.n_mlp, channel_multiplier=cfg.channel_multiplier,
        truncation=cfg.truncation,
        truncation_mean_samples=cfg.truncation_mean_samples,
        device=device,
    )

    # one fixed set of W+ latents shared across all mappers, for direct comparison
    g_cpu = torch.Generator(device=device).manual_seed(args.seed_base)
    z = torch.randn(args.n, cfg.latent_dim, device=device, generator=g_cpu)
    with torch.no_grad():
        w = G.z_to_w(z)
        if cfg.truncation < 1.0:
            w = G.mean_latent + cfg.truncation * (w - G.mean_latent)
        wp = G.w_to_wplus(w)
        src = G.synthesize(wp).clamp_(-1, 1)

    ckpts = sorted(mappers_dir.glob("*.pt"))
    if not ckpts:
        _log.error("no mappers found in %s", mappers_dir); sys.exit(1)
    _log.info("found %d mappers", len(ckpts))

    for ck in ckpts:
        try:
            m = MapperInferer(str(ck), device=device)
        except Exception as e:
            _log.warning("failed to load %s: %s", ck.name, e); continue
        with torch.no_grad():
            edited = G.synthesize(m.edit(wp, strength=args.strength)).clamp_(-1, 1)
        grid = torch.cat([src, edited], dim=0)
        vutils.save_image(grid, out / f"{ck.stem}.png",
                          nrow=args.n, normalize=True, value_range=(-1, 1))
        _log.info("[%s] %r -> %s", ck.stem, m.description, out / f"{ck.stem}.png")


if __name__ == "__main__":
    main()
