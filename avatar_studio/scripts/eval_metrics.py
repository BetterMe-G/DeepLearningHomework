"""Quantitative evaluation for trained StyleCLIP mappers.

For each checkpoint in `checkpoints/mappers/`, this script:
  1. Samples N fixed W+ latents (shared across all mappers for direct comparison)
  2. Renders before/after image pairs
  3. Computes three metrics per mapper:
     * ID  similarity (ArcFace cosine, source vs edited)  — higher = identity preserved
     * CLIP score     (cosine, edited vs the training text prompt) — higher = text aligned
     * LPIPS distance (perceptual, source vs edited)      — lower  = smaller change

Output:
  * `eval_metrics/results.csv` — one row per mapper
  * `eval_metrics/metrics.png` — three bar charts (ID / CLIP / LPIPS)

Usage (run from the avatar_studio/ root):
    python scripts/eval_metrics.py
    python scripts/eval_metrics.py --n 16 --strength 0.1
    python scripts/eval_metrics.py --no_lpips                # skip LPIPS
    python scripts/eval_metrics.py --strength 0.05 0.15      # sweep strengths

Requires:
    pip install lpips matplotlib
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import torch

# Only stdlib + torch at import time. The heavy avatar_studio.* imports
# (which may need to JIT-compile CUDA extensions on first use) are deferred
# to main() so that `--help` and other introspection stay snappy and
# don't fail on machines that don't have a C++ toolchain.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avatar_studio.utils.logger import get_logger  # lightweight, no JIT

_log = get_logger("eval_metrics")

# Same name->prompt mapping as scripts/train_all_mappers.sh so the CLIP
# score is scored against the *training* prompt of each mapper.
PROMPTS: dict[str, str] = {
    "holographic":     "a person with iridescent holographic hair",
    "cyber_tattoo":    "a person with cyberpunk neon face tattoos",
    "golden_freckles": "a person with golden freckles constellation across the face",
    "pastel_split":    "a person with pastel pink and platinum split dyed hair",
    "dark_academia":   "a person with dark academia aesthetic and vintage round glasses",
    "egirl_heart":     "a person with e-girl aesthetic, heart shaped cheek blush",
    "baroque_oil":     "a baroque oil painting portrait of a person",
    "kabuki":          "a person with kabuki theater white face makeup and red accent lines",
    "crystal_skin":    "a person with crystals growing from the skin",
    "neon_noir":       "a person in dramatic neon noir lighting, pink and blue rim light",
}


def pick_device(requested: str) -> str:
    """Honor `cuda` if available, else `mps` on Apple Silicon, else `cpu`."""
    if requested.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if requested.startswith("mps"):
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--mappers_dir", default=None,
                   help="defaults to cfg.checkpoints.mappers_dir")
    p.add_argument("--out_dir", default="eval_metrics")
    p.add_argument("--n", type=int, default=16, help="samples per mapper (shared seeds)")
    p.add_argument("--strength", type=float, default=0.1,
                   help="mapper edit strength; matches eval_mappers.py default")
    p.add_argument("--seed_base", type=int, default=20240901)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--no_clip",  action="store_true")
    p.add_argument("--no_id",    action="store_true")
    return p.parse_args()


def _safe_load_id(arcface_ckpt: str, device: str):
    try:
        m = IDLoss(arcface_ckpt, device)
        _log.info("ID loss ready (ArcFace)")
        return m
    except FileNotFoundError as e:
        _log.warning("ArcFace ckpt missing (%s) — ID metric disabled", e)
        return None
    except Exception as e:
        _log.warning("ArcFace load failed (%s) — ID metric disabled", e)
        return None


def _safe_load_clip(device: str):
    try:
        m = GlobalCLIPLoss(device)
        _log.info("CLIP loss ready (ViT-B/32)")
        return m
    except Exception as e:
        _log.warning("CLIP load failed (%s) — CLIP metric disabled", e)
        return None


def _safe_load_lpips(device: str):
    try:
        import lpips
        fn = lpips.LPIPS(net="alex").to(device)
        for p_ in fn.parameters():
            p_.requires_grad = False
        fn.eval()
        _log.info("LPIPS ready (alex net)")
        return fn
    except ImportError:
        _log.warning("`pip install lpips` to enable LPIPS — LPIPS metric disabled")
        return None
    except Exception as e:
        _log.warning("LPIPS init failed (%s) — LPIPS metric disabled", e)
        return None


def main() -> None:
    # Parse args FIRST so that `--help` / bad flags exit before we touch
    # any module that needs to JIT-compile CUDA extensions.
    args = parse_args()

    # Lazy imports — these may JIT-compile CUDA extensions on first call.
    from avatar_studio.config import load_config, resolve_path
    from avatar_studio.models.stylegan2 import StyleGAN2Generator
    from avatar_studio.models.id_loss import IDLoss
    from avatar_studio.models.clip_loss import GlobalCLIPLoss
    from avatar_studio.edit.mapper import MapperInferer

    cfg = load_config(args.config)
    device = pick_device(cfg.device)
    mappers_dir = Path(args.mappers_dir or resolve_path(cfg, cfg.checkpoints.mappers_dir))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    _log.info("device=%s  n=%d  strength=%.3f  mappers_dir=%s",
              device, args.n, args.strength, mappers_dir)

    # ---- Models ----
    G = StyleGAN2Generator(
        ckpt_path=resolve_path(cfg, cfg.checkpoints.stylegan2),
        image_size=cfg.image_size, latent_dim=cfg.latent_dim,
        n_mlp=cfg.n_mlp, channel_multiplier=cfg.channel_multiplier,
        truncation=cfg.truncation,
        truncation_mean_samples=cfg.truncation_mean_samples,
        device=device,
    )
    id_loss    = None if args.no_id    else _safe_load_id(resolve_path(cfg, cfg.checkpoints.arcface), device)
    clip_loss  = None if args.no_clip  else _safe_load_clip(device)
    lpips_fn   = None if args.no_lpips else _safe_load_lpips(device)

    # ---- Shared source W+ and source images ----
    g = torch.Generator(device=device).manual_seed(args.seed_base)
    z = torch.randn(args.n, cfg.latent_dim, device=device, generator=g)
    with torch.no_grad():
        w = G.z_to_w(z)
        if cfg.truncation < 1.0:
            w = G.mean_latent + cfg.truncation * (w - G.mean_latent)
        wp_src = G.w_to_wplus(w)
        src = G.synthesize(wp_src).clamp_(-1, 1)

    # Pre-encode source ID once (L2-normalised for cosine)
    f_src_id = None
    if id_loss is not None:
        with torch.no_grad():
            f_src_id = id_loss.extract(src)
            f_src_id = f_src_id / f_src_id.norm(dim=-1, keepdim=True)

    # Pre-encode all text prompts once
    text_feats: dict[str, torch.Tensor] = {}
    if clip_loss is not None:
        for name, prompt in PROMPTS.items():
            with torch.no_grad():
                text_feats[name] = clip_loss.encode_text_mean(prompt)  # (1, D)

    # ---- Iterate mappers ----
    ckpts = sorted(mappers_dir.glob("*.pt"))
    if not ckpts:
        _log.error("no mappers found in %s", mappers_dir); sys.exit(1)
    _log.info("found %d mappers", len(ckpts))

    rows: list[dict] = []
    for ck in ckpts:
        name = ck.stem
        prompt = PROMPTS.get(name, f"a photo of a person with {name.replace('_', ' ')}")

        try:
            mapper = MapperInferer(str(ck), device=device)
        except Exception as e:
            _log.warning("failed to load %s: %s — skipping", ck.name, e); continue

        with torch.no_grad():
            wp_edit = mapper.edit(wp_src, strength=args.strength)
            edited  = G.synthesize(wp_edit).clamp_(-1, 1)

        # ID similarity
        id_score = float("nan")
        if id_loss is not None and f_src_id is not None:
            with torch.no_grad():
                f_edit = id_loss.extract(edited)
                f_edit = f_edit / f_edit.norm(dim=-1, keepdim=True)
                id_score = (f_edit * f_src_id).sum(dim=-1).mean().item()

        # CLIP score (cosine to prompt)
        clip_score = float("nan")
        if clip_loss is not None:
            with torch.no_grad():
                f_img = clip_loss.encode_image(edited)        # (N, D)
            tf = text_feats.get(name) or clip_loss.encode_text_mean(prompt)
            clip_score = (f_img * tf).sum(dim=-1).mean().item()

        # LPIPS distance
        lpips_score = float("nan")
        if lpips_fn is not None:
            with torch.no_grad():
                d = lpips_fn(src, edited).flatten()          # (N,)
                lpips_score = d.mean().item()

        rows.append({
            "mapper": name,
            "prompt": prompt,
            "id_sim": id_score,
            "clip":   clip_score,
            "lpips":  lpips_score,
        })
        _log.info("[%s]  ID=%.3f  CLIP=%.3f  LPIPS=%.3f",
                  name, id_score, clip_score, lpips_score)

    # ---- CSV ----
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mapper", "prompt", "id_sim", "clip", "lpips"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    _log.info("wrote %s", csv_path)

    # ---- Stdout table ----
    print()
    print("=" * 90)
    print(f"{'mapper':<18}{'ID ↑':>10}{'CLIP ↑':>10}{'LPIPS ↓':>10}   prompt")
    print("-" * 90)
    for r in rows:
        print(f"{r['mapper']:<18}"
              f"{r['id_sim']:>10.3f}"
              f"{r['clip']:>10.3f}"
              f"{r['lpips']:>10.3f}"
              f"   {r['prompt']}")
    print("=" * 90)

    # ---- Bar chart ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["mapper"] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        axes[0].bar(names, [r["id_sim"] for r in rows], color="#156082")
        axes[0].set_title("ID similarity ↑ (ArcFace cos, src vs edited)")
        axes[0].set_ylim(0, 1); axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(names, [r["clip"] for r in rows], color="#E97132")
        axes[1].set_title("CLIP score ↑ (cos, edited vs prompt)")
        axes[1].set_ylim(0, 0.4); axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(names, [r["lpips"] for r in rows], color="#0F9ED5")
        axes[2].set_title("LPIPS ↓ (perceptual distance)")
        axes[2].tick_params(axis="x", rotation=45)

        fig.tight_layout()
        chart_path = out / "metrics.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        _log.info("wrote %s", chart_path)
    except ImportError:
        _log.warning("matplotlib not installed — skipping chart (CSV still saved)")
    except Exception as e:
        _log.warning("matplotlib chart failed: %s", e)


if __name__ == "__main__":
    main()
