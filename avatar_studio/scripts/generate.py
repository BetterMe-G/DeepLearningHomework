"""CLI for generating avatars.

Examples:

    # 1. random face
    python scripts/generate.py --out out/random.png

    # 2. text-driven via trained mapper
    python scripts/generate.py --mapper red_curly --out out/red.png

    # 3. real photo + style swap
    python scripts/generate.py --ref alice.jpg --style anime --out out/alice_anime.png

    # 4. full combo
    python scripts/generate.py --ref alice.jpg --style anime --mapper red_curly \
        --strength 0.15 --out out/alice_anime_red.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_studio.pipeline import AvatarPipeline
from avatar_studio.utils.logger import get_logger


_log = get_logger("generate")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--text", default=None, help="text prompt")
    p.add_argument("--ref", default=None, help="path to reference photo")
    p.add_argument("--style", default=None, help="style name (matches checkpoints/styles/<name>.pt)")
    p.add_argument("--mapper", default=None, help="mapper name (matches checkpoints/mappers/<name>.pt)")
    p.add_argument("--strength", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="outputs/result.png")
    args = p.parse_args()

    pipe = AvatarPipeline.from_config(args.config)
    res = pipe.generate(
        text=args.text,
        ref_image=args.ref,
        style=args.style,
        mapper=args.mapper,
        strength=args.strength,
        seed=args.seed,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.image.save(args.out)
    _log.info("wrote %s  (style=%s, text=%r)", args.out, res.style, res.text)


if __name__ == "__main__":
    main()
