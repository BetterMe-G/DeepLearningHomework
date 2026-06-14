"""AvatarPipeline — top-level facade for the avatar studio.

Combines:
  - StyleGAN2-FFHQ (base) or a NADA/JoJoGAN-finetuned G (style swap)
  - e4e inversion (optional reference photo)
  - StyleCLIP Mapper text edits

Typical usage:

    pipe = AvatarPipeline.from_config()                   # load weights once
    img = pipe.generate(text="a person with blue hair",   # text only
                        style="anime",                    # optional style
                        ref_image="alice.jpg",            # optional photo
                        seed=42)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import threading
import PIL.Image
import torch

from .config import load_config, resolve_path
from .models.stylegan2 import StyleGAN2Generator
from .models.e4e import E4EInverter, invert_image
from .edit.mapper import MapperInferer
from .utils.image import align_face, pil_to_tensor, tensor_to_pil
from .utils.logger import get_logger


_log = get_logger("avatar.pipeline")


@dataclass
class GenerateResult:
    image: PIL.Image.Image
    wplus: torch.Tensor               # final W+ used
    seed: Optional[int] = None
    used_ref: bool = False
    style: Optional[str] = None
    text: Optional[str] = None


class AvatarPipeline:
    """Heavy, long-lived object. Construct once at server startup."""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.device = self.cfg.device

        _log.info("loading StyleGAN2 ...")
        self.G_base = StyleGAN2Generator(
            ckpt_path=resolve_path(self.cfg, self.cfg.checkpoints.stylegan2),
            image_size=self.cfg.image_size,
            latent_dim=self.cfg.latent_dim,
            n_mlp=self.cfg.n_mlp,
            channel_multiplier=self.cfg.channel_multiplier,
            truncation=self.cfg.truncation,
            truncation_mean_samples=self.cfg.truncation_mean_samples,
            device=self.device,
        )

        # caches
        self._style_Gs: dict[str, StyleGAN2Generator] = {}
        self._mapper_cache: dict[str, MapperInferer] = {}
        self._e4e: Optional[E4EInverter] = None
        self._lock = threading.RLock()                 # multi-threaded API safety

    # --------------------------- generators ---------------------------

    def _load_style_G(self, style: str) -> StyleGAN2Generator:
        if style in self._style_Gs:
            return self._style_Gs[style]
        path = Path(resolve_path(self.cfg, self.cfg.checkpoints.styles_dir)) / f"{style}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Style checkpoint not found: {path}. "
                                    f"Train one with StyleGAN-NADA first.")
        _log.info("loading style G '%s' from %s", style, path)
        G = StyleGAN2Generator(
            ckpt_path=str(path),
            image_size=self.cfg.image_size,
            latent_dim=self.cfg.latent_dim,
            n_mlp=self.cfg.n_mlp,
            channel_multiplier=self.cfg.channel_multiplier,
            truncation=self.cfg.truncation,
            truncation_mean_samples=self.cfg.truncation_mean_samples,
            device=self.device,
        )
        self._style_Gs[style] = G
        return G

    # --------------------------- mappers ------------------------------

    def _load_mapper(self, name: str) -> MapperInferer:
        if name in self._mapper_cache:
            return self._mapper_cache[name]
        path = Path(resolve_path(self.cfg, self.cfg.checkpoints.mappers_dir)) / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Mapper checkpoint not found: {path}")
        m = MapperInferer(str(path), device=self.device)
        self._mapper_cache[name] = m
        return m

    # --------------------------- inversion ---------------------------

    def _invert(self, ref_image: str | PIL.Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(ref_image, str):
            aligned = align_face(ref_image,
                                 resolve_path(self.cfg, self.cfg.checkpoints.dlib_landmarks),
                                 output_size=self.cfg.image_size)
        else:
            aligned = ref_image.convert("RGB")
            if aligned.size != (self.cfg.image_size, self.cfg.image_size):
                aligned = aligned.resize((self.cfg.image_size, self.cfg.image_size),
                                         PIL.Image.LANCZOS)
        cache_dir = Path(resolve_path(self.cfg, self.cfg.checkpoints.inversion_cache))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "last_inversion.pt"
        if self._e4e is None:
            _log.info("loading e4e encoder ...")
            self._e4e = E4EInverter(
                resolve_path(self.cfg, self.cfg.checkpoints.e4e),
                device=self.device,
            )
        wplus = invert_image(aligned,
                             ckpt_path=resolve_path(self.cfg, self.cfg.checkpoints.e4e),
                             cache_path=str(cache_path),
                             device=self.device,
                             inverter=self._e4e)
        target = pil_to_tensor(aligned, size=self.cfg.image_size).to(self.device)
        return wplus, target

    # --------------------------- core API ----------------------------

    @torch.no_grad()
    def _sample_w(self, seed: Optional[int]) -> torch.Tensor:
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            z = torch.randn(1, self.cfg.latent_dim, device=self.device, generator=g)
            w = self.G_base.z_to_w(z)
            t = self.cfg.truncation
            if t < 1.0:
                w = self.G_base.mean_latent + t * (w - self.G_base.mean_latent)
        else:
            w = self.G_base.sample_w(1)
        return self.G_base.w_to_wplus(w)

    @torch.no_grad()
    def generate(self,
                 text: Optional[str] = None,
                 ref_image: Optional[str | PIL.Image.Image] = None,
                 style: Optional[str] = None,
                 mapper: Optional[str] = None,
                 strength: float = 0.1,
                 seed: Optional[int] = None,
                 ) -> GenerateResult:
        """Run end-to-end generation.

        Mutually compatible flags (you can mix):
          text + mapper                  → mapper-based attribute edit
          ref_image                      → e4e inversion (sets w from photo)
          style                          → swap to a NADA-trained G

        Without any flags: returns a random truncated face from the base G.
        """
        with self._lock:
            # ---- 1. choose G ----
            G = self._load_style_G(style) if style else self.G_base

            # ---- 2. compute pivot w+ ----
            if ref_image is not None:
                wplus, _target = self._invert(ref_image)
            else:
                wplus = self._sample_w(seed)

            # ---- 3. text edit ----
            if text and not mapper:
                raise ValueError("text edit requires --mapper in mapper-only mode.")
            if text and mapper:
                m = self._load_mapper(mapper)
                wplus = m.edit(wplus, strength=strength)

            # ---- 4. render ----
            img = G.synthesize(wplus)
            pil = tensor_to_pil(img)

            return GenerateResult(
                image=pil, wplus=wplus.detach().cpu(),
                seed=seed, used_ref=ref_image is not None,
                style=style, text=text,
            )

    # --------------------------- factories ---------------------------

    @classmethod
    def from_config(cls, path: Optional[str] = None) -> "AvatarPipeline":
        return cls(load_config(path))
