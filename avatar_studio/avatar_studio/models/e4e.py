"""e4e (encoder4editing) wrapper for photo -> W+ inversion.

Uses the vendored pSp framework (`avatar_studio/vendor/psp.py`). The e4e
checkpoint already bundles its training `opts`; we only override
`checkpoint_path` and `device` so the loader picks up our local file.
"""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import PIL.Image
import torchvision.transforms as T

from ..vendor.psp import pSp


_E4E_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class E4EInverter(nn.Module):
    """Holds a loaded pSp+e4e model and exposes a single `invert(pil) -> W+` call."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        ckpt = torch.load(ckpt_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = ckpt_path
        opts["device"] = device
        # batch_size only matters at training time; force 1 for inference safety.
        opts["batch_size"] = 1
        self.net = pSp(Namespace(**opts)).to(device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def invert(self, pil_img: PIL.Image.Image) -> torch.Tensor:
        """Return W+ latent (1, n_latent, 512) for an aligned face image."""
        x = _E4E_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        _, w_plus = self.net(x, randomize_noise=False, return_latents=True,
                             resize=False, input_code=False)
        return w_plus


def invert_image(pil_img: PIL.Image.Image,
                 ckpt_path: str,
                 cache_path: Optional[str] = None,
                 device: str = "cuda",
                 inverter: Optional[E4EInverter] = None) -> torch.Tensor:
    """Convenience wrapper. Caches the W+ tensor to disk if `cache_path` is given."""
    if cache_path and Path(cache_path).is_file():
        return torch.load(cache_path, map_location=device)
    inv = inverter if inverter is not None else E4EInverter(ckpt_path, device=device)
    w = inv.invert(pil_img)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(w.detach().cpu(), cache_path)
    return w
