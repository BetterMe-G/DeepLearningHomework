"""StyleGAN2 generator wrapper.

The underlying rosinality StyleGAN2 implementation lives under
`avatar_studio/vendor/stylegan2/` (vendored from the encoder4editing fork
so the latent space stays compatible with the e4e encoder).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..vendor.stylegan2.model import Generator as _SG2Generator, Discriminator as _SG2Discriminator


class StyleGAN2Generator(nn.Module):
    """Convenience facade around the rosinality SG2 Generator."""

    def __init__(self,
                 ckpt_path: str,
                 image_size: int = 1024,
                 latent_dim: int = 512,
                 n_mlp: int = 8,
                 channel_multiplier: int = 2,
                 truncation: float = 0.7,
                 truncation_mean_samples: int = 4096,
                 device: str = "cuda"):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.truncation = truncation
        self.device = device

        self.G = _SG2Generator(image_size, latent_dim, n_mlp,
                               channel_multiplier=channel_multiplier).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        # rosinality-format checkpoints store the EMA copy under "g_ema".
        state = ckpt.get("g_ema", ckpt.get("g", ckpt))
        self.G.load_state_dict(state, strict=False)
        self.G.eval()

        with torch.no_grad():
            self.mean_latent = self.G.mean_latent(truncation_mean_samples)

    @property
    def n_latent(self) -> int:
        return self.G.n_latent

    def sample_z(self, n: int = 1) -> torch.Tensor:
        return torch.randn(n, self.latent_dim, device=self.device)

    def z_to_w(self, z: torch.Tensor) -> torch.Tensor:
        """Map z to W (single 512-D code per sample)."""
        return self.G.style(z)

    def w_to_wplus(self, w: torch.Tensor) -> torch.Tensor:
        """Broadcast a single W to W+ (per-layer)."""
        if w.dim() == 2:
            w = w.unsqueeze(1).repeat(1, self.n_latent, 1)
        return w

    def sample_w(self, n: int = 1, truncation: float | None = None) -> torch.Tensor:
        """Sample a truncated W code, shape (n, 512)."""
        z = self.sample_z(n)
        w = self.z_to_w(z)
        t = self.truncation if truncation is None else truncation
        if t < 1.0:
            w = self.mean_latent + t * (w - self.mean_latent)
        return w

    def forward(self,
                styles,
                input_is_latent: bool = True,
                truncation: float | None = None,
                noise=None,
                randomize_noise: bool = False) -> torch.Tensor:
        """Run the synthesis. `styles` is a list of W+ tensors as the underlying G expects."""
        if isinstance(styles, torch.Tensor):
            styles = [styles]
        t = 1.0 if truncation is None or input_is_latent else (self.truncation if truncation is None else truncation)
        img, _ = self.G(styles,
                        input_is_latent=input_is_latent,
                        truncation=t,
                        truncation_latent=self.mean_latent,
                        noise=noise,
                        randomize_noise=randomize_noise)
        return img

    def synthesize(self, wplus: torch.Tensor) -> torch.Tensor:
        """Render a W+ tensor (B, n_latent, 512) to image in [-1, 1]."""
        return self.forward(wplus, input_is_latent=True)

class StyleGAN2Discriminator(nn.Module):
    """Optional: pretrained D, useful as a perceptual feature extractor (à la JoJoGAN)."""
    def __init__(self, ckpt_path: str, image_size: int = 1024,
                 channel_multiplier: int = 2, device: str = "cuda"):
        super().__init__()
        self.D = _SG2Discriminator(image_size, channel_multiplier=channel_multiplier).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.D.load_state_dict(ckpt.get("d", ckpt), strict=False)
        self.D.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.D(x)
