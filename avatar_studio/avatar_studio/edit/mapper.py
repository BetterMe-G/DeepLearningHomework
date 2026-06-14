"""StyleCLIP Latent Mapper.

Architecture (matches Patashnik et al., ICCV 2021):
  three sub-mappers operate on disjoint groups of the W+ codes:
      coarse  : layers  0..3   (pose, shape)
      medium  : layers  4..7   (features, expression)
      fine    : layers  8..n   (color, texture)
  each is a 4-layer MLP with PixelNorm + Leaky ReLU.

We train *one mapper per text prompt*. At inference time, given a latent w+
(from random sampling or e4e inversion), the mapper outputs Δw+ and the
final image is G(w+ + Δw+).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    # Matches StyleCLIP/models/stylegan2/model.py:PixelNorm — reduces over dim=1.
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Equalised-lr linear layer (StyleGAN convention)."""

    def __init__(self, in_dim, out_dim, lr_mul=1.0, activation: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)
        if self.activation:
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)
        return out


class SubMapper(nn.Module):
    """PixelNorm + 4 × EqualLinear(activation=fused_lrelu).

    Mirrors StyleCLIP/mapper/latent_mappers.py:Mapper exactly: four activated
    EqualLinear layers, no extra unactivated head. EqualLinear treats the last
    dim as in_features so we can feed (B, L, 512) directly.
    """

    def __init__(self, n_layers: int = 4, dim: int = 512, lr_mul: float = 0.01):
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(n_layers):
            layers.append(EqualLinear(dim, dim, lr_mul=lr_mul, activation=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class MapperConfig:
    n_latent: int = 18           # 1024 SG2 → 18 layers
    no_coarse: bool = False
    no_medium: bool = False
    no_fine: bool = False
    n_layers: int = 4
    hidden_dim: int = 512


class LevelsMapper(nn.Module):
    """Three SubMappers (coarse/medium/fine)."""

    def __init__(self, cfg: MapperConfig):
        super().__init__()
        self.cfg = cfg
        # 4/4/(n-8) split, mirroring the official StyleCLIP repo.
        self.coarse = None if cfg.no_coarse else SubMapper(cfg.n_layers, cfg.hidden_dim)
        self.medium = None if cfg.no_medium else SubMapper(cfg.n_layers, cfg.hidden_dim)
        self.fine   = None if cfg.no_fine   else SubMapper(cfg.n_layers, cfg.hidden_dim)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: (B, n_latent, 512) → Δw same shape."""
        n = self.cfg.n_latent
        # splits — keep robust for either 16 (256/512) or 18 (1024) layers
        c_end, m_end = 4, 8
        outs = []
        if self.coarse is not None:
            outs.append(self.coarse(w[:, :c_end]))
        else:
            outs.append(torch.zeros_like(w[:, :c_end]))
        if self.medium is not None:
            outs.append(self.medium(w[:, c_end:m_end]))
        else:
            outs.append(torch.zeros_like(w[:, c_end:m_end]))
        if self.fine is not None:
            outs.append(self.fine(w[:, m_end:n]))
        else:
            outs.append(torch.zeros_like(w[:, m_end:n]))
        return torch.cat(outs, dim=1)


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

from ..models.stylegan2 import StyleGAN2Generator
from ..models.clip_loss import GlobalCLIPLoss
from ..models.id_loss import IDLoss


@dataclass
class TrainConfig:
    description: str                     # text prompt to train against
    iterations: int = 50_000
    batch_size: int = 2
    lr: float = 0.5
    lambda_clip: float = 1.0
    lambda_id:   float = 0.1
    lambda_l2:   float = 0.8
    log_every:   int = 200
    save_every:  int = 5000
    truncation:  float = 0.7


class MapperTrainer:
    def __init__(self,
                 generator: StyleGAN2Generator,
                 clip_loss: GlobalCLIPLoss,
                 id_loss: Optional[IDLoss],
                 mapper_cfg: MapperConfig,
                 train_cfg: TrainConfig,
                 device: str = "cuda"):
        self.G = generator
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad = False

        self.mapper = LevelsMapper(mapper_cfg).to(device)
        self.clip_loss = clip_loss
        self.id_loss = id_loss
        self.cfg = train_cfg
        self.device = device

        self.optim = torch.optim.Adam(self.mapper.parameters(),
                                      lr=train_cfg.lr, betas=(0.9, 0.999))

    def _sample_wplus(self, b: int) -> torch.Tensor:
        w = self.G.sample_w(b, truncation=self.cfg.truncation)
        return self.G.w_to_wplus(w)

    def step(self) -> dict:
        self.mapper.train()
        w = self._sample_wplus(self.cfg.batch_size)            # (B, L, 512)
        with torch.no_grad():
            src_img = self.G.synthesize(w)
        dw = self.mapper(w) * 0.1                              # scale down for stability
        edit_img = self.G.synthesize(w + dw)

        l_clip = self.clip_loss(edit_img, self.cfg.description)
        l_l2   = (dw ** 2).mean()
        l_id   = self.id_loss(edit_img, src_img) if self.id_loss is not None else torch.tensor(0.0, device=self.device)

        loss = (self.cfg.lambda_clip * l_clip
                + self.cfg.lambda_l2 * l_l2
                + self.cfg.lambda_id * l_id)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {"loss": loss.item(),
                "clip": l_clip.item(),
                "l2":   l_l2.item(),
                "id":   float(l_id) if isinstance(l_id, torch.Tensor) else l_id}

    def state_dict(self) -> dict:
        return {
            "mapper": self.mapper.state_dict(),
            "mapper_cfg": self.mapper.cfg.__dict__,
            "description": self.cfg.description,
        }


# --------------------------------------------------------------------------
# Inference helper
# --------------------------------------------------------------------------

class MapperInferer:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        sd = torch.load(ckpt_path, map_location=device)
        cfg = MapperConfig(**sd["mapper_cfg"])
        self.mapper = LevelsMapper(cfg).to(device).eval()
        self.mapper.load_state_dict(sd["mapper"])
        self.description = sd.get("description", "")
        self.device = device

    @torch.no_grad()
    def delta(self, wplus: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
        return self.mapper(wplus) * strength

    @torch.no_grad()
    def edit(self, wplus: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
        return wplus + self.delta(wplus, strength)
