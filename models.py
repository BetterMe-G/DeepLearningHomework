"""
DCGAN Generator and Discriminator (Radford et al., 2015).
Generator: z (B, z_dim, 1, 1) -> image (B, 3, 64, 64) in [-1, 1].
Discriminator: image (B, 3, 64, 64) -> scalar logit.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN


def weights_init(m):
    """DCGAN weight init: Normal(0, 0.02) for conv/convT; Normal(1, 0.02) for BN."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_dim=100, g_feat=64, channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # input: z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, g_feat * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_feat * 8),
            nn.ReLU(True),
            # (g_feat*8) x 4 x 4
            nn.ConvTranspose2d(g_feat * 8, g_feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_feat * 4),
            nn.ReLU(True),
            # (g_feat*4) x 8 x 8
            nn.ConvTranspose2d(g_feat * 4, g_feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_feat * 2),
            nn.ReLU(True),
            # (g_feat*2) x 16 x 16
            nn.ConvTranspose2d(g_feat * 2, g_feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_feat),
            nn.ReLU(True),
            # g_feat x 32 x 32
            nn.ConvTranspose2d(g_feat, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # channels x 64 x 64
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """Spectral-norm D. BN is dropped because SN already constrains layer scale
    and the two often fight each other. Set use_sn=False to disable SN (e.g. for WGAN-GP)."""
    def __init__(self, d_feat=64, channels=3, use_sn=True):
        super().__init__()
        def _c(layer):
            return SN(layer) if use_sn else layer
        self.main = nn.Sequential(
            # channels x 64 x 64
            _c(nn.Conv2d(channels, d_feat, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # d_feat x 32 x 32
            _c(nn.Conv2d(d_feat, d_feat * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # (d_feat*2) x 16 x 16
            _c(nn.Conv2d(d_feat * 2, d_feat * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # (d_feat*4) x 8 x 8
            _c(nn.Conv2d(d_feat * 4, d_feat * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # (d_feat*8) x 4 x 4
            _c(nn.Conv2d(d_feat * 8, 1, 4, 1, 0, bias=False)),
            # -> 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1)
