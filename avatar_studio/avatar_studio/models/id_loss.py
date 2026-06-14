"""ArcFace IR-SE-50 identity loss.

Backbone reuses the canonical implementation vendored from
`InsightFace_Pytorch/model.py` (TreB1eN/InsightFace_Pytorch) at
`avatar_studio/vendor/insightface_model.py` — that is the same source the
StyleCLIP / encoder4editing ecosystem expects `model_ir_se50.pth`
weights for.

We add a thin loss wrapper that does the StyleCLIP-style face crop and
computes `1 - cos(arcface(edit), arcface(source))`.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..vendor.insightface_model import Backbone


class IDLoss(nn.Module):
    """1 - cos(arcface(edit), arcface(source)) used by the StyleCLIP mapper trainer."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        self.net = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se").to(device)
        self.net.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        # StyleCLIP-style preprocessing: downsample to 256, crop the face box, then to 112.
        self.pool256 = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))

    def _crop(self, img: torch.Tensor) -> torch.Tensor:
        # img is (B,3,H,W) in [-1,1] from StyleGAN — center crop the face region.
        img = self.pool256(img)
        img = img[:, :, 35:223, 32:220]   # StyleCLIP authors' crop window
        img = self.face_pool(img)
        return img

    def extract(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(self._crop(img))

    def forward(self, edited: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f_src = self.extract(source)
        f_edit = self.extract(edited)
        return (1 - (f_edit * f_src).sum(dim=-1)).mean()
