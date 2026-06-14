"""CLIP-based losses used for text-driven editing.

Two losses are exposed:
  * `GlobalCLIPLoss` — cosine distance from CLIP(image) to CLIP(text).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import clip


_IMAGENET_TEMPLATES = (
    "a photo of a {}.",
    "a cropped photo of the face of a {}.",
    "a portrait of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a good photo of a {}.",
)


class _CLIPBase(nn.Module):
    """Holds a frozen CLIP model + image preprocessing for tensor inputs in [-1, 1]."""

    def __init__(self, device: str = "cuda", clip_model: str = "ViT-B/32"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(clip_model, device=device, jit=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # CLIP standard normalization (after we de-normalize from [-1,1] to [0,1])
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1),
        )
        # use a differentiable resize so gradients flow through to the generator
        self.upsample = nn.Upsample(size=224, mode="bicubic", align_corners=False)
        # Ensure all buffers/modules sit on the same device.
        self.to(device)

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        img = (img + 1) / 2                        # [-1,1] -> [0,1]
        img = self.upsample(img)
        img = (img - self.mean) / self.std
        return img

    @torch.no_grad()
    def encode_text_mean(self, text: str) -> torch.Tensor:
        prompts = [t.format(text) for t in _IMAGENET_TEMPLATES]
        tok = clip.tokenize(prompts).to(self.device)
        feats = self.model.encode_text(tok).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.mean(dim=0, keepdim=True)     # (1, D)

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        f = self.model.encode_image(self._preprocess(img)).float()
        return f / f.norm(dim=-1, keepdim=True)


class GlobalCLIPLoss(_CLIPBase):
    """1 - cos(CLIP(image), CLIP(text)). Used by StyleCLIP latent optimization."""

    def forward(self, img: torch.Tensor, text: str) -> torch.Tensor:
        text_feat = self.encode_text_mean(text)
        img_feat = self.encode_image(img)
        return (1 - (img_feat * text_feat).sum(dim=-1)).mean()
