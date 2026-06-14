"""Helpers shared by train / generate / interpolate / eval scripts."""
import os
import random
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm(x):
    """[-1, 1] -> [0, 1]"""
    return (x.clamp(-1, 1) + 1) / 2


def save_image_grid(tensor, path, nrow=8):
    """Save a grid of images (input in [-1,1]) to disk as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(denorm(tensor), path, nrow=nrow)


def save_tensor_as_png(tensor, path):
    """Save a single CHW image tensor (in [-1,1]) as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = denorm(tensor).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def slerp(val, low, high):
    """Spherical linear interpolation between two latent vectors.
    val: scalar in [0, 1]; low/high: 1-D tensors with the same shape."""
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos((low_norm * high_norm).sum().clamp(-1, 1))
    so = torch.sin(omega)
    if so.item() < 1e-6:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def lerp(val, low, high):
    """Linear interpolation between two latent vectors."""
    return (1.0 - val) * low + val * high
