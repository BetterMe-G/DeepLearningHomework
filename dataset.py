"""
Dataset loaders.

Supports:
  - 'lfw'    : LFW (Kaggle: atulanandjha/lfwpeople). Expects images recursively
               inside `data_root` (we accept arbitrary sub-folder structure
               because the Kaggle archive nests images under person folders,
               e.g. lfw-deepfunneled/<person>/*.jpg ).
  - 'celeba' : CelebA aligned & cropped images flat inside `data_root`.
  - 'folder' : Generic ImageFolder-style root (subdirs of class names).
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FlatImageDataset(Dataset):
    """Loads every image found recursively under `root`. Ignores labels."""

    def __init__(self, root, transform=None):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(
                f"Data root '{root}' not found. "
                f"Download LFW from "
                f"https://www.kaggle.com/datasets/atulanandjha/lfwpeople "
                f"and extract under this directory."
            )
        self.paths = sorted(
            str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}.")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_transform(image_size, center_crop=None, hflip_p=0.0):
    """Standard DCGAN preprocessing: resize -> center-crop -> [-1, 1]."""
    ops = []
    if center_crop is not None:
        # First resize the shorter side, then center crop to a square.
        ops.append(T.Resize(center_crop))
        ops.append(T.CenterCrop(center_crop))
    ops.append(T.Resize(image_size))
    ops.append(T.CenterCrop(image_size))
    if hflip_p > 0:
        ops.append(T.RandomHorizontalFlip(p=hflip_p))
    ops.append(T.ToTensor())
    ops.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(ops)


def get_dataloader(cfg):
    """Return a DataLoader based on the dataset name in cfg."""
    if cfg.dataset.lower() == "lfw":
        # LFW images are 250x250 with the face roughly centered;
        # crop the central 178 then resize for a face-tight 64x64.
        tfm = build_transform(
            cfg.image_size, center_crop=178, hflip_p=getattr(cfg, "hflip_p", 0.0)
        )
        ds = FlatImageDataset(cfg.data_root, transform=tfm)

    elif cfg.dataset.lower() == "celeba":
        # CelebA aligned: 178x218, classic recipe is center-crop 178.
        tfm = build_transform(
            cfg.image_size, center_crop=178, hflip_p=getattr(cfg, "hflip_p", 0.0)
        )
        ds = FlatImageDataset(cfg.data_root, transform=tfm)

    elif cfg.dataset.lower() == "folder":
        tfm = build_transform(cfg.image_size, hflip_p=getattr(cfg, "hflip_p", 0.0))
        ds = FlatImageDataset(cfg.data_root, transform=tfm)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return ds, loader
