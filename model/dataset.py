"""PyTorch dataset for loading synthetic flash / no-flash pairs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FlashNoFlashDataset(Dataset):
    """Loads pre-generated synthetic data from disk.

    Expected directory layout (created by synthetic_data_generator.py)::

        root_dir/
            flash/          00000.png …
            no_flash/       00000.png …
            no_flash_clean/ 00000.png …
            depth/          00000.npy …

    Each ``__getitem__`` returns a dict with keys:
        flash, no_flash, depth, target  — all (C, H, W) float32 in [0, 1].
    """

    def __init__(self, root_dir: str | Path, augment: bool = False):
        self.root = Path(root_dir)
        self.augment = augment

        # Discover sample indices from the flash/ subdirectory
        flash_dir = self.root / "flash"
        self.indices = sorted(
            int(p.stem) for p in flash_dir.glob("*.png")
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load a BGR PNG, convert to RGB float32 [0, 1]."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img[:, :, ::-1].astype(np.float32) / 255.0

    def _load_depth(self, path: Path) -> np.ndarray:
        """Load a raw .npy depth map and normalise to [0, 1]."""
        d = np.load(str(path)).astype(np.float32)
        lo, hi = d.min(), d.max()
        if hi - lo > 1e-6:
            d = (d - lo) / (hi - lo)
        else:
            d = np.zeros_like(d)
        return d

    def __getitem__(self, index: int) -> dict:
        idx = self.indices[index]
        tag = f"{idx:05d}"

        flash = self._load_rgb(self.root / "flash" / f"{tag}.png")
        no_flash = self._load_rgb(self.root / "no_flash" / f"{tag}.png")
        target = self._load_rgb(self.root / "no_flash_clean" / f"{tag}.png")
        depth = self._load_depth(self.root / "depth" / f"{tag}.npy")

        # Augmentation: random horizontal and vertical flips
        if self.augment:
            if np.random.random() > 0.5:
                flash = flash[:, ::-1, :].copy()
                no_flash = no_flash[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()
                depth = depth[:, ::-1].copy()
            if np.random.random() > 0.5:
                flash = flash[::-1, :, :].copy()
                no_flash = no_flash[::-1, :, :].copy()
                target = target[::-1, :, :].copy()
                depth = depth[::-1, :].copy()

        # HWC → CHW
        flash = torch.from_numpy(flash.transpose(2, 0, 1))
        no_flash = torch.from_numpy(no_flash.transpose(2, 0, 1))
        target = torch.from_numpy(target.transpose(2, 0, 1))
        depth = torch.from_numpy(depth[None])  # (1, H, W)

        return {
            "flash": flash,
            "no_flash": no_flash,
            "depth": depth,
            "target": target,
        }
