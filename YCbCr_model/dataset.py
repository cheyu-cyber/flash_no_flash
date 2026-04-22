"""PyTorch dataset for loading synthetic flash / no-flash pairs (YCbCr)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from YCbCr_model.color import rgb_to_ycbcr


class FlashNoFlashDataset(Dataset):
    """Loads pre-generated synthetic data from disk, returning YCbCr tensors.

    Expected directory layout (created by synthetic_data_generator.py)::

        root_dir/
            flash/          00000.png …
            no_flash/       00000.png …
            no_flash_clean/ 00000.png …

    Each ``__getitem__`` returns a dict with keys:
        flash, no_flash, target  — all (3, H, W) float32 YCbCr in [0, 1].
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

    def __getitem__(self, index: int) -> dict:
        idx = self.indices[index]
        tag = f"{idx:05d}"

        flash = self._load_rgb(self.root / "flash" / f"{tag}.png")
        no_flash = self._load_rgb(self.root / "no_flash" / f"{tag}.png")
        target = self._load_rgb(self.root / "no_flash_clean" / f"{tag}.png")

        # Augmentation: random horizontal and vertical flips (in RGB space,
        # before the color transform — result is equivalent either way).
        if self.augment:
            if np.random.random() > 0.5:
                flash = flash[:, ::-1, :].copy()
                no_flash = no_flash[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()
            if np.random.random() > 0.5:
                flash = flash[::-1, :, :].copy()
                no_flash = no_flash[::-1, :, :].copy()
                target = target[::-1, :, :].copy()

        # HWC → CHW, RGB → YCbCr
        flash = rgb_to_ycbcr(torch.from_numpy(flash.transpose(2, 0, 1)))
        no_flash = rgb_to_ycbcr(torch.from_numpy(no_flash.transpose(2, 0, 1)))
        target = rgb_to_ycbcr(torch.from_numpy(target.transpose(2, 0, 1)))

        return {
            "flash": flash,
            "no_flash": no_flash,
            "target": target,
        }
