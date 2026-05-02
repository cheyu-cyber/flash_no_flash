from __future__ import annotations

import torch


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """ Convert an RGB tensor in [0,1] to YCbCr in [0,1]. """
    r = rgb[..., 0:1, :, :]
    g = rgb[..., 1:2, :, :]
    b = rgb[..., 2:3, :, :]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=-3)


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr tensor in [0,1] to RGB in [0,1]."""
    y  = ycbcr[..., 0:1, :, :]
    cb = ycbcr[..., 1:2, :, :] - 0.5
    cr = ycbcr[..., 2:3, :, :] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=-3).clamp(0.0, 1.0)
