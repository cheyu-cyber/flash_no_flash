"""Loss functions for the Gated U-Net (RGB variant).

Combined loss = L1 + SSIM + gate entropy regularisation.

Aligned with the YCbCr variant — no VGG perceptual term.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import ModelConfig


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1D Gaussian kernel normalised to sum to 1."""
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """2D Gaussian kernel for depthwise convolution, shape (channels, 1, size, size)."""
    k1d = _gaussian_kernel_1d(size, sigma, device, dtype)
    k2d = k1d[:, None] * k1d[None, :]
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Mean SSIM between two (B, C, H, W) image tensors in [0, 1]."""
    channels = x.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, channels, x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    return (num / den).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """L1 + SSIM reconstruction loss.

    Operates on RGB tensors in [0, 1].
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_l1 = cfg.loss_l1_weight
        self.w_ssim = cfg.loss_ssim_weight

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        gates: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        output : (B, 3, H, W) RGB
        target : (B, 3, H, W) RGB
        gates  : list of (B, C_k, H_k, W_k) gate activations
                 (accepted for signature compatibility with downstream
                 code that passes them; not used in the loss).
        """
        # --- L1 ---
        l1 = F.l1_loss(output, target)

        # --- SSIM (1 - SSIM so lower = better) ---
        ssim_val = ssim(output, target)
        ssim_loss = 1.0 - ssim_val

        total = self.w_l1 * l1 + self.w_ssim * ssim_loss

        loss_dict = {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "total": total.item(),
        }
        return total, loss_dict
