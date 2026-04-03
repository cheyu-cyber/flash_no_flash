"""Loss functions for the Gated U-Net.

Combined loss = L1 + SSIM + VGG perceptual + gate entropy regularisation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
    k2d = k1d[:, None] * k1d[None, :]  # outer product
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute mean SSIM between two (B, C, H, W) image tensors in [0, 1].

    Returns a scalar tensor (1 = identical, 0 = no similarity).
    """
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
# VGG feature extractor (frozen)
# ---------------------------------------------------------------------------

class VGGFeatures(nn.Module):
    """Extract relu1_2, relu2_2, relu3_3 features from a pretrained VGG-16."""

    # ImageNet normalisation constants
    MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Indices of the target layers in vgg16.features
    LAYER_INDICES = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15}

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        max_idx = max(self.LAYER_INDICES.values()) + 1
        self.slices = nn.ModuleDict()
        prev = 0
        for name, idx in sorted(self.LAYER_INDICES.items(), key=lambda x: x[1]):
            self.slices[name] = nn.Sequential(*list(vgg.features.children())[prev:idx + 1])
            prev = idx + 1

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Normalise from [0,1] to ImageNet space
        mean = self.MEAN.to(x.device, x.dtype)
        std = self.STD.to(x.device, x.dtype)
        x = (x - mean) / std

        feats = {}
        for name in sorted(self.LAYER_INDICES, key=lambda n: self.LAYER_INDICES[n]):
            x = self.slices[name](x)
            feats[name] = x
        return feats


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """L1 + SSIM + perceptual (VGG) + gate entropy regularisation.

    Parameters
    ----------
    cfg : ModelConfig
        Provides the loss weights.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_l1 = cfg.loss_l1_weight
        self.w_ssim = cfg.loss_ssim_weight
        self.w_perc = cfg.loss_perceptual_weight
        self.w_gate = cfg.loss_gate_reg_weight
        self.vgg = VGGFeatures()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        gates: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        output : (B, 3, H, W)
        target : (B, 3, H, W)
        gates  : list of (B, C_k, H_k, W_k) gate activations

        Returns
        -------
        loss      : scalar tensor
        loss_dict : dict with individual loss component values (for logging)
        """
        # --- L1 ---
        l1 = F.l1_loss(output, target)

        # --- SSIM (1 - SSIM so that lower = better) ---
        ssim_val = ssim(output, target)
        ssim_loss = 1.0 - ssim_val

        # --- Perceptual ---
        with torch.no_grad():
            target_feats = self.vgg(target)
        output_feats = self.vgg(output)
        perc = sum(
            F.l1_loss(output_feats[k], target_feats[k])
            for k in output_feats
        )

        # --- Gate entropy regularisation ---
        # Minimising binary entropy pushes gates toward 0 or 1 (decisive)
        eps = 1e-7
        gate_entropy = torch.tensor(0.0, device=output.device)
        for g in gates:
            g_clamped = g.clamp(eps, 1.0 - eps)
            ent = -(g_clamped * g_clamped.log() + (1 - g_clamped) * (1 - g_clamped).log())
            gate_entropy = gate_entropy + ent.mean()
        gate_entropy = gate_entropy / max(len(gates), 1)

        total = (self.w_l1 * l1
                + self.w_ssim * ssim_loss
                + self.w_perc * perc
                + self.w_gate * gate_entropy)

        loss_dict = {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "perceptual": perc.item(),
            "gate_entropy": gate_entropy.item(),
            "total": total.item(),
        }
        return total, loss_dict
