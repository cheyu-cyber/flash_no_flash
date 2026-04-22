"""Loss functions for the Gated U-Net (YCbCr variant).

Combined loss = channel-weighted L1 (heavier on Cb/Cr) + SSIM on Y only
+ gate entropy regularisation.

Design rationale
----------------
Operating in YCbCr lets us weight the losses by what each channel actually
carries. We exploit that as follows:

* **L1 with channel weights** — Y is weighted 1.0 and Cb/Cr each 2.0, so
  chroma is pinned tightly to the target (prevents color shift) while the
  luminance channel stays free to denoise/reconstruct brightness.
* **SSIM on Y only** — SSIM captures local luminance structure. Applying
  it only to Y matches its classical formulation and keeps the structural
  term from double-counting chroma, which L1 already handles.
* **No VGG perceptual loss** — dropped deliberately. For flash/no-flash
  denoising the chroma is mostly region-smooth, so the VGG-on-RGB texture
  prior is not worth the cost or the cross-channel gradient blending.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import YCbCrModelConfig


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
    """Channel-weighted L1 + SSIM-on-Y + gate entropy regularisation.

    Operates on YCbCr tensors in [0, 1]. Per-channel L1 weights come from
    ``cfg.loss_l1_{y,cb,cr}_weight`` — chroma is typically weighted higher
    so color is pinned tightly while the Y channel stays free.
    """

    def __init__(self, cfg: YCbCrModelConfig):
        super().__init__()
        self.w_l1 = cfg.loss_l1_weight
        self.w_ssim = cfg.loss_ssim_weight
        self.w_gate = cfg.loss_gate_reg_weight
        # Register channel weights as a buffer so they move with .to(device).
        self.register_buffer(
            "l1_chan_w",
            torch.tensor([
                cfg.loss_l1_y_weight,
                cfg.loss_l1_cb_weight,
                cfg.loss_l1_cr_weight,
            ]).reshape(1, 3, 1, 1),
        )

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        gates: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        output : (B, 3, H, W) YCbCr
        target : (B, 3, H, W) YCbCr
        gates  : list of (B, C_k, H_k, W_k) gate activations
        """
        # --- Channel-weighted L1 ---
        # Weighted mean of |output - target| with per-channel weights.
        abs_diff = (output - target).abs()
        weighted = abs_diff * self.l1_chan_w
        l1 = weighted.sum() / (abs_diff.numel() * self.l1_chan_w.mean())

        # --- SSIM on Y channel only ---
        ssim_val = ssim(output[:, 0:1], target[:, 0:1])
        ssim_loss = 1.0 - ssim_val

        # --- Gate entropy (push gates toward 0 or 1 — decisive) ---
        eps = 1e-7
        gate_entropy = torch.tensor(0.0, device=output.device)
        for g in gates:
            g_clamped = g.clamp(eps, 1.0 - eps)
            ent = -(g_clamped * g_clamped.log() + (1 - g_clamped) * (1 - g_clamped).log())
            gate_entropy = gate_entropy + ent.mean()
        gate_entropy = gate_entropy / max(len(gates), 1)

        total = (self.w_l1 * l1
                + self.w_ssim * ssim_loss
                + self.w_gate * gate_entropy)

        loss_dict = {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "gate_entropy": gate_entropy.item(),
            "total": total.item(),
        }
        return total, loss_dict
