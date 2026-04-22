"""Dual-Encoder Gated U-Net for flash / no-flash denoising.

Architecture overview
---------------------
Two independent encoders extract multi-scale features from the flash and
ambient (no-flash) images.  At each decoder level a *gated skip connection*
learns where to trust the ambient tone vs. the flash structure:

    g_k = sigmoid(Conv([F_k, A_k]))
    skip_k = g_k * A_k + (1 - g_k) * F_k

A single multi-head self-attention block at the bottleneck (32x32) provides
cheap global context.  The decoder upsamples back to 512x512 and outputs the
denoised ambient image via a Sigmoid head.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import YCbCrModelConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv3x3 -> GN -> ReLU -> Conv3x3 -> GN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Multi-level convolutional encoder (for flash or ambient)."""

    def __init__(self, in_channels: int = 3, channels: Tuple[int, ...] = (64, 128, 256, 512)):
        super().__init__()
        self.levels = nn.ModuleList()
        ch = in_channels
        for out_ch in channels:
            self.levels.append(ConvBlock(ch, out_ch))
            ch = out_ch
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for level in self.levels:
            x = level(x)
            skips.append(x)      # skip before pooling
            x = self.pool(x)
        return x, skips          # x = pooled output after last level


# ---------------------------------------------------------------------------
# Bottleneck attention
# ---------------------------------------------------------------------------

class BottleneckAttention(nn.Module):
    """Multi-head self-attention at the bottleneck resolution (e.g. 32x32)."""

    def __init__(self, channels: int = 512, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, dim, HW)

        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Gated skip connection
# ---------------------------------------------------------------------------

class GatedSkipConnection(nn.Module):
    """Learned gate that blends flash and ambient features.

    g = sigmoid(Conv([F, A]))
    output = g * A + (1 - g) * F
    """

    def __init__(self, flash_ch: int, ambient_ch: int):
        super().__init__()
        total = flash_ch + ambient_ch
        self.gate = nn.Sequential(
            nn.Conv2d(total, total // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(total // 4, flash_ch, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f_skip: torch.Tensor,
        a_skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(torch.cat([f_skip, a_skip], dim=1))
        fused = g * a_skip + (1 - g) * f_skip
        return fused, g


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderLevel(nn.Module):
    """Upsample -> concatenate gated skip -> ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv_block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GatedUNet(nn.Module):
    """Dual-encoder gated U-Net for flash / no-flash denoising.

    Parameters
    ----------
    cfg : YCbCrModelConfig
        Architecture hyper-parameters loaded from config.json.
    """

    def __init__(self, cfg: YCbCrModelConfig):
        super().__init__()
        enc_ch = cfg.encoder_channels           # (64, 128, 256, 512)
        dec_ch = cfg.decoder_channels           # (256, 128, 64, 32)
        n_levels = len(enc_ch)

        # --- encoders ---
        self.flash_encoder = Encoder(3, enc_ch)
        self.ambient_encoder = Encoder(3, enc_ch)

        # --- bottleneck ---
        self.bottleneck_merge = nn.Sequential(
            nn.Conv2d(enc_ch[-1] * 2, cfg.bottleneck_channels, 1, bias=False),
            nn.GroupNorm(8, cfg.bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_attn = BottleneckAttention(cfg.bottleneck_channels, cfg.attention_heads)
        self.bottleneck_conv = ConvBlock(cfg.bottleneck_channels, cfg.bottleneck_channels)

        # --- gated skip connections (one per encoder level) ---
        self.gates = nn.ModuleList([
            GatedSkipConnection(enc_ch[k], enc_ch[k])
            for k in range(n_levels)
        ])

        # --- decoder levels (top-down: level 3 → 0) ---
        self.decoder_levels = nn.ModuleList()
        prev_ch = cfg.bottleneck_channels
        for i, k in enumerate(reversed(range(n_levels))):
            skip_ch = enc_ch[k]
            out_ch = dec_ch[i]
            self.decoder_levels.append(DecoderLevel(prev_ch, skip_ch, out_ch))
            prev_ch = out_ch

        # --- output head ---
        self.output_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        flash: torch.Tensor,
        no_flash: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        flash    : (B, 3, H, W)
        no_flash : (B, 3, H, W)

        Returns
        -------
        output : (B, 3, H, W)  denoised ambient image
        gates  : list of (B, C_k, H_k, W_k) gate activation maps
        """
        # Encode
        f_out, f_skips = self.flash_encoder(flash)
        a_out, a_skips = self.ambient_encoder(no_flash)

        # Bottleneck
        x = torch.cat([f_out, a_out], dim=1)
        x = self.bottleneck_merge(x)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_conv(x)

        # Decode with gated skips (level 3 → 2 → 1 → 0)
        n_levels = len(f_skips)
        gates: List[torch.Tensor] = []
        for i, k in enumerate(reversed(range(n_levels))):
            fused_skip, g = self.gates[k](f_skips[k], a_skips[k])
            gates.append(g)
            x = self.decoder_levels[i](x, fused_skip)

        output = self.output_head(x)
        return output, gates
