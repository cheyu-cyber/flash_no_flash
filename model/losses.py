"""Loss functions for the Gated U-Net.

Combined loss = L1 reconstruction + VGG perceptual + gate entropy regularisation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from utils.config import ModelConfig


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
    """L1 + perceptual (VGG) + gate entropy regularisation.

    Parameters
    ----------
    cfg : ModelConfig
        Provides the three loss weights.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_l1 = cfg.loss_l1_weight
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
        l1 = torch.nn.functional.l1_loss(output, target)

        # --- Perceptual ---
        with torch.no_grad():
            target_feats = self.vgg(target)
        output_feats = self.vgg(output)
        perc = sum(
            torch.nn.functional.l1_loss(output_feats[k], target_feats[k])
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

        total = self.w_l1 * l1 + self.w_perc * perc + self.w_gate * gate_entropy

        loss_dict = {
            "l1": l1.item(),
            "perceptual": perc.item(),
            "gate_entropy": gate_entropy.item(),
            "total": total.item(),
        }
        return total, loss_dict
