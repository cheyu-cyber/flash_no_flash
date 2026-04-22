"""Run inference on validation samples and real photos, save visual comparisons.

The YCbCr model operates on YCbCr tensors internally; all inputs and outputs
are converted to/from RGB at the boundary so the visualisations remain in
RGB for human viewing.

Run from the project root:

    python -m YCbCr_model.visualize
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import load_config
from YCbCr_model.network import GatedUNet
from YCbCr_model.dataset import FlashNoFlashDataset
from YCbCr_model.color import rgb_to_ycbcr, ycbcr_to_rgb


REAL_DATA_DIR = Path("data/flash_data_JBF_Detail_transfer")

# Scenes: (name, flash_file, noflash_file, reference_files...)
REAL_SCENES = [
    ("carpet",     "carpet_00_flash.tif",     "carpet_01_noflash.tif",
     {"JBF": "carpet_02_bilateral.tif", "Petschnigg05": "carpet_03_our_result.tif"}),
    ("cave01",     "cave01_00_flash.tif",     "cave01_01_noflash.tif",
     {"Petschnigg05": "cave01_03_our_result.tif", "Reference": "cave01_04_reference.tif"}),
    ("lamp",       "lamp_00_flash.tif",       "lamp_01_noflash.tif",
     {"JBF": "lamp_02_bilateral.tif", "Petschnigg05": "lamp_03_our_result.tif"}),
    ("potsdetail", "potsdetail_00_flash.tif", "potsdetail_01_noflash.tif",
     {"JBF": "potsdetail_02_bilateral.tif", "Petschnigg05": "potsdetail_03_our_result.tif"}),
    ("puppets",    "puppets_00_flash.tif",    "puppets_01_noflash.tif",
     {"Petschnigg05": "puppets_03_our_result.tif"}),
]


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor [0,1] -> (H, W, C) numpy for display."""
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def ycbcr_tensor_to_rgb_numpy(t: torch.Tensor) -> np.ndarray:
    """YCbCr (C, H, W) tensor [0,1] -> RGB (H, W, C) numpy for display."""
    rgb = ycbcr_to_rgb(t.unsqueeze(0)).squeeze(0)
    return rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def load_tif_rgb(path: Path) -> np.ndarray:
    """Load a TIF image as RGB float32 [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img[:, :, ::-1].astype(np.float32) / 255.0


def prepare_input(img: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Resize HWC [0,1] image to (H, W) model input size and return CHW tensor."""
    h, w = size
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized.transpose(2, 0, 1))


def save_comparison(
    out_path: Path,
    title: str,
    flash_np: np.ndarray,
    noflash_np: np.ndarray,
    output_np: np.ndarray,
    gates: list,
    target_np: np.ndarray | None = None,
    extra_refs: dict[str, np.ndarray] | None = None,
):
    """Save a comparison figure with inputs, output, gate stats, and optional references."""
    n_extra = len(extra_refs) if extra_refs else 0
    has_target = target_np is not None
    n_cols = 3 + (1 if has_target else 0) + n_extra
    fig, axes = plt.subplots(2, max(n_cols, 3), figsize=(5 * max(n_cols, 3), 10))
    fig.suptitle(title, fontsize=14)

    # Top row: inputs + output + references
    col = 0
    axes[0, col].imshow(flash_np); axes[0, col].set_title("Flash (input)"); col += 1
    axes[0, col].imshow(noflash_np); axes[0, col].set_title("No-Flash (input)"); col += 1
    axes[0, col].imshow(output_np); axes[0, col].set_title("Ours (model output)"); col += 1
    if has_target:
        axes[0, col].imshow(target_np); axes[0, col].set_title("Target"); col += 1
    if extra_refs:
        for ref_name, ref_img in extra_refs.items():
            axes[0, col].imshow(ref_img); axes[0, col].set_title(ref_name); col += 1

    for ax in axes[0]:
        ax.axis("off")

    # Bottom row: gate heatmaps + metrics
    n_levels = len(gates)
    for lvl in range(min(n_levels, n_cols - 1)):
        gate_map = gates[lvl].squeeze(0).mean(dim=0).cpu().numpy()
        im = axes[1, lvl].imshow(gate_map, cmap="RdBu_r", vmin=0, vmax=1)
        g_mean = gate_map.mean()
        axes[1, lvl].set_title(f"Gate L{lvl} ({gate_map.shape[0]}x{gate_map.shape[1]})\n"
                                f"flash={1 - g_mean:.1%} / noflash={g_mean:.1%}")
        axes[1, lvl].axis("off")

    # Metrics text in last bottom cell
    ax_txt = axes[1, -1]
    ax_txt.axis("off")
    gate_text = ""
    if has_target:
        mse = np.mean((output_np - target_np) ** 2)
        psnr_val = 10 * math.log10(1.0 / max(mse, 1e-10))
        mae = np.mean(np.abs(output_np - target_np))
        gate_text += f"PSNR: {psnr_val:.2f} dB\nMAE: {mae:.4f}\n\n"
    gate_text += "Gate mix:\n"
    for lvl, g in enumerate(gates):
        g_mean = g.mean().item()
        gate_text += f"  L{lvl}: flash={1 - g_mean:.1%}  noflash={g_mean:.1%}\n"
    ax_txt.text(0.05, 0.5, gate_text, fontsize=11, family="monospace",
                verticalalignment="center", transform=ax_txt.transAxes)
    ax_txt.set_title("Metrics")

    # Hide unused axes
    for row in axes:
        for ax in row:
            if not ax.images and not ax.texts:
                ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    cfg = load_config()
    mcfg = cfg.ycbcr_model
    input_size = tuple(cfg.image_size)  # (H, W) that the model was trained on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GatedUNet(mcfg).to(device)
    ckpt_path = Path(mcfg.checkpoint_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?')})")

    out_dir = Path("logs_ycbcr/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Synthetic validation samples
    # ----------------------------------------------------------------
    print("\n=== Synthetic validation samples ===")
    data_root = Path(cfg.generation.output_dir)
    val_ds = FlashNoFlashDataset(data_root / "val", augment=False)
    n_samples = min(8, len(val_ds))

    for i in range(n_samples):
        batch = val_ds[i]
        flash = batch["flash"].unsqueeze(0).to(device)
        no_flash = batch["no_flash"].unsqueeze(0).to(device)

        output, gates = model(flash, no_flash)
        # Convert YCbCr → RGB for display and metrics
        output_np = ycbcr_tensor_to_rgb_numpy(output.squeeze(0))
        target_np = ycbcr_tensor_to_rgb_numpy(batch["target"])

        mse = np.mean((output_np - target_np) ** 2)
        psnr_val = 10 * math.log10(1.0 / max(mse, 1e-10))

        save_comparison(
            out_dir / f"synthetic_{i:03d}.png",
            f"Synthetic Sample {i}",
            ycbcr_tensor_to_rgb_numpy(batch["flash"]),
            ycbcr_tensor_to_rgb_numpy(batch["no_flash"]),
            output_np,
            gates,
            target_np=target_np,
        )
        print(f"  Sample {i}: PSNR={psnr_val:.2f} dB")

    # ----------------------------------------------------------------
    # 2. Real photo pairs from JBF Detail Transfer dataset
    # ----------------------------------------------------------------
    print("\n=== Real photos (JBF Detail Transfer) ===")

    for scene_name, flash_file, noflash_file, ref_files in REAL_SCENES:
        flash_path = REAL_DATA_DIR / flash_file
        noflash_path = REAL_DATA_DIR / noflash_file

        if not flash_path.exists() or not noflash_path.exists():
            print(f"  Skipping {scene_name}: files not found")
            continue

        # Load full-res for display
        flash_full = load_tif_rgb(flash_path)
        noflash_full = load_tif_rgb(noflash_path)

        # Resize to model input and convert RGB → YCbCr before feeding the model
        flash_t = rgb_to_ycbcr(prepare_input(flash_full, input_size).unsqueeze(0)).to(device)
        noflash_t = rgb_to_ycbcr(prepare_input(noflash_full, input_size).unsqueeze(0)).to(device)

        output, gates = model(flash_t, noflash_t)
        # Model output is YCbCr — convert back to RGB for display
        output_np = ycbcr_tensor_to_rgb_numpy(output.squeeze(0))

        in_h, in_w = input_size

        # Load reference results (resized to model input for comparison)
        extra_refs = {}
        for ref_name, ref_file in ref_files.items():
            ref_path = REAL_DATA_DIR / ref_file
            if ref_path.exists():
                ref_img = load_tif_rgb(ref_path)
                extra_refs[ref_name] = cv2.resize(ref_img, (in_w, in_h),
                                                  interpolation=cv2.INTER_AREA)

        # Display inputs resized to model input for consistent layout
        flash_disp = cv2.resize(flash_full, (in_w, in_h),
                                interpolation=cv2.INTER_AREA)
        noflash_disp = cv2.resize(noflash_full, (in_w, in_h),
                                  interpolation=cv2.INTER_AREA)

        save_comparison(
            out_dir / f"real_{scene_name}.png",
            f"Real: {scene_name} ({flash_full.shape[1]}x{flash_full.shape[0]} -> {in_w}x{in_h})",
            flash_disp,
            noflash_disp,
            output_np,
            gates,
            extra_refs=extra_refs,
        )
        print(f"  {scene_name}: saved")

    print(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
