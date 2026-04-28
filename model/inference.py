"""Run inference on real photos and synthetic validation samples.

Outputs (all under ``logs/visualizations/``):

* ``real/<name>_full.png``       — model output only, 1024x768.
* ``real/<name>_combined.png``   — 2x4 grid:
    row 1: flash | no_flash | ours | petschnigg (or blank)
    row 2: gate L0 | gate L1 | gate L2 | gate L3
* ``synthetic/<idx>_full.png``      — model output only, 1024x768.
* ``synthetic/<idx>_combined.png``  — 2x4 grid:
    row 1: flash | no_flash | ours | no_flash_clean
    row 2: gate L0 | gate L1 | gate L2 | gate L3
* ``synthetic/loss.txt``         — per-sample loss + PSNR.

Run from the project root:

    python -m model.inference
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import load_config
from model.network import GatedUNet
from model.dataset import FlashNoFlashDataset
from model.losses import CombinedLoss
from model.align import align_pair_and_crop


REAL_DATA_DIR = Path("data/real_data")
OUT_ROOT = Path("logs/visualizations")
OUTPUT_W, OUTPUT_H = 1024, 768
N_SYNTHETIC = 5
# Camera handheld between flash/no-flash shots → register no-flash to flash
# before feeding the pair to the model. Synthetic pairs are pixel-perfect.
ALIGN_REAL = True


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1].astype(np.float32) / 255.0


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor in [0,1] -> (H, W, C) numpy."""
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save_full_output(out_path: Path, img: np.ndarray) -> None:
    """Save a HWC [0,1] RGB image as a PNG at its native size."""
    bgr = (img[:, :, ::-1].clip(0, 1) * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path), bgr)


# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------

def save_combined(
    out_path: Path,
    title: str,
    flash_np: np.ndarray,
    noflash_np: np.ndarray,
    output_np: np.ndarray,
    fourth_np: np.ndarray | None,
    fourth_title: str,
    gates: list[torch.Tensor],
) -> None:
    """2 rows x 4 cols: inputs/output/reference + 4 gate maps."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(title, fontsize=14)

    # Row 1: flash, no_flash, ours, fourth
    axes[0, 0].imshow(np.clip(flash_np, 0, 1));   axes[0, 0].set_title("Flash")
    axes[0, 1].imshow(np.clip(noflash_np, 0, 1)); axes[0, 1].set_title("No-Flash")
    axes[0, 2].imshow(np.clip(output_np, 0, 1));  axes[0, 2].set_title("Ours")
    if fourth_np is not None:
        axes[0, 3].imshow(np.clip(fourth_np, 0, 1))
    axes[0, 3].set_title(fourth_title)
    for ax in axes[0]:
        ax.axis("off")

    # Row 2: gate L0..L3 (encoder order — earliest level = highest resolution)
    for lvl in range(4):
        ax = axes[1, lvl]
        if lvl < len(gates):
            g = gates[lvl].squeeze(0).mean(dim=0).cpu().numpy()
            ax.imshow(g, cmap="RdBu_r", vmin=0, vmax=1)
            mean = g.mean()
            ax.set_title(f"Gate L{lvl} ({g.shape[0]}x{g.shape[1]})\n"
                         f"flash={1 - mean:.1%} / noflash={mean:.1%}")
        else:
            ax.set_title(f"Gate L{lvl}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Real-photo pairing
# ---------------------------------------------------------------------------

def discover_real_pairs() -> list[tuple[str, Path, Path, Path | None]]:
    """Find (name, flash_path, noflash_path, petschnigg_path_or_None) for every pair."""
    flash_dir = REAL_DATA_DIR / "flash"
    noflash_dir = REAL_DATA_DIR / "no_flash"
    petsch_dir = REAL_DATA_DIR / "petschnigg"

    pairs = []
    for fp in sorted(flash_dir.iterdir()):
        if not fp.is_file():
            continue
        stem = fp.stem
        ext = fp.suffix

        # Two filename conventions:
        #   "{name}_00_flash" + "{name}_01_noflash"  (TIF reference scenes)
        #   "{name}_flash"    + "{name}_noflash"     (JPG photos)
        if "_00_flash" in stem:
            base = stem.replace("_00_flash", "")
            noflash_name = f"{base}_01_noflash{ext}"
        elif stem.endswith("_flash"):
            base = stem[: -len("_flash")]
            noflash_name = f"{base}_noflash{ext}"
        else:
            continue

        np_path = noflash_dir / noflash_name
        if not np_path.exists():
            continue

        pp_path = petsch_dir / f"{base}_03_our_result.tif"
        pairs.append((base, fp, np_path, pp_path if pp_path.exists() else None))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main() -> None:
    cfg = load_config()
    mcfg = cfg.model
    in_h, in_w = cfg.image_size  # (768, 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model ---
    model = GatedUNet(mcfg).to(device)
    ckpt_path = Path(mcfg.checkpoint_dir) / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?')})")

    loss_fn = CombinedLoss(mcfg).to(device)

    real_dir = OUT_ROOT / "real"
    syn_dir = OUT_ROOT / "synthetic"
    real_dir.mkdir(parents=True, exist_ok=True)
    syn_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Real data
    # --------------------------------------------------------------
    print("\n=== Real photos ===")
    for name, fp, np_path, pp_path in discover_real_pairs():
        flash_full = load_rgb(fp)
        noflash_full = load_rgb(np_path)
        native_shape = flash_full.shape[:2]

        # Align at native resolution → crop the warp-invalid border off both
        # images → only then resize. This way every pixel the model sees
        # corresponds to real content from both flash and no-flash.
        crop_box = None
        align_info = ""
        if ALIGN_REAL:
            flash_full, noflash_full, crop_box, (dx, dy, score, applied) = \
                align_pair_and_crop(flash_full, noflash_full)
            ch, cw = flash_full.shape[:2]
            tag = "" if applied else "  REJECTED (low confidence / huge shift)"
            align_info = f"  align dx={dx:+.1f} dy={dy:+.1f} r={score:.3f} crop→{cw}x{ch}{tag}"

        flash_disp = cv2.resize(flash_full, (in_w, in_h), interpolation=cv2.INTER_AREA)
        noflash_disp = cv2.resize(noflash_full, (in_w, in_h), interpolation=cv2.INTER_AREA)

        flash_t = torch.from_numpy(flash_disp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        noflash_t = torch.from_numpy(noflash_disp.transpose(2, 0, 1)).unsqueeze(0).to(device)

        output, gates = model(flash_t, noflash_t)
        output_np = tensor_to_numpy(output.squeeze(0))  # (in_h, in_w, 3)

        if pp_path is not None:
            petsch = load_rgb(pp_path)
            # Petschnigg result lives in the original flash frame, so it
            # must take the same crop as flash to stay registered.
            if crop_box is not None and petsch.shape[:2] == native_shape:
                top, bottom, left, right = crop_box
                petsch = petsch[top:bottom, left:right]
            fourth = cv2.resize(petsch, (in_w, in_h), interpolation=cv2.INTER_AREA)
            fourth_title = "Petschnigg"
        else:
            fourth = None
            fourth_title = "(no Petschnigg)"

        # Full-resolution output (1024x768)
        out_full = cv2.resize(output_np, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_CUBIC) \
            if (in_w, in_h) != (OUTPUT_W, OUTPUT_H) else output_np
        save_full_output(real_dir / f"{name}_full.png", out_full)

        save_combined(
            real_dir / f"{name}_combined.png",
            f"Real: {name}",
            flash_disp,
            noflash_disp,
            output_np,
            fourth,
            fourth_title,
            gates,
        )
        tag = "with petschnigg" if pp_path is not None else "no petschnigg"
        print(f"  {name}: saved ({tag}){align_info}")

    # --------------------------------------------------------------
    # Synthetic data
    # --------------------------------------------------------------
    print("\n=== Synthetic samples ===")
    val_ds = FlashNoFlashDataset(Path(cfg.generation.output_dir) / "val", augment=False)
    n = min(N_SYNTHETIC, len(val_ds))

    loss_lines = []
    for i in range(n):
        sample = val_ds[i]
        flash = sample["flash"].unsqueeze(0).to(device)
        noflash = sample["no_flash"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)

        output, gates = model(flash, noflash)
        _, loss_dict = loss_fn(output, target, gates)

        output_np = tensor_to_numpy(output.squeeze(0))
        target_np = tensor_to_numpy(sample["target"])
        flash_np = tensor_to_numpy(sample["flash"])
        noflash_np = tensor_to_numpy(sample["no_flash"])

        mse = float(np.mean((output_np - target_np) ** 2))
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))

        out_full = cv2.resize(output_np, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_CUBIC) \
            if output_np.shape[:2] != (OUTPUT_H, OUTPUT_W) else output_np
        save_full_output(syn_dir / f"{i:03d}_full.png", out_full)

        save_combined(
            syn_dir / f"{i:03d}_combined.png",
            f"Synthetic Sample {i}  PSNR={psnr:.2f} dB  loss={loss_dict['total']:.4f}",
            flash_np,
            noflash_np,
            output_np,
            target_np,
            "no_flash_clean (target)",
            gates,
        )

        loss_lines.append(
            f"sample {i:03d}  loss_total={loss_dict['total']:.6f}  "
            f"l1={loss_dict['l1']:.6f}  ssim={loss_dict['ssim']:.6f}  "
            f"gate_entropy={loss_dict['gate_entropy']:.6f}  "
            f"psnr={psnr:.4f} dB"
        )
        print(f"  sample {i:03d}: PSNR={psnr:.2f} dB  loss={loss_dict['total']:.4f}")

    # Aggregate stats + per-sample lines
    if loss_lines:
        loss_path = syn_dir / "loss.txt"
        with loss_path.open("w") as f:
            f.write(f"checkpoint: {ckpt_path}\n")
            f.write(f"epoch: {ckpt.get('epoch', '?')}\n")
            f.write(f"num_samples: {n}\n\n")
            for line in loss_lines:
                f.write(line + "\n")
        print(f"  wrote {loss_path}")

    print(f"\nAll outputs written under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
