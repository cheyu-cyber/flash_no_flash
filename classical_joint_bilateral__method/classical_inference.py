"""Classical (Petschnigg) inference over all flash / no-flash pairs.

Mirrors ``model/inference.py`` for the classical Joint Bilateral pipeline:

* iterate every pair under ``data/test_data/{flash,no_flash}``
* register no-flash to flash at native resolution (handheld between shots)
* resize the aligned pair to 1024 x 768
* run :func:`flash_no_flash_pipeline` (Eqs. 5-7 from Petschnigg et al.)
* write the final image plus a side-by-side flash | no-flash | denoised PNG

Run from this directory::

    python classical_inference.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.structure import flash_no_flash_pipeline, srgb_to_linear

# Swap the bilateral filters in models.structure for the torch versions so
# 1024x768 stays usable. The pipeline resolves bilateral_filter and
# joint_bilateral_filter via the structure module's globals, so monkey-
# patching those names is transparent to the pipeline body.
USE_TORCH = False
try:
    import torch
    from models import structure as _structure
    from models.algo_torch import (
        bilateral_filter_torch,
        joint_bilateral_filter_torch,
    )

    _DTYPE = torch.float32  # plenty for 8-bit image inputs

    def _bf(image, sigma_d, sigma_r, radius=None):
        return bilateral_filter_torch(image, sigma_d, sigma_r, radius, dtype=_DTYPE)

    def _jbf(ambient, flash, sigma_d, sigma_r, radius=None):
        return joint_bilateral_filter_torch(
            ambient, flash, sigma_d, sigma_r, radius, dtype=_DTYPE
        )

    _structure.bilateral_filter = _bf
    _structure.joint_bilateral_filter = _jbf
    USE_TORCH = True
except Exception as exc:  # pragma: no cover
    print(f"[warn] torch fast path unavailable, falling back to numpy: {exc!r}")

# Optional translation alignment (handheld camera between shots).
HAVE_ALIGN = False
try:
    from model.align import align_pair_and_crop
    HAVE_ALIGN = True
except Exception as exc:  # pragma: no cover
    print(f"[warn] alignment unavailable, skipping: {exc!r}")


# --- paths / hyperparameters ---
DATA_ROOT   = PROJECT_ROOT / "data" / "test_data"
FLASH_DIR   = DATA_ROOT / "flash"
NOFLASH_DIR = DATA_ROOT / "no_flash"
OUT_DIR     = PROJECT_ROOT / "logs" / "classical"
OUT_FULL    = OUT_DIR / "full"
OUT_COMBO   = OUT_DIR / "combined"

OUTPUT_W, OUTPUT_H = 1024, 768

SIGMA_D              = 2.0
SIGMA_R_BILATERAL    = 0.03
SIGMA_R_JOINT        = 0.001
TAU_SHADOW           = 0.08
SATURATION_THRESHOLD = 0.95
RADIUS               = max(1, int(np.ceil(3.0 * SIGMA_D)))


def load_rgb01(path: Path) -> np.ndarray:
    """Read an image as RGB float32 in [0, 1] (cv2 returns BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1].astype(np.float32) / 255.0


def save_rgb01(path: Path, img: np.ndarray) -> None:
    bgr = (img[:, :, ::-1].clip(0, 1) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), bgr)


def save_combined(path: Path, flash: np.ndarray, noflash: np.ndarray, output: np.ndarray) -> None:
    """Horizontal concat: flash | no-flash | classical denoised."""
    sep = np.ones((flash.shape[0], 6, 3), dtype=flash.dtype)
    grid = np.concatenate([flash, sep, noflash, sep, output], axis=1)
    save_rgb01(path, grid)


def discover_pairs() -> list[tuple[str, Path, Path]]:
    """Match every flash file with its no-flash counterpart.

    The dataset mixes three filename conventions:

    * ``<name>_00_flash.<ext>`` <-> ``<name>_01_noflash.<ext>`` (TIF scenes)
    * ``<name>_flash.<ext>``    <-> ``<name>_noflash.<ext>``    (JPG photos)
    * ``<name>.<ext>``          <-> ``<name>.<ext>``            (numeric PNGs)
    """
    pairs = []
    for fp in sorted(FLASH_DIR.iterdir()):
        if not fp.is_file():
            continue
        stem, ext = fp.stem, fp.suffix
        if "_00_flash" in stem:
            base = stem.replace("_00_flash", "")
            np_name = f"{base}_01_noflash{ext}"
        elif stem.endswith("_flash"):
            base = stem[: -len("_flash")]
            np_name = f"{base}_noflash{ext}"
        else:
            base = stem
            np_name = fp.name  # numeric / identical-name convention
        np_path = NOFLASH_DIR / np_name
        if np_path.exists():
            pairs.append((base, fp, np_path))
    return pairs


def _maybe_align(flash: np.ndarray, ambient: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if not HAVE_ALIGN:
        return flash, ambient, ""
    try:
        af, aa, _crop, (dx, dy, score, applied) = align_pair_and_crop(flash, ambient)
        tag = "" if applied else " REJECTED"
        info = f"  align dx={dx:+.1f} dy={dy:+.1f} r={score:.3f}{tag}"
        return af, aa, info
    except Exception as exc:
        return flash, ambient, f"  align ERROR: {exc!r}"


def main() -> None:
    OUT_FULL.mkdir(parents=True, exist_ok=True)
    OUT_COMBO.mkdir(parents=True, exist_ok=True)

    print(f"data       : {DATA_ROOT}")
    print(f"output     : {OUT_DIR}")
    print(f"resolution : {OUTPUT_W}x{OUTPUT_H}")
    print(f"torch      : {'on (float32)' if USE_TORCH else 'off (numpy)'}")
    print(f"alignment  : {'on' if HAVE_ALIGN else 'off'}")
    print(f"sigmas     : sigma_d={SIGMA_D}  sr_b={SIGMA_R_BILATERAL}  "
          f"sr_j={SIGMA_R_JOINT}  radius={RADIUS}")
    print()

    pairs = discover_pairs()
    print(f"found {len(pairs)} flash/no-flash pairs")
    print()

    t0 = time.perf_counter()
    for i, (name, fp, np_path) in enumerate(pairs):
        flash   = load_rgb01(fp)
        ambient = load_rgb01(np_path)

        flash, ambient, align_info = _maybe_align(flash, ambient)

        flash   = cv2.resize(flash,   (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_AREA)
        ambient = cv2.resize(ambient, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_AREA)

        # Eq. 8 expects linear-light pair for the mask threshold; sRGB-linearize.
        ambient_lin = srgb_to_linear(ambient)
        flash_lin   = srgb_to_linear(flash)

        result = flash_no_flash_pipeline(
            ambient=ambient,
            flash=flash,
            sigma_d=SIGMA_D,
            sigma_r_bilateral=SIGMA_R_BILATERAL,
            sigma_r_joint=SIGMA_R_JOINT,
            ambient_linear=ambient_lin,
            flash_linear=flash_lin,
            tau_shadow=TAU_SHADOW,
            saturation_threshold=SATURATION_THRESHOLD,
            radius=RADIUS,
            morph_radius=1,
            feather_radius=2,
        )

        save_rgb01(OUT_FULL / f"{name}.png", result.result)
        save_combined(OUT_COMBO / f"{name}.png", flash, ambient, result.result)
        print(f"  [{i + 1:02d}/{len(pairs)}] {name}{align_info}")

    elapsed = time.perf_counter() - t0
    print(f"\nWrote {len(pairs)} outputs to {OUT_DIR}")
    print(f"elapsed: {elapsed:.1f}s  ({elapsed / max(1, len(pairs)):.2f}s per pair)")


if __name__ == "__main__":
    main()
