"""Generate synthetic flash / no-flash training and validation datasets.

Reads all parameters from config.json (no argparse).  Run directly:

    python synthetic_data_generator.py

Outputs are saved under the directory specified by generation.output_dir in
config.json, with the following structure:

    <output_dir>/
      train/  (or val/)
        scene/          - underlying reflectance
        depth/          - depth maps (raw .npy + visualised .png)
        flash/          - flash images (noisy)
        no_flash/       - ambient images (noisy)
        flash_clean/    - flash images (clean)
        no_flash_clean/ - ambient images (clean)
        shadow/         - shadow maps
        specular/       - specular highlight maps
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import load_config, SyntheticDataConfig
from utils.data_generator import FlashNoFlashGenerator, SceneSample


SUBDIRS = [
    "scene",
    "depth",
    "flash",
    "no_flash",
    "flash_clean",
    "no_flash_clean",
    "shadow",
    "specular",
]


def _ensure_dirs(base: Path, partition: str) -> Path:
    """Create the directory tree for a partition and return its root."""
    part_dir = base / partition
    for sub in SUBDIRS:
        (part_dir / sub).mkdir(parents=True, exist_ok=True)
    return part_dir


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] float array to uint8."""
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """Normalise a depth map to [0, 255] for visualisation."""
    lo, hi = depth.min(), depth.max()
    if hi - lo < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    normalised = (depth - lo) / (hi - lo)
    return (normalised * 255).astype(np.uint8)


def save_sample(sample: SceneSample, part_dir: Path, idx: int) -> None:
    """Write all channels of a single sample to disk."""
    # RGB images (convert to BGR for OpenCV)
    cv2.imwrite(str(part_dir / "scene" / f"{idx:05d}.png"), _to_uint8(sample.scene[:, :, ::-1]))
    cv2.imwrite(str(part_dir / "flash" / f"{idx:05d}.png"), _to_uint8(sample.flash[:, :, ::-1]))
    cv2.imwrite(str(part_dir / "no_flash" / f"{idx:05d}.png"), _to_uint8(sample.no_flash[:, :, ::-1]))
    cv2.imwrite(str(part_dir / "flash_clean" / f"{idx:05d}.png"), _to_uint8(sample.flash_clean[:, :, ::-1]))
    cv2.imwrite(str(part_dir / "no_flash_clean" / f"{idx:05d}.png"), _to_uint8(sample.no_flash_clean[:, :, ::-1]))

    # Single-channel maps
    cv2.imwrite(str(part_dir / "depth" / f"{idx:05d}.png"), _depth_to_vis(sample.depth))
    np.save(str(part_dir / "depth" / f"{idx:05d}.npy"), sample.depth)
    cv2.imwrite(str(part_dir / "shadow" / f"{idx:05d}.png"), _to_uint8(sample.shadow_map))
    cv2.imwrite(str(part_dir / "specular" / f"{idx:05d}.png"), _to_uint8(sample.specular_map))


def generate_partition(
    cfg: SyntheticDataConfig,
    partition: str,
    num_samples: int,
    base_dir: Path,
    rng: np.random.Generator,
) -> None:
    """Generate and save all samples for one partition (train or val)."""
    part_dir = _ensure_dirs(base_dir, partition)
    generator = FlashNoFlashGenerator(cfg, rng=rng)

    print(f"Generating {num_samples} {partition} samples …")
    for i in tqdm(range(num_samples), desc=partition):
        sample = generator.generate()
        save_sample(sample, part_dir, i)

    print(f"  Saved to {part_dir}")


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    base_dir = Path(cfg.generation.output_dir)

    generate_partition(cfg, "train", cfg.generation.num_train, base_dir, rng)
    generate_partition(cfg, "val", cfg.generation.num_val, base_dir, rng)

    print("Done.")


if __name__ == "__main__":
    main()
