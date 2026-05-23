"""Generate synthetic flash / no-flash training and validation datasets.

Reads all parameters from config.json (no argparse).  Run directly:

    python synthetic_data_generator.py

Outputs are saved under the directory specified by generation.output_dir in
config.json. By default only the three subdirs required for training are
written; diagnostic maps are opt-in via generation.save_extras.

    <output_dir>/
      train/  (or val/)
        flash/          - flash images (noisy)            [REQUIRED]
        no_flash/       - ambient images (noisy)          [REQUIRED]
        no_flash_clean/ - ambient images (clean, target)  [REQUIRED]

        # Opt-in via generation.save_extras in config.json. Each name below
        # turns on writing of one directory.
        scene/          - underlying reflectance
        depth/          - depth maps (raw .npy + visualised .png)
        shadow/         - shadow maps
        specular/       - specular highlight maps
        flash_clean/    - flash images (clean)
"""

from __future__ import annotations

# Pin BLAS / OpenMP thread pools to 1 BEFORE importing numpy/scipy/cv2.
# Each worker process runs single-threaded numeric code; without this,
# N worker processes × default-thread BLAS = heavy core oversubscription.
# setdefault lets the shell env still override if explicitly set.
import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Cap OpenCV's thread pool in the parent too; it has its own pool separate
# from OpenMP/BLAS and does not honour the env vars above.
cv2.setNumThreads(1)

from utils.config import load_config, SyntheticDataConfig
from utils.data_generator import FlashNoFlashGenerator, SceneSample


# Always-saved subdirs (required by both training datasets).
REQUIRED_SUBDIRS = ["flash", "no_flash", "no_flash_clean"]

# Opt-in subdirs, controlled by generation.save_extras in config.json.
OPTIONAL_SUBDIRS = ["scene", "depth", "shadow", "specular", "flash_clean"]


def _ensure_dirs(base: Path, partition: str, extras: List[str]) -> Path:
    """Create the directory tree for a partition and return its root."""
    part_dir = base / partition
    for sub in REQUIRED_SUBDIRS:
        (part_dir / sub).mkdir(parents=True, exist_ok=True)
    for sub in extras:
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


def save_sample(
    sample: SceneSample,
    part_dir: Path,
    idx: int,
    extras: set[str],
) -> None:
    """Write the required channels of a single sample, plus any opt-in extras."""
    # Required for training (loaded by model/dataset.py and YCbCr_model/dataset.py).
    cv2.imwrite(str(part_dir / "flash" / f"{idx:05d}.png"), _to_uint8(sample.flash[:, :, ::-1]))
    cv2.imwrite(str(part_dir / "no_flash" / f"{idx:05d}.png"), _to_uint8(sample.no_flash[:, :, ::-1]))
    cv2.imwrite(str(part_dir / "no_flash_clean" / f"{idx:05d}.png"), _to_uint8(sample.no_flash_clean[:, :, ::-1]))

    # Opt-in diagnostic outputs.
    if "scene" in extras:
        cv2.imwrite(str(part_dir / "scene" / f"{idx:05d}.png"), _to_uint8(sample.scene[:, :, ::-1]))
    if "flash_clean" in extras:
        cv2.imwrite(str(part_dir / "flash_clean" / f"{idx:05d}.png"), _to_uint8(sample.flash_clean[:, :, ::-1]))
    if "depth" in extras:
        cv2.imwrite(str(part_dir / "depth" / f"{idx:05d}.png"), _depth_to_vis(sample.depth))
        np.save(str(part_dir / "depth" / f"{idx:05d}.npy"), sample.depth)
    if "shadow" in extras:
        cv2.imwrite(str(part_dir / "shadow" / f"{idx:05d}.png"), _to_uint8(sample.shadow_map))
    if "specular" in extras:
        cv2.imwrite(str(part_dir / "specular" / f"{idx:05d}.png"), _to_uint8(sample.specular_map))


# ---------------------------------------------------------------------------
# Parallel worker state
# ---------------------------------------------------------------------------

_WORKER_GENERATOR: FlashNoFlashGenerator | None = None


def _worker_init(cfg: SyntheticDataConfig) -> None:
    """Per-process initialiser: build one generator, reused across samples.

    The generator's constructor pre-computes pixel coordinate grids and focal
    lengths, which are non-trivial at larger resolutions; we do that once per
    worker and swap in a fresh RNG per task for deterministic seeding.
    """
    global _WORKER_GENERATOR
    # BLAS / OpenMP threads are already pinned via env vars set at import
    # time (see top of file). Pin OpenCV too — it has its own thread pool.
    cv2.setNumThreads(1)
    _WORKER_GENERATOR = FlashNoFlashGenerator(cfg, rng=np.random.default_rng(0))


def _worker_generate(task: Tuple[int, np.random.SeedSequence, str, Tuple[str, ...]]) -> int:
    """Generate one sample with a task-specific seed and save it."""
    idx, seed_seq, part_dir_str, extras_tuple = task
    gen = _WORKER_GENERATOR
    assert gen is not None, "_worker_init must run before _worker_generate"
    gen.rng = np.random.default_rng(seed_seq)
    sample = gen.generate()
    save_sample(sample, Path(part_dir_str), idx, set(extras_tuple))
    return idx


def _expected_files_for_index(
    part_dir: Path, idx: int, extras: List[str]
) -> List[Path]:
    """List every file that a fully-saved sample at ``idx`` should have on
    disk, given the currently-requested extras. Used to decide whether a
    sample needs regenerating."""
    tag = f"{idx:05d}"
    files = [
        part_dir / "flash"          / f"{tag}.png",
        part_dir / "no_flash"       / f"{tag}.png",
        part_dir / "no_flash_clean" / f"{tag}.png",
    ]
    extras_set = set(extras)
    if "scene"       in extras_set: files.append(part_dir / "scene"       / f"{tag}.png")
    if "flash_clean" in extras_set: files.append(part_dir / "flash_clean" / f"{tag}.png")
    if "shadow"      in extras_set: files.append(part_dir / "shadow"      / f"{tag}.png")
    if "specular"    in extras_set: files.append(part_dir / "specular"    / f"{tag}.png")
    if "depth"       in extras_set:
        # depth writes both a PNG visualisation and the raw .npy float array.
        files.append(part_dir / "depth" / f"{tag}.png")
        files.append(part_dir / "depth" / f"{tag}.npy")
    return files


def _completed_indices(
    part_dir: Path, num_samples: int, extras: List[str]
) -> set[int]:
    """Return the set of sample indices in [0, num_samples) for which every
    expected file is already present on disk."""
    done: set[int] = set()
    for i in range(num_samples):
        if all(p.exists() for p in _expected_files_for_index(part_dir, i, extras)):
            done.add(i)
    return done


def generate_partition(
    cfg: SyntheticDataConfig,
    partition: str,
    num_samples: int,
    base_dir: Path,
    seed_seqs: List[np.random.SeedSequence],
    num_workers: int,
) -> None:
    """Generate and save all samples for one partition (train or val).

    Resumable: indices whose required files (and any opt-in extras) are
    already present on disk are skipped. The per-index seed comes from
    ``seed_seqs[i]`` which is deterministic in ``(cfg.seed, num_samples)``,
    so a skipped sample is bit-identical to what a fresh regeneration would
    produce.
    """
    extras_tuple = tuple(cfg.generation.save_extras)
    extras_list = list(extras_tuple)
    part_dir = _ensure_dirs(base_dir, partition, extras_list)

    # Scan existing files and build the task list of only-missing indices.
    done = _completed_indices(part_dir, num_samples, extras_list)
    tasks = [
        (i, seed_seqs[i], str(part_dir), extras_tuple)
        for i in range(num_samples)
        if i not in done
    ]
    todo = len(tasks)

    if todo == 0:
        print(f"All {num_samples} {partition} samples already present in {part_dir}; nothing to do.")
        return

    if done:
        print(
            f"Resuming {partition}: {len(done)} of {num_samples} already complete; "
            f"generating the remaining {todo} with {num_workers} workers …"
        )
    else:
        print(f"Generating {num_samples} {partition} samples with {num_workers} workers …")

    # Chunk work so per-task IPC overhead stays small, but keep chunks small
    # enough that the progress bar updates responsively.
    chunksize = max(1, todo // (num_workers * 8))

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(cfg,),
    ) as ex:
        for _ in tqdm(
            ex.map(_worker_generate, tasks, chunksize=chunksize),
            total=todo,
            desc=partition,
        ):
            pass

    print(f"  Saved to {part_dir}")


def main() -> None:
    # --- Load everything config-related up front ---
    cfg = load_config()
    base_dir = Path(cfg.generation.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Worker count: respect the SGE/qsub allocation (NSLOTS) on SCC, since
    # os.cpu_count() returns the whole node (e.g. 32) rather than the slots
    # reserved for this job. Fall back to affinity/cpu_count off-cluster.
    # Overridable via NUM_DATAGEN_WORKERS.
    nslots = os.environ.get("NSLOTS")
    if nslots:
        allowed = int(nslots)
    else:
        try:
            allowed = len(os.sched_getaffinity(0))
        except AttributeError:
            allowed = os.cpu_count() or 2
    default_workers = max(1, allowed)
    num_workers = int(os.environ.get("NUM_DATAGEN_WORKERS", default_workers))

    # Per-sample deterministic seeding: spawn a SeedSequence tree from cfg.seed.
    # Each sample gets an independent stream, reproducible for a given
    # (cfg.seed, num_train, num_val) regardless of worker count.
    root_ss = np.random.SeedSequence(cfg.seed)
    train_root, val_root = root_ss.spawn(2)
    train_seeds = train_root.spawn(cfg.generation.num_train)
    val_seeds = val_root.spawn(cfg.generation.num_val)

    # --- Run image generation ---
    generate_partition(cfg, "train", cfg.generation.num_train, base_dir, train_seeds, num_workers)
    generate_partition(cfg, "val", cfg.generation.num_val, base_dir, val_seeds, num_workers)

    print("Done.")


if __name__ == "__main__":
    main()
