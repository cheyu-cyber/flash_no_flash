"""Training script for the Gated U-Net.

Run from the project root:

    python -m model.train
"""

from __future__ import annotations

import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from model.network import GatedUNet
from model.dataset import FlashNoFlashDataset
from model.losses import CombinedLoss


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


class CSVLogger:
    """Append metrics to a CSV file, one row per epoch."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        write_header = not path.exists()
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if write_header:
            self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR (dB) between two [0, 1] image tensors."""
    mse = torch.mean((output - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def gate_stats(gates: List[torch.Tensor]) -> dict:
    """Compute per-level gate means (no-flash fraction) from gate activations."""
    stats = {}
    for level, g in enumerate(gates):
        mean_g = g.mean().item()
        stats[f"gate_L{level}_noflash"] = mean_g
        stats[f"gate_L{level}_flash"] = 1.0 - mean_g
    return stats


def train_one_epoch(
    model: GatedUNet,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
    logger: logging.Logger,
) -> dict:
    model.train()
    running = {"l1": 0.0, "perceptual": 0.0, "gate_entropy": 0.0, "total": 0.0}
    running_gates: dict[str, float] = {}
    n_batches = 0

    for i, batch in enumerate(loader):
        flash = batch["flash"].to(device)
        no_flash = batch["no_flash"].to(device)
        target = batch["target"].to(device)

        output, gates = model(flash, no_flash)
        loss, loss_dict = criterion(output, target, gates)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in running:
            running[k] += loss_dict[k]

        gs = gate_stats(gates)
        for k, v in gs.items():
            running_gates[k] = running_gates.get(k, 0.0) + v

        n_batches += 1

        if (i + 1) % log_interval == 0:
            avg = {k: v / n_batches for k, v in running.items()}
            logger.info(
                f"  [epoch {epoch+1} | batch {i+1}/{len(loader)}] "
                f"loss={avg['total']:.4f}  l1={avg['l1']:.4f}  "
                f"perc={avg['perceptual']:.4f}  gate={avg['gate_entropy']:.4f}"
            )

    avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg_gates = {k: v / max(n_batches, 1) for k, v in running_gates.items()}
    return {**avg_losses, **avg_gates}


@torch.no_grad()
def validate(
    model: GatedUNet,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    model.eval()
    running = {"l1": 0.0, "perceptual": 0.0, "gate_entropy": 0.0, "total": 0.0, "psnr": 0.0}
    running_gates: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        flash = batch["flash"].to(device)
        no_flash = batch["no_flash"].to(device)
        target = batch["target"].to(device)

        output, gates = model(flash, no_flash)
        _, loss_dict = criterion(output, target, gates)

        for k in loss_dict:
            running[k] += loss_dict[k]
        running["psnr"] += psnr(output, target)

        gs = gate_stats(gates)
        for k, v in gs.items():
            running_gates[k] = running_gates.get(k, 0.0) + v

        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg_gates = {k: v / max(n_batches, 1) for k, v in running_gates.items()}
    return {**avg_losses, **avg_gates}


def main() -> None:
    cfg = load_config()
    mcfg = cfg.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- logging ----------
    log_dir = Path("logs")
    logger = setup_logging(log_dir)

    # Determine gate column names from encoder config
    n_levels = len(mcfg.encoder_channels)
    gate_cols = []
    for lvl in range(n_levels):
        gate_cols += [f"gate_L{lvl}_flash", f"gate_L{lvl}_noflash"]

    csv_logger = CSVLogger(
        log_dir / "metrics.csv",
        fieldnames=[
            "epoch", "lr", "elapsed_s",
            "train_loss", "train_l1", "train_perceptual", "train_gate_entropy",
            *[f"train_{c}" for c in gate_cols],
            "val_loss", "val_l1", "val_perceptual", "val_gate_entropy", "val_psnr",
            *[f"val_{c}" for c in gate_cols],
        ],
    )

    logger.info(f"Device: {device}")

    # ---------- data ----------
    data_root = Path(cfg.generation.output_dir)
    train_ds = FlashNoFlashDataset(data_root / "train", augment=True)
    val_ds = FlashNoFlashDataset(data_root / "val", augment=False)
    logger.info(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=mcfg.batch_size,
        shuffle=True,
        num_workers=mcfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=mcfg.batch_size,
        shuffle=False,
        num_workers=mcfg.num_workers,
        pin_memory=True,
    )

    # ---------- model ----------
    model = GatedUNet(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ---------- optimiser & scheduler ----------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=mcfg.learning_rate, weight_decay=mcfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=mcfg.num_epochs)
    criterion = CombinedLoss(mcfg).to(device)

    # ---------- checkpoint dir ----------
    ckpt_dir = Path(mcfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # ---------- training loop ----------
    for epoch in range(mcfg.num_epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mcfg.log_interval, logger
        )
        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # Log train summary
        gate_summary = "  ".join(
            f"L{lvl} flash={train_metrics.get(f'gate_L{lvl}_flash', 0):.1%} "
            f"noflash={train_metrics.get(f'gate_L{lvl}_noflash', 0):.1%}"
            for lvl in range(n_levels)
        )
        logger.info(
            f"Epoch {epoch+1}/{mcfg.num_epochs} ({elapsed:.1f}s) — "
            f"train_loss={train_metrics['total']:.4f}  lr={lr:.2e}"
        )
        logger.info(f"  Gate contributions: {gate_summary}")

        # Build CSV row (fill val columns later if needed)
        csv_row: dict = {
            "epoch": epoch + 1,
            "lr": f"{lr:.2e}",
            "elapsed_s": f"{elapsed:.1f}",
            "train_loss": f"{train_metrics['total']:.6f}",
            "train_l1": f"{train_metrics['l1']:.6f}",
            "train_perceptual": f"{train_metrics['perceptual']:.6f}",
            "train_gate_entropy": f"{train_metrics['gate_entropy']:.6f}",
        }
        for c in gate_cols:
            csv_row[f"train_{c}"] = f"{train_metrics.get(c, 0):.6f}"

        # Validation
        if (epoch + 1) % mcfg.val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, device)

            val_gate_summary = "  ".join(
                f"L{lvl} flash={val_metrics.get(f'gate_L{lvl}_flash', 0):.1%} "
                f"noflash={val_metrics.get(f'gate_L{lvl}_noflash', 0):.1%}"
                for lvl in range(n_levels)
            )
            logger.info(
                f"  val_loss={val_metrics['total']:.4f}  "
                f"val_psnr={val_metrics['psnr']:.2f} dB"
            )
            logger.info(f"  Val gate contributions: {val_gate_summary}")

            csv_row["val_loss"] = f"{val_metrics['total']:.6f}"
            csv_row["val_l1"] = f"{val_metrics['l1']:.6f}"
            csv_row["val_perceptual"] = f"{val_metrics['perceptual']:.6f}"
            csv_row["val_gate_entropy"] = f"{val_metrics['gate_entropy']:.6f}"
            csv_row["val_psnr"] = f"{val_metrics['psnr']:.4f}"
            for c in gate_cols:
                csv_row[f"val_{c}"] = f"{val_metrics.get(c, 0):.6f}"

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    ckpt_dir / "best.pt",
                )
                logger.info(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")

        csv_logger.log(csv_row)

    # Save final checkpoint
    torch.save(
        {
            "epoch": mcfg.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / "last.pt",
    )
    csv_logger.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
