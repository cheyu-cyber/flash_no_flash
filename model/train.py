"""Training script for the Gated U-Net.

Run from the project root:

    python -m model.train
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from model.network import GatedUNet
from model.dataset import FlashNoFlashDataset
from model.losses import CombinedLoss


def psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR (dB) between two [0, 1] image tensors."""
    mse = torch.mean((output - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def train_one_epoch(
    model: GatedUNet,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> dict:
    model.train()
    running = {"l1": 0.0, "perceptual": 0.0, "gate_entropy": 0.0, "total": 0.0}
    n_batches = 0

    for i, batch in enumerate(loader):
        flash = batch["flash"].to(device)
        no_flash = batch["no_flash"].to(device)
        depth = batch["depth"].to(device)
        target = batch["target"].to(device)

        output, gates = model(flash, no_flash, depth)
        loss, loss_dict = criterion(output, target, gates)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in running:
            running[k] += loss_dict[k]
        n_batches += 1

        if (i + 1) % log_interval == 0:
            avg = {k: v / n_batches for k, v in running.items()}
            print(
                f"  [epoch {epoch+1} | batch {i+1}/{len(loader)}] "
                f"loss={avg['total']:.4f}  l1={avg['l1']:.4f}  "
                f"perc={avg['perceptual']:.4f}  gate={avg['gate_entropy']:.4f}"
            )

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: GatedUNet,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    model.eval()
    running = {"l1": 0.0, "perceptual": 0.0, "gate_entropy": 0.0, "total": 0.0, "psnr": 0.0}
    n_batches = 0

    for batch in loader:
        flash = batch["flash"].to(device)
        no_flash = batch["no_flash"].to(device)
        depth = batch["depth"].to(device)
        target = batch["target"].to(device)

        output, gates = model(flash, no_flash, depth)
        _, loss_dict = criterion(output, target, gates)

        for k in loss_dict:
            running[k] += loss_dict[k]
        running["psnr"] += psnr(output, target)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def main() -> None:
    cfg = load_config()
    mcfg = cfg.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- data ----------
    data_root = Path(cfg.generation.output_dir)
    train_ds = FlashNoFlashDataset(data_root / "train", augment=True)
    val_ds = FlashNoFlashDataset(data_root / "val", augment=False)
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

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
    print(f"Model parameters: {n_params:,}")

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
            model, train_loader, criterion, optimizer, device, epoch, mcfg.log_interval
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{mcfg.num_epochs} ({elapsed:.1f}s) — "
            f"train_loss={train_metrics['total']:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation
        if (epoch + 1) % mcfg.val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            print(
                f"  val_loss={val_metrics['total']:.4f}  "
                f"val_psnr={val_metrics['psnr']:.2f} dB"
            )

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
                print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")

    # Save final checkpoint
    torch.save(
        {
            "epoch": mcfg.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / "last.pt",
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
