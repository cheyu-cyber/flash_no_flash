# flash_no_flash

A learned flash / no-flash reconstruction pipeline. A dual-encoder gated U-Net
takes a noisy ambient (no-flash) capture and a high-SNR flash capture of the
same scene and produces a denoised image that keeps the ambient colour and the
flash brightness.

## Setup

All knobs live in `config.json` at the project root. Image size, scene
parameter ranges, model hyperparameters, training schedule, output paths.
Edit it before running anything.

## 1. Generate synthetic training data

```bash
python synthetic_data_generator.py
```
or 
```bash
python -m synthetic_data_generator
```

Reads `config.json` and writes `num_train + num_val` paired samples to the
`generation.output_dir` directory (default `./data/synthetic`). Each sample
includes the flash, no-flash, clean target, depth map, shadow map, and
specular map.

## 2. Train

RGB variant:

```bash
python -m model.train
```

YCbCr variant (per-channel weighted L1, SSIM on Y only):

```bash
python -m YCbCr_model.train
```

Both honour `config.json`'s `model` / `ycbcr_model` blocks for optimiser
settings, batch size, epoch count, and resume checkpoint. Per-epoch metrics
go to `logs/metrics.csv` (RGB) or `logs_ycbcr/metrics.csv` (YCbCr);
checkpoints go to `checkpoints/` or `checkpoints_ycbcr/`.

## 3. Inference

RGB variant:

```bash
python -m model.inference
```

YCbCr variant:

```bash
python -m YCbCr_model.inference
```

Both scripts:

- load the latest checkpoint from the configured `checkpoint_dir`,
- run the model on every aligned pair in `data/real_data/flash/` and
  `data/real_data/no_flash/`, plus the held-out synthetic validation split,
- write per-sample full reconstructions and 2×4 diagnostic montages
  (inputs / output / Petschnigg baseline if available, plus per-level gate
  maps) to `logs/visualizations/` (RGB) or `logs_ycbcr/visualizations/`
  (YCbCr).

## Directory layout

```
config.json                     all runtime parameters
synthetic_data_generator.py     produces ./data/synthetic
utils/                          config loader + data generator (parallelised)
model/                          RGB variant: network, dataset, losses, train, inference
YCbCr_model/                    YCbCr variant of the same
data/real_data/                 real flash / no-flash pairs (Petschnigg test set)
checkpoints*/                   saved weights
logs*/                          training metrics and inference visualisations
```
