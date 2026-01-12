# DeepJSCC (MVP)

This repository provides a minimal, end-to-end Deep Joint Source-Channel Coding (DeepJSCC) system for semantic image transmission.
It trains a single neural model to map images directly to channel symbols and reconstruct them after a differentiable physical channel.

## System Overview

**Transmitter (Encoder)**

- CNN compresses an image into a low-dimensional latent tensor.
- A power constraint is enforced via per-sample normalization (average power = 1).

**Channel (Differentiable)**

- AWGN noise is injected based on an SNR value (in dB).

**Receiver (Decoder)**

- CNN reconstructs the image from the noisy latent.
- Output uses a sigmoid activation, producing values in [0, 1].

## What This MVP Produces

- A trained checkpoint (`best.pt`) under `output_dir`.
- A PSNR-vs-SNR sweep written as `psnr_sweep.json` and `psnr_sweep.csv` under `output_dir`.
- A training log (`metrics.jsonl`) under `output_dir`.

## Requirements

- Python 3.11+
- Windows is supported; default `num_workers=0` avoids multiprocessing issues.

## Installation

This project uses `uv` for dependency management.

```bash
uv add typer pydantic pyyaml torch torchvision numpy tqdm
uv add --dev pytest ruff
```

Notes:

- PyTorch wheels are platform/driver dependent; if `uv add torch` fails on your machine, install an appropriate wheel per the official PyTorch instructions, then re-run `uv add` for the remaining packages.

## Configuration

All runtime configuration is provided by a single YAML file:

- [config.yaml](config.yaml)

The config is validated using Pydantic at runtime. Typical engineering knobs:

- **Compression / bandwidth**: `model.latent_channels`
- **Channel conditions**:
  - Training distribution: `channel.train_snr.mode` and range (`min_db`, `max_db`) or fixed `snr_db`
  - Evaluation sweep: `eval.snr_db_list`
- **Throughput vs stability (Windows)**: `dataset.num_workers` (keep `0` unless you need more)
- **Reproducibility**: `seed`

## Running

Validate the config:

```bash
uv run deepjscc validate-config --config config.yaml
```

Train:

```bash
uv run deepjscc train --config config.yaml
```

Evaluate PSNR across the configured SNR list:

```bash
uv run deepjscc eval --config config.yaml
```

## Outputs and Artifacts

All outputs are written under `output_dir` from the config, for example `runs/mvp/`.

Expected artifacts:

- `config.resolved.yaml`: the validated, resolved config used for the run
- `metrics.jsonl`: per-epoch training metrics
- `best.pt`: best checkpoint by validation PSNR
- `epoch_*.pt`: periodic checkpoints
- `psnr_sweep.json` and `psnr_sweep.csv`: evaluation results

## Baseline Results

Configuration:

- Dataset: CIFAR-10 (50k train, 10k test)
- Model: 64 base channels, 16 latent channels
- Training: 10 epochs, batch size 128, uniform SNR [0, 20] dB
- Device: CPU (tested on Windows)

Training convergence:

- Epoch 1: PSNR 19.8 dB (loss 0.0076)
- Epoch 10: PSNR 22.1 dB (loss 0.0062)
- Training time: ~2.5 minutes per epoch (CPU, Windows)

Evaluation (PSNR vs SNR):

| SNR (dB) | PSNR (dB) |
| -------- | --------- |
| -5       | 15.07     |
| 0        | 18.36     |
| 5        | 21.32     |
| 10       | 23.26     |
| 15       | 24.19     |
| 20       | 24.55     |

Observations:

- System maintains non-zero reconstruction quality at negative SNR.
- PSNR increase is approximately log-linear with SNR in the tested range.
- At 20 dB SNR, PSNR approaches the compression bottleneck imposed by 16 latent channels.

## Reproducibility Notes

- This project sets Python/NumPy/PyTorch seeds.
- GPU kernels can still introduce non-determinism depending on CUDA/cuDNN settings; the MVP does not force full determinism.
- For comparisons, keep `model.latent_channels`, training SNR distribution, and dataset preprocessing fixed.

## Interpreting Results

The key engineering signal is graceful degradation:

- At lower SNR, reconstruction quality decreases but remains non-catastrophic.
- A classical digital baseline (e.g., JPEG + channel coding) typically shows a sharper cliff effect under similar constraints.

For this MVP the primary metric is PSNR; SSIM/MS-SSIM and task-oriented metrics can be added once the baseline is stable.

## Troubleshooting

- **Training is slow on Windows**: keep `dataset.num_workers=0`; try smaller `dataset.batch_size`.
- **Out-of-memory (GPU)**: reduce `dataset.batch_size` and/or `model.base_channels`.
- **Dataset downloads repeatedly**: verify `dataset.data_dir` and file permissions.
- **Warnings from torchvision/NumPy**: these are upstream deprecations and do not block the MVP.
