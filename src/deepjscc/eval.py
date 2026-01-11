from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from .channel import awgn, power_normalize
from .config import DeepJSCCConfig
from .metrics import psnr
from .model import DeepJSCCAutoencoder
from .train import build_model
from .utils import AverageMeter, ensure_dir, resolve_device


def load_checkpoint(model: DeepJSCCAutoencoder, ckpt_path: Path, *, map_location: torch.device) -> dict:
    payload = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(payload["model"])
    return payload


@torch.no_grad()
def psnr_sweep(cfg: DeepJSCCConfig, *, ckpt_path: Path, test_loader) -> dict[str, float]:
    device = resolve_device(cfg.device)
    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, map_location=device)
    model.eval()

    results: dict[str, float] = {}

    for snr in cfg.eval.snr_db_list:
        meter = AverageMeter()
        for batch_idx, (x, _) in enumerate(test_loader, start=1):
            if cfg.eval.max_batches is not None and batch_idx > cfg.eval.max_batches:
                break

            x = x.to(device)
            z = model.encode(x)
            if model.power_normalize:
                z = power_normalize(z, power=1.0)
            y = awgn(z, snr_db=float(snr), power=1.0)
            x_hat = model.decode(y)
            meter = meter.update(psnr(x_hat, x).item(), n=x.shape[0])

        results[str(snr)] = meter.avg

    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    (out_dir / "psnr_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "psnr_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", "psnr"])
        for snr_str, val in results.items():
            writer.writerow([snr_str, val])

    return results
