from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from .channel import awgn, power_normalize
from .config import DeepJSCCConfig, dump_resolved_config
from .metrics import psnr
from .model import Decoder, DeepJSCCAutoencoder, Encoder
from .utils import AverageMeter, ensure_dir, resolve_device, set_seed


def _sample_train_snr_db(cfg: DeepJSCCConfig, batch_size: int, device: torch.device) -> torch.Tensor:
    spec = cfg.channel.train_snr
    if spec.mode == "fixed":
        return torch.full((batch_size, 1, 1, 1), float(spec.snr_db), device=device)
    snr = torch.empty((batch_size, 1, 1, 1), device=device).uniform_(float(spec.min_db), float(spec.max_db))
    return snr


def build_model(cfg: DeepJSCCConfig) -> DeepJSCCAutoencoder:
    enc = Encoder(base_channels=cfg.model.base_channels, latent_channels=cfg.model.latent_channels)
    dec = Decoder(
        base_channels=cfg.model.base_channels,
        latent_channels=cfg.model.latent_channels,
        output_activation=cfg.model.output_activation,
    )
    return DeepJSCCAutoencoder(enc, dec, power_normalize=cfg.model.power_normalize)


def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


@torch.no_grad()
def validate(model: DeepJSCCAutoencoder, loader, *, device: torch.device, snr_db: float) -> dict[str, float]:
    model.eval()

    psnr_meter = AverageMeter()
    for x, _ in loader:
        x = x.to(device)
        z = model.encode(x)
        if model.power_normalize:
            z = power_normalize(z, power=1.0)
        y = awgn(z, snr_db=snr_db, power=1.0)
        x_hat = model.decode(y)
        p = psnr(x_hat, x).item()
        psnr_meter = psnr_meter.update(p, n=x.shape[0])

    return {"psnr": psnr_meter.avg}


def fit(cfg: DeepJSCCConfig, train_loader, test_loader) -> Path:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    dump_resolved_config(out_dir / "config.resolved.yaml", cfg)

    model = build_model(cfg).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    best_psnr = float("-inf")
    best_path = out_dir / "best.pt"

    log_path = out_dir / "metrics.jsonl"
    log_f = log_path.open("a", encoding="utf-8")

    try:
        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            psnr_meter = AverageMeter()

            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs}")
            for step, (x, _) in enumerate(pbar, start=1):
                x = x.to(device)

                z = model.encode(x)
                if model.power_normalize:
                    z = power_normalize(z, power=1.0)

                snr_db = _sample_train_snr_db(cfg, x.shape[0], device)
                y = awgn(z, snr_db=snr_db, power=1.0)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    x_hat = model.decode(y)
                    loss = torch.mean((x_hat - x) ** 2)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                if cfg.train.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))

                scaler.step(optimizer)
                scaler.update()

                batch_psnr = psnr(x_hat.detach(), x).item()
                psnr_meter = psnr_meter.update(batch_psnr, n=x.shape[0])

                if step % cfg.train.log_every == 0:
                    pbar.set_postfix(loss=float(loss.item()), psnr=psnr_meter.avg)

            record = {"epoch": epoch, "train_psnr": psnr_meter.avg}

            if epoch % cfg.train.eval_every == 0:
                # Use the midpoint of the training SNR range (or fixed value).
                train_snr = cfg.channel.train_snr
                if train_snr.mode == "fixed":
                    val_snr = float(train_snr.snr_db)
                else:
                    val_snr = 0.5 * (float(train_snr.min_db) + float(train_snr.max_db))

                val = validate(model, test_loader, device=device, snr_db=val_snr)
                record.update({"val_psnr": val["psnr"], "val_snr_db": val_snr})

                if val["psnr"] > best_psnr:
                    best_psnr = val["psnr"]
                    _save_checkpoint(best_path, model, optimizer, epoch)

            if epoch % cfg.train.save_every == 0:
                _save_checkpoint(out_dir / f"epoch_{epoch}.pt", model, optimizer, epoch)

            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

    finally:
        log_f.close()

    if not best_path.exists():
        _save_checkpoint(best_path, model, optimizer, cfg.train.epochs)

    return best_path
