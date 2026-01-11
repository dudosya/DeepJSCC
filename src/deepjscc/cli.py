from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .data import build_cifar10_loaders
from .eval import psnr_sweep
from .train import fit

app = typer.Typer(add_completion=False)


@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a single YAML config file.",
    ),
) -> None:
    cfg = load_config(config)
    typer.echo(f"Config OK: output_dir={cfg.output_dir} device={cfg.device}")


@app.command()
def train(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a single YAML config file.",
    ),
) -> None:
    cfg = load_config(config)
    train_loader, test_loader = build_cifar10_loaders(cfg)
    best_path = fit(cfg, train_loader, test_loader)
    typer.echo(f"Saved best checkpoint: {best_path}")


@app.command("eval")
def evaluate(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a single YAML config file.",
    ),
    ckpt: Path | None = typer.Option(
        None,
        "--ckpt",
        exists=False,
        dir_okay=False,
        help="Checkpoint path (defaults to <output_dir>/best.pt).",
    ),
) -> None:
    cfg = load_config(config)
    _, test_loader = build_cifar10_loaders(cfg)

    ckpt_path = ckpt or (Path(cfg.output_dir) / "best.pt")
    if not ckpt_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {ckpt_path}")

    results = psnr_sweep(cfg, ckpt_path=ckpt_path, test_loader=test_loader)
    typer.echo("PSNR sweep written to output_dir (psnr_sweep.json/csv)")
    for snr_db, p in results.items():
        typer.echo(f"snr_db={snr_db}: psnr={p:.3f}")
