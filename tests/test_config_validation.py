from __future__ import annotations

from pathlib import Path

import pytest

from deepjscc.config import load_config


def test_load_config_valid(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
seed: 1
device: cpu
output_dir: runs/test

dataset:
  name: cifar10
  data_dir: data
  batch_size: 4
  num_workers: 0
  pin_memory: false
  download: false

model:
  base_channels: 8
  latent_channels: 4
  power_normalize: true
  output_activation: sigmoid

channel:
  type: awgn
  train_snr:
    mode: fixed
    snr_db: 10

train:
  epochs: 1
  lr: 0.001
  weight_decay: 0
  log_every: 1
  save_every: 1
  eval_every: 1
  amp: false
  grad_clip_norm: null

eval:
  snr_db_list: [0, 10]
  max_batches: 1
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.dataset.batch_size == 4
    assert cfg.channel.type == "awgn"


def test_load_config_rejects_invalid_channel(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
channel:
  type: rayleigh
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        _ = load_config(cfg_path)
