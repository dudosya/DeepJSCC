from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


DeviceLiteral = Literal["auto", "cpu", "cuda"]
SNRMode = Literal["fixed", "uniform"]


class DatasetConfig(BaseModel):
    name: Literal["cifar10"] = "cifar10"
    data_dir: Path = Path("data")
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = True

    @field_validator("batch_size")
    @classmethod
    def _batch_size_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be > 0")
        return value

    @field_validator("num_workers")
    @classmethod
    def _num_workers_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("num_workers must be >= 0")
        return value


class ModelConfig(BaseModel):
    base_channels: int = 64
    latent_channels: int = 16
    power_normalize: bool = True
    output_activation: Literal["sigmoid"] = "sigmoid"

    @field_validator("base_channels", "latent_channels")
    @classmethod
    def _channels_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("channels must be > 0")
        return value


class TrainSNRConfig(BaseModel):
    mode: SNRMode = "uniform"
    snr_db: float = 10.0
    min_db: float = 0.0
    max_db: float = 20.0

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> "TrainSNRConfig":
        if self.mode == "fixed":
            return self
        if self.min_db >= self.max_db:
            raise ValueError("train_snr.min_db must be < train_snr.max_db")
        return self


class ChannelConfig(BaseModel):
    type: Literal["awgn"] = "awgn"
    train_snr: TrainSNRConfig = Field(default_factory=TrainSNRConfig)


class TrainConfig(BaseModel):
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    log_every: int = 100
    save_every: int = 1
    eval_every: int = 1
    amp: bool = False
    grad_clip_norm: float | None = None

    @field_validator("epochs")
    @classmethod
    def _epochs_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("epochs must be >= 1")
        return value

    @field_validator("lr")
    @classmethod
    def _lr_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("lr must be > 0")
        return value


class EvalConfig(BaseModel):
    snr_db_list: list[float] = Field(default_factory=lambda: [-5, 0, 5, 10, 15, 20])
    max_batches: int | None = None

    @field_validator("snr_db_list")
    @classmethod
    def _snr_db_list_nonempty(cls, value: list[float]) -> list[float]:
        if len(value) == 0:
            raise ValueError("snr_db_list must be non-empty")
        return value


class DeepJSCCConfig(BaseModel):
    seed: int = 0
    device: DeviceLiteral = "auto"
    output_dir: Path = Path("runs/mvp")

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @field_validator("seed")
    @classmethod
    def _seed_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value


def load_config(path: Path) -> DeepJSCCConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping/object")

    try:
        return DeepJSCCConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid config {path}:\n{exc}") from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_resolved_config(path: Path, cfg: DeepJSCCConfig) -> None:
    payload: dict[str, Any] = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
