from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> "AverageMeter":
        return AverageMeter(self.total + float(value) * n, self.count + n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
