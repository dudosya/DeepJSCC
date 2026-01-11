from __future__ import annotations

import torch


def mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean((x_hat - x) ** 2)


def psnr(x_hat: torch.Tensor, x: torch.Tensor, *, data_range: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    err = mse(x_hat, x).clamp_min(eps)
    return 10.0 * torch.log10((data_range**2) / err)
