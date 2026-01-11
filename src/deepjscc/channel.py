from __future__ import annotations

import torch


def power_normalize(x: torch.Tensor, *, power: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    # Normalize per-sample average power to `power`.
    dims = tuple(range(1, x.ndim))
    mean_power = (x**2).mean(dim=dims, keepdim=True)
    scale = torch.sqrt(torch.as_tensor(power, device=x.device, dtype=x.dtype) / (mean_power + eps))
    return x * scale


def awgn(x: torch.Tensor, *, snr_db: torch.Tensor | float, power: float = 1.0) -> torch.Tensor:
    snr_db_t = torch.as_tensor(snr_db, device=x.device, dtype=x.dtype)
    snr_linear = torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype), snr_db_t / 10.0)
    noise_var = torch.as_tensor(power, device=x.device, dtype=x.dtype) / snr_linear
    noise_std = torch.sqrt(noise_var)
    noise = torch.randn_like(x) * noise_std
    return x + noise
