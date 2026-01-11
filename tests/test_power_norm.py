from __future__ import annotations

import torch

from deepjscc.channel import power_normalize


def test_power_normalize_sets_unit_power() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 16, 4, 4)
    y = power_normalize(x, power=1.0)
    mean_power = (y**2).mean(dim=(1, 2, 3))
    assert torch.allclose(mean_power, torch.ones_like(mean_power), atol=1e-3, rtol=1e-3)
