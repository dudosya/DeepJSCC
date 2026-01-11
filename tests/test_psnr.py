from __future__ import annotations

import torch

from deepjscc.metrics import psnr


def test_psnr_identical_is_high() -> None:
    x = torch.zeros(2, 3, 32, 32)
    p = psnr(x, x).item()
    assert p > 90
