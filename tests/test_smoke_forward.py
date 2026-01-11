from __future__ import annotations

import torch

from deepjscc.channel import awgn, power_normalize
from deepjscc.config import DeepJSCCConfig
from deepjscc.train import build_model


def test_smoke_forward_shapes() -> None:
    cfg = DeepJSCCConfig()
    cfg.dataset.batch_size = 2
    model = build_model(cfg)

    x = torch.rand(2, 3, 32, 32)
    z = model.encode(x)
    z = power_normalize(z)
    y = awgn(z, snr_db=10.0)
    x_hat = model.decode(y)

    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()
    x_hat_detached = x_hat.detach()
    assert float(x_hat_detached.min()) >= 0.0
    assert float(x_hat_detached.max()) <= 1.0
