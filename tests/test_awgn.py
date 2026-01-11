from __future__ import annotations

import torch

from deepjscc.channel import awgn


def test_awgn_noise_variance_matches_snr() -> None:
    torch.manual_seed(0)
    x = torch.zeros(1024, 16)
    snr_db = 10.0
    y = awgn(x, snr_db=snr_db, power=1.0)

    # For zero signal, output variance ~ noise_var = power / snr_linear.
    snr_linear = 10 ** (snr_db / 10.0)
    expected_var = 1.0 / snr_linear
    emp_var = float(y.var(unbiased=False).item())
    assert abs(emp_var - expected_var) / expected_var < 0.2
