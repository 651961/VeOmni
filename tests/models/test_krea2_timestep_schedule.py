import math

import pytest
import torch

from veomni.models.diffusers.krea2.krea2_condition.modeling_krea2_condition import (
    _krea2_dynamic_mu,
    _krea2_shifted_timesteps,
)


def test_krea2_dynamic_mu_matches_raw_1024_resolution():
    assert _krea2_dynamic_mu(seq_len=4096, x1=256, x2=6400, y1=0.5, y2=1.15) == pytest.approx(0.90625)


def test_krea2_shifted_timesteps_matches_official_formula():
    steps = 10
    mu = 0.8
    actual = _krea2_shifted_timesteps(
        seq_len=4096,
        steps=steps,
        x1=256,
        x2=6400,
        y1=0.5,
        y2=1.15,
        mu=mu,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)[:-1]
    exp_mu = math.exp(mu)
    expected = exp_mu / (exp_mu + (1.0 / base - 1.0))
    torch.testing.assert_close(actual, expected)
