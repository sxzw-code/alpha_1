"""Tests for Ornstein–Uhlenbeck mean reversion."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.market import OrnsteinUhlenbeck


def test_ou_path_dimensions() -> None:
    model = OrnsteinUhlenbeck(x0=100.0, mu=100.0, theta=1.0, sigma=2.0, dt=0.01, seed=1)
    path = model.generate_path(40, include_initial=True)
    assert path.shape == (41,)
    assert path[0] == pytest.approx(100.0)


def test_ou_paths_vectorized_dimensions() -> None:
    model = OrnsteinUhlenbeck(x0=50.0, mu=60.0, theta=0.5, sigma=1.5, dt=0.01, seed=2)
    paths = model.generate_paths(10, 30, include_initial=False)
    assert paths.shape == (10, 30)


def test_ou_reproducibility() -> None:
    kwargs = dict(x0=100.0, mu=100.0, theta=2.0, sigma=5.0, dt=1 / 252)
    a = OrnsteinUhlenbeck(**kwargs, seed=42)
    b = OrnsteinUhlenbeck(**kwargs, seed=42)
    assert np.allclose(a.generate_paths(5, 100), b.generate_paths(5, 100))


def test_ou_mean_reversion_toward_mu() -> None:
    """With strong theta and moderate noise, terminal mean should approach μ."""
    model = OrnsteinUhlenbeck(
        x0=0.0, mu=10.0, theta=5.0, sigma=0.5, dt=0.01, seed=123
    )
    paths = model.generate_paths(2000, 500, include_initial=False)
    terminal_mean = float(paths[:, -1].mean())
    assert terminal_mean == pytest.approx(10.0, abs=0.5)


def test_ou_rejects_negative_theta() -> None:
    with pytest.raises(ValueError, match="theta"):
        OrnsteinUhlenbeck(x0=1.0, mu=1.0, theta=-0.1, sigma=1.0, dt=0.01)


def test_ou_theta_zero_is_arithmetic_brownian() -> None:
    model = OrnsteinUhlenbeck(x0=0.0, mu=99.0, theta=0.0, sigma=1.0, dt=1.0, seed=5)
    # μ is ignored when theta=0; path is pure noise walk from x0.
    path = model.generate_path(1, include_initial=True)
    assert path.shape == (2,)
    assert path[0] == 0.0


def test_ou_vectorized_matches_recurrence() -> None:
    model = OrnsteinUhlenbeck(x0=100.0, mu=105.0, theta=1.5, sigma=3.0, dt=1 / 252)
    z = np.random.default_rng(0).standard_normal((12, 80))
    closed = model._paths_from_shocks(z)
    x = np.full(12, model.x0)
    looped = np.empty_like(z)
    for t in range(z.shape[1]):
        x = model._transition(x, z[:, t])
        looped[:, t] = x
    assert np.allclose(closed, looped, rtol=1e-10, atol=1e-10)


def test_ou_theta_zero_vectorized_cumsum() -> None:
    model = OrnsteinUhlenbeck(x0=10.0, mu=0.0, theta=0.0, sigma=2.0, dt=0.25)
    z = np.random.default_rng(1).standard_normal((4, 20))
    paths = model._paths_from_shocks(z)
    expected = 10.0 + model.sigma * np.sqrt(0.25) * np.cumsum(z, axis=1)
    assert np.allclose(paths, expected)
