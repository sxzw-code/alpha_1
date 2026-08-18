"""Tests for Geometric Brownian Motion."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.market import GeometricBrownianMotion


def test_gbm_path_dimensions_include_initial() -> None:
    model = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252, seed=1)
    path = model.generate_path(50, include_initial=True)
    assert path.shape == (51,)
    assert path[0] == pytest.approx(100.0)


def test_gbm_paths_vectorized_dimensions() -> None:
    model = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252, seed=2)
    paths = model.generate_paths(25, 100, include_initial=True)
    assert paths.shape == (25, 101)
    assert np.all(paths[:, 0] == 100.0)


def test_gbm_prices_remain_positive() -> None:
    model = GeometricBrownianMotion(s0=50.0, mu=-0.1, sigma=0.8, dt=1 / 252, seed=3)
    paths = model.generate_paths(100, 500, include_initial=True)
    assert np.all(paths > 0.0)


def test_gbm_step_updates_state() -> None:
    model = GeometricBrownianMotion(s0=100.0, mu=0.0, sigma=0.2, dt=0.01, seed=4)
    assert model.step_index == 0
    assert model.current_price == 100.0
    state = model.step()
    assert state.step == 1
    assert model.step_index == 1
    assert model.current_price == state.price
    assert state.price > 0.0


def test_gbm_reproducibility_identical_seeds() -> None:
    a = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.25, dt=1 / 252, seed=99)
    b = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.25, dt=1 / 252, seed=99)
    assert np.allclose(a.generate_path(200), b.generate_path(200))


def test_gbm_different_seeds_diverge() -> None:
    a = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.25, dt=1 / 252, seed=1)
    b = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.25, dt=1 / 252, seed=2)
    assert not np.allclose(a.generate_path(50), b.generate_path(50))


def test_gbm_rejects_non_positive_s0() -> None:
    with pytest.raises(ValueError, match="s0"):
        GeometricBrownianMotion(s0=0.0, mu=0.0, sigma=0.2, dt=0.01)


def test_gbm_stepping_does_not_affect_generate_paths() -> None:
    model = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=0.01, seed=7)
    model.step()
    model.step()
    live_price = model.current_price
    path = model.generate_path(10, rng=np.random.default_rng(0), include_initial=True)
    assert path[0] == pytest.approx(100.0)
    assert model.current_price == pytest.approx(live_price)


def test_gbm_paths_without_initial() -> None:
    model = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252, seed=8)
    paths = model.generate_paths(10, 40, include_initial=False)
    assert paths.shape == (10, 40)
    assert np.all(paths > 0.0)


def test_gbm_paths_are_independent() -> None:
    model = GeometricBrownianMotion(s0=100.0, mu=0.0, sigma=0.3, dt=1 / 252, seed=21)
    paths = model.generate_paths(50, 200, include_initial=True)
    assert not np.allclose(paths[0], paths[1])
    log_r = np.diff(np.log(paths), axis=1)
    corr = np.corrcoef(log_r[0], log_r[1])[0, 1]
    assert abs(corr) < 0.25
    # Off-diagonal mean correlation of terminal log-prices vs path 0 should be small
    terminals = np.log(paths[:, -1])
    # variance across paths is positive
    assert float(np.std(terminals)) > 0.0
