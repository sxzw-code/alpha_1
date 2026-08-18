"""Tests for performance analytics."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.analytics import (
    analyze,
    analyze_drawdown,
    max_drawdown,
    max_drawdown_duration,
    period_returns,
    stacked_equity_metrics,
)


def test_known_total_return() -> None:
    equity = np.array([80.0, 100.0])
    metrics = analyze(equity)
    assert metrics.total_return == pytest.approx(0.25)
    assert metrics.final_equity == pytest.approx(100.0)


def test_known_drawdown() -> None:
    equity = np.array([100.0, 120.0, 90.0, 90.0])
    dd = analyze_drawdown(equity)
    assert max_drawdown(equity) == pytest.approx(-0.25)
    assert dd.max_drawdown == pytest.approx(-0.25)
    assert dd.series == pytest.approx(np.array([0.0, 0.0, -0.25, -0.25]))
    assert max_drawdown_duration(dd.series) == 2
    assert dd.max_drawdown_duration == 2


def test_sharpe_matches_closed_form() -> None:
    # Four 1% / -1% oscillations starting at 100.
    r = np.array([0.01, -0.01, 0.01, -0.01])
    equity = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + r]))
    ppy = 252.0
    metrics = analyze(equity, periods_per_year=ppy, risk_free_rate=0.0)
    total = float(equity[-1] / equity[0] - 1.0)
    n = 4
    cagr = (1.0 + total) ** (ppy / n) - 1.0
    ann_vol = float(np.std(r, ddof=1)) * np.sqrt(ppy)
    assert metrics.total_return == pytest.approx(total)
    assert metrics.annualized_return == pytest.approx(cagr)
    assert metrics.annualized_volatility == pytest.approx(ann_vol)
    assert metrics.sharpe_ratio == pytest.approx(cagr / ann_vol)


def test_zero_volatility_flat_equity() -> None:
    equity = np.array([100.0, 100.0, 100.0, 100.0])
    metrics = analyze(equity)
    assert metrics.total_return == pytest.approx(0.0)
    assert metrics.annualized_return == pytest.approx(0.0)
    assert metrics.annualized_volatility == pytest.approx(0.0)
    assert metrics.sharpe_ratio == pytest.approx(0.0)
    assert metrics.max_drawdown == pytest.approx(0.0)
    assert metrics.n_trades == 0
    assert np.isnan(metrics.win_rate)
    assert np.isnan(metrics.average_trade_pnl)


def test_zero_volatility_positive_drift_is_inf_sharpe() -> None:
    equity = np.array([100.0, 101.0, 102.01])  # constant +1%
    metrics = analyze(equity)
    assert metrics.annualized_volatility == pytest.approx(0.0)
    assert metrics.sharpe_ratio == float("inf")


def test_short_history_single_observation() -> None:
    metrics = analyze(np.array([50_000.0]))
    assert metrics.total_return == pytest.approx(0.0)
    assert metrics.annualized_volatility == pytest.approx(0.0)
    assert metrics.sharpe_ratio == pytest.approx(0.0)


def test_empty_equity() -> None:
    metrics = analyze(np.array([]))
    assert metrics.total_return == pytest.approx(0.0)
    assert metrics.n_trades == 0


def test_period_returns() -> None:
    rets = period_returns(np.array([100.0, 110.0, 99.0]))
    assert rets == pytest.approx(np.array([0.10, 99.0 / 110.0 - 1.0]))


def test_stacked_equity_matches_univariate() -> None:
    rng = np.random.default_rng(0)
    paths = 100.0 * np.exp(np.cumsum(0.001 + 0.02 * rng.standard_normal((6, 30)), axis=1))
    paths = np.concatenate([np.full((6, 1), 100.0), paths], axis=1)
    stacked = stacked_equity_metrics(paths, periods_per_year=252.0)
    for i in range(6):
        m = analyze(paths[i], periods_per_year=252.0)
        assert stacked["total_return"][i] == pytest.approx(m.total_return)
        assert stacked["max_drawdown"][i] == pytest.approx(m.max_drawdown)
        assert stacked["sharpe_ratio"][i] == pytest.approx(m.sharpe_ratio, rel=1e-10)
