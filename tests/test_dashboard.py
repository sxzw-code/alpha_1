"""Tests for dashboard factories and charts (no Streamlit runtime)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from alpha.analytics.metrics import analyze_result
from alpha.dashboard.app import _in_streamlit, main
from alpha.dashboard.charts import (
    drawdown_figure,
    equity_figure,
    histogram_figure,
    price_figure,
)
from alpha.dashboard.factories import (
    DashboardConfig,
    build_engine,
    build_monte_carlo,
    execution_config,
)
from alpha.dashboard.formatting import fmt_money, fmt_pct, fmt_sharpe, rolling_mean


def _cfg(**overrides: object) -> DashboardConfig:
    base = dict(
        model_name="Geometric Brownian Motion",
        s0=100.0,
        mu=0.08,
        sigma=0.20,
        x0=100.0,
        ou_mu=100.0,
        theta=1.0,
        ou_sigma=5.0,
        n_steps=40,
        seed=7,
        strategy_name="Moving Average Crossover",
        fast_window=5,
        slow_window=12,
        lookback=10,
        entry_z=2.0,
        exit_z=0.5,
        trade_quantity=10.0,
        initial_capital=100_000.0,
        frictions=True,
        commission_bps=5.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        compare_friction=False,
    )
    base.update(overrides)
    return DashboardConfig(**base)  # type: ignore[arg-type]


def test_validation_slow_window() -> None:
    cfg = _cfg(fast_window=20, slow_window=10)
    assert any("Slow window" in e for e in cfg.validation_errors())


def test_frictionless_toggle() -> None:
    on = execution_config(_cfg(frictions=True))
    off = execution_config(_cfg(frictions=False))
    assert on.commission_bps == 5.0
    assert off.is_frictionless


def test_single_simulation_via_factory() -> None:
    cfg = _cfg()
    result = build_engine(cfg).run(cfg.n_steps)
    metrics = analyze_result(result, periods_per_year=252.0)
    assert result.prices.shape[0] == cfg.n_steps + 1
    assert result.equity.shape == result.prices.shape
    assert np.isfinite(metrics.final_equity)
    fig = price_figure(result, cfg)
    assert isinstance(fig, go.Figure)
    assert isinstance(equity_figure(result), go.Figure)
    assert isinstance(drawdown_figure(result), go.Figure)


def test_small_monte_carlo_via_factory() -> None:
    cfg = _cfg(strategy_name="Buy and Hold", n_steps=25)
    mc = build_monte_carlo(cfg, use_frictions=True)
    result = mc.run(n_paths=8, n_steps=25, seed=3, store_equity_paths=True)
    assert result.final_equity.values.shape == (8,)
    assert result.equity_paths is not None
    fig = histogram_figure(
        result.total_return, title="returns", x_title="R", is_percent=True
    )
    assert isinstance(fig, go.Figure)


def test_formatters() -> None:
    assert fmt_pct(0.1234) == "12.34%"
    assert fmt_money(1000) == "$1,000.00"
    assert fmt_sharpe(float("inf")) == "+∞"
    assert fmt_sharpe(float("nan")) == "n/a"


def test_rolling_mean() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    ma = rolling_mean(x, 2)
    assert np.isnan(ma[0])
    assert ma[1:] == pytest.approx([1.5, 2.5, 3.5])


def test_app_import_is_safe() -> None:
    assert _in_streamlit() is False
    main()  # no-op outside Streamlit
