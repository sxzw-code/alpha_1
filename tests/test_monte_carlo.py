"""Tests for Monte Carlo simulation."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.execution import ExecutionConfig, SimpleExecutionModel
from alpha.market import GeometricBrownianMotion, PathReplayModel
from alpha.portfolio import Portfolio
from alpha.simulation import MonteCarloSimulator, SimulationEngine
from alpha.strategies import BuyAndHoldStrategy, HoldStrategy, MovingAverageCrossover


def _gbm_factory() -> GeometricBrownianMotion:
    return GeometricBrownianMotion(s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252)


def test_monte_carlo_output_dimensions() -> None:
    mc = MonteCarloSimulator(
        model_factory=_gbm_factory,
        strategy_factory=lambda: BuyAndHoldStrategy(quantity=10.0),
        initial_capital=100_000.0,
    )
    result = mc.run(n_paths=32, n_steps=40, seed=7, store_equity_paths=True)
    assert result.n_paths == 32
    assert result.n_steps == 40
    assert result.final_equity.values.shape == (32,)
    assert result.total_return.values.shape == (32,)
    assert result.annualized_return.values.shape == (32,)
    assert result.sharpe_ratio.values.shape == (32,)
    assert result.max_drawdown.values.shape == (32,)
    assert result.total_transaction_costs.values.shape == (32,)
    assert result.n_trades.values.shape == (32,)
    assert result.equity_paths is not None
    assert result.equity_paths.shape == (32, 41)


def test_monte_carlo_reproducibility() -> None:
    def make() -> MonteCarloSimulator:
        return MonteCarloSimulator(
            model_factory=_gbm_factory,
            strategy_factory=lambda: MovingAverageCrossover(3, 8, trade_quantity=5.0),
            execution_config=ExecutionConfig(commission_bps=5.0, slippage_bps=1.0),
            initial_capital=100_000.0,
        )

    a = make().run(n_paths=20, n_steps=60, seed=42)
    b = make().run(n_paths=20, n_steps=60, seed=42)
    assert np.allclose(a.final_equity.values, b.final_equity.values)
    assert np.allclose(a.sharpe_ratio.values, b.sharpe_ratio.values)
    assert np.allclose(a.max_drawdown.values, b.max_drawdown.values)
    assert np.allclose(a.n_trades.values, b.n_trades.values)


def test_percentile_ordering() -> None:
    mc = MonteCarloSimulator(
        model_factory=_gbm_factory,
        strategy_factory=lambda: BuyAndHoldStrategy(quantity=8.0),
        initial_capital=100_000.0,
    )
    result = mc.run(n_paths=64, n_steps=80, seed=3)
    for dist in (
        result.total_return,
        result.annualized_return,
        result.sharpe_ratio,
        result.max_drawdown,
        result.final_equity,
    ):
        assert dist.p5 <= dist.p25 + 1e-12
        assert dist.p25 <= dist.median + 1e-12
        assert dist.median <= dist.p75 + 1e-12
        assert dist.p75 <= dist.p95 + 1e-12
        assert dist.min <= dist.p5 + 1e-12
        assert dist.p95 <= dist.max + 1e-12


def test_paths_are_independent() -> None:
    mc = MonteCarloSimulator(
        model_factory=_gbm_factory,
        strategy_factory=lambda: HoldStrategy(),
        initial_capital=100_000.0,
    )
    result = mc.run(n_paths=16, n_steps=100, seed=11, store_equity_paths=True)
    assert result.equity_paths is not None
    # Hold strategy: equity is flat, so independence is in the generated prices
    # reconstructed from a fresh model with the same seed.
    model = _gbm_factory()
    prices = model.generate_paths(16, 100, rng=np.random.default_rng(11), include_initial=True)
    assert not np.allclose(prices[0], prices[1])
    log_r = np.diff(np.log(prices), axis=1)
    corr = np.corrcoef(log_r[0], log_r[1])[0, 1]
    assert abs(corr) < 0.35


def test_state_reset_between_paths() -> None:
    created = {"strategies": 0, "portfolios": 0}

    def strategy_factory() -> BuyAndHoldStrategy:
        created["strategies"] += 1
        return BuyAndHoldStrategy(quantity=4.0)

    def portfolio_factory() -> Portfolio:
        created["portfolios"] += 1
        return Portfolio(initial_capital=50_000.0)

    mc = MonteCarloSimulator(
        model_factory=_gbm_factory,
        strategy_factory=strategy_factory,
        portfolio_factory=portfolio_factory,
    )
    n_paths = 9
    mc.run(n_paths=n_paths, n_steps=15, seed=5)
    assert created["strategies"] == n_paths
    assert created["portfolios"] == n_paths


def test_mc_path_matches_single_engine_replay() -> None:
    n_paths, n_steps = 5, 45
    mc = MonteCarloSimulator(
        model_factory=_gbm_factory,
        strategy_factory=lambda: BuyAndHoldStrategy(quantity=10.0),
        execution_config=ExecutionConfig.frictionless(),
        initial_capital=100_000.0,
    )
    result = mc.run(n_paths=n_paths, n_steps=n_steps, seed=99, store_equity_paths=True)
    prices = _gbm_factory().generate_paths(
        n_paths, n_steps, rng=np.random.default_rng(99), include_initial=True
    )
    engine = SimulationEngine(
        model=PathReplayModel(prices[2], dt=1 / 252),
        portfolio=Portfolio(initial_capital=100_000.0),
        strategy=BuyAndHoldStrategy(quantity=10.0),
        execution=SimpleExecutionModel.frictionless(),
        rng=np.random.default_rng(0),
        record_steps=False,
    )
    single = engine.run(n_steps)
    assert result.equity_paths is not None
    assert np.allclose(result.equity_paths[2], single.equity)


def test_hold_strategy_mc_zero_costs_and_flat_equity() -> None:
    mc = MonteCarloSimulator(
        model_factory=_gbm_factory,
        strategy_factory=lambda: HoldStrategy(),
        initial_capital=25_000.0,
    )
    result = mc.run(n_paths=12, n_steps=30, seed=1)
    assert np.allclose(result.final_equity.values, 25_000.0)
    assert np.allclose(result.total_transaction_costs.values, 0.0)
    assert np.allclose(result.n_trades.values, 0.0)
    assert np.allclose(result.max_drawdown.values, 0.0)
