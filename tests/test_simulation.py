"""Tests for the simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.market import GeometricBrownianMotion
from alpha.portfolio import Portfolio
from alpha.simulation import SimulationEngine
from alpha.strategies import BuyAndHoldStrategy, HoldStrategy


def _make_engine(seed: int = 42, strategy=None) -> SimulationEngine:
    model = GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252)
    portfolio = Portfolio(initial_capital=100_000.0)
    strat = strategy if strategy is not None else HoldStrategy()
    return SimulationEngine(model=model, portfolio=portfolio, strategy=strat, seed=seed)


def test_engine_step_advances_state() -> None:
    engine = _make_engine()
    assert engine.step_index == 0
    result = engine.step()
    assert result.step == 1
    assert engine.step_index == 1
    assert result.market.price > 0.0
    assert result.portfolio.equity > 0.0
    assert len(engine.results().prices) == 2  # initial + one step


def test_engine_run_n_steps() -> None:
    engine = _make_engine()
    result = engine.run(n_steps=50)
    assert result.n_steps == 50
    assert result.prices.shape == (51,)
    assert result.equity.shape == (51,)
    assert result.timestamps.shape == (51,)
    assert len(result.step_results) == 50


def test_engine_reset_restores_initial_state() -> None:
    engine = _make_engine(seed=7)
    engine.run(20)
    engine.reset()
    assert engine.step_index == 0
    assert engine.portfolio.cash == pytest.approx(100_000.0)
    assert engine.model.current_price == pytest.approx(100.0)
    assert len(engine.results().prices) == 1


def test_engine_reset_reproducibility() -> None:
    engine = _make_engine(seed=11, strategy=BuyAndHoldStrategy(quantity=10))
    first = engine.run(30)
    engine.reset()
    second = engine.run(30)
    assert np.allclose(first.prices, second.prices)
    assert np.allclose(first.equity, second.equity)


def test_engine_identical_seeds_match() -> None:
    a = _make_engine(seed=99, strategy=BuyAndHoldStrategy(quantity=5))
    b = _make_engine(seed=99, strategy=BuyAndHoldStrategy(quantity=5))
    ra, rb = a.run(40), b.run(40)
    assert np.allclose(ra.prices, rb.prices)
    assert np.allclose(ra.equity, rb.equity)


def test_buy_and_hold_executes_once() -> None:
    engine = _make_engine(seed=1, strategy=BuyAndHoldStrategy(quantity=10))
    r1 = engine.step()
    assert r1.fill is not None
    assert r1.portfolio.quantity == pytest.approx(10.0)
    r2 = engine.step()
    assert r2.fill is None
    assert r2.portfolio.quantity == pytest.approx(10.0)


def test_hold_strategy_never_trades() -> None:
    engine = _make_engine(seed=2, strategy=HoldStrategy())
    result = engine.run(25)
    assert engine.portfolio.quantity == pytest.approx(0.0)
    assert all(sr.fill is None for sr in result.step_results)
    # Equity stays at cash when flat.
    assert np.allclose(result.equity, 100_000.0)
