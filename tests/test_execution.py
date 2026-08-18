"""Tests for execution costs and slippage."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.execution import (
    ExecutionConfig,
    Order,
    OrderSide,
    SimpleExecutionModel,
)
from alpha.market import GeometricBrownianMotion
from alpha.portfolio import Portfolio
from alpha.simulation import SimulationEngine, compare_friction
from alpha.strategies import BuyAndHoldStrategy, MovingAverageCrossover


def _order(side: OrderSide, qty: float = 10.0) -> Order:
    return Order(side=side, quantity=qty, step=1, timestamp=1.0)


def test_buy_slippage_increases_execution_price() -> None:
    model = SimpleExecutionModel(
        config=ExecutionConfig(commission_bps=0.0, slippage_bps=10.0, spread_bps=0.0)
    )
    fill = model.execute(_order(OrderSide.BUY), market_price=100.0)
    assert fill is not None
    assert fill.execution_price > fill.market_price
    assert fill.execution_price == pytest.approx(100.1)


def test_sell_slippage_decreases_execution_price() -> None:
    model = SimpleExecutionModel(
        config=ExecutionConfig(commission_bps=0.0, slippage_bps=10.0, spread_bps=0.0)
    )
    fill = model.execute(_order(OrderSide.SELL), market_price=100.0)
    assert fill is not None
    assert fill.execution_price < fill.market_price
    assert fill.execution_price == pytest.approx(99.9)


def test_commission_calculated_on_notional() -> None:
    model = SimpleExecutionModel(
        config=ExecutionConfig(commission_bps=5.0, slippage_bps=0.0, spread_bps=0.0)
    )
    fill = model.execute(_order(OrderSide.BUY, qty=10.0), market_price=100.0)
    assert fill is not None
    # 5 bps of 1000 notional = 0.5
    assert fill.commission == pytest.approx(0.5)
    assert fill.total_transaction_cost == pytest.approx(0.5)


def test_spread_adds_to_adverse_price() -> None:
    model = SimpleExecutionModel(
        config=ExecutionConfig(commission_bps=0.0, slippage_bps=2.0, spread_bps=3.0)
    )
    fill = model.execute(_order(OrderSide.BUY), market_price=100.0)
    assert fill is not None
    assert fill.execution_price == pytest.approx(100.05)  # 5 bps total adverse


def test_portfolio_cash_includes_commission() -> None:
    pf = Portfolio(initial_capital=10_000.0)
    model = SimpleExecutionModel(
        config=ExecutionConfig(commission_bps=10.0, slippage_bps=0.0)
    )
    pending = model.execute(_order(OrderSide.BUY, qty=10.0), market_price=100.0)
    assert pending is not None
    fill = pf.apply_fill(pending)
    # notional 1000 + commission 1.0
    assert pf.cash == pytest.approx(8_999.0)
    assert fill.commission == pytest.approx(1.0)
    assert pf.total_transaction_costs == pytest.approx(1.0)
    assert pf.realized_pnl == pytest.approx(-1.0)
    # Marked at mid: equity reflects commission drag.
    assert pf.equity(100.0) == pytest.approx(9_999.0)


def test_transaction_costs_reduce_final_equity() -> None:
    seed = 42
    n_steps = 100
    qty = 10.0

    frictionless = SimulationEngine(
        model=GeometricBrownianMotion(s0=100.0, mu=0.08, sigma=0.2, dt=1 / 252),
        portfolio=Portfolio(initial_capital=100_000.0),
        strategy=BuyAndHoldStrategy(quantity=qty),
        execution=SimpleExecutionModel.frictionless(),
        seed=seed,
    ).run(n_steps)

    realistic = SimulationEngine(
        model=GeometricBrownianMotion(s0=100.0, mu=0.08, sigma=0.2, dt=1 / 252),
        portfolio=Portfolio(initial_capital=100_000.0),
        strategy=BuyAndHoldStrategy(quantity=qty),
        execution=SimpleExecutionModel.realistic(
            commission_bps=10.0, slippage_bps=5.0, spread_bps=2.0
        ),
        seed=seed,
    ).run(n_steps)

    assert np.allclose(frictionless.prices, realistic.prices)
    assert realistic.total_transaction_costs > 0.0
    assert frictionless.total_transaction_costs == pytest.approx(0.0)
    assert realistic.final_equity < frictionless.final_equity


def test_compare_friction_identical_price_paths() -> None:
    comparison = compare_friction(
        model_factory=lambda: GeometricBrownianMotion(
            s0=100.0, mu=0.05, sigma=0.25, dt=1 / 252
        ),
        strategy_factory=lambda: MovingAverageCrossover(5, 20, trade_quantity=10.0),
        portfolio_factory=lambda: Portfolio(initial_capital=100_000.0),
        n_steps=150,
        seed=7,
        realistic_execution=SimpleExecutionModel.realistic(),
    )
    assert np.allclose(
        comparison.frictionless.prices, comparison.realistic.prices
    )
    rows = comparison.summary_rows()
    assert rows["total_transaction_costs"][0] == pytest.approx(0.0)
    assert rows["total_transaction_costs"][1] >= 0.0
    if comparison.realistic.n_trades > 0:
        assert (
            comparison.realistic.final_equity
            <= comparison.frictionless.final_equity + 1e-9
        )


def test_insufficient_cash_with_costs() -> None:
    pf = Portfolio(initial_capital=1_000.0)
    model = SimpleExecutionModel(
        config=ExecutionConfig(commission_bps=50.0, slippage_bps=0.0)
    )
    # 10 * 100 = 1000 notional, but commission pushes over cash.
    pending = model.execute(_order(OrderSide.BUY, qty=10.0), market_price=100.0)
    assert pending is not None
    with pytest.raises(ValueError, match="Insufficient cash"):
        pf.apply_fill(pending)
