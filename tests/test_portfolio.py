"""Tests for portfolio accounting."""

from __future__ import annotations

import pytest

from alpha.execution import Order, OrderSide
from alpha.portfolio import Portfolio


def _buy(qty: float, step: int = 1) -> Order:
    return Order(side=OrderSide.BUY, quantity=qty, step=step, timestamp=float(step))


def _sell(qty: float, step: int = 2) -> Order:
    return Order(side=OrderSide.SELL, quantity=qty, step=step, timestamp=float(step))


def test_portfolio_buy_updates_cash_and_position() -> None:
    pf = Portfolio(initial_capital=10_000.0)
    fill = pf.execute(_buy(10), fill_price=100.0)
    assert fill is not None
    assert pf.cash == pytest.approx(9_000.0)
    assert pf.quantity == pytest.approx(10.0)
    assert pf.average_entry_price == pytest.approx(100.0)
    assert pf.market_value(100.0) == pytest.approx(1_000.0)
    assert pf.equity(100.0) == pytest.approx(10_000.0)
    assert pf.unrealized_pnl(110.0) == pytest.approx(100.0)
    assert pf.realized_pnl == pytest.approx(0.0)


def test_portfolio_sell_realizes_pnl() -> None:
    pf = Portfolio(initial_capital=10_000.0)
    pf.execute(_buy(10), fill_price=100.0)
    pf.execute(_sell(10), fill_price=110.0)
    assert pf.quantity == pytest.approx(0.0)
    assert pf.cash == pytest.approx(10_100.0)
    assert pf.realized_pnl == pytest.approx(100.0)
    assert pf.unrealized_pnl(110.0) == pytest.approx(0.0)
    assert pf.total_pnl(110.0) == pytest.approx(100.0)
    assert pf.equity(110.0) == pytest.approx(10_100.0)
    assert len(pf.trade_history) == 2


def test_portfolio_partial_close() -> None:
    pf = Portfolio(initial_capital=10_000.0)
    pf.execute(_buy(10), fill_price=100.0)
    pf.execute(_sell(4), fill_price=120.0)
    assert pf.quantity == pytest.approx(6.0)
    assert pf.average_entry_price == pytest.approx(100.0)
    assert pf.realized_pnl == pytest.approx(80.0)
    assert pf.unrealized_pnl(120.0) == pytest.approx(120.0)


def test_portfolio_insufficient_cash() -> None:
    pf = Portfolio(initial_capital=500.0)
    with pytest.raises(ValueError, match="Insufficient cash"):
        pf.execute(_buy(10), fill_price=100.0)


def test_portfolio_reset() -> None:
    pf = Portfolio(initial_capital=5_000.0)
    pf.execute(_buy(5), fill_price=100.0)
    pf.mark_to_market(100.0, step=1, timestamp=1.0)
    pf.reset()
    assert pf.cash == pytest.approx(5_000.0)
    assert pf.quantity == pytest.approx(0.0)
    assert pf.realized_pnl == pytest.approx(0.0)
    assert pf.trade_history == []
    assert pf.equity_history == []


def test_hold_order_is_noop() -> None:
    pf = Portfolio(initial_capital=1_000.0)
    order = Order(side=OrderSide.HOLD, quantity=0.0)
    assert pf.execute(order, fill_price=50.0) is None
    assert pf.cash == pytest.approx(1_000.0)
