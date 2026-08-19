"""Tests for moving-average and mean-reversion strategies."""

from __future__ import annotations

import pytest

from alpha.execution.orders import OrderSide
from alpha.market.base import MarketState
from alpha.portfolio.portfolio import PortfolioSnapshot
from alpha.strategies import MeanReversionStrategy, MovingAverageCrossover


def _state(step: int, price: float) -> MarketState:
    return MarketState(step=step, timestamp=float(step), price=price)


def _flat_portfolio(price: float, step: int = 0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        step=step,
        timestamp=float(step),
        cash=100_000.0,
        quantity=0.0,
        average_entry_price=0.0,
        market_price=price,
        market_value=0.0,
        equity=100_000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        cumulative_return=0.0,
        total_transaction_costs=0.0,
    )


def _long_portfolio(price: float, qty: float, step: int = 0) -> PortfolioSnapshot:
    mv = qty * price
    return PortfolioSnapshot(
        step=step,
        timestamp=float(step),
        cash=100_000.0 - mv,
        quantity=qty,
        average_entry_price=price,
        market_price=price,
        market_value=mv,
        equity=100_000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        cumulative_return=0.0,
        total_transaction_costs=0.0,
    )


def test_ma_warmup_holds_until_slow_window() -> None:
    strat = MovingAverageCrossover(fast_window=2, slow_window=3, trade_quantity=10)
    for i, px in enumerate([10.0, 11.0], start=1):
        sig = strat.generate_signal(_state(i, px), _flat_portfolio(px, i))
        assert sig.side is OrderSide.HOLD
        assert sig.reason == "ma_warmup"


def test_ma_cross_up_buys_when_flat() -> None:
    """Construct a clear golden cross after warmup."""
    strat = MovingAverageCrossover(fast_window=2, slow_window=4, trade_quantity=5.0)
    # Falling then rising so fast crosses above slow.
    prices = [10.0, 9.0, 8.0, 7.0, 9.0, 12.0, 14.0]
    signals = []
    for i, px in enumerate(prices, start=1):
        signals.append(
            strat.generate_signal(_state(i, px), _flat_portfolio(px, i))
        )

    buy_steps = [i + 1 for i, s in enumerate(signals) if s.side is OrderSide.BUY]
    assert buy_steps, "expected at least one BUY on cross-up"
    buy = signals[buy_steps[0] - 1]
    assert buy.quantity == 5.0
    assert buy.reason == "ma_cross_up"


def test_ma_cross_down_sells_when_long() -> None:
    strat = MovingAverageCrossover(fast_window=2, slow_window=4, trade_quantity=5.0)
    # Decline → rally (golden cross) → selloff (death cross).
    prices = [
        20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 18.0, 22.0, 26.0,
        28.0, 30.0, 28.0, 24.0, 20.0, 16.0,
    ]
    signals = []
    position = 0.0
    for i, px in enumerate(prices, start=1):
        pf = (
            _flat_portfolio(px, i)
            if position <= 0.0
            else _long_portfolio(px, position, i)
        )
        sig = strat.generate_signal(_state(i, px), pf)
        signals.append(sig)
        if sig.side is OrderSide.BUY:
            position = sig.quantity
        elif sig.side is OrderSide.SELL:
            position = 0.0

    assert any(s.side is OrderSide.BUY for s in signals)
    assert any(s.side is OrderSide.SELL for s in signals)
    sell = next(s for s in signals if s.side is OrderSide.SELL)
    assert sell.quantity == 5.0
    assert sell.reason == "ma_cross_down"


def test_ma_no_lookahead_signal_depends_only_on_past() -> None:
    """Signal at t using prefix prices equals an independent run on that prefix."""
    full = [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0, 15.0, 13.0, 12.0]
    t = 7

    prefix_strat = MovingAverageCrossover(2, 4, trade_quantity=1.0)
    sig_prefix = None
    for i, px in enumerate(full[:t], start=1):
        sig_prefix = prefix_strat.generate_signal(_state(i, px), _flat_portfolio(px, i))

    # Same prefix, fresh strategy — must match (deterministic, past-only).
    check = MovingAverageCrossover(2, 4, trade_quantity=1.0)
    sig_check = None
    for i, px in enumerate(full[:t], start=1):
        sig_check = check.generate_signal(_state(i, px), _flat_portfolio(px, i))

    assert sig_prefix is not None and sig_check is not None
    assert sig_prefix.side is sig_check.side
    assert sig_prefix.quantity == sig_check.quantity

    # Extending with future prices must not rewrite history: replaying the
    # prefix alone always reproduces the same step-t signal.
    full_strat = MovingAverageCrossover(2, 4, trade_quantity=1.0)
    sig_at_t = None
    for i, px in enumerate(full, start=1):
        sig = full_strat.generate_signal(_state(i, px), _flat_portfolio(px, i))
        if i == t:
            sig_at_t = sig
    assert sig_at_t is not None
    assert sig_at_t.side is sig_prefix.side
    assert sig_at_t.quantity == sig_prefix.quantity


def test_mean_reversion_buys_on_low_zscore() -> None:
    strat = MeanReversionStrategy(
        lookback=5, trade_quantity=3.0, entry_z=1.5, exit_z=0.25
    )
    # Stable then sharp drop below mean.
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 90.0]
    signals = [
        strat.generate_signal(_state(i, px), _flat_portfolio(px, i))
        for i, px in enumerate(prices, start=1)
    ]
    assert signals[-1].side is OrderSide.BUY
    assert signals[-1].quantity == 3.0


def test_mean_reversion_exits_near_mean() -> None:
    strat = MeanReversionStrategy(
        lookback=5, trade_quantity=3.0, entry_z=1.5, exit_z=0.25
    )
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 99.0]
    position = 0.0
    signals = []
    for i, px in enumerate(prices, start=1):
        pf = (
            _flat_portfolio(px, i)
            if position <= 0.0
            else _long_portfolio(px, position, i)
        )
        sig = strat.generate_signal(_state(i, px), pf)
        signals.append(sig)
        if sig.side is OrderSide.BUY:
            position = sig.quantity
        elif sig.side is OrderSide.SELL:
            position = 0.0
    assert signals[-2].side is OrderSide.BUY
    assert signals[-1].side is OrderSide.SELL


def _collect_signals(factory, prices: list[float]):
    strat = factory()
    return [
        strat.generate_signal(_state(i, px), _flat_portfolio(px, i))
        for i, px in enumerate(prices, start=1)
    ]


def test_ma_future_prices_cannot_change_past_signals() -> None:
    prefix = [10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0]
    factory = lambda: MovingAverageCrossover(2, 4, trade_quantity=1.0)
    up = _collect_signals(factory, prefix + [50.0, 80.0])
    down = _collect_signals(factory, prefix + [1.0, 0.5])
    for a, b in zip(up[: len(prefix)], down[: len(prefix)], strict=True):
        assert a.side is b.side
        assert a.quantity == b.quantity


def test_mean_reversion_future_prices_cannot_change_past_signals() -> None:
    prefix = [100.0, 101.0, 99.0, 100.0, 98.0, 90.0, 91.0]
    factory = lambda: MeanReversionStrategy(5, 2.0, entry_z=1.5, exit_z=0.25)
    up = _collect_signals(factory, prefix + [400.0, 500.0])
    down = _collect_signals(factory, prefix + [10.0, 5.0])
    for a, b in zip(up[: len(prefix)], down[: len(prefix)], strict=True):
        assert a.side is b.side
        assert a.quantity == b.quantity


def test_mean_reversion_holds_during_warmup() -> None:
    strat = MeanReversionStrategy(lookback=5, trade_quantity=1.0)
    for i, px in enumerate([100.0, 101.0, 99.0, 102.0], start=1):
        sig = strat.generate_signal(_state(i, px), _flat_portfolio(px, i))
        assert sig.side is OrderSide.HOLD
        assert sig.reason == "zscore_warmup"


def test_invalid_strategy_parameters() -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossover(10, 10, 1.0)
    with pytest.raises(ValueError):
        MeanReversionStrategy(lookback=5, trade_quantity=1.0, entry_z=0.5, exit_z=1.0)

