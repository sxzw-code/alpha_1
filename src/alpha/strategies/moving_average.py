"""Moving-average crossover strategy."""

from __future__ import annotations

from collections import deque
from typing import Optional

from alpha.execution.orders import OrderSide, Signal
from alpha.market.base import MarketState
from alpha.portfolio.portfolio import PortfolioSnapshot
from alpha.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """Long-only fast/slow moving-average crossover.

    BUY when the fast MA crosses **above** the slow MA while flat.
    SELL (close) when the fast MA crosses **below** the slow MA while long.

    Uses only prices observed up to the current bar (no look-ahead).
    A crossover is evaluated only after both windows are warm and a prior
    MA pair exists, so the first fully-formed bar does not spuriously trade.
    """

    def __init__(
        self,
        fast_window: int,
        slow_window: int,
        trade_quantity: float,
    ) -> None:
        if fast_window < 1:
            raise ValueError(f"fast_window must be >= 1, got {fast_window}")
        if slow_window <= fast_window:
            raise ValueError(
                f"slow_window must be > fast_window, got {slow_window} <= {fast_window}"
            )
        if trade_quantity <= 0.0:
            raise ValueError(f"trade_quantity must be positive, got {trade_quantity}")
        self._fast_window = int(fast_window)
        self._slow_window = int(slow_window)
        self._trade_quantity = float(trade_quantity)
        self._prices: deque[float] = deque(maxlen=self._slow_window)
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None

    @property
    def fast_window(self) -> int:
        return self._fast_window

    @property
    def slow_window(self) -> int:
        return self._slow_window

    def reset(self) -> None:
        self._prices.clear()
        self._prev_fast = None
        self._prev_slow = None

    def _sma(self, window: int) -> float:
        vals = list(self._prices)[-window:]
        return sum(vals) / float(window)

    def generate_signal(
        self,
        market_state: MarketState,
        portfolio_state: Optional[PortfolioSnapshot] = None,
    ) -> Signal:
        self._prices.append(float(market_state.price))
        if len(self._prices) < self._slow_window:
            return Signal.hold(
                asset_id=market_state.asset_id, reason="ma_warmup"
            )

        fast = self._sma(self._fast_window)
        slow = self._sma(self._slow_window)
        position = 0.0 if portfolio_state is None else portfolio_state.quantity

        signal = Signal.hold(asset_id=market_state.asset_id, reason="no_cross")
        if self._prev_fast is not None and self._prev_slow is not None:
            crossed_up = self._prev_fast <= self._prev_slow and fast > slow
            crossed_down = self._prev_fast >= self._prev_slow and fast < slow
            if crossed_up and position <= 0.0:
                signal = Signal(
                    side=OrderSide.BUY,
                    quantity=self._trade_quantity,
                    asset_id=market_state.asset_id,
                    reason="ma_cross_up",
                )
            elif crossed_down and position > 0.0:
                signal = Signal(
                    side=OrderSide.SELL,
                    quantity=float(position),
                    asset_id=market_state.asset_id,
                    reason="ma_cross_down",
                )

        self._prev_fast = fast
        self._prev_slow = slow
        return signal
