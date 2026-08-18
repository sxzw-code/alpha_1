"""Z-score mean-reversion strategy."""

from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Optional

from alpha.execution.orders import OrderSide, Signal
from alpha.market.base import MarketState
from alpha.portfolio.portfolio import PortfolioSnapshot
from alpha.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """Long-only rolling z-score mean reversion.

    BUY when ``z <= -entry_z`` while flat (price sufficiently below its mean).
    SELL / exit when ``z >= -exit_z`` while long (price has reverted toward mean).

    Uses only prices observed up to the current bar (no look-ahead).
    """

    def __init__(
        self,
        lookback: int,
        trade_quantity: float,
        *,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if trade_quantity <= 0.0:
            raise ValueError(f"trade_quantity must be positive, got {trade_quantity}")
        if entry_z <= 0.0:
            raise ValueError(f"entry_z must be positive, got {entry_z}")
        if exit_z < 0.0:
            raise ValueError(f"exit_z must be >= 0, got {exit_z}")
        if exit_z >= entry_z:
            raise ValueError(
                f"exit_z must be < entry_z for a valid band, got {exit_z} >= {entry_z}"
            )
        self._lookback = int(lookback)
        self._trade_quantity = float(trade_quantity)
        self._entry_z = float(entry_z)
        self._exit_z = float(exit_z)
        self._prices: deque[float] = deque(maxlen=self._lookback)

    @property
    def lookback(self) -> int:
        return self._lookback

    @property
    def entry_z(self) -> float:
        return self._entry_z

    @property
    def exit_z(self) -> float:
        return self._exit_z

    def reset(self) -> None:
        self._prices.clear()

    def _zscore(self, price: float) -> Optional[float]:
        if len(self._prices) < self._lookback:
            return None
        vals = list(self._prices)
        mean = sum(vals) / float(self._lookback)
        var = sum((x - mean) ** 2 for x in vals) / float(self._lookback)
        std = sqrt(var)
        if std <= 1e-12:
            return 0.0
        return (price - mean) / std

    def generate_signal(
        self,
        market_state: MarketState,
        portfolio_state: Optional[PortfolioSnapshot] = None,
    ) -> Signal:
        price = float(market_state.price)
        self._prices.append(price)
        z = self._zscore(price)
        if z is None:
            return Signal.hold(
                asset_id=market_state.asset_id, reason="zscore_warmup"
            )

        position = 0.0 if portfolio_state is None else portfolio_state.quantity
        if position <= 0.0 and z <= -self._entry_z:
            return Signal(
                side=OrderSide.BUY,
                quantity=self._trade_quantity,
                asset_id=market_state.asset_id,
                reason=f"zscore_entry z={z:.3f}",
            )
        if position > 0.0 and z >= -self._exit_z:
            return Signal(
                side=OrderSide.SELL,
                quantity=float(position),
                asset_id=market_state.asset_id,
                reason=f"zscore_exit z={z:.3f}",
            )
        return Signal.hold(asset_id=market_state.asset_id, reason=f"zscore_hold z={z:.3f}")
