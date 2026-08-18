"""Abstract strategy interface.

Strategies observe market state (and optionally portfolio context) and emit
signals. They must not mutate the portfolio; the simulation engine converts
signals into orders and routes fills through portfolio accounting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from alpha.execution.orders import OrderSide, Signal
from alpha.market.base import MarketState
from alpha.portfolio.portfolio import PortfolioSnapshot


class Strategy(ABC):
    """Generate trading intent from market observations."""

    @abstractmethod
    def generate_signal(
        self,
        market_state: MarketState,
        portfolio_state: Optional[PortfolioSnapshot] = None,
    ) -> Signal:
        """Return a signal for the current observation.

        Parameters
        ----------
        market_state:
            Latest price / timestamp from the market model.
        portfolio_state:
            Optional mark-to-market snapshot. Strategies may read it but
            must not mutate the underlying portfolio.
        """

    def reset(self) -> None:
        """Clear any internal strategy state. Override when stateful."""


class HoldStrategy(Strategy):
    """Always emit HOLD — useful as a baseline / no-trade control."""

    def generate_signal(
        self,
        market_state: MarketState,
        portfolio_state: Optional[PortfolioSnapshot] = None,
    ) -> Signal:
        return Signal.hold(asset_id=market_state.asset_id, reason="hold_strategy")


class BuyAndHoldStrategy(Strategy):
    """Passive long benchmark: buy a fixed quantity once, then hold.

    Useful for comparing active strategies against buy-and-hold exposure
    on the same simulated path.
    """

    def __init__(self, quantity: float = 1.0) -> None:
        if quantity <= 0.0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        self._quantity = float(quantity)
        self._bought = False

    def reset(self) -> None:
        self._bought = False

    def generate_signal(
        self,
        market_state: MarketState,
        portfolio_state: Optional[PortfolioSnapshot] = None,
    ) -> Signal:
        if self._bought:
            return Signal.hold(asset_id=market_state.asset_id, reason="already_long")
        if portfolio_state is not None and portfolio_state.quantity != 0.0:
            self._bought = True
            return Signal.hold(asset_id=market_state.asset_id, reason="already_long")
        self._bought = True
        return Signal(
            side=OrderSide.BUY,
            quantity=self._quantity,
            asset_id=market_state.asset_id,
            reason="buy_and_hold_entry",
        )
