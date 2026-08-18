"""Order, signal, and fill datatypes.

Complex limit-order matching is intentionally out of scope.
Transaction costs and slippage are applied by the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    MARKET = "MARKET"


@dataclass(frozen=True, slots=True)
class Signal:
    """Strategy intent: side + unsigned target quantity.

    Strategies emit signals only; they never mutate the portfolio.
    """

    side: OrderSide
    quantity: float = 0.0
    asset_id: str = "ASSET"
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantity < 0.0:
            raise ValueError(f"Signal quantity must be >= 0, got {self.quantity}")
        if self.side is OrderSide.HOLD and self.quantity != 0.0:
            raise ValueError("HOLD signals must have quantity 0")
        if self.side is not OrderSide.HOLD and self.quantity == 0.0:
            raise ValueError(f"{self.side.value} signals require quantity > 0")

    @classmethod
    def hold(cls, *, asset_id: str = "ASSET", reason: Optional[str] = None) -> Signal:
        return cls(side=OrderSide.HOLD, quantity=0.0, asset_id=asset_id, reason=reason)

    def to_order(
        self,
        *,
        step: int,
        timestamp: float,
        requested_price: Optional[float] = None,
    ) -> Order:
        return Order(
            side=self.side,
            quantity=self.quantity,
            order_type=OrderType.MARKET,
            asset_id=self.asset_id,
            step=step,
            timestamp=timestamp,
            requested_price=requested_price,
            reason=self.reason,
        )


@dataclass(frozen=True, slots=True)
class Order:
    """Executable order derived from a strategy signal."""

    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    asset_id: str = "ASSET"
    step: int = 0
    timestamp: float = 0.0
    requested_price: Optional[float] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantity < 0.0:
            raise ValueError(f"Order quantity must be >= 0, got {self.quantity}")
        if self.side is OrderSide.HOLD and self.quantity != 0.0:
            raise ValueError("HOLD orders must have quantity 0")

    @property
    def is_hold(self) -> bool:
        return self.side is OrderSide.HOLD or self.quantity == 0.0

    def signed_quantity(self) -> float:
        if self.side is OrderSide.BUY:
            return self.quantity
        if self.side is OrderSide.SELL:
            return -self.quantity
        return 0.0


@dataclass(frozen=True, slots=True)
class Fill:
    """Executed trade produced by the execution layer, applied by the portfolio.

    ``execution_price`` is the price used for inventory / cash notional.
    ``market_price`` is the observed mid / model price before frictions.
    ``realized_pnl`` is filled in by the portfolio when the fill is applied
    (trade P&L after commissions).
    """

    order: Order
    market_price: float
    execution_price: float
    fill_quantity: float
    commission: float = 0.0
    slippage: float = 0.0
    total_transaction_cost: float = 0.0
    realized_pnl: float = 0.0
    timestamp: float = 0.0
    step: int = 0

    @property
    def fill_price(self) -> float:
        """Alias for ``execution_price`` (backward-compatible name)."""
        return self.execution_price

    @property
    def side(self) -> OrderSide:
        return self.order.side
