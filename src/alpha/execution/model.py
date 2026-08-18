"""Execution models: map orders + market price → fills with frictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from alpha.execution.orders import Fill, Order, OrderSide


def _bps_to_fraction(bps: float) -> float:
    return bps * 1.0e-4


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Configurable commission, slippage, and optional half-spread.

    All cost parameters are in basis points of notional (or of mid price
    for per-unit adjustments).

    Parameters
    ----------
    commission_bps:
        Commission charged as a fraction of executed notional.
    slippage_bps:
        Adverse price movement vs the observed market price.
        BUY pays more; SELL receives less.
    spread_bps:
        Optional half-spread in bps applied in the same adverse direction
        (simple bid/ask proxy — not a full LOB).
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("commission_bps", self.commission_bps),
            ("slippage_bps", self.slippage_bps),
            ("spread_bps", self.spread_bps),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

    @classmethod
    def frictionless(cls) -> ExecutionConfig:
        """Zero commission, slippage, and spread."""
        return cls(commission_bps=0.0, slippage_bps=0.0, spread_bps=0.0)

    @classmethod
    def realistic(
        cls,
        *,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
    ) -> ExecutionConfig:
        """Typical retail/light-institutional friction defaults."""
        return cls(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
        )

    @property
    def is_frictionless(self) -> bool:
        return (
            self.commission_bps == 0.0
            and self.slippage_bps == 0.0
            and self.spread_bps == 0.0
        )


class ExecutionModel(Protocol):
    """Protocol for order → fill conversion."""

    def execute(self, order: Order, market_price: float) -> Optional[Fill]:
        """Return a fill, or ``None`` for HOLD / zero-quantity orders."""


@dataclass(frozen=True, slots=True)
class SimpleExecutionModel:
    """Market-order execution with bps commission, slippage, and spread.

    Does not simulate a limit-order book. Slippage and spread adjust the
    execution price adversely relative to the observed market price;
    commission is a separate cash fee on notional.
    """

    config: ExecutionConfig = ExecutionConfig()

    @classmethod
    def frictionless(cls) -> SimpleExecutionModel:
        return cls(config=ExecutionConfig.frictionless())

    @classmethod
    def realistic(
        cls,
        *,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
    ) -> SimpleExecutionModel:
        return cls(
            config=ExecutionConfig.realistic(
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
            )
        )

    def execution_price(self, side: OrderSide, market_price: float) -> float:
        """Adverse price adjustment for BUY / SELL."""
        if market_price <= 0.0:
            raise ValueError(f"market_price must be positive, got {market_price}")
        adverse = _bps_to_fraction(self.config.slippage_bps + self.config.spread_bps)
        if side is OrderSide.BUY:
            return market_price * (1.0 + adverse)
        if side is OrderSide.SELL:
            return market_price * (1.0 - adverse)
        return market_price

    def execute(self, order: Order, market_price: float) -> Optional[Fill]:
        if order.is_hold:
            return None
        if market_price <= 0.0:
            raise ValueError(f"market_price must be positive, got {market_price}")

        exec_price = self.execution_price(order.side, market_price)
        notional = order.quantity * exec_price
        commission = notional * _bps_to_fraction(self.config.commission_bps)
        slippage_cost = abs(exec_price - market_price) * order.quantity
        total_cost = commission + slippage_cost

        return Fill(
            order=order,
            market_price=float(market_price),
            execution_price=float(exec_price),
            fill_quantity=float(order.quantity),
            commission=float(commission),
            slippage=float(slippage_cost),
            total_transaction_cost=float(total_cost),
            realized_pnl=0.0,
            timestamp=order.timestamp,
            step=order.step,
        )
