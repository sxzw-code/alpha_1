"""Stateful single-asset portfolio with P&L and history tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpha.execution.orders import Fill, Order, OrderSide
from alpha.portfolio.position import Position


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Immutable record of an executed trade."""

    step: int
    timestamp: float
    side: OrderSide
    quantity: float
    market_price: float
    execution_price: float
    commission: float
    slippage: float
    total_transaction_cost: float
    asset_id: str
    realized_pnl: float
    cash_after: float


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Point-in-time portfolio performance view for UI / analytics."""

    step: int
    timestamp: float
    cash: float
    quantity: float
    average_entry_price: float
    market_price: float
    market_value: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    cumulative_return: float
    total_transaction_costs: float


@dataclass(slots=True)
class EquityPoint:
    step: int
    timestamp: float
    equity: float
    price: float
    cash: float
    quantity: float


class Portfolio:
    """Cash + single-asset inventory with full accounting.

    Interfaces are intentionally asset-id aware so a future multi-asset
    book can hold a mapping of :class:`Position` without rewriting callers.

    Transaction costs (commission) reduce cash and realized P&L once.
    Slippage is embedded in ``execution_price`` and therefore flows through
    average entry / realized trade P&L without a second deduction.
    """

    def __init__(
        self,
        initial_capital: float,
        *,
        asset_id: str = "ASSET",
        allow_short: bool = False,
    ) -> None:
        if initial_capital <= 0.0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        self._initial_capital = float(initial_capital)
        self._cash = float(initial_capital)
        self._asset_id = asset_id
        self._allow_short = allow_short
        self._position = Position(asset_id=asset_id)
        self._realized_pnl = 0.0
        self._total_transaction_costs = 0.0
        self._trade_history: list[TradeRecord] = []
        self._equity_history: list[EquityPoint] = []
        self._last_price: Optional[float] = None

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def quantity(self) -> float:
        return self._position.quantity

    @property
    def position(self) -> Position:
        return self._position

    @property
    def average_entry_price(self) -> float:
        return self._position.average_entry_price

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def total_transaction_costs(self) -> float:
        return self._total_transaction_costs

    @property
    def allow_short(self) -> bool:
        return self._allow_short

    @property
    def trade_history(self) -> list[TradeRecord]:
        return list(self._trade_history)

    @property
    def equity_history(self) -> list[EquityPoint]:
        return list(self._equity_history)

    def market_value(self, price: float) -> float:
        return self._position.market_value(price)

    def equity(self, price: float) -> float:
        return self._cash + self.market_value(price)

    def unrealized_pnl(self, price: float) -> float:
        return self._position.unrealized_pnl(price)

    def total_pnl(self, price: float) -> float:
        return self._realized_pnl + self.unrealized_pnl(price)

    def cumulative_return(self, price: float) -> float:
        return self.equity(price) / self._initial_capital - 1.0

    def mark_to_market(
        self, price: float, *, step: int = 0, timestamp: float = 0.0
    ) -> PortfolioSnapshot:
        """Update last price, append equity history, and return a snapshot."""
        if price <= 0.0:
            raise ValueError(f"price must be positive, got {price}")
        self._last_price = float(price)
        snap = self.snapshot(price, step=step, timestamp=timestamp)
        self._equity_history.append(
            EquityPoint(
                step=step,
                timestamp=timestamp,
                equity=snap.equity,
                price=price,
                cash=snap.cash,
                quantity=snap.quantity,
            )
        )
        return snap

    def snapshot(
        self, price: float, *, step: int = 0, timestamp: float = 0.0
    ) -> PortfolioSnapshot:
        mv = self.market_value(price)
        unrealized = self.unrealized_pnl(price)
        equity = self._cash + mv
        return PortfolioSnapshot(
            step=step,
            timestamp=timestamp,
            cash=self._cash,
            quantity=self._position.quantity,
            average_entry_price=self._position.average_entry_price,
            market_price=price,
            market_value=mv,
            equity=equity,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized,
            total_pnl=self._realized_pnl + unrealized,
            cumulative_return=equity / self._initial_capital - 1.0,
            total_transaction_costs=self._total_transaction_costs,
        )

    def apply_fill(self, fill: Fill) -> Fill:
        """Apply an execution-layer fill to cash and inventory.

        Returns a new :class:`Fill` with ``realized_pnl`` populated.
        Commission is deducted from cash once and charged to realized P&L.
        Slippage is already in ``execution_price`` — not deducted again.
        """
        order = fill.order
        if order.is_hold or fill.fill_quantity == 0.0:
            raise ValueError("Cannot apply a HOLD / zero-quantity fill")
        if fill.execution_price <= 0.0:
            raise ValueError(
                f"execution_price must be positive, got {fill.execution_price}"
            )
        if order.asset_id != self._asset_id:
            raise ValueError(
                f"Order asset_id {order.asset_id!r} does not match portfolio "
                f"asset {self._asset_id!r}"
            )
        if fill.commission < 0.0:
            raise ValueError(f"commission must be >= 0, got {fill.commission}")

        signed_qty = order.signed_quantity()
        if not self._allow_short and self._position.quantity + signed_qty < -1e-12:
            raise ValueError("Short selling is disabled for this portfolio")

        notional = abs(signed_qty) * fill.execution_price
        if order.side is OrderSide.BUY:
            required = notional + fill.commission
            if required > self._cash + 1e-12:
                raise ValueError(
                    f"Insufficient cash: need {required:.6f}, have {self._cash:.6f}"
                )

        trade_pnl = self._position.apply_fill(signed_qty, fill.execution_price)
        # Commission reduces cash and realized P&L once (no double-count with slippage).
        net_realized = trade_pnl - fill.commission
        self._realized_pnl += net_realized
        self._cash -= signed_qty * fill.execution_price
        self._cash -= fill.commission
        self._total_transaction_costs += fill.total_transaction_cost
        self._last_price = fill.market_price

        applied = Fill(
            order=order,
            market_price=fill.market_price,
            execution_price=fill.execution_price,
            fill_quantity=fill.fill_quantity,
            commission=fill.commission,
            slippage=fill.slippage,
            total_transaction_cost=fill.total_transaction_cost,
            realized_pnl=net_realized,
            timestamp=fill.timestamp,
            step=fill.step,
        )
        self._trade_history.append(
            TradeRecord(
                step=order.step,
                timestamp=order.timestamp,
                side=order.side,
                quantity=order.quantity,
                market_price=fill.market_price,
                execution_price=fill.execution_price,
                commission=fill.commission,
                slippage=fill.slippage,
                total_transaction_cost=fill.total_transaction_cost,
                asset_id=order.asset_id,
                realized_pnl=net_realized,
                cash_after=self._cash,
            )
        )
        return applied

    def execute(
        self,
        order: Order,
        fill_price: float,
        *,
        market_price: Optional[float] = None,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> Optional[Fill]:
        """Convenience path: build a fill at ``fill_price`` and apply it.

        Prefer routing orders through :class:`~alpha.execution.model.SimpleExecutionModel`
        in simulation. This helper remains for unit tests and frictionless fills.
        """
        if order.is_hold or order.quantity == 0.0:
            return None
        mid = float(market_price) if market_price is not None else float(fill_price)
        total_cost = float(commission) + float(slippage)
        pending = Fill(
            order=order,
            market_price=mid,
            execution_price=float(fill_price),
            fill_quantity=float(order.quantity),
            commission=float(commission),
            slippage=float(slippage),
            total_transaction_cost=total_cost,
            timestamp=order.timestamp,
            step=order.step,
        )
        return self.apply_fill(pending)

    def reset(self) -> None:
        """Restore initial capital and clear history."""
        self._cash = self._initial_capital
        self._position = Position(asset_id=self._asset_id)
        self._realized_pnl = 0.0
        self._total_transaction_costs = 0.0
        self._trade_history.clear()
        self._equity_history.clear()
        self._last_price = None
