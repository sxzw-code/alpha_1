"""Execution models: map orders + market price → fills with frictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from alpha.execution.context import ExecutionContext
from alpha.execution.market_impact import (
    MarketImpactModel,
    SquareRootMarketImpactModel,
)
from alpha.execution.orders import Fill, Order, OrderSide


def _bps_to_fraction(bps: float) -> float:
    return bps * 1.0e-4


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Configurable commission, slippage, spread, and optional market impact.

    All bps parameters are basis points of notional (or of mid price for
    per-unit adjustments).

    Market impact (optional)
    ------------------------
    When ``market_impact_enabled`` is True, a square-root impact term::

        I = η × σ_daily × sqrt(Q / V)

    is applied after spread and fixed slippage. ``annual_volatility`` may be
    set manually; otherwise the engine passes model volatility via
    :class:`~alpha.execution.context.ExecutionContext`. ``average_daily_volume``
    is a shares-per-day liquidity proxy (manual for synthetic markets).
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    market_impact_enabled: bool = False
    impact_coefficient: float = 0.10
    average_daily_volume: float = 1_000_000.0
    annual_volatility: Optional[float] = None
    participation_warning_threshold: float = 0.10
    periods_per_year: float = 252.0

    def __post_init__(self) -> None:
        for name, value in (
            ("commission_bps", self.commission_bps),
            ("slippage_bps", self.slippage_bps),
            ("spread_bps", self.spread_bps),
            ("impact_coefficient", self.impact_coefficient),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.average_daily_volume <= 0.0:
            raise ValueError(
                f"average_daily_volume must be positive, got {self.average_daily_volume}"
            )
        if self.participation_warning_threshold < 0.0:
            raise ValueError(
                "participation_warning_threshold must be >= 0, got "
                f"{self.participation_warning_threshold}"
            )
        if self.periods_per_year <= 0.0:
            raise ValueError(
                f"periods_per_year must be positive, got {self.periods_per_year}"
            )
        if self.annual_volatility is not None and self.annual_volatility < 0.0:
            raise ValueError(
                f"annual_volatility must be >= 0, got {self.annual_volatility}"
            )

    @classmethod
    def frictionless(cls) -> ExecutionConfig:
        """Zero commission, slippage, spread, and market impact."""
        return cls(
            commission_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            market_impact_enabled=False,
        )

    @classmethod
    def realistic(
        cls,
        *,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
    ) -> ExecutionConfig:
        """Typical retail/light-institutional friction defaults (no impact)."""
        return cls(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            market_impact_enabled=False,
        )

    @classmethod
    def liquidity_aware(
        cls,
        *,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
        impact_coefficient: float = 0.10,
        average_daily_volume: float = 1_000_000.0,
        annual_volatility: Optional[float] = None,
    ) -> ExecutionConfig:
        """Realistic frictions plus square-root market impact."""
        return cls(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            market_impact_enabled=True,
            impact_coefficient=impact_coefficient,
            average_daily_volume=average_daily_volume,
            annual_volatility=annual_volatility,
        )

    @property
    def is_frictionless(self) -> bool:
        return (
            self.commission_bps == 0.0
            and self.slippage_bps == 0.0
            and self.spread_bps == 0.0
            and not self.market_impact_enabled
        )


class ExecutionModel(Protocol):
    """Protocol for order → fill conversion."""

    def execute(
        self,
        order: Order,
        market_price: float,
        *,
        context: Optional[ExecutionContext] = None,
    ) -> Optional[Fill]:
        """Return a fill, or ``None`` for HOLD / zero-quantity orders."""


@dataclass(frozen=True, slots=True)
class SimpleExecutionModel:
    """Market-order execution with bps frictions and optional market impact.

    Execution price construction (documented order)::

        mid (= market_price)
          → half-spread adjustment
          → fixed slippage (additive bps on mid, backward compatible)
          → square-root market impact (optional, multiplicative)
          → execution_price

    Commission is a separate cash fee on executed notional.
    """

    config: ExecutionConfig = ExecutionConfig()
    impact_model: Optional[MarketImpactModel] = None

    def __post_init__(self) -> None:
        if self.impact_model is None and self.config.market_impact_enabled:
            object.__setattr__(
                self,
                "impact_model",
                SquareRootMarketImpactModel(
                    coefficient=self.config.impact_coefficient,
                    participation_warning_threshold=(
                        self.config.participation_warning_threshold
                    ),
                ),
            )

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

    @classmethod
    def liquidity_aware(
        cls,
        *,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
        impact_coefficient: float = 0.10,
        average_daily_volume: float = 1_000_000.0,
        annual_volatility: Optional[float] = None,
    ) -> SimpleExecutionModel:
        return cls(
            config=ExecutionConfig.liquidity_aware(
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                impact_coefficient=impact_coefficient,
                average_daily_volume=average_daily_volume,
                annual_volatility=annual_volatility,
            )
        )

    def execution_price(
        self,
        side: OrderSide,
        market_price: float,
        *,
        quantity: float = 0.0,
        context: Optional[ExecutionContext] = None,
    ) -> float:
        """Adverse price adjustment for BUY / SELL (includes impact when enabled)."""
        fill = self.execute(
            Order(side=side, quantity=quantity, step=0, timestamp=0.0),
            market_price,
            context=context,
        )
        if fill is None:
            return market_price
        return fill.execution_price

    def _resolve_volatility(
        self, context: Optional[ExecutionContext]
    ) -> float:
        if context is not None and context.annual_volatility is not None:
            return float(context.annual_volatility)
        if self.config.annual_volatility is not None:
            return float(self.config.annual_volatility)
        return 0.0

    def _resolve_adv(self, context: Optional[ExecutionContext]) -> float:
        if context is not None and context.average_daily_volume is not None:
            return float(context.average_daily_volume)
        return float(self.config.average_daily_volume)

    def _resolve_periods(self, context: Optional[ExecutionContext]) -> float:
        if context is not None:
            return float(context.periods_per_year)
        return float(self.config.periods_per_year)

    def execute(
        self,
        order: Order,
        market_price: float,
        *,
        context: Optional[ExecutionContext] = None,
    ) -> Optional[Fill]:
        if order.is_hold:
            return None
        if market_price <= 0.0:
            raise ValueError(f"market_price must be positive, got {market_price}")

        qty = float(order.quantity)
        spread_frac = _bps_to_fraction(self.config.spread_bps)
        slip_frac = _bps_to_fraction(self.config.slippage_bps)
        combined_frac = _bps_to_fraction(
            self.config.spread_bps + self.config.slippage_bps
        )

        if order.side is OrderSide.BUY:
            after_spread = market_price * (1.0 + spread_frac)
            after_slippage = market_price * (1.0 + combined_frac)
            exec_price = after_slippage
        elif order.side is OrderSide.SELL:
            after_spread = market_price * (1.0 - spread_frac)
            after_slippage = market_price * (1.0 - combined_frac)
            exec_price = after_slippage
        else:
            after_spread = market_price
            after_slippage = market_price
            exec_price = market_price

        spread_cost = abs(after_spread - market_price) * qty
        fixed_slippage_cost = abs(after_slippage - after_spread) * qty

        impact_cost = 0.0
        impact_bps = 0.0
        participation = 0.0
        adv = self._resolve_adv(context)
        impact_fraction = 0.0

        if self.config.market_impact_enabled and self.impact_model is not None:
            annual_vol = self._resolve_volatility(context)
            impact = self.impact_model.calculate_impact(
                side=order.side,
                quantity=qty,
                market_price=market_price,
                annual_volatility=annual_vol,
                average_daily_volume=adv,
                periods_per_year=self._resolve_periods(context),
            )
            impact_fraction = impact.impact_fraction
            impact_bps = impact.impact_bps
            participation = impact.participation_rate
            adv = impact.average_daily_volume
            if order.side is OrderSide.BUY:
                exec_price = after_slippage * (1.0 + impact_fraction)
            elif order.side is OrderSide.SELL:
                exec_price = after_slippage * (1.0 - impact_fraction)
            impact_cost = abs(exec_price - after_slippage) * qty

        implicit_total = spread_cost + fixed_slippage_cost + impact_cost
        notional = qty * exec_price
        commission = notional * _bps_to_fraction(self.config.commission_bps)
        total_cost = commission + implicit_total

        return Fill(
            order=order,
            market_price=float(market_price),
            execution_price=float(exec_price),
            fill_quantity=qty,
            commission=float(commission),
            slippage=float(implicit_total),
            total_transaction_cost=float(total_cost),
            spread_cost=float(spread_cost),
            fixed_slippage_cost=float(fixed_slippage_cost),
            market_impact_cost=float(impact_cost),
            market_impact_bps=float(impact_bps),
            participation_rate=float(participation),
            average_daily_volume=float(adv),
            price_after_spread=float(after_spread),
            price_after_slippage=float(after_slippage),
            realized_pnl=0.0,
            timestamp=order.timestamp,
            step=order.step,
        )
