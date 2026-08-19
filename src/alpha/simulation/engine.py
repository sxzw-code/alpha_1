"""Stateful simulation engine: market → strategy → order → execution → portfolio."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import numpy as np
from numpy.random import Generator

from alpha.execution.model import ExecutionModel, SimpleExecutionModel
from alpha.execution.context import ExecutionContext
from alpha.execution.orders import Fill, Order, OrderSide, Signal
from alpha.market.base import MarketState, PriceModel
from alpha.market.gbm import GeometricBrownianMotion
from alpha.market.historical import HistoricalMarketReplay
from alpha.portfolio.portfolio import Portfolio, PortfolioSnapshot
from alpha.strategies.base import Strategy

ExecutionTiming = Literal["same_bar", "next_bar_open"]
LiquiditySource = Literal["historical", "manual"]
VolatilitySource = Literal["historical", "manual", "model"]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Structured output of a single engine step — suitable for a live UI."""

    step: int
    timestamp: float
    market: MarketState
    signal: Signal
    order: Order
    fill: Optional[Fill]
    portfolio: PortfolioSnapshot


@dataclass(slots=True)
class SimulationResult:
    """Aggregate output of a multi-step run.

    Arrays are aligned with the price/equity history (index 0 = initial mark
    before any strategy action). Per-step trading details live in
    ``step_results`` (length ``n_steps``).
    """

    n_steps: int
    seed: Optional[int]
    prices: np.ndarray
    equity: np.ndarray
    cash: np.ndarray
    positions: np.ndarray
    timestamps: np.ndarray
    steps: np.ndarray
    cumulative_return: np.ndarray
    signals: list[Signal]
    trades: list[Fill]
    total_transaction_costs: float
    final_portfolio: PortfolioSnapshot
    step_results: list[StepResult] = field(default_factory=list)
    execution_label: str = "custom"

    @property
    def final_equity(self) -> float:
        return float(self.final_portfolio.equity)

    @property
    def final_price(self) -> float:
        return float(self.prices[-1])

    @property
    def n_trades(self) -> int:
        return len(self.trades)


@dataclass(frozen=True, slots=True)
class FrictionComparison:
    """Side-by-side frictionless vs realistic results on the same price path."""

    frictionless: SimulationResult
    realistic: SimulationResult

    def summary_rows(self) -> dict[str, tuple[float, float]]:
        """Metric name → (frictionless, realistic) for UI tables."""
        f, r = self.frictionless, self.realistic
        return {
            "final_equity": (f.final_equity, r.final_equity),
            "cumulative_return": (
                float(f.cumulative_return[-1]),
                float(r.cumulative_return[-1]),
            ),
            "total_transaction_costs": (
                f.total_transaction_costs,
                r.total_transaction_costs,
            ),
            "n_trades": (float(f.n_trades), float(r.n_trades)),
        }


class SimulationEngine:
    """Orchestrates one simulation path under a price model and strategy.

    Pipeline per step::

        price model → market update → strategy signal → order
            → execution model → fill → portfolio update → performance snapshot

    Execution timing
    ----------------
    ``same_bar`` (default for synthetic paths): each step reveals the new
    market price, the strategy signals, and the fill occurs at that same
    observation (plus configured slippage). This is a close-to-close /
    same-bar fill convention — slightly optimistic versus next-bar execution.

    ``next_bar_open`` (default for historical backtests): the strategy
    observes the **close** of bar *t*, generates a signal, and actionable
    orders fill at the **open** of bar *t+1*. Pending orders never use
    future bar information.

    Designed so a UI can call :meth:`step` repeatedly and read
    :class:`StepResult` / :meth:`performance` without batching.
    """

    def __init__(
        self,
        model: PriceModel,
        portfolio: Portfolio,
        strategy: Strategy,
        *,
        execution: Optional[ExecutionModel] = None,
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
        record_steps: bool = True,
        execution_label: Optional[str] = None,
        execution_timing: ExecutionTiming = "same_bar",
        liquidity_source: LiquiditySource = "manual",
        volatility_source: VolatilitySource = "model",
        adv_window: int = 20,
        vol_window: int = 20,
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Provide at most one of seed or rng")
        self._model = model
        self._portfolio = portfolio
        self._strategy = strategy
        self._execution: ExecutionModel = (
            execution if execution is not None else SimpleExecutionModel.frictionless()
        )
        self._seed = seed
        self._rng: Generator = (
            rng if rng is not None else np.random.default_rng(seed)
        )
        self._record_steps = record_steps
        if isinstance(model, HistoricalMarketReplay) and execution_timing == "same_bar":
            self._execution_timing: ExecutionTiming = "next_bar_open"
        else:
            self._execution_timing = execution_timing
        self._liquidity_source = liquidity_source
        self._volatility_source = volatility_source
        self._adv_window = adv_window
        self._vol_window = vol_window
        self._pending_order: Optional[Order] = None
        if execution_label is not None:
            self._execution_label = execution_label
        elif isinstance(self._execution, SimpleExecutionModel):
            cfg = self._execution.config
            if cfg.is_frictionless:
                self._execution_label = "frictionless"
            elif cfg.market_impact_enabled:
                self._execution_label = "liquidity-aware"
            else:
                self._execution_label = "realistic"
        else:
            self._execution_label = "custom"
        self._step_index = 0
        self._prices: list[float] = []
        self._equity: list[float] = []
        self._cash: list[float] = []
        self._positions: list[float] = []
        self._timestamps: list[float] = []
        self._step_indices: list[int] = []
        self._step_results: list[StepResult] = []
        self._trades: list[Fill] = []
        self._last_result: Optional[StepResult] = None
        # Engine owns the RNG stream; model live-stepping consumes from it.
        self._model.reset(rng=self._rng)
        self._record_initial_mark()

    def _record_initial_mark(self) -> None:
        """Mark portfolio at t=0 before any strategy action."""
        state = self._model.state()
        snap = self._portfolio.mark_to_market(
            state.price,
            step=state.step,
            timestamp=state.timestamp,
            record=self._record_steps,
        )
        self._append_history(state.price, snap)

    def _append_history(self, price: float, snap: PortfolioSnapshot) -> None:
        self._prices.append(price)
        self._equity.append(snap.equity)
        self._cash.append(snap.cash)
        self._positions.append(snap.quantity)
        self._timestamps.append(snap.timestamp)
        self._step_indices.append(snap.step)

    @property
    def model(self) -> PriceModel:
        return self._model

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def execution(self) -> ExecutionModel:
        return self._execution

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def last_result(self) -> Optional[StepResult]:
        return self._last_result

    def performance(self) -> PortfolioSnapshot:
        """Current portfolio snapshot at the latest market price."""
        return self._portfolio.snapshot(
            self._model.current_price,
            step=self._model.step_index,
            timestamp=self._model.timestamp,
        )

    def _execution_context(self, *, market_price: float) -> Optional[ExecutionContext]:
        """Build liquidity / volatility context for the execution layer."""
        if not isinstance(self._execution, SimpleExecutionModel):
            return None
        cfg = self._execution.config
        if not cfg.market_impact_enabled and cfg.annual_volatility is None:
            if not isinstance(self._model, HistoricalMarketReplay):
                return None

        annual_vol = cfg.annual_volatility
        adv = cfg.average_daily_volume
        current_volume: Optional[float] = None
        ppy = 1.0 / self._model.dt if self._model.dt > 0.0 else 252.0

        if isinstance(self._model, HistoricalMarketReplay):
            current_volume = self._model.current_volume
            if self._liquidity_source == "historical":
                hist_adv = self._model.historical_adv()
                if hist_adv is not None and hist_adv > 0.0:
                    adv = hist_adv
            if self._volatility_source == "historical":
                hist_vol = self._model.historical_annual_volatility()
                if hist_vol is not None:
                    annual_vol = hist_vol
            elif self._volatility_source == "manual" and cfg.annual_volatility is not None:
                annual_vol = cfg.annual_volatility
        elif annual_vol is None and isinstance(self._model, GeometricBrownianMotion):
            if self._volatility_source != "manual":
                annual_vol = self._model.sigma

        if annual_vol is None and not cfg.market_impact_enabled:
            return None

        return ExecutionContext(
            annual_volatility=annual_vol,
            average_daily_volume=adv,
            current_volume=current_volume,
            periods_per_year=ppy,
        )

    def _execution_price(self, market: MarketState) -> float:
        """Price used for immediate (same-bar) fills."""
        return float(market.price)

    def _open_execution_price(self, market: MarketState) -> float:
        """Price used for next-bar-open fills."""
        if market.open is not None and market.open > 0.0:
            return float(market.open)
        return float(market.price)

    def _is_actionable(self, signal: Signal) -> bool:
        return signal.side in (OrderSide.BUY, OrderSide.SELL) and signal.quantity > 0.0

    def _execute_pending(self, market: MarketState) -> Optional[Fill]:
        if self._pending_order is None:
            return None
        order = self._pending_order
        self._pending_order = None
        exec_price = self._open_execution_price(market)
        exec_order = replace(order, requested_price=exec_price, step=market.step, timestamp=market.timestamp)
        pending = self._execution.execute(
            exec_order,
            exec_price,
            context=self._execution_context(market_price=exec_price),
        )
        if pending is None:
            return None
        fill = self._portfolio.apply_fill(pending)
        self._trades.append(fill)
        return fill

    def step(self) -> StepResult:
        """Advance one observation through the full pipeline."""
        market = self._model.step(self._rng)

        pending_fill: Optional[Fill] = None
        if self._execution_timing == "next_bar_open":
            pending_fill = self._execute_pending(market)

        pre_trade = self._portfolio.snapshot(
            market.price, step=market.step, timestamp=market.timestamp
        )
        signal = self._strategy.generate_signal(market, pre_trade)
        order = signal.to_order(
            step=market.step,
            timestamp=market.timestamp,
            requested_price=market.price,
        )

        fill: Optional[Fill] = pending_fill
        if self._execution_timing == "next_bar_open":
            if self._is_actionable(signal):
                self._pending_order = order
        else:
            pending = self._execution.execute(
                order,
                self._execution_price(market),
                context=self._execution_context(market_price=market.price),
            )
            if pending is not None:
                fill = self._portfolio.apply_fill(pending)
                self._trades.append(fill)

        post_trade = self._portfolio.mark_to_market(
            market.price,
            step=market.step,
            timestamp=market.timestamp,
            record=self._record_steps,
        )
        result = StepResult(
            step=market.step,
            timestamp=market.timestamp,
            market=market,
            signal=signal,
            order=order,
            fill=fill,
            portfolio=post_trade,
        )
        self._step_index = market.step
        self._append_history(market.price, post_trade)
        if self._record_steps:
            self._step_results.append(result)
        self._last_result = result
        return result

    def run(self, n_steps: int) -> SimulationResult:
        """Run ``n_steps`` advances and return structured results."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        for _ in range(n_steps):
            self.step()
        return self.results()

    def results(self) -> SimulationResult:
        """Build a :class:`SimulationResult` from current histories."""
        equity = np.asarray(self._equity, dtype=float)
        initial = float(self._portfolio.initial_capital)
        return SimulationResult(
            n_steps=self._step_index,
            seed=self._seed,
            prices=np.asarray(self._prices, dtype=float),
            equity=equity,
            cash=np.asarray(self._cash, dtype=float),
            positions=np.asarray(self._positions, dtype=float),
            timestamps=np.asarray(self._timestamps, dtype=float),
            steps=np.asarray(self._step_indices, dtype=int),
            cumulative_return=equity / initial - 1.0,
            signals=[sr.signal for sr in self._step_results],
            trades=list(self._trades),
            total_transaction_costs=self._portfolio.total_transaction_costs,
            final_portfolio=self.performance(),
            step_results=list(self._step_results),
            execution_label=self._execution_label,
        )

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset model, portfolio, strategy, and histories.

        If ``seed`` is provided, both the engine and model RNGs are
        re-seeded for a fresh reproducible run.
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            self._model.reset(rng=self._rng)
        else:
            if self._seed is not None:
                self._rng = np.random.default_rng(self._seed)
                self._model.reset(rng=self._rng)
            else:
                self._model.reset()
        self._portfolio.reset()
        self._strategy.reset()
        self._step_index = 0
        self._prices.clear()
        self._equity.clear()
        self._cash.clear()
        self._positions.clear()
        self._timestamps.clear()
        self._step_indices.clear()
        self._step_results.clear()
        self._trades.clear()
        self._last_result = None
        self._pending_order = None
        self._record_initial_mark()


def compare_friction(
    *,
    model_factory,
    strategy_factory,
    portfolio_factory,
    n_steps: int,
    seed: int,
    realistic_execution: Optional[ExecutionModel] = None,
) -> FrictionComparison:
    """Run the same seeded path under frictionless and realistic execution.

    Factories are zero-arg callables so each leg gets fresh model / strategy /
    portfolio instances while sharing the same RNG seed (identical prices).
    """
    realistic = realistic_execution or SimpleExecutionModel.realistic()

    eng_a = SimulationEngine(
        model=model_factory(),
        portfolio=portfolio_factory(),
        strategy=strategy_factory(),
        execution=SimpleExecutionModel.frictionless(),
        seed=seed,
        execution_label="frictionless",
    )
    frictionless = eng_a.run(n_steps)

    eng_b = SimulationEngine(
        model=model_factory(),
        portfolio=portfolio_factory(),
        strategy=strategy_factory(),
        execution=realistic,
        seed=seed,
        execution_label="realistic",
    )
    realistic_result = eng_b.run(n_steps)

    if not np.allclose(frictionless.prices, realistic_result.prices):
        raise RuntimeError(
            "Price paths diverged between frictionless and realistic runs; "
            "check that execution does not consume RNG."
        )
    return FrictionComparison(
        frictionless=frictionless, realistic=realistic_result
    )
