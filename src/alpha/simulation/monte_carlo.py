"""Monte Carlo evaluation of a strategy across independent price paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from alpha.analytics.metrics import stacked_equity_metrics, trade_statistics
from alpha.execution.model import ExecutionConfig, ExecutionModel, SimpleExecutionModel
from alpha.market.base import PriceModel
from alpha.market.replay import PathReplayModel
from alpha.portfolio.portfolio import Portfolio
from alpha.simulation.engine import SimulationEngine, SimulationResult
from alpha.strategies.base import Strategy

_PERCENTILES = (5.0, 25.0, 75.0, 95.0)


@dataclass(frozen=True, slots=True)
class MetricDistribution:
    """Empirical distribution of one scalar metric across Monte Carlo paths."""

    name: str
    values: np.ndarray
    mean: float
    median: float
    std: float
    p5: float
    p25: float
    p75: float
    p95: float
    min: float
    max: float

    @property
    def n_paths(self) -> int:
        return int(self.values.size)


def summarize_distribution(name: str, values: np.ndarray) -> MetricDistribution:
    """Mean / median / std / percentiles of a 1-d sample (NaNs dropped)."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        nan = float("nan")
        return MetricDistribution(
            name=name,
            values=raw,
            mean=nan,
            median=nan,
            std=nan,
            p5=nan,
            p25=nan,
            p75=nan,
            p95=nan,
            min=nan,
            max=nan,
        )
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    p5, p25, p75, p95 = (float(x) for x in np.percentile(finite, _PERCENTILES))
    return MetricDistribution(
        name=name,
        values=raw,
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        std=std,
        p5=p5,
        p25=p25,
        p75=p75,
        p95=p95,
        min=float(np.min(finite)),
        max=float(np.max(finite)),
    )


@dataclass(slots=True)
class MonteCarloResult:
    """Distributions of path-level performance statistics."""

    n_paths: int
    n_steps: int
    seed: int
    periods_per_year: float
    risk_free_rate: float
    final_equity: MetricDistribution
    total_return: MetricDistribution
    annualized_return: MetricDistribution
    sharpe_ratio: MetricDistribution
    max_drawdown: MetricDistribution
    total_transaction_costs: MetricDistribution
    n_trades: MetricDistribution
    win_rate: MetricDistribution
    average_trade_pnl: MetricDistribution
    equity_paths: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Human-readable report matching the intended Monte Carlo table."""
        n = f"{self.n_paths:,}"
        lines = [
            f"{n} simulations  ({self.n_steps} steps, seed={self.seed})",
            "",
            "Annualized Return",
            f"  Mean:          {_pct(self.annualized_return.mean)}",
            f"  Median:        {_pct(self.annualized_return.median)}",
            f"  5th pct:       {_pct(self.annualized_return.p5)}",
            f"  95th pct:      {_pct(self.annualized_return.p95)}",
            "",
            "Sharpe Ratio",
            f"  Mean:          {_num(self.sharpe_ratio.mean)}",
            f"  Median:        {_num(self.sharpe_ratio.median)}",
            f"  5th pct:       {_num(self.sharpe_ratio.p5)}",
            f"  95th pct:      {_num(self.sharpe_ratio.p95)}",
            "",
            "Maximum Drawdown",
            f"  Mean:          {_pct(self.max_drawdown.mean)}",
            f"  Median:        {_pct(self.max_drawdown.median)}",
            f"  Worst:         {_pct(self.max_drawdown.min)}",
            f"  5th pct:       {_pct(self.max_drawdown.p5)}",
            "",
            "Total Return",
            f"  Mean:          {_pct(self.total_return.mean)}",
            f"  Median:        {_pct(self.total_return.median)}",
            "",
            "Final Equity",
            f"  Mean:          {_money(self.final_equity.mean)}",
            f"  Median:        {_money(self.final_equity.median)}",
            "",
            "Transaction Costs",
            f"  Mean:          {_money(self.total_transaction_costs.mean)}",
            f"  Median:        {_money(self.total_transaction_costs.median)}",
            "",
            "Trades per path",
            f"  Mean:          {_num(self.n_trades.mean)}",
            f"  Median:        {_num(self.n_trades.median)}",
        ]
        return "\n".join(lines)


def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:>8.1%}"


def _num(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:>8.2f}"


def _money(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"${x:,.2f}"


class MonteCarloSimulator:
    """Evaluate one strategy on many independent simulated market paths.

    Price paths are generated in a single vectorized call
    (``PriceModel.generate_paths``). Each path then gets a **fresh**
    strategy, portfolio, and execution model; prices are replayed through
    the existing simulation pipeline so strategy/execution semantics stay
    identical to a single-path :class:`SimulationEngine` run.

    Parameters
    ----------
    model_factory:
        Zero-arg callable returning a new :class:`PriceModel`. Used once
        to generate the path matrix (the factory's own seed is ignored;
        ``run(seed=...)`` owns the RNG).
    strategy_factory:
        Zero-arg callable returning a new :class:`Strategy` per path.
    execution_config:
        Optional friction settings. Ignored if ``execution_factory`` is set.
    execution_factory:
        Optional zero-arg callable returning a new execution model per path.
    initial_capital:
        Starting cash for the default portfolio factory.
    """

    def __init__(
        self,
        model_factory: Callable[[], PriceModel],
        strategy_factory: Callable[[], Strategy],
        *,
        execution_config: Optional[ExecutionConfig] = None,
        execution_factory: Optional[Callable[[], ExecutionModel]] = None,
        portfolio_factory: Optional[Callable[[], Portfolio]] = None,
        initial_capital: float = 100_000.0,
        allow_short: bool = False,
        periods_per_year: float = 252.0,
        risk_free_rate: float = 0.0,
    ) -> None:
        if initial_capital <= 0.0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if periods_per_year <= 0.0:
            raise ValueError(
                f"periods_per_year must be positive, got {periods_per_year}"
            )
        if execution_config is not None and execution_factory is not None:
            raise ValueError("Provide at most one of execution_config or execution_factory")
        self._model_factory = model_factory
        self._strategy_factory = strategy_factory
        self._initial_capital = float(initial_capital)
        self._allow_short = allow_short
        self._periods_per_year = float(periods_per_year)
        self._risk_free_rate = float(risk_free_rate)
        if execution_factory is not None:
            self._execution_factory = execution_factory
        elif execution_config is not None:
            config = execution_config
            self._execution_factory = lambda: SimpleExecutionModel(config=config)
        else:
            self._execution_factory = SimpleExecutionModel.frictionless
        if portfolio_factory is not None:
            self._portfolio_factory = portfolio_factory
        else:
            capital = self._initial_capital
            short = self._allow_short
            self._portfolio_factory = lambda: Portfolio(
                initial_capital=capital, allow_short=short
            )

    def run(
        self,
        n_paths: int,
        n_steps: int,
        seed: int,
        *,
        store_equity_paths: bool = False,
        rng: Optional[Generator] = None,
    ) -> MonteCarloResult:
        """Simulate ``n_paths`` independent trajectories of length ``n_steps``.

        Identical ``seed`` values reproduce the same path matrix and the same
        strategy outcomes. Pass ``rng`` *or* ``seed``, not both.
        """
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        gen = rng if rng is not None else np.random.default_rng(seed)

        template = self._model_factory()
        paths = template.generate_paths(
            n_paths, n_steps, rng=gen, include_initial=True
        )
        dt = template.dt
        asset_id = template.asset_id

        equity_paths = np.empty((n_paths, n_steps + 1), dtype=float)
        costs = np.empty(n_paths, dtype=float)
        n_trades = np.empty(n_paths, dtype=float)
        win_rates = np.empty(n_paths, dtype=float)
        avg_pnls = np.empty(n_paths, dtype=float)

        dummy_rng = np.random.default_rng(0)
        for i in range(n_paths):
            result = self._run_one_path(
                prices=paths[i],
                dt=dt,
                asset_id=asset_id,
                n_steps=n_steps,
                dummy_rng=dummy_rng,
            )
            equity_paths[i] = result.equity
            costs[i] = result.total_transaction_costs
            n_tr, _n_comp, win_rate, avg_pnl = trade_statistics(result.trades)
            n_trades[i] = n_tr
            win_rates[i] = win_rate
            avg_pnls[i] = avg_pnl

        stacked = stacked_equity_metrics(
            equity_paths,
            periods_per_year=self._periods_per_year,
            risk_free_rate=self._risk_free_rate,
        )
        return MonteCarloResult(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=int(seed),
            periods_per_year=self._periods_per_year,
            risk_free_rate=self._risk_free_rate,
            final_equity=summarize_distribution("final_equity", stacked["final_equity"]),
            total_return=summarize_distribution("total_return", stacked["total_return"]),
            annualized_return=summarize_distribution(
                "annualized_return", stacked["annualized_return"]
            ),
            sharpe_ratio=summarize_distribution("sharpe_ratio", stacked["sharpe_ratio"]),
            max_drawdown=summarize_distribution("max_drawdown", stacked["max_drawdown"]),
            total_transaction_costs=summarize_distribution(
                "total_transaction_costs", costs
            ),
            n_trades=summarize_distribution("n_trades", n_trades),
            win_rate=summarize_distribution("win_rate", win_rates),
            average_trade_pnl=summarize_distribution("average_trade_pnl", avg_pnls),
            equity_paths=equity_paths if store_equity_paths else None,
        )

    def _run_one_path(
        self,
        *,
        prices: np.ndarray,
        dt: float,
        asset_id: str,
        n_steps: int,
        dummy_rng: Generator,
    ) -> SimulationResult:
        """Fresh model / strategy / portfolio / execution for one path."""
        engine = SimulationEngine(
            model=PathReplayModel(prices, dt=dt, asset_id=asset_id),
            portfolio=self._portfolio_factory(),
            strategy=self._strategy_factory(),
            execution=self._execution_factory(),
            rng=dummy_rng,
            record_steps=False,
        )
        return engine.run(n_steps)
