"""Build backend objects from dashboard settings. No simulation logic here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from alpha.execution.model import ExecutionConfig, SimpleExecutionModel
from alpha.market.base import PriceModel
from alpha.market.gbm import GeometricBrownianMotion
from alpha.market.mean_reversion import OrnsteinUhlenbeck
from alpha.portfolio.portfolio import Portfolio
from alpha.simulation.engine import SimulationEngine
from alpha.simulation.monte_carlo import MonteCarloSimulator
from alpha.strategies.base import BuyAndHoldStrategy, Strategy
from alpha.strategies.mean_reversion import MeanReversionStrategy
from alpha.strategies.moving_average import MovingAverageCrossover

ModelName = Literal["Geometric Brownian Motion", "Ornstein–Uhlenbeck"]
StrategyName = Literal["Moving Average Crossover", "Mean Reversion", "Buy and Hold"]

DT = 1.0 / 252.0


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Validated UI settings mapped onto the existing Alpha APIs."""

    model_name: ModelName
    s0: float
    mu: float
    sigma: float
    x0: float
    ou_mu: float
    theta: float
    ou_sigma: float
    n_steps: int
    seed: int
    strategy_name: StrategyName
    fast_window: int
    slow_window: int
    lookback: int
    entry_z: float
    exit_z: float
    trade_quantity: float
    initial_capital: float
    frictions: bool
    commission_bps: float
    slippage_bps: float
    spread_bps: float
    compare_friction: bool

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.n_steps < 2:
            errors.append("Number of steps must be at least 2.")
        if self.s0 <= 0.0 or self.x0 <= 0.0:
            errors.append("Initial price / value must be positive.")
        if self.sigma < 0.0 or self.ou_sigma < 0.0:
            errors.append("Volatility cannot be negative.")
        if self.theta < 0.0:
            errors.append("Mean-reversion speed must be ≥ 0.")
        if self.initial_capital <= 0.0:
            errors.append("Initial capital must be positive.")
        if self.trade_quantity <= 0.0:
            errors.append("Trade quantity must be positive.")
        if self.strategy_name == "Moving Average Crossover":
            if self.fast_window < 1:
                errors.append("Fast window must be ≥ 1.")
            if self.slow_window <= self.fast_window:
                errors.append("Slow window must be greater than fast window.")
        if self.strategy_name == "Mean Reversion":
            if self.lookback < 2:
                errors.append("Lookback must be ≥ 2.")
            if self.entry_z <= 0.0:
                errors.append("Entry z-score must be positive.")
            if self.exit_z < 0.0:
                errors.append("Exit z-score must be ≥ 0.")
            if self.exit_z >= self.entry_z:
                errors.append("Exit z-score must be below entry z-score.")
        if self.commission_bps < 0.0 or self.slippage_bps < 0.0 or self.spread_bps < 0.0:
            errors.append("Cost parameters cannot be negative.")
        return errors


def execution_config(cfg: DashboardConfig, *, force_frictionless: bool = False) -> ExecutionConfig:
    if force_frictionless or not cfg.frictions:
        return ExecutionConfig.frictionless()
    return ExecutionConfig(
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        spread_bps=cfg.spread_bps,
    )


def realistic_execution_config(cfg: DashboardConfig) -> ExecutionConfig:
    """Costs implied by the sidebar, even if the main-run toggle is off."""
    return ExecutionConfig(
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        spread_bps=cfg.spread_bps,
    )


def model_factory(cfg: DashboardConfig) -> Callable[[], PriceModel]:
    if cfg.model_name == "Geometric Brownian Motion":
        return lambda: GeometricBrownianMotion(
            s0=cfg.s0, mu=cfg.mu, sigma=cfg.sigma, dt=DT
        )
    return lambda: OrnsteinUhlenbeck(
        x0=cfg.x0, mu=cfg.ou_mu, theta=cfg.theta, sigma=cfg.ou_sigma, dt=DT
    )


def strategy_factory(cfg: DashboardConfig) -> Callable[[], Strategy]:
    if cfg.strategy_name == "Moving Average Crossover":
        return lambda: MovingAverageCrossover(
            cfg.fast_window, cfg.slow_window, cfg.trade_quantity
        )
    if cfg.strategy_name == "Mean Reversion":
        return lambda: MeanReversionStrategy(
            cfg.lookback,
            cfg.trade_quantity,
            entry_z=cfg.entry_z,
            exit_z=cfg.exit_z,
        )
    return lambda: BuyAndHoldStrategy(quantity=cfg.trade_quantity)


def portfolio_factory(cfg: DashboardConfig) -> Callable[[], Portfolio]:
    return lambda: Portfolio(initial_capital=cfg.initial_capital, allow_short=False)


def build_engine(
    cfg: DashboardConfig,
    *,
    execution: Optional[SimpleExecutionModel] = None,
    record_steps: bool = True,
) -> SimulationEngine:
    exec_model = execution or SimpleExecutionModel(config=execution_config(cfg))
    return SimulationEngine(
        model=model_factory(cfg)(),
        portfolio=portfolio_factory(cfg)(),
        strategy=strategy_factory(cfg)(),
        execution=exec_model,
        seed=cfg.seed,
        record_steps=record_steps,
    )


def build_monte_carlo(
    cfg: DashboardConfig,
    *,
    use_frictions: bool,
) -> MonteCarloSimulator:
    costs = (
        realistic_execution_config(cfg)
        if use_frictions
        else ExecutionConfig.frictionless()
    )
    return MonteCarloSimulator(
        model_factory=model_factory(cfg),
        strategy_factory=strategy_factory(cfg),
        execution_config=costs,
        initial_capital=cfg.initial_capital,
        periods_per_year=1.0 / DT,
        risk_free_rate=0.0,
    )
