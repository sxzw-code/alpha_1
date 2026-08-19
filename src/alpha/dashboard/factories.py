"""Build backend objects from dashboard settings. No simulation logic here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd

from alpha.execution.model import ExecutionConfig, SimpleExecutionModel
from alpha.market.base import PriceModel
from alpha.market.gbm import GeometricBrownianMotion
from alpha.market.historical import HistoricalMarketReplay, PriceBasis
from alpha.market.mean_reversion import OrnsteinUhlenbeck
from alpha.portfolio.portfolio import Portfolio
from alpha.simulation.engine import (
    LiquiditySource,
    SimulationEngine,
    VolatilitySource,
)
from alpha.simulation.monte_carlo import MonteCarloSimulator
from alpha.strategies.base import BuyAndHoldStrategy, Strategy
from alpha.strategies.mean_reversion import MeanReversionStrategy
from alpha.strategies.moving_average import MovingAverageCrossover

ModelName = Literal["Geometric Brownian Motion", "Ornstein–Uhlenbeck"]
StrategyName = Literal["Moving Average Crossover", "Mean Reversion", "Buy and Hold"]
MarketSourceName = Literal["Synthetic", "Historical"]
DataProviderName = Literal["Yahoo Finance", "CSV file"]
MonteCarloSourceName = Literal["GBM", "OU", "Historical Bootstrap"]

DT = 1.0 / 252.0


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Validated UI settings mapped onto the existing Alpha APIs."""

    market_source: MarketSourceName
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
    market_impact_enabled: bool
    impact_coefficient: float
    average_daily_volume: float
    manual_annual_volatility: Optional[float]
    compare_friction: bool
    # Historical mode
    data_provider: DataProviderName = "Yahoo Finance"
    symbol: str = "AAPL"
    hist_start: str = "2020-01-01"
    hist_end: str = "2024-12-31"
    price_basis: PriceBasis = "adjusted"
    csv_path: str = "tests/fixtures/sample_ohlcv.csv"
    liquidity_source: LiquiditySource = "historical"
    volatility_source: VolatilitySource = "historical"
    adv_window: int = 20
    vol_window: int = 20
    monte_carlo_source: MonteCarloSourceName = "GBM"
    bootstrap_block_size: int = 10

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.market_source == "Synthetic":
            if self.n_steps < 2:
                errors.append("Number of steps must be at least 2.")
            if self.s0 <= 0.0 or self.x0 <= 0.0:
                errors.append("Initial price / value must be positive.")
            if self.sigma < 0.0 or self.ou_sigma < 0.0:
                errors.append("Volatility cannot be negative.")
            if self.theta < 0.0:
                errors.append("Mean-reversion speed must be ≥ 0.")
        else:
            if not self.symbol.strip():
                errors.append("Symbol is required for historical mode.")
            if self.data_provider == "CSV file" and not Path(self.csv_path).exists():
                errors.append(f"CSV file not found: {self.csv_path}")
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
        if self.impact_coefficient < 0.0:
            errors.append("Impact coefficient cannot be negative.")
        if self.average_daily_volume <= 0.0:
            errors.append("Average daily volume must be positive.")
        if self.manual_annual_volatility is not None and self.manual_annual_volatility < 0.0:
            errors.append("Manual volatility cannot be negative.")
        return errors

    @property
    def is_historical(self) -> bool:
        return self.market_source == "Historical"


def _annual_volatility(cfg: DashboardConfig) -> Optional[float]:
    if cfg.manual_annual_volatility is not None:
        return float(cfg.manual_annual_volatility)
    if cfg.market_source == "Synthetic" and cfg.model_name == "Geometric Brownian Motion":
        return float(cfg.sigma)
    return None


def _liquidity_source(cfg: DashboardConfig) -> LiquiditySource:
    if cfg.is_historical and cfg.liquidity_source == "historical":
        return "historical"
    return "manual"


def _volatility_source(cfg: DashboardConfig) -> VolatilitySource:
    if cfg.is_historical:
        if cfg.volatility_source == "historical":
            return "historical"
        if cfg.volatility_source == "manual":
            return "manual"
    if cfg.manual_annual_volatility is not None:
        return "manual"
    return "model"


def execution_config(cfg: DashboardConfig, *, force_frictionless: bool = False) -> ExecutionConfig:
    if force_frictionless or not cfg.frictions:
        return ExecutionConfig.frictionless()
    return ExecutionConfig(
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        spread_bps=cfg.spread_bps,
        market_impact_enabled=cfg.market_impact_enabled,
        impact_coefficient=cfg.impact_coefficient,
        average_daily_volume=cfg.average_daily_volume,
        annual_volatility=_annual_volatility(cfg),
        periods_per_year=1.0 / DT,
    )


def realistic_execution_config(cfg: DashboardConfig) -> ExecutionConfig:
    """Costs implied by the sidebar, even if the main-run toggle is off."""
    return ExecutionConfig(
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        spread_bps=cfg.spread_bps,
        market_impact_enabled=cfg.market_impact_enabled,
        impact_coefficient=cfg.impact_coefficient,
        average_daily_volume=cfg.average_daily_volume,
        annual_volatility=_annual_volatility(cfg),
        periods_per_year=1.0 / DT,
    )


def load_historical_data(cfg: DashboardConfig) -> pd.DataFrame:
    """Load normalized OHLCV for dashboard historical mode."""
    if cfg.data_provider == "CSV file":
        from alpha.data.csv_source import CSVMarketDataSource

        src = CSVMarketDataSource(cfg.csv_path)
    else:
        from alpha.data.yfinance_source import YFinanceMarketDataSource

        src = YFinanceMarketDataSource(use_cache=True)
    return src.load(cfg.symbol, cfg.hist_start, cfg.hist_end, interval="1d")


def historical_model_from_data(cfg: DashboardConfig, data: pd.DataFrame) -> HistoricalMarketReplay:
    return HistoricalMarketReplay(
        data,
        symbol=cfg.symbol,
        price_basis=cfg.price_basis,
        adv_window=cfg.adv_window,
        vol_window=cfg.vol_window,
    )


def model_factory(cfg: DashboardConfig) -> Callable[[], PriceModel]:
    if cfg.is_historical:
        data = load_historical_data(cfg)
        return lambda: historical_model_from_data(cfg, data)
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
    asset = cfg.symbol.upper() if cfg.is_historical else "ASSET"
    return lambda: Portfolio(
        initial_capital=cfg.initial_capital, allow_short=False, asset_id=asset
    )


def effective_n_steps(cfg: DashboardConfig, data: Optional[pd.DataFrame] = None) -> int:
    if cfg.is_historical:
        if data is not None:
            return max(len(data) - 1, 1)
        df = load_historical_data(cfg)
        return max(len(df) - 1, 1)
    return cfg.n_steps


def build_engine(
    cfg: DashboardConfig,
    *,
    execution: Optional[SimpleExecutionModel] = None,
    record_steps: bool = True,
    historical_data: Optional[pd.DataFrame] = None,
) -> SimulationEngine:
    exec_model = execution or SimpleExecutionModel(config=execution_config(cfg))
    if cfg.is_historical:
        data = historical_data if historical_data is not None else load_historical_data(cfg)
        model = historical_model_from_data(cfg, data)
        timing = "next_bar_open"
    else:
        model = model_factory(cfg)()
        timing = "same_bar"
    return SimulationEngine(
        model=model,
        portfolio=portfolio_factory(cfg)(),
        strategy=strategy_factory(cfg)(),
        execution=exec_model,
        seed=cfg.seed,
        record_steps=record_steps,
        execution_timing=timing,
        liquidity_source=_liquidity_source(cfg),
        volatility_source=_volatility_source(cfg),
        adv_window=cfg.adv_window,
        vol_window=cfg.vol_window,
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
