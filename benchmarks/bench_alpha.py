#!/usr/bin/env python3
"""Reproducible Alpha benchmarks. Numbers are measured, never hardcoded.

    python benchmarks/bench_alpha.py
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

from alpha.execution import SimpleExecutionModel
from alpha.execution.model import ExecutionConfig
from alpha.market import GeometricBrownianMotion, OrnsteinUhlenbeck
from alpha.portfolio import Portfolio
from alpha.simulation import MonteCarloSimulator, SimulationEngine
from alpha.strategies import BuyAndHoldStrategy, HoldStrategy, MovingAverageCrossover


def _timed(fn, *, warmup: int = 1, repeats: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_gbm_paths(n_paths: int = 10_000, n_steps: int = 1_000) -> None:
    model = GeometricBrownianMotion(s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252, seed=1)

    def run() -> None:
        model.generate_paths(n_paths, n_steps, rng=np.random.default_rng(1))

    elapsed = _timed(run)
    print("GBM path generation (vectorized NumPy)")
    print(f"  {n_paths:,} paths × {n_steps:,} steps: {elapsed:.4f} s")
    print(f"  {n_paths * n_steps / elapsed:,.0f} price points / s")
    print()


def bench_ou_paths(n_paths: int = 10_000, n_steps: int = 1_000) -> None:
    model = OrnsteinUhlenbeck(
        x0=100.0, mu=100.0, theta=1.0, sigma=5.0, dt=1 / 252, seed=1
    )

    def run() -> None:
        model.generate_paths(n_paths, n_steps, rng=np.random.default_rng(1))

    elapsed = _timed(run)
    print("OU path generation (vectorized NumPy)")
    print(f"  {n_paths:,} paths × {n_steps:,} steps: {elapsed:.4f} s")
    print(f"  {n_paths * n_steps / elapsed:,.0f} points / s")
    print()


def bench_engine_steps(n_steps: int = 20_000) -> float:
    def make() -> SimulationEngine:
        return SimulationEngine(
            model=GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252),
            portfolio=Portfolio(initial_capital=100_000.0),
            strategy=HoldStrategy(),
            execution=SimpleExecutionModel.frictionless(),
            seed=1,
            record_steps=False,
        )

    make().run(min(200, n_steps))
    engine = make()
    t0 = time.perf_counter()
    engine.run(n_steps)
    elapsed = time.perf_counter() - t0
    rate = n_steps / elapsed
    print("Simulation engine (HoldStrategy, frictionless GBM)")
    print(f"  {n_steps:,} step updates in {elapsed:.4f} s")
    print(f"  {rate:,.0f} updates / s")
    print(f"  exceeds 100 updates/s: {'YES' if rate >= 100.0 else 'NO'}")
    print()

    engine_ma = SimulationEngine(
        model=GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252),
        portfolio=Portfolio(initial_capital=100_000.0),
        strategy=MovingAverageCrossover(10, 30, trade_quantity=10.0),
        execution=SimpleExecutionModel.realistic(),
        seed=1,
        record_steps=False,
    )
    t0 = time.perf_counter()
    engine_ma.run(n_steps)
    elapsed_ma = time.perf_counter() - t0
    print("Simulation engine (MA crossover + realistic costs)")
    print(f"  {n_steps:,} step updates in {elapsed_ma:.4f} s")
    print(f"  {n_steps / elapsed_ma:,.0f} updates / s")
    print()

    engine_impact = SimulationEngine(
        model=GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252),
        portfolio=Portfolio(initial_capital=100_000.0),
        strategy=MovingAverageCrossover(10, 30, trade_quantity=10.0),
        execution=SimpleExecutionModel(
            config=ExecutionConfig.liquidity_aware(
                commission_bps=5.0,
                slippage_bps=2.0,
                spread_bps=1.0,
                impact_coefficient=0.10,
                average_daily_volume=1_000_000.0,
                annual_volatility=0.20,
            )
        ),
        seed=1,
        record_steps=False,
    )
    t0 = time.perf_counter()
    engine_impact.run(n_steps)
    elapsed_impact = time.perf_counter() - t0
    print("Simulation engine (MA + liquidity-aware market impact)")
    print(f"  {n_steps:,} step updates in {elapsed_impact:.4f} s")
    print(f"  {n_steps / elapsed_impact:,.0f} updates / s")
    print(f"  exceeds 100 updates/s: {'YES' if n_steps / elapsed_impact >= 100.0 else 'NO'}")
    print()
    return rate


def bench_monte_carlo() -> None:
    def make_mc() -> MonteCarloSimulator:
        return MonteCarloSimulator(
            model_factory=lambda: GeometricBrownianMotion(
                s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252
            ),
            strategy_factory=lambda: BuyAndHoldStrategy(quantity=10.0),
            execution_config=SimpleExecutionModel.realistic().config,
            initial_capital=100_000.0,
        )

    print("Monte Carlo (Buy & Hold, 252 steps, realistic execution)")
    print("  Note: path generation is vectorized; per-path replay is Python.")
    for n_paths in (1_000, 10_000):
        mc = make_mc()
        mc.run(n_paths=min(20, n_paths), n_steps=252, seed=1)
        mc = make_mc()
        t0 = time.perf_counter()
        mc.run(n_paths=n_paths, n_steps=252, seed=1)
        elapsed = time.perf_counter() - t0
        print(
            f"  {n_paths:,} paths: {elapsed:.4f} s  "
            f"({n_paths / elapsed:,.1f} paths / s)"
        )

    impact_cfg = ExecutionConfig.liquidity_aware(
        commission_bps=5.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        impact_coefficient=0.10,
        average_daily_volume=1_000_000.0,
        annual_volatility=0.20,
    )
    mc_impact = MonteCarloSimulator(
        model_factory=lambda: GeometricBrownianMotion(
            s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252
        ),
        strategy_factory=lambda: BuyAndHoldStrategy(quantity=100.0),
        execution_config=impact_cfg,
        initial_capital=100_000.0,
    )
    print("Monte Carlo with market impact (1,000 paths × 252 steps)")
    mc_impact.run(n_paths=20, n_steps=252, seed=1)
    t0 = time.perf_counter()
    mc_impact.run(n_paths=1_000, n_steps=252, seed=1)
    elapsed_impact = time.perf_counter() - t0
    print(f"  1,000 paths: {elapsed_impact:.4f} s  ({1_000 / elapsed_impact:,.1f} paths / s)")
    print()


def bench_historical_replay(n_bars: int = 2_520) -> None:
    """Replay synthetic OHLCV-shaped data through HistoricalMarketReplay."""
    import pandas as pd

    from alpha.market.historical import HistoricalMarketReplay
    from alpha.strategies import HoldStrategy

    dates = pd.bdate_range("2015-01-01", periods=n_bars)
    close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0.05, 1.0, n_bars))
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adjusted_close": close,
            "volume": np.full(n_bars, 1_000_000.0),
        }
    )
    model = HistoricalMarketReplay(df, symbol="BENCH")

    def run() -> None:
        model.reset()
        eng = SimulationEngine(
            model=model,
            portfolio=Portfolio(initial_capital=100_000.0),
            strategy=HoldStrategy(),
            execution=SimpleExecutionModel.frictionless(),
            execution_timing="next_bar_open",
            record_steps=False,
        )
        eng.run(model.n_steps_available)

    elapsed = _timed(run, warmup=0, repeats=3)
    steps = n_bars - 1
    print("Historical OHLCV replay (HoldStrategy, next-bar-open)")
    print(f"  ~{n_bars:,} daily bars ({steps:,} steps): {elapsed:.4f} s")
    print(f"  {steps / elapsed:,.0f} bars replayed / s")
    print()


def main() -> None:
    print("Alpha benchmarks")
    print("=" * 64)
    print(f"Python {sys.version.split()[0]}  NumPy {np.__version__}  {platform.machine()}")
    print("Times are wall-clock via time.perf_counter (best of 3 where noted).")
    print("Do not copy these numbers into docs without re-running this script.")
    print()
    bench_gbm_paths()
    bench_ou_paths()
    bench_engine_steps()
    bench_historical_replay(n_bars=1_260)
    bench_historical_replay(n_bars=2_520)
    bench_monte_carlo()


if __name__ == "__main__":
    main()
