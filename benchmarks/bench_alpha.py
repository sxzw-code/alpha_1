#!/usr/bin/env python3
"""Benchmarks for vectorized path generation and simulation throughput.

Run from the repo root after ``pip install -e .``::

    python benchmarks/bench_alpha.py
"""

from __future__ import annotations

import time

import numpy as np

from alpha.execution import SimpleExecutionModel
from alpha.market import GeometricBrownianMotion, OrnsteinUhlenbeck
from alpha.portfolio import Portfolio
from alpha.simulation import MonteCarloSimulator, SimulationEngine
from alpha.strategies import BuyAndHoldStrategy, HoldStrategy, MovingAverageCrossover


def _timed(fn, *, warmup: int = 1, repeats: int = 3) -> float:
    """Return the best ``perf_counter`` elapsed seconds over ``repeats``."""
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
    print("GBM path generation")
    print(f"  {n_paths:,} paths × {n_steps:,} steps: {elapsed:.4f} seconds")
    print(f"  {n_paths * n_steps / elapsed:,.0f} price points / sec")
    print()


def bench_ou_paths(n_paths: int = 10_000, n_steps: int = 1_000) -> None:
    model = OrnsteinUhlenbeck(
        x0=100.0, mu=100.0, theta=1.0, sigma=5.0, dt=1 / 252, seed=1
    )

    def run() -> None:
        model.generate_paths(n_paths, n_steps, rng=np.random.default_rng(1))

    elapsed = _timed(run)
    print("OU path generation")
    print(f"  {n_paths:,} paths × {n_steps:,} steps: {elapsed:.4f} seconds")
    print(f"  {n_paths * n_steps / elapsed:,.0f} points / sec")
    print()


def bench_engine_steps(n_steps: int = 20_000) -> None:
    def make() -> SimulationEngine:
        return SimulationEngine(
            model=GeometricBrownianMotion(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252),
            portfolio=Portfolio(initial_capital=100_000.0),
            strategy=HoldStrategy(),
            execution=SimpleExecutionModel.frictionless(),
            seed=1,
            record_steps=False,
        )

    make().run(min(200, n_steps))  # warmup imports / caches

    engine = make()
    t0 = time.perf_counter()
    engine.run(n_steps)
    elapsed = time.perf_counter() - t0
    rate = n_steps / elapsed
    print("Simulation engine (HoldStrategy, frictionless GBM)")
    print(f"  {n_steps:,} step updates in {elapsed:.4f} seconds")
    print(f"  {rate:,.0f} updates / sec")
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
    print(f"  {n_steps:,} step updates in {elapsed_ma:.4f} seconds")
    print(f"  {n_steps / elapsed_ma:,.0f} updates / sec")
    print()


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
    for n_paths in (1_000, 10_000):
        mc = make_mc()
        mc.run(n_paths=min(20, n_paths), n_steps=252, seed=1)  # warmup
        mc = make_mc()
        t0 = time.perf_counter()
        mc.run(n_paths=n_paths, n_steps=252, seed=1)
        elapsed = time.perf_counter() - t0
        print(f"  {n_paths:,} paths: {elapsed:.4f} sec  ({n_paths / elapsed:,.1f} paths / sec)")
    print()


def main() -> None:
    print("Alpha benchmarks  (best of 3 where noted; engine/MC timed once after warmup)")
    print("=" * 64)
    print()
    bench_gbm_paths()
    bench_ou_paths()
    bench_engine_steps()
    bench_monte_carlo()


if __name__ == "__main__":
    main()
