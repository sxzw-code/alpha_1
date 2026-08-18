#!/usr/bin/env python3
"""Example: Monte Carlo evaluation of a moving-average crossover on GBM.

Run from the repo root after ``pip install -e .``::

    python examples/run_monte_carlo.py
"""

from __future__ import annotations

from alpha.analytics import analyze_result
from alpha.execution import ExecutionConfig, SimpleExecutionModel
from alpha.market import GeometricBrownianMotion
from alpha.portfolio import Portfolio
from alpha.simulation import MonteCarloSimulator, SimulationEngine
from alpha.strategies import MovingAverageCrossover


def main() -> None:
    n_paths = 1_000
    n_steps = 252
    seed = 42

    mc = MonteCarloSimulator(
        model_factory=lambda: GeometricBrownianMotion(
            s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252
        ),
        strategy_factory=lambda: MovingAverageCrossover(
            fast_window=10, slow_window=30, trade_quantity=50.0
        ),
        execution_config=ExecutionConfig(
            commission_bps=5.0, slippage_bps=2.0, spread_bps=1.0
        ),
        initial_capital=100_000.0,
        periods_per_year=252.0,
        risk_free_rate=0.0,
    )
    result = mc.run(n_paths=n_paths, n_steps=n_steps, seed=seed)
    print(result.summary())

    # Single-path metrics on one engine run for comparison.
    engine = SimulationEngine(
        model=GeometricBrownianMotion(s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252),
        portfolio=Portfolio(initial_capital=100_000.0),
        strategy=MovingAverageCrossover(10, 30, trade_quantity=50.0),
        execution=SimpleExecutionModel.realistic(
            commission_bps=5.0, slippage_bps=2.0, spread_bps=1.0
        ),
        seed=seed,
    )
    one = engine.run(n_steps)
    stats = analyze_result(one, periods_per_year=252.0)
    print()
    print("=== One GBM path (same seed, MA crossover) ===")
    print(f"total return:     {stats.total_return:.2%}")
    print(f"ann. return:      {stats.annualized_return:.2%}")
    print(f"Sharpe:           {stats.sharpe_ratio:.2f}")
    print(f"max drawdown:     {stats.max_drawdown:.2%}")
    print(f"trades:           {stats.n_trades}")
    print(f"transaction costs:${stats.total_transaction_costs:,.2f}")


if __name__ == "__main__":
    main()
