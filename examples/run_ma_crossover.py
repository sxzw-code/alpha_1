#!/usr/bin/env python3
"""Example: GBM + moving-average crossover with transaction costs.

Run from the repo root after ``pip install -e .``::

    python examples/run_ma_crossover.py
"""

from __future__ import annotations

from alpha.execution import SimpleExecutionModel
from alpha.market import GeometricBrownianMotion
from alpha.portfolio import Portfolio
from alpha.simulation import SimulationEngine, compare_friction
from alpha.strategies import MovingAverageCrossover


def main() -> None:
    seed = 42
    n_steps = 252
    capital = 100_000.0

    model = GeometricBrownianMotion(s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252)
    portfolio = Portfolio(initial_capital=capital)
    strategy = MovingAverageCrossover(
        fast_window=10, slow_window=30, trade_quantity=50.0
    )
    execution = SimpleExecutionModel.realistic(
        commission_bps=5.0, slippage_bps=2.0, spread_bps=1.0
    )

    engine = SimulationEngine(
        model=model,
        portfolio=portfolio,
        strategy=strategy,
        execution=execution,
        seed=seed,
    )
    result = engine.run(n_steps)

    print("=== Single path (realistic execution) ===")
    print(f"steps:              {result.n_steps}")
    print(f"final price:        {result.final_price:,.4f}")
    print(f"final equity:       ${result.final_equity:,.2f}")
    print(f"cumulative return:  {result.cumulative_return[-1]:.2%}")
    print(f"trades:             {result.n_trades}")
    print(f"transaction costs:  ${result.total_transaction_costs:,.2f}")

    comparison = compare_friction(
        model_factory=lambda: GeometricBrownianMotion(
            s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252
        ),
        strategy_factory=lambda: MovingAverageCrossover(10, 30, trade_quantity=50.0),
        portfolio_factory=lambda: Portfolio(initial_capital=capital),
        n_steps=n_steps,
        seed=seed,
        realistic_execution=execution,
    )
    rows = comparison.summary_rows()
    print()
    print("=== Frictionless vs Realistic (identical price path) ===")
    print(f"{'metric':<28} {'frictionless':>14} {'realistic':>14}")
    print("-" * 58)
    print(
        f"{'Final equity':<28} "
        f"${rows['final_equity'][0]:>12,.2f} ${rows['final_equity'][1]:>12,.2f}"
    )
    print(
        f"{'Cumulative return':<28} "
        f"{rows['cumulative_return'][0]:>13.2%} {rows['cumulative_return'][1]:>13.2%}"
    )
    print(
        f"{'Transaction costs':<28} "
        f"${rows['total_transaction_costs'][0]:>12,.2f} "
        f"${rows['total_transaction_costs'][1]:>12,.2f}"
    )
    print(
        f"{'Trades':<28} "
        f"{int(rows['n_trades'][0]):>14} {int(rows['n_trades'][1]):>14}"
    )


if __name__ == "__main__":
    main()
