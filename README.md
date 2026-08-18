# Alpha

Real-time trading simulation engine for evaluating algorithmic strategies under stochastic price models.

This is an **educational / research simulator**. It is not investment advice and not a live brokerage.

## Dashboard

Launch the interactive UI (Streamlit + Plotly):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

After launch, save a screenshot to `docs/screenshots/dashboard.png` if you want one in this README.

The dashboard calls the existing engine, analytics, and Monte Carlo modules. It does not reimplement simulation logic.

## Features

- **Stochastic market models**: Geometric Brownian Motion (GBM) and Ornstein–Uhlenbeck, with vectorized multi-path generation
- **Strategies**: moving-average crossover, z-score mean reversion, buy-and-hold
- **Execution**: commission (bps), slippage (bps), optional half-spread; frictionless vs realistic on the same path
- **Analytics**: total / annualized return, volatility, Sharpe, drawdown path, trade blotter
- **Monte Carlo**: independent paths, histograms, percentile tables, sample-path plots
- **Simulated real-time mode**: `Start` / `Step` / `Reset` on the existing step-based engine (not live market data)
- **Dashboard**: Streamlit + Plotly research UI

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Library quick start

```python
from alpha.execution import ExecutionConfig
from alpha.market import GeometricBrownianMotion
from alpha.simulation import MonteCarloSimulator
from alpha.strategies import MovingAverageCrossover

mc = MonteCarloSimulator(
    model_factory=lambda: GeometricBrownianMotion(s0=100.0, mu=0.08, sigma=0.20, dt=1 / 252),
    strategy_factory=lambda: MovingAverageCrossover(10, 30, trade_quantity=50.0),
    execution_config=ExecutionConfig(commission_bps=5.0, slippage_bps=2.0),
    initial_capital=100_000.0,
)
result = mc.run(n_paths=1_000, n_steps=252, seed=42)
print(result.summary())
```

CLI examples:

- `examples/run_ma_crossover.py` — frictionless vs realistic on one path
- `examples/run_monte_carlo.py` — Monte Carlo distribution
- `python benchmarks/bench_alpha.py` — path-generation and engine throughput

## Market models

**Geometric Brownian Motion**

\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
\]

Exact step: \(S_{t+\Delta t} = S_t \exp[(\mu-\tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z]\). Dashboard \(\mu\) and \(\sigma\) are annual; \(\Delta t = 1/252\).

**Ornstein–Uhlenbeck**

\[
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
\]

Mean-reverting Gaussian process. \(\theta\) is the pull toward the long-run mean \(\mu\).

## Monte Carlo

Paths are drawn independently from the selected model (vectorized). Each path gets a **new** strategy, portfolio, and execution state. Reported distributions include mean, median, and 5th/95th percentiles of return, Sharpe, drawdown, and final equity.

## Transaction costs

- **Commission**: cash fee in basis points of executed notional (5 bps = 0.05%).
- **Slippage**: BUY fills above the observed mid; SELL fills below it.
- Optional **half-spread** is applied in the same adverse direction. This is not a limit-order book.

## Project layout

```
src/alpha/
  market/       # GBM, OU, path replay
  strategies/   # MA crossover, mean reversion, buy-and-hold
  portfolio/    # cash, inventory, P&L
  execution/    # orders, fills, costs
  simulation/   # step engine + Monte Carlo
  analytics/    # risk / performance metrics
  dashboard/    # Streamlit UI (calls the modules above)
app.py          # streamlit run app.py
```

## Tests

```bash
pytest
```
