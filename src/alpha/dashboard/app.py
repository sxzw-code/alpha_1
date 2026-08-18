"""Alpha interactive dashboard (Streamlit).

Launch from the repository root::

    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from alpha.analytics.metrics import PerformanceMetrics, analyze_result
from alpha.dashboard.charts import (
    drawdown_figure,
    equity_figure,
    friction_bar_figure,
    histogram_figure,
    price_figure,
    sample_paths_figure,
)
from alpha.dashboard.factories import (
    DashboardConfig,
    build_engine,
    build_monte_carlo,
    model_factory,
    portfolio_factory,
    realistic_execution_config,
    strategy_factory,
)
from alpha.dashboard.formatting import (
    fmt_int,
    fmt_money,
    fmt_pct,
    fmt_sharpe,
    trades_frame,
)
from alpha.execution.model import SimpleExecutionModel
from alpha.simulation.engine import SimulationEngine, SimulationResult, compare_friction
from alpha.simulation.monte_carlo import MonteCarloResult

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Source Sans 3", sans-serif;
}
.block-container { padding-top: 1.1rem; max-width: 1400px; }
header { visibility: hidden; }

.alpha-header {
  display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 1px solid #243040; padding-bottom: 0.7rem; margin-bottom: 0.9rem;
}
.alpha-title {
  font-size: 1.55rem; font-weight: 600; letter-spacing: 0.18em;
  color: #e8edf4; margin: 0;
}
.alpha-title span { color: #c4a35a; }
.alpha-sub { color: #8b9bb0; font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase; }

.metric-row { display: grid; grid-template-columns: repeat(7, 1fr); gap: 0.65rem; margin: 0.4rem 0 1rem; }
@media (max-width: 1100px) { .metric-row { grid-template-columns: repeat(3, 1fr); } }
.metric-card {
  background: #151b24; border: 1px solid #243040; border-radius: 8px;
  padding: 0.7rem 0.8rem 0.75rem;
}
.metric-card .lbl {
  color: #8b9bb0; font-size: 0.68rem; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 0.25rem;
}
.metric-card .val {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 1.05rem; color: #e8edf4; font-weight: 500;
}
.metric-card .val.pos { color: #3dd68c; }
.metric-card .val.neg { color: #f07178; }

.sim-note {
  color: #8b9bb0; font-size: 0.8rem; border-left: 2px solid #c4a35a;
  padding-left: 0.6rem; margin: 0.2rem 0 0.8rem;
}
section[data-testid="stSidebar"] { background: #10161e; }
</style>
"""


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    """Render the dashboard. Safe to import: no Streamlit calls unless running."""
    if not _in_streamlit():
        return
    import streamlit as st

    st.set_page_config(
        page_title="Alpha — Trading Simulator",
        page_icon="α",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state(st)
    st.markdown(
        '<div class="alpha-header">'
        '<p class="alpha-title">α <span>ALPHA</span></p>'
        '<p class="alpha-sub">Trading Simulator &nbsp;·&nbsp; Research only</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    cfg = _sidebar(st)
    errors = cfg.validation_errors()
    if errors:
        for msg in errors:
            st.error(msg)

    left, right = st.columns([1.15, 2.55], gap="large")
    with left:
        _run_controls(st, cfg, errors)
        _live_panel(st, cfg, errors)
    with right:
        _single_run_panel(st, cfg)

    result: Optional[SimulationResult] = st.session_state.get("single_result")
    metrics: Optional[PerformanceMetrics] = st.session_state.get("single_metrics")
    if result is not None and metrics is not None:
        _metric_cards(st, metrics)
        _trade_table(st, result)

    _friction_section(st)
    _monte_carlo_section(st, cfg, errors)


def _init_state(st: Any) -> None:
    defaults = {
        "single_result": None,
        "single_metrics": None,
        "comparison": None,
        "mc_result": None,
        "mc_price_paths": None,
        "live_engine": None,
        "live_auto": False,
        "live_cfg_key": None,
        "error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sidebar(st: Any) -> DashboardConfig:
    with st.sidebar:
        st.markdown("**Simulation settings**")
        model_name = st.selectbox(
            "Market model",
            ("Geometric Brownian Motion", "Ornstein–Uhlenbeck"),
            help="GBM: dS = μS dt + σS dW. OU: dX = θ(μ−X) dt + σ dW.",
        )
        if model_name == "Geometric Brownian Motion":
            s0 = st.number_input("Initial price S₀", min_value=0.01, value=100.0, step=1.0)
            mu = st.number_input(
                "Annual drift μ",
                value=0.08,
                step=0.01,
                format="%.3f",
                help="Expected annual return of the GBM drift term.",
            )
            sigma = st.number_input(
                "Annual volatility σ",
                min_value=0.0,
                value=0.20,
                step=0.01,
                format="%.3f",
            )
            x0, ou_mu, theta, ou_sigma = 100.0, 100.0, 1.0, 5.0
        else:
            s0, mu, sigma = 100.0, 0.08, 0.20
            x0 = st.number_input("Initial value X₀", min_value=0.01, value=100.0, step=1.0)
            ou_mu = st.number_input("Long-run mean μ", value=100.0, step=1.0)
            theta = st.number_input(
                "Mean-reversion speed θ",
                min_value=0.0,
                value=1.0,
                step=0.1,
                help="Higher θ pulls X back to μ faster.",
            )
            ou_sigma = st.number_input("OU volatility σ", min_value=0.0, value=5.0, step=0.1)

        n_steps = st.slider("Simulation steps", min_value=20, max_value=1000, value=252, step=1)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

        st.divider()
        st.markdown("**Strategy**")
        strategy_name = st.selectbox(
            "Strategy",
            ("Moving Average Crossover", "Mean Reversion", "Buy and Hold"),
        )
        trade_quantity = st.number_input(
            "Trade quantity",
            min_value=0.01,
            value=50.0,
            step=1.0,
            help="Shares transacted on each entry (long-only).",
        )
        fast_window, slow_window = 10, 30
        lookback, entry_z, exit_z = 20, 2.0, 0.5
        if strategy_name == "Moving Average Crossover":
            fast_window = int(st.number_input("Fast window", min_value=1, value=10, step=1))
            slow_window = int(
                st.number_input("Slow window", min_value=2, value=30, step=1)
            )
        elif strategy_name == "Mean Reversion":
            lookback = int(st.number_input("Lookback", min_value=2, value=20, step=1))
            entry_z = float(
                st.number_input("Entry z-score", min_value=0.05, value=2.0, step=0.1)
            )
            exit_z = float(
                st.number_input("Exit z-score", min_value=0.0, value=0.5, step=0.1)
            )

        st.divider()
        st.markdown("**Execution**")
        initial_capital = st.number_input(
            "Initial capital", min_value=1.0, value=100_000.0, step=1000.0, format="%.0f"
        )
        frictions = st.toggle(
            "Include trading frictions",
            value=True,
            help="When off, the single-path run uses zero commission and slippage.",
        )
        commission_bps = st.number_input(
            "Commission (bps of notional)",
            min_value=0.0,
            value=5.0,
            step=0.5,
            help="Cash fee = bps × executed notional. Example: 5 bps = 0.05%.",
        )
        slippage_bps = st.number_input(
            "Slippage (bps)",
            min_value=0.0,
            value=2.0,
            step=0.5,
            help="BUY pays more than mid; SELL receives less.",
        )
        with st.expander("Advanced costs"):
            spread_bps = st.number_input(
                "Half-spread (bps)",
                min_value=0.0,
                value=1.0,
                step=0.5,
                help="Simple bid/ask proxy, added in the same adverse direction as slippage.",
            )
        compare_friction = st.checkbox(
            "Compare frictionless vs realistic on the same path",
            value=True,
        )

        st.divider()
        st.caption(
            "Educational / research simulator. Not investment advice and not a live brokerage."
        )

    return DashboardConfig(
        model_name=model_name,  # type: ignore[arg-type]
        s0=float(s0),
        mu=float(mu),
        sigma=float(sigma),
        x0=float(x0),
        ou_mu=float(ou_mu),
        theta=float(theta),
        ou_sigma=float(ou_sigma),
        n_steps=int(n_steps),
        seed=int(seed),
        strategy_name=strategy_name,  # type: ignore[arg-type]
        fast_window=int(fast_window),
        slow_window=int(slow_window),
        lookback=int(lookback),
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        trade_quantity=float(trade_quantity),
        initial_capital=float(initial_capital),
        frictions=bool(frictions),
        commission_bps=float(commission_bps),
        slippage_bps=float(slippage_bps),
        spread_bps=float(spread_bps),
        compare_friction=bool(compare_friction),
    )


def _run_controls(st: Any, cfg: DashboardConfig, errors: list[str]) -> None:
    st.markdown("##### Single path")
    st.markdown(
        '<p class="sim-note">Runs the existing SimulationEngine for N steps on one seeded stochastic path.</p>',
        unsafe_allow_html=True,
    )
    disabled = bool(errors)
    if st.button("Run simulation", type="primary", use_container_width=True, disabled=disabled):
        _safe(st, lambda: _execute_single(st, cfg))
    if st.session_state.error:
        st.error(st.session_state.error)


def _live_panel(st: Any, cfg: DashboardConfig, errors: list[str]) -> None:
    st.markdown("##### Simulated real-time market")
    st.caption("Steps the engine once per tick. This is not live-market trading.")
    disabled = bool(errors)
    speed = st.slider("Simulation speed (steps / sec)", 1, 25, 8, disabled=disabled)
    r1, r2 = st.columns(2)
    r3, r4 = st.columns(2)
    with r1:
        if st.button("Start", use_container_width=True, disabled=disabled):
            _safe(st, lambda: _ensure_live_engine(st, cfg))
            st.session_state.live_auto = True
    with r2:
        if st.button("Stop", use_container_width=True, disabled=disabled):
            st.session_state.live_auto = False
    with r3:
        if st.button("Step", use_container_width=True, disabled=disabled):
            st.session_state.live_auto = False
            _safe(st, lambda: _live_step(st, cfg))
    with r4:
        if st.button("Reset", use_container_width=True, disabled=disabled):
            st.session_state.live_auto = False
            _safe(st, lambda: _reset_live(st, cfg))

    engine: Optional[SimulationEngine] = st.session_state.live_engine
    if engine is not None:
        st.caption(
            f"Step {engine.step_index} / {cfg.n_steps}  ·  "
            f"price {engine.model.current_price:,.4f}  ·  "
            f"equity {fmt_money(engine.performance().equity)}"
        )
        if engine.step_index >= cfg.n_steps:
            st.session_state.live_auto = False
            st.info("Simulated path complete. Reset to start again.")
        elif st.session_state.live_auto:
            _safe(st, lambda: _live_step(st, cfg))
            time.sleep(1.0 / max(speed, 1))
            st.rerun()


def _single_run_panel(st: Any, cfg: DashboardConfig) -> None:
    result: Optional[SimulationResult] = st.session_state.single_result
    if result is None:
        st.info("Configure the sidebar and click **Run simulation** or use **Step**.")
        return
    st.plotly_chart(price_figure(result, cfg), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(equity_figure(result), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(drawdown_figure(result), use_container_width=True, config=PLOTLY_CONFIG)


def _metric_cards(st: Any, metrics: PerformanceMetrics) -> None:
    def _cls(value: float, *, invert: bool = False) -> str:
        if not np.isfinite(value) or abs(value) < 1e-15:
            return ""
        positive = value > 0
        if invert:
            positive = not positive
        return "pos" if positive else "neg"

    cards = [
        ("Total return", fmt_pct(metrics.total_return), _cls(metrics.total_return)),
        ("Ann. return", fmt_pct(metrics.annualized_return), _cls(metrics.annualized_return)),
        ("Sharpe", fmt_sharpe(metrics.sharpe_ratio), _cls(metrics.sharpe_ratio)),
        ("Max drawdown", fmt_pct(metrics.max_drawdown), _cls(metrics.max_drawdown, invert=True)),
        ("Final value", fmt_money(metrics.final_equity), ""),
        ("Trades", fmt_int(metrics.n_trades), ""),
        ("Costs", fmt_money(metrics.total_transaction_costs), "neg" if metrics.total_transaction_costs > 0 else ""),
    ]
    html = '<div class="metric-row">' + "".join(
        f'<div class="metric-card"><div class="lbl">{lab}</div>'
        f'<div class="val {cls}">{val}</div></div>'
        for lab, val, cls in cards
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _trade_table(st: Any, result: SimulationResult) -> None:
    with st.expander(f"Trade blotter  ({result.n_trades} fills)", expanded=False):
        frame = trades_frame(result)
        if frame.empty:
            st.caption("No fills on this path.")
            return
        shown = frame.copy()
        for col in ("market_price", "execution_price", "slippage", "commission", "realized_pnl"):
            shown[col] = shown[col].map(lambda x: f"{x:,.4f}")
        st.dataframe(shown, use_container_width=True, hide_index=True)


def _friction_section(st: Any) -> None:
    comparison = st.session_state.comparison
    if comparison is None:
        return
    st.markdown("##### Friction comparison")
    st.caption("Identical seed and strategy; only the execution model changes.")
    f_metrics = analyze_result(comparison.frictionless, periods_per_year=252.0)
    r_metrics = analyze_result(comparison.realistic, periods_per_year=252.0)
    table = pd.DataFrame(
        {
            "Frictionless": [
                fmt_pct(f_metrics.total_return),
                fmt_pct(f_metrics.annualized_return),
                fmt_sharpe(f_metrics.sharpe_ratio),
                fmt_pct(f_metrics.max_drawdown),
                fmt_money(f_metrics.final_equity),
                fmt_money(f_metrics.total_transaction_costs),
            ],
            "Realistic": [
                fmt_pct(r_metrics.total_return),
                fmt_pct(r_metrics.annualized_return),
                fmt_sharpe(r_metrics.sharpe_ratio),
                fmt_pct(r_metrics.max_drawdown),
                fmt_money(r_metrics.final_equity),
                fmt_money(r_metrics.total_transaction_costs),
            ],
        },
        index=[
            "Return",
            "Ann. return",
            "Sharpe",
            "Max drawdown",
            "Final equity",
            "Transaction costs",
        ],
    )
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.dataframe(table, use_container_width=True)
    with c2:
        st.plotly_chart(
            friction_bar_figure(f_metrics.final_equity, r_metrics.final_equity),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )


def _monte_carlo_section(st: Any, cfg: DashboardConfig, errors: list[str]) -> None:
    st.markdown("##### Monte Carlo analysis")
    st.caption(
        "Independent paths from the same model and strategy. "
        "Path generation is vectorized; each path still uses a fresh portfolio and strategy."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        n_paths = st.slider("Number of paths", 50, 1500, 200, 50)
    with c2:
        mc_steps = st.slider("Path length (steps)", 20, 756, cfg.n_steps, 1)
    with c3:
        mc_seed = st.number_input("Monte Carlo seed", min_value=0, value=cfg.seed, step=1)

    if st.button("Run Monte Carlo", disabled=bool(errors)):
        def _run() -> None:
            with st.spinner(f"Simulating {n_paths:,} paths × {mc_steps} steps…"):
                _execute_mc(st, cfg, n_paths=int(n_paths), n_steps=int(mc_steps), seed=int(mc_seed))

        _safe(st, _run)

    mc: Optional[MonteCarloResult] = st.session_state.mc_result
    if mc is None:
        return

    st.markdown(
        f"**{mc.n_paths:,} simulations** &nbsp;·&nbsp; {mc.n_steps} steps &nbsp;·&nbsp; seed {mc.seed}"
    )
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(
            histogram_figure(
                mc.final_equity,
                title="Final portfolio value",
                x_title="Equity",
                is_money=True,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
        st.plotly_chart(
            histogram_figure(
                mc.sharpe_ratio,
                title="Sharpe ratio",
                x_title="Sharpe",
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with h2:
        st.plotly_chart(
            histogram_figure(
                mc.total_return,
                title="Total return",
                x_title="Return",
                is_percent=True,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
        st.plotly_chart(
            histogram_figure(
                mc.max_drawdown,
                title="Maximum drawdown",
                x_title="Drawdown",
                is_percent=True,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    summary = pd.DataFrame(
        {
            "5th": [
                fmt_pct(mc.total_return.p5),
                fmt_sharpe(mc.sharpe_ratio.p5),
                fmt_pct(mc.max_drawdown.p5),
                fmt_money(mc.final_equity.p5),
            ],
            "Median": [
                fmt_pct(mc.total_return.median),
                fmt_sharpe(mc.sharpe_ratio.median),
                fmt_pct(mc.max_drawdown.median),
                fmt_money(mc.final_equity.median),
            ],
            "Mean": [
                fmt_pct(mc.total_return.mean),
                fmt_sharpe(mc.sharpe_ratio.mean),
                fmt_pct(mc.max_drawdown.mean),
                fmt_money(mc.final_equity.mean),
            ],
            "95th": [
                fmt_pct(mc.total_return.p95),
                fmt_sharpe(mc.sharpe_ratio.p95),
                fmt_pct(mc.max_drawdown.p95),
                fmt_money(mc.final_equity.p95),
            ],
        },
        index=["Return", "Sharpe", "Max drawdown", "Final equity"],
    )
    st.dataframe(summary, use_container_width=True)

    prices = st.session_state.mc_price_paths
    eq = mc.equity_paths
    p1, p2 = st.columns(2)
    with p1:
        if prices is not None:
            st.plotly_chart(
                sample_paths_figure(
                    prices, title="Sample price paths (subset)", n_show=30, seed=mc.seed
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
    with p2:
        if eq is not None:
            st.plotly_chart(
                sample_paths_figure(
                    eq,
                    title="Sample equity paths (subset)",
                    n_show=30,
                    seed=mc.seed,
                    is_money=True,
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )


def _execute_single(st: Any, cfg: DashboardConfig) -> None:
    engine = build_engine(cfg)
    result = engine.run(cfg.n_steps)
    st.session_state.single_result = result
    st.session_state.single_metrics = analyze_result(result, periods_per_year=252.0)
    st.session_state.live_engine = engine
    st.session_state.live_cfg_key = cfg
    st.session_state.live_auto = False
    if cfg.compare_friction:
        realistic = SimpleExecutionModel(config=realistic_execution_config(cfg))
        st.session_state.comparison = compare_friction(
            model_factory=model_factory(cfg),
            strategy_factory=strategy_factory(cfg),
            portfolio_factory=portfolio_factory(cfg),
            n_steps=cfg.n_steps,
            seed=cfg.seed,
            realistic_execution=realistic,
        )
    else:
        st.session_state.comparison = None


def _execute_mc(st: Any, cfg: DashboardConfig, *, n_paths: int, n_steps: int, seed: int) -> None:
    mc = build_monte_carlo(cfg, use_frictions=cfg.frictions)
    result = mc.run(n_paths=n_paths, n_steps=n_steps, seed=seed, store_equity_paths=True)
    st.session_state.mc_result = result
    st.session_state.mc_price_paths = model_factory(cfg)().generate_paths(
        n_paths, n_steps, rng=np.random.default_rng(seed), include_initial=True
    )


def _ensure_live_engine(st: Any, cfg: DashboardConfig) -> SimulationEngine:
    engine: Optional[SimulationEngine] = st.session_state.live_engine
    if engine is None or st.session_state.live_cfg_key != cfg:
        engine = build_engine(cfg)
        st.session_state.live_engine = engine
        st.session_state.live_cfg_key = cfg
        st.session_state.single_result = engine.results()
        st.session_state.single_metrics = analyze_result(
            engine.results(), periods_per_year=252.0
        )
    return engine


def _live_step(st: Any, cfg: DashboardConfig) -> None:
    engine = _ensure_live_engine(st, cfg)
    if engine.step_index >= cfg.n_steps:
        st.session_state.live_auto = False
        return
    engine.step()
    result = engine.results()
    st.session_state.single_result = result
    st.session_state.single_metrics = analyze_result(result, periods_per_year=252.0)


def _reset_live(st: Any, cfg: DashboardConfig) -> None:
    engine = build_engine(cfg)
    st.session_state.live_engine = engine
    st.session_state.live_cfg_key = cfg
    st.session_state.single_result = engine.results()
    st.session_state.single_metrics = analyze_result(engine.results(), periods_per_year=252.0)


def _safe(st: Any, fn) -> None:
    try:
        st.session_state.error = None
        fn()
    except ValueError as exc:
        st.session_state.error = str(exc)
        st.session_state.live_auto = False
    except Exception as exc:  # noqa: BLE001 — UI must not show stack traces
        st.session_state.error = f"Simulation failed: {exc}"
        st.session_state.live_auto = False


if __name__ == "__main__":
    if _in_streamlit():
        main()
    else:
        print("Launch with:  streamlit run app.py")
