"""Alpha interactive dashboard (Streamlit).

Launch from the repository root::

    streamlit run app.py
"""

from __future__ import annotations

import html
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from alpha.analytics.metrics import PerformanceMetrics, analyze_result, decompose_transaction_costs
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
    effective_n_steps,
    execution_config,
    load_historical_data,
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
    fmt_signed_money,
    fmt_signed_num,
    fmt_signed_pct,
    impact_trades_frame,
    trades_frame,
)
from alpha.execution.sensitivity import order_size_sensitivity
from alpha.execution.model import SimpleExecutionModel
from alpha.simulation.engine import SimulationEngine, SimulationResult, compare_friction
from alpha.simulation.monte_carlo import MonteCarloResult
from alpha.strategies import MovingAverageCrossover
from alpha.strategies.base import BuyAndHoldStrategy

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Source Sans 3", sans-serif;
}
.stApp { background: #070b10; }
.block-container { padding: 0.55rem 1.35rem 1.6rem; max-width: 1480px; }
[data-testid="stHeader"] { background: transparent; }
header { visibility: hidden; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

section[data-testid="stSidebar"] {
  background: #0b1016;
  border-right: 1px solid #1c2733;
}
section[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.28rem; }

.side-kicker {
  font-size: 0.68rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #c4a35a;
  margin: 0.65rem 0 0.15rem;
  font-weight: 600;
}
.field-label {
  font-size: 0.72rem;
  color: #7d8b9a;
  margin: 0.15rem 0 0.05rem;
}

.term-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 0.45rem 1.1rem;
  border: 1px solid #1c2733;
  background: linear-gradient(180deg, #10161e 0%, #0b1016 100%);
  padding: 0.5rem 0.85rem;
  margin-bottom: 0.65rem;
}
.brand { display: flex; align-items: baseline; gap: 0.55rem; padding: 0.15rem 0 0.35rem; }
.brand .logo {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  color: #c4a35a; font-size: 1.25rem; font-weight: 500;
}
.brand .name {
  font-size: 1.22rem; font-weight: 600; letter-spacing: 0.22em; color: #e8edf4;
}
.brand .tag {
  color: #7d8b9a; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
}
.head-meta {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  color: #9aabbc; font-size: 0.76rem; letter-spacing: 0.02em;
  padding: 0.45rem 0;
}
.head-status {
  display: flex; align-items: center; justify-content: flex-end;
  gap: 0.75rem; padding: 0.28rem 0;
}
.head-status .mono {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.78rem; color: #d7e0ea;
}
hr.head-rule { border: none; border-top: 1px solid #1c2733; margin: 0 0 0.55rem; }
.pill {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  border: 1px solid #1c2733; padding: 0.18rem 0.5rem;
}
.pill.ok { color: #3dd68c; border-color: #1e4a36; background: #0d1f16; }
.pill.live { color: #c4a35a; border-color: #5a4a24; background: #1a150c; }
.pill.idle { color: #7d8b9a; }
.pill.bad { color: #f07178; border-color: #5a2a2e; background: #1a0e10; }

.metric-row {
  display: grid;
  gap: 0.5rem;
  margin: 0.15rem 0 0.7rem;
}
.metric-row.n5 { grid-template-columns: repeat(5, 1fr); }
.metric-row.n4 { grid-template-columns: repeat(4, 1fr); }
@media (max-width: 1100px) {
  .metric-row.n5, .metric-row.n4 { grid-template-columns: repeat(2, 1fr); }
}
.metric-card {
  background: #10161e;
  border: 1px solid #1c2733;
  padding: 0.55rem 0.7rem 0.6rem;
}
.metric-card .lbl {
  color: #7d8b9a; font-size: 0.64rem; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 0.18rem;
}
.metric-card .val {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 1.12rem; color: #e8edf4; font-weight: 500;
}
.metric-card .sub {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem; color: #7d8b9a; margin-top: 0.12rem;
}
.metric-card .val.pos { color: #3dd68c; }
.metric-card .val.neg { color: #f07178; }

.sec {
  display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 1px solid #1c2733; margin: 0.35rem 0 0.55rem; padding-bottom: 0.28rem;
}
.sec-title {
  font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: #c4a35a; font-weight: 600;
}
.sec-right {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.72rem; color: #7d8b9a;
}

.term-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.term-table th, .term-table td {
  border-bottom: 1px solid #1c2733; padding: 0.42rem 0.55rem; text-align: right;
}
.term-table th:first-child, .term-table td:first-child {
  text-align: left; color: #7d8b9a; letter-spacing: 0.06em; text-transform: uppercase;
  font-size: 0.7rem;
}
.term-table th {
  color: #7d8b9a; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; font-size: 0.66rem;
}
.term-table td {
  font-family: "IBM Plex Mono", ui-monospace, monospace; color: #d7e0ea;
}
.term-table td.pos { color: #3dd68c; }
.term-table td.neg { color: #f07178; }

.note {
  color: #7d8b9a; font-size: 0.78rem; margin: 0 0 0.4rem;
}

button[kind="primary"] { font-weight: 600; }

.stTabs [data-baseweb="tab-list"] {
  gap: 0.15rem;
  border-bottom: 1px solid #1c2733;
}
.stTabs [data-baseweb="tab"] {
  font-size: 0.74rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 0.35rem 0.7rem;
  color: #7d8b9a;
}
.stTabs [aria-selected="true"] { color: #e8edf4; }
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
        page_title="Alpha — Research Terminal",
        page_icon="α",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state(st)
    cfg = _sidebar(st)
    errors = cfg.validation_errors()
    if errors:
        for msg in errors:
            st.error(msg)
    elif not st.session_state.demo_ran:
        st.session_state.demo_ran = True
        _safe(st, lambda: _execute_single(st, cfg))

    result: Optional[SimulationResult] = st.session_state.get("single_result")
    metrics: Optional[PerformanceMetrics] = st.session_state.get("single_metrics")
    engine: Optional[SimulationEngine] = st.session_state.get("live_engine")

    _header(st, cfg, errors, result=result, engine=engine)
    _toolbar(st, cfg, errors)
    if st.session_state.error:
        st.error(st.session_state.error)

    if result is not None and metrics is not None:
        _metric_cards(st, metrics)
        st.plotly_chart(price_figure(result, cfg), width="stretch", config=PLOTLY_CONFIG)
        st.plotly_chart(equity_figure(result), width="stretch", config=PLOTLY_CONFIG)
    else:
        st.info("Set market and strategy controls, then run a simulation.")

    perf_tab, trades_tab, mc_tab, exec_tab = st.tabs(
        ["Performance", "Trades", "Monte Carlo", "Execution Analysis"]
    )
    with perf_tab:
        _performance_tab(st, result, metrics)
    with trades_tab:
        _trades_tab(st, result)
    with mc_tab:
        _monte_carlo_section(st, cfg, errors)
    with exec_tab:
        _execution_analysis_section(st, cfg, result)


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
        "demo_ran": False,
        "historical_data": None,
        "historical_load_error": None,
        "historical_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _label(st: Any, text: str) -> None:
    st.markdown(f'<p class="field-label">{html.escape(text)}</p>', unsafe_allow_html=True)


def _sidebar(st: Any) -> DashboardConfig:
    with st.sidebar:
        st.markdown('<div class="side-kicker">Market</div>', unsafe_allow_html=True)
        _label(st, "Market source")
        market_source = st.radio(
            "Market source",
            ("Synthetic", "Historical"),
            horizontal=True,
            label_visibility="collapsed",
        )
        model_name = "Geometric Brownian Motion"
        s0, mu, sigma = 100.0, 0.08, 0.20
        x0, ou_mu, theta, ou_sigma = 100.0, 100.0, 1.0, 5.0
        n_steps = 252
        data_provider = "Yahoo Finance"
        symbol = "AAPL"
        hist_start = "2020-01-01"
        hist_end = "2024-12-31"
        price_basis = "adjusted"
        csv_path = "tests/fixtures/sample_ohlcv.csv"
        liquidity_source = "historical"
        volatility_source = "historical"
        adv_window = 20
        vol_window = 20

        if market_source == "Synthetic":
            _label(st, "Model")
            model_name = st.selectbox(
                "Market model",
                ("Geometric Brownian Motion", "Ornstein–Uhlenbeck"),
                label_visibility="collapsed",
                help="GBM: dS = μS dt + σS dW. OU: dX = θ(μ−X) dt + σ dW.",
            )
            if model_name == "Geometric Brownian Motion":
                _label(st, "Initial price S₀")
                s0 = st.number_input(
                    "Initial price S₀", min_value=0.01, value=100.0, step=1.0,
                    label_visibility="collapsed",
                )
                c1, c2 = st.columns(2)
                with c1:
                    _label(st, "Drift μ (ann.)")
                    mu = st.number_input(
                        "Annual drift μ", value=0.08, step=0.01, format="%.3f",
                        label_visibility="collapsed",
                        help="Annualized drift. Dashboard Δt = 1/252, so μ is per year.",
                    )
                with c2:
                    _label(st, "Vol σ (ann.)")
                    sigma = st.number_input(
                        "Annual volatility σ", min_value=0.0, value=0.20, step=0.01,
                        format="%.3f", label_visibility="collapsed",
                        help="Annualized GBM vol. Must use the same time unit as 1/Δt.",
                    )
                x0, ou_mu, theta, ou_sigma = 100.0, 100.0, 1.0, 5.0
            else:
                s0, mu, sigma = 100.0, 0.08, 0.20
                _label(st, "Initial value X₀")
                x0 = st.number_input(
                    "Initial value X₀", min_value=0.01, value=100.0, step=1.0,
                    label_visibility="collapsed",
                )
                _label(st, "Long-run mean μ")
                ou_mu = st.number_input(
                    "Long-run mean μ", value=100.0, step=1.0, label_visibility="collapsed"
                )
                c1, c2 = st.columns(2)
                with c1:
                    _label(st, "θ (per year)")
                    theta = st.number_input(
                        "Mean-reversion speed θ (per year)", min_value=0.0, value=1.0,
                        step=0.1, label_visibility="collapsed",
                        help="With Δt=1/252, θ is per year. Half-life ≈ ln(2)/θ years.",
                    )
                with c2:
                    _label(st, "Level vol σ")
                    ou_sigma = st.number_input(
                        "OU level vol σ", min_value=0.0, value=5.0, step=0.1,
                        label_visibility="collapsed",
                        help="Diffusion of the level X, not a return volatility.",
                    )
        else:
            _label(st, "Data provider")
            data_provider = st.selectbox(
                "Data provider",
                ("Yahoo Finance", "CSV file"),
                label_visibility="collapsed",
            )
            _label(st, "Symbol")
            symbol = st.text_input("Symbol", value="AAPL", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                _label(st, "Start date")
                hist_start = st.text_input(
                    "Start date", value="2020-01-01", label_visibility="collapsed"
                )
            with c2:
                _label(st, "End date")
                hist_end = st.text_input(
                    "End date", value="2024-12-31", label_visibility="collapsed"
                )
            _label(st, "Price basis")
            price_basis = st.selectbox(
                "Price basis",
                ("adjusted", "raw"),
                label_visibility="collapsed",
                help="Use adjusted close for returns when available.",
            )
            if data_provider == "CSV file":
                _label(st, "CSV path")
                csv_path = st.text_input(
                    "CSV path",
                    value="tests/fixtures/sample_ohlcv.csv",
                    label_visibility="collapsed",
                )
            if st.button("Load data", width="stretch"):
                try:
                    from alpha.dashboard.factories import load_historical_data

                    tmp_cfg = DashboardConfig(
                        market_source="Historical",
                        model_name="Geometric Brownian Motion",
                        s0=100.0, mu=0.08, sigma=0.20, x0=100.0, ou_mu=100.0,
                        theta=1.0, ou_sigma=5.0, n_steps=252, seed=42,
                        strategy_name="Moving Average Crossover",
                        fast_window=10, slow_window=30, lookback=20,
                        entry_z=2.0, exit_z=0.5, trade_quantity=100.0,
                        initial_capital=100_000.0, frictions=True,
                        commission_bps=5.0, slippage_bps=2.0, spread_bps=1.0,
                        market_impact_enabled=False, impact_coefficient=0.1,
                        average_daily_volume=1_000_000.0,
                        manual_annual_volatility=None, compare_friction=False,
                        data_provider=data_provider,  # type: ignore[arg-type]
                        symbol=symbol, hist_start=hist_start, hist_end=hist_end,
                        price_basis=price_basis,  # type: ignore[arg-type]
                        csv_path=csv_path,
                    )
                    st.session_state.historical_data = load_historical_data(tmp_cfg)
                    st.session_state.historical_load_error = None
                except Exception as exc:  # noqa: BLE001
                    st.session_state.historical_load_error = str(exc)
                    st.session_state.historical_data = None

            if st.session_state.get("historical_load_error"):
                st.error(st.session_state.historical_load_error)
            elif st.session_state.get("historical_data") is not None:
                df = st.session_state.historical_data
                st.caption(
                    f"{symbol} · {df['timestamp'].iloc[0].date()} → "
                    f"{df['timestamp'].iloc[-1].date()} · {len(df):,} bars"
                )

        if market_source == "Synthetic":
            c1, c2 = st.columns(2)
            with c1:
                _label(st, "Steps")
                n_steps = st.slider(
                    "Simulation steps", 20, 1000, 252, 1, label_visibility="collapsed",
                    help="252 steps ≈ one trading year at Δt = 1/252.",
                )
            with c2:
                _label(st, "Seed")
                seed = st.number_input(
                    "Random seed", min_value=0, value=42, step=1, label_visibility="collapsed"
                )
        else:
            seed = 42
            if st.session_state.get("historical_data") is not None:
                n_steps = max(len(st.session_state.historical_data) - 1, 1)
            else:
                n_steps = 252

        st.markdown('<div class="side-kicker">Strategy</div>', unsafe_allow_html=True)
        _label(st, "Rule")
        strategy_name = st.selectbox(
            "Strategy",
            ("Moving Average Crossover", "Mean Reversion", "Buy and Hold"),
            label_visibility="collapsed",
        )
        _label(st, "Trade quantity")
        trade_quantity = st.number_input(
            "Trade quantity", min_value=0.01, value=100.0, step=1.0,
            label_visibility="collapsed",
            help="Shares per entry (~10% of $100k at $100). Long-only.",
        )
        fast_window, slow_window = 10, 30
        lookback, entry_z, exit_z = 20, 2.0, 0.5
        if strategy_name == "Moving Average Crossover":
            c1, c2 = st.columns(2)
            with c1:
                _label(st, "Fast window")
                fast_window = int(
                    st.number_input(
                        "Fast window", min_value=1, value=10, step=1,
                        label_visibility="collapsed",
                    )
                )
            with c2:
                _label(st, "Slow window")
                slow_window = int(
                    st.number_input(
                        "Slow window", min_value=2, value=30, step=1,
                        label_visibility="collapsed",
                    )
                )
        elif strategy_name == "Mean Reversion":
            _label(st, "Lookback")
            lookback = int(
                st.number_input(
                    "Lookback", min_value=2, value=20, step=1, label_visibility="collapsed"
                )
            )
            c1, c2 = st.columns(2)
            with c1:
                _label(st, "Entry z")
                entry_z = float(
                    st.number_input(
                        "Entry z-score", min_value=0.05, value=2.0, step=0.1,
                        label_visibility="collapsed",
                    )
                )
            with c2:
                _label(st, "Exit z")
                exit_z = float(
                    st.number_input(
                        "Exit z-score", min_value=0.0, value=0.5, step=0.1,
                        label_visibility="collapsed",
                    )
                )

        st.markdown('<div class="side-kicker">Execution</div>', unsafe_allow_html=True)
        _label(st, "Initial capital")
        initial_capital = st.number_input(
            "Initial capital", min_value=1.0, value=100_000.0, step=1000.0,
            format="%.0f", label_visibility="collapsed",
        )
        frictions = st.toggle(
            "Trading frictions",
            value=True,
            help="When off, the single-path run uses zero commission and slippage.",
        )
        c1, c2 = st.columns(2)
        with c1:
            _label(st, "Commission bps")
            commission_bps = st.number_input(
                "Commission (bps of notional)", min_value=0.0, value=5.0, step=0.5,
                label_visibility="collapsed",
                help="Cash fee = bps × executed notional. Example: 5 bps = 0.05%.",
            )
        with c2:
            _label(st, "Slippage bps")
            slippage_bps = st.number_input(
                "Slippage (bps)", min_value=0.0, value=2.0, step=0.5,
                label_visibility="collapsed",
                help="BUY pays more than mid; SELL receives less.",
            )
        with st.expander("Advanced"):
            spread_bps = st.number_input(
                "Half-spread (bps)", min_value=0.0, value=1.0, step=0.5,
                help="Simple bid/ask proxy, added in the same adverse direction as slippage.",
            )
            compare_friction = st.checkbox(
                "Frictionless vs costs (same path)",
                value=True,
            )
        st.markdown('<div class="side-kicker">Market impact</div>', unsafe_allow_html=True)
        market_impact_enabled = st.toggle(
            "Market impact",
            value=False,
            help="Square-root impact: larger orders vs ADV receive worse fills.",
        )
        average_daily_volume = st.number_input(
            "ADV (shares / day)",
            min_value=1.0,
            value=1_000_000.0,
            step=50_000.0,
            format="%.0f",
            help="Manual ADV fallback. Historical mode can use rolling volume.",
        )
        impact_coefficient = st.number_input(
            "Impact coefficient η",
            min_value=0.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Scales I = η σ_daily sqrt(Q/V). Empirical approximation, not a LOB model.",
        )
        if market_source == "Historical":
            liquidity_source = st.radio(
                "Liquidity source",
                ("historical", "manual"),
                horizontal=True,
                format_func=lambda x: "Historical rolling ADV" if x == "historical" else "Manual ADV",
            )
            volatility_source = st.radio(
                "Volatility source",
                ("historical", "manual"),
                horizontal=True,
                format_func=lambda x: "Historical rolling σ" if x == "historical" else "Manual σ",
            )
            vol_source = "Manual" if volatility_source == "manual" else "Historical"
        else:
            vol_source = st.radio(
                "Volatility source",
                ("Model (GBM σ)", "Manual"),
                horizontal=True,
                help="OU uses manual vol when impact is enabled.",
            )
        manual_annual_volatility: Optional[float] = None
        if vol_source in ("Manual", "manual") or (
            market_source == "Historical" and volatility_source == "manual"
        ):
            manual_annual_volatility = float(
                st.number_input(
                    "Manual annual σ",
                    min_value=0.0,
                    value=0.20,
                    step=0.01,
                    format="%.3f",
                )
            )
        st.caption("Simulated paths. Not a live market. Not investment advice.")

    return DashboardConfig(
        market_source=market_source,  # type: ignore[arg-type]
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
        market_impact_enabled=bool(market_impact_enabled),
        impact_coefficient=float(impact_coefficient),
        average_daily_volume=float(average_daily_volume),
        manual_annual_volatility=manual_annual_volatility,
        compare_friction=bool(compare_friction),
        data_provider=data_provider,  # type: ignore[arg-type]
        symbol=str(symbol),
        hist_start=str(hist_start),
        hist_end=str(hist_end),
        price_basis=price_basis,  # type: ignore[arg-type]
        csv_path=str(csv_path),
        liquidity_source=liquidity_source,  # type: ignore[arg-type]
        volatility_source=volatility_source if market_source == "Historical" else "model",  # type: ignore[arg-type]
        adv_window=int(adv_window),
        vol_window=int(vol_window),
    )


def _strategy_short(cfg: DashboardConfig) -> str:
    if cfg.strategy_name == "Moving Average Crossover":
        return f"MA {cfg.fast_window}/{cfg.slow_window}"
    if cfg.strategy_name == "Mean Reversion":
        return f"MR z={cfg.entry_z:g}/{cfg.exit_z:g}"
    return "Buy & Hold"


def _model_short(cfg: DashboardConfig) -> str:
    if cfg.is_historical:
        return f"HIST {cfg.symbol}"
    if cfg.model_name == "Geometric Brownian Motion":
        return "GBM"
    return "OU"


def _header(
    st: Any,
    cfg: DashboardConfig,
    errors: list[str],
    *,
    result: Optional[SimulationResult],
    engine: Optional[SimulationEngine],
) -> None:
    if errors:
        mode, pill = "INVALID", "bad"
    elif st.session_state.live_auto:
        mode, pill = "STEPPING", "live"
    elif engine is not None and engine.step_index >= cfg.n_steps:
        mode, pill = "COMPLETE", "ok"
    elif result is not None:
        mode, pill = "READY", "ok"
    else:
        mode, pill = "IDLE", "idle"

    if engine is not None:
        step_txt = f"{engine.step_index}/{cfg.n_steps}"
        last_txt = f"LAST {engine.model.current_price:,.2f}"
        eq_txt = f"EQ {fmt_money(engine.performance().equity)}"
    elif result is not None and result.prices.size:
        step_txt = f"{int(result.n_steps)}/{cfg.n_steps}"
        last_txt = f"LAST {float(result.prices[-1]):,.2f}"
        eq_txt = f"EQ {fmt_money(float(result.equity[-1]))}"
    else:
        step_txt, last_txt, eq_txt = f"0/{cfg.n_steps}", "LAST —", "EQ —"

    meta = (
        f"{_model_short(cfg)}  ·  {_strategy_short(cfg)}  ·  "
        f"seed {cfg.seed}  ·  {cfg.n_steps} steps  ·  "
        f"{'frictions on' if cfg.frictions else 'frictionless'}"
    )
    left, mid, right = st.columns([1.15, 2.15, 2.35], gap="small")
    with left:
        st.markdown(
            '<div class="brand"><span class="logo">α</span>'
            '<span class="name">ALPHA</span>'
            '<span class="tag">research terminal</span></div>',
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown(
            f'<div class="head-meta">{html.escape(meta)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="head-status">'
            f'<span class="pill {pill}">{html.escape(mode)}</span>'
            f'<span class="mono">{html.escape(step_txt)}</span>'
            f'<span class="mono">{html.escape(last_txt)}</span>'
            f'<span class="mono">{html.escape(eq_txt)}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown('<hr class="head-rule">', unsafe_allow_html=True)


def _toolbar(st: Any, cfg: DashboardConfig, errors: list[str]) -> None:
    disabled = bool(errors)
    c_run, c_live, c_spd = st.columns([1.15, 2.4, 1.55], gap="small")
    with c_run:
        if st.button("Run simulation", type="primary", width="stretch", disabled=disabled):
            _safe(st, lambda: _execute_single(st, cfg))
    with c_live:
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            if st.button("Start", width="stretch", disabled=disabled):
                _safe(st, lambda: _ensure_live_engine(st, cfg))
                st.session_state.live_auto = True
        with c2:
            if st.button("Stop", width="stretch", disabled=disabled):
                st.session_state.live_auto = False
        with c3:
            if st.button("Step", width="stretch", disabled=disabled):
                st.session_state.live_auto = False
                _safe(st, lambda: _live_step(st, cfg))
        with c4:
            if st.button("Reset", width="stretch", disabled=disabled):
                st.session_state.live_auto = False
                _safe(st, lambda: _reset_live(st, cfg))
    with c_spd:
        speed = st.slider("Speed (steps/s)", 1, 25, 8, disabled=disabled)

    engine: Optional[SimulationEngine] = st.session_state.live_engine
    if engine is not None and engine.step_index >= cfg.n_steps:
        st.session_state.live_auto = False
    elif st.session_state.live_auto and not disabled:
        _safe(st, lambda: _live_step(st, cfg))
        time.sleep(1.0 / max(int(speed), 1))
        st.rerun()


def _tone(value: float, *, invert: bool = False) -> str:
    if not np.isfinite(value) or abs(value) < 1e-15:
        return ""
    positive = value > 0
    if invert:
        positive = not positive
    return "pos" if positive else "neg"


def _metric_html(cards: list[tuple[str, str, str, str]], *, n: int) -> str:
    parts = [f'<div class="metric-row n{n}">']
    for lab, val, cls, sub in cards:
        extra = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
        parts.append(
            f'<div class="metric-card"><div class="lbl">{html.escape(lab)}</div>'
            f'<div class="val {cls}">{html.escape(val)}</div>{extra}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _metric_cards(st: Any, metrics: PerformanceMetrics) -> None:
    cards = [
        ("Total return", fmt_pct(metrics.total_return), _tone(metrics.total_return), ""),
        ("Sharpe", fmt_sharpe(metrics.sharpe_ratio), _tone(metrics.sharpe_ratio), ""),
        (
            "Max DD",
            fmt_pct(metrics.max_drawdown),
            _tone(metrics.max_drawdown, invert=True),
            "",
        ),
        ("Final equity", fmt_money(metrics.final_equity), "", f"{fmt_int(metrics.n_trades)} fills"),
        (
            "Costs",
            fmt_money(metrics.total_transaction_costs),
            "neg" if metrics.total_transaction_costs > 0 else "",
            "",
        ),
    ]
    st.markdown(_metric_html(cards, n=5), unsafe_allow_html=True)


def _section(st: Any, title: str, right: str = "") -> None:
    st.markdown(
        f'<div class="sec"><span class="sec-title">{html.escape(title)}</span>'
        f'<span class="sec-right">{html.escape(right)}</span></div>',
        unsafe_allow_html=True,
    )


def _performance_tab(
    st: Any,
    result: Optional[SimulationResult],
    metrics: Optional[PerformanceMetrics],
) -> None:
    if result is None or metrics is None:
        st.caption("Run a simulation to populate path statistics.")
        return
    _section(st, "Path statistics", f"{result.n_steps} steps  ·  {result.n_trades} fills")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.plotly_chart(drawdown_figure(result), width="stretch", config=PLOTLY_CONFIG)
    with right:
        win = fmt_pct(metrics.win_rate) if np.isfinite(metrics.win_rate) else "n/a"
        avg = fmt_money(metrics.average_trade_pnl) if np.isfinite(metrics.average_trade_pnl) else "n/a"
        rows = [
            ("Ann. return", fmt_pct(metrics.annualized_return)),
            ("Ann. volatility", fmt_pct(metrics.annualized_volatility)),
            ("Sharpe", fmt_sharpe(metrics.sharpe_ratio)),
            ("Max drawdown", fmt_pct(metrics.max_drawdown)),
            ("DD duration", f"{metrics.max_drawdown_duration} steps"),
            ("Fills", fmt_int(metrics.n_trades)),
            ("Completed round trips", fmt_int(metrics.n_completed_trades)),
            ("Win rate", win),
            ("Avg round-trip P&L", avg),
            ("Transaction costs", fmt_money(metrics.total_transaction_costs)),
        ]
        body = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows
        )
        st.markdown(
            f'<table class="term-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
            f"<tbody>{body}</tbody></table>",
            unsafe_allow_html=True,
        )


def _trades_tab(st: Any, result: Optional[SimulationResult]) -> None:
    if result is None:
        st.caption("Run a simulation to populate the blotter.")
        return
    _section(st, "Trade blotter", f"{result.n_trades} fills")
    frame = trades_frame(result)
    if frame.empty:
        st.caption("No fills on this path.")
        return
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        height=min(420, 52 + 28 * max(len(frame), 1)),
        column_config={
            "step": st.column_config.NumberColumn("Step", format="%d"),
            "timestamp": st.column_config.NumberColumn("Time", format="%.4f"),
            "side": st.column_config.TextColumn("Side"),
            "quantity": st.column_config.NumberColumn("Qty", format="%.2f"),
            "market_price": st.column_config.NumberColumn("Mid", format="%.4f"),
            "execution_price": st.column_config.NumberColumn("Fill", format="%.4f"),
            "slippage": st.column_config.NumberColumn("Slippage $", format="$%.4f"),
            "commission": st.column_config.NumberColumn("Commission", format="$%.4f"),
            "realized_pnl": st.column_config.NumberColumn("Realized P&L", format="$%.2f"),
        },
    )


def _execution_analysis_section(
    st: Any,
    cfg: DashboardConfig,
    result: Optional[SimulationResult],
) -> None:
    if result is not None and result.n_trades > 0:
        _section(st, "Cost decomposition", f"{result.n_trades} fills")
        breakdown = decompose_transaction_costs(result.trades)
        rows = [
            ("Commission", fmt_money(breakdown.commission)),
            ("Spread", fmt_money(breakdown.spread_cost)),
            ("Fixed slippage", fmt_money(breakdown.fixed_slippage_cost)),
            ("Market impact", fmt_money(breakdown.market_impact_cost)),
            ("Total execution costs", fmt_money(breakdown.total)),
        ]
        body = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows
        )
        st.markdown(
            f'<table class="term-table"><thead><tr><th>Component</th><th>Amount</th></tr></thead>'
            f"<tbody>{body}</tbody></table>",
            unsafe_allow_html=True,
        )

        impact_frame = impact_trades_frame(result)
        if not impact_frame.empty and impact_frame["market_impact_cost"].sum() > 0.0:
            _section(st, "Market impact per trade")
            shown = impact_frame.copy()
            shown["participation_rate"] = shown["participation_rate"].map(
                lambda x: f"{x:.4%}" if np.isfinite(x) else "n/a"
            )
            shown["market_impact_bps"] = shown["market_impact_bps"].map(
                lambda x: f"{x:.2f}" if np.isfinite(x) else "n/a"
            )
            st.dataframe(shown, width="stretch", hide_index=True)

    comparison = st.session_state.comparison
    if comparison is None:
        st.caption(
            "Enable “Frictionless vs costs” under Execution → Advanced, then run a simulation."
        )
    else:
        _section(st, "Friction comparison", "identical seed and strategy")
        f_metrics = analyze_result(comparison.frictionless, periods_per_year=252.0)
        r_metrics = analyze_result(comparison.realistic, periods_per_year=252.0)

        specs: list[tuple[str, float, float, str]] = [
            ("Return", f_metrics.total_return, r_metrics.total_return, "pct"),
            ("Sharpe", f_metrics.sharpe_ratio, r_metrics.sharpe_ratio, "sharpe"),
            ("Max DD", f_metrics.max_drawdown, r_metrics.max_drawdown, "dd"),
            ("Final equity", f_metrics.final_equity, r_metrics.final_equity, "money"),
            (
                "Costs",
                f_metrics.total_transaction_costs,
                r_metrics.total_transaction_costs,
                "cost",
            ),
        ]
        rows = []
        for name, left, right, kind in specs:
            delta = right - left
            if kind == "pct":
                l_s, r_s, d_s = fmt_pct(left), fmt_pct(right), fmt_signed_pct(delta)
                cls = _tone(delta)
            elif kind == "sharpe":
                l_s, r_s, d_s = fmt_sharpe(left), fmt_sharpe(right), fmt_signed_num(delta)
                cls = _tone(delta)
            elif kind == "dd":
                l_s, r_s, d_s = fmt_pct(left), fmt_pct(right), fmt_signed_pct(delta)
                cls = _tone(delta, invert=True)
            elif kind == "cost":
                l_s, r_s, d_s = fmt_money(left), fmt_money(right), fmt_signed_money(delta)
                cls = _tone(delta, invert=True)
            else:
                l_s, r_s, d_s = fmt_money(left), fmt_money(right), fmt_signed_money(delta)
                cls = _tone(delta)
            rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>{html.escape(l_s)}</td><td>{html.escape(r_s)}</td>"
                f'<td class="{cls}">{html.escape(d_s)}</td></tr>'
            )
        table = (
            '<table class="term-table"><thead><tr>'
            "<th>Metric</th><th>Frictionless</th><th>With costs</th><th>Difference</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
        c1, c2 = st.columns([1.45, 1], gap="large")
        with c1:
            st.markdown(table, unsafe_allow_html=True)
            st.markdown(
                '<p class="note">Difference = with costs − frictionless. '
                "Parametric TCA including optional square-root market impact.</p>",
                unsafe_allow_html=True,
            )
        with c2:
            st.plotly_chart(
                friction_bar_figure(f_metrics.final_equity, r_metrics.final_equity),
                width="stretch",
                config=PLOTLY_CONFIG,
            )

    if cfg.market_impact_enabled and cfg.strategy_name != "Mean Reversion":
        _section(st, "Order size sensitivity", "same path  ·  varying quantity")
        quantities = [
            max(1.0, cfg.trade_quantity / 10.0),
            cfg.trade_quantity,
            cfg.trade_quantity * 10.0,
            min(cfg.average_daily_volume * 0.01, cfg.trade_quantity * 100.0),
        ]
        quantities = sorted({float(q) for q in quantities if q > 0.0})
        exec_cfg = execution_config(cfg)
        if cfg.strategy_name == "Moving Average Crossover":
            strat_factory = lambda q: MovingAverageCrossover(
                cfg.fast_window, cfg.slow_window, float(q)
            )
        else:
            strat_factory = lambda q: BuyAndHoldStrategy(quantity=float(q))

        try:
            sens = order_size_sensitivity(
                model_factory=model_factory(cfg),
                strategy_factory=strat_factory,
                portfolio_factory=portfolio_factory(cfg),
                quantities=quantities,
                n_steps=min(cfg.n_steps, 252),
                seed=cfg.seed,
                execution_config=exec_cfg,
            )
            sens_display = sens.copy()
            sens_display["participation_rate"] = sens_display["participation_rate"].map(
                lambda x: f"{x:.4%}"
            )
            sens_display["total_return"] = sens_display["total_return"].map(fmt_pct)
            sens_display["market_impact_cost"] = sens_display["market_impact_cost"].map(fmt_money)
            sens_display["total_execution_cost"] = sens_display["total_execution_cost"].map(
                fmt_money
            )
            st.dataframe(sens_display, width="stretch", hide_index=True)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Sensitivity run skipped: {exc}")


def _monte_carlo_section(st: Any, cfg: DashboardConfig, errors: list[str]) -> None:
    _section(st, "Monte Carlo", "independent paths  ·  fresh book each run")
    st.markdown(
        '<p class="note">Path generation is vectorized; each path still uses a new '
        "strategy, portfolio, and execution model. Percentiles are sample quantiles, "
        "not a parametric VaR model.</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.85])
    with c1:
        n_paths = st.slider("Paths", 50, 1500, 200, 50)
    with c2:
        mc_steps = st.slider("Path length", 20, 756, cfg.n_steps, 1)
    with c3:
        mc_seed = st.number_input("MC seed", min_value=0, value=cfg.seed, step=1)
    with c4:
        st.write("")
        run_mc = st.button("Run Monte Carlo", type="primary", width="stretch", disabled=bool(errors))

    if run_mc:
        def _run() -> None:
            with st.spinner(f"Simulating {n_paths:,} paths × {mc_steps} steps…"):
                _execute_mc(st, cfg, n_paths=int(n_paths), n_steps=int(mc_steps), seed=int(mc_seed))

        _safe(st, _run)

    mc: Optional[MonteCarloResult] = st.session_state.mc_result
    if mc is None:
        st.caption("Run Monte Carlo to populate expected return, tails, and distributions.")
        return

    _section(
        st,
        "Distribution summary",
        f"{mc.n_paths:,} paths  ·  {mc.n_steps} steps  ·  seed {mc.seed}",
    )
    cards = [
        (
            "Expected return",
            fmt_pct(mc.total_return.mean),
            _tone(mc.total_return.mean),
            "mean path return",
        ),
        (
            "Median",
            fmt_pct(mc.total_return.median),
            _tone(mc.total_return.median),
            "median path return",
        ),
        (
            "5% percentile",
            fmt_pct(mc.total_return.p5),
            _tone(mc.total_return.p5),
            "left-tail return (VaR-style)",
        ),
        (
            "Worst DD",
            fmt_pct(mc.max_drawdown.min),
            _tone(mc.max_drawdown.min, invert=True),
            "min of path max drawdowns",
        ),
    ]
    st.markdown(_metric_html(cards, n=4), unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        st.plotly_chart(
            histogram_figure(
                mc.total_return,
                title="RETURN DISTRIBUTION",
                x_title="Total return",
                is_percent=True,
                mark_p5=True,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    with r1c2:
        st.plotly_chart(
            histogram_figure(
                mc.final_equity,
                title="FINAL EQUITY DISTRIBUTION",
                x_title="Equity",
                is_money=True,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1:
        st.plotly_chart(
            histogram_figure(
                mc.sharpe_ratio,
                title="SHARPE DISTRIBUTION",
                x_title="Sharpe",
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    with r2c2:
        st.plotly_chart(
            histogram_figure(
                mc.max_drawdown,
                title="DRAWDOWN DISTRIBUTION",
                x_title="Max drawdown",
                is_percent=True,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

    with st.expander("Sample paths"):
        prices = st.session_state.mc_price_paths
        eq = mc.equity_paths
        p1, p2 = st.columns(2)
        with p1:
            if prices is not None:
                st.plotly_chart(
                    sample_paths_figure(
                        prices, title="SAMPLE PRICE PATHS", n_show=30, seed=mc.seed
                    ),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )
        with p2:
            if eq is not None:
                st.plotly_chart(
                    sample_paths_figure(
                        eq,
                        title="SAMPLE EQUITY PATHS",
                        n_show=30,
                        seed=mc.seed,
                        is_money=True,
                    ),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )


def _execute_single(st: Any, cfg: DashboardConfig) -> None:
    hist_data = st.session_state.get("historical_data")
    if cfg.is_historical and hist_data is None:
        hist_data = load_historical_data(cfg)
        st.session_state.historical_data = hist_data
    n_steps = effective_n_steps(cfg, hist_data)
    engine = build_engine(cfg, historical_data=hist_data)
    result = engine.run(n_steps)
    st.session_state.single_result = result
    st.session_state.single_metrics = analyze_result(result, periods_per_year=252.0)
    st.session_state.live_engine = engine
    st.session_state.live_cfg_key = cfg
    st.session_state.live_auto = False
    if cfg.compare_friction and not cfg.is_historical:
        realistic = SimpleExecutionModel(config=realistic_execution_config(cfg))
        st.session_state.comparison = compare_friction(
            model_factory=model_factory(cfg),
            strategy_factory=strategy_factory(cfg),
            portfolio_factory=portfolio_factory(cfg),
            n_steps=n_steps,
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
