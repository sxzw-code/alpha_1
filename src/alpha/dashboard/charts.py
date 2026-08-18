"""Plotly figures for the Alpha dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from alpha.analytics.metrics import analyze_drawdown
from alpha.dashboard.factories import DashboardConfig
from alpha.dashboard.formatting import rolling_mean, rolling_std
from alpha.simulation.engine import SimulationResult
from alpha.simulation.monte_carlo import MetricDistribution, MonteCarloResult

PAPER = "#0b0f14"
PLOT = "#10161e"
GRID = "#243040"
TEXT = "#d5deea"
MUTED = "#8b9bb0"
ACCENT = "#c4a35a"
PRICE = "#7eb6ff"
FAST = "#e0c36e"
SLOW = "#c17cff"
BAND = "rgba(193, 124, 255, 0.12)"
EQUITY = "#3dd68c"
DRAWDOWN = "#f07178"
BUY = "#3dd68c"
SELL = "#ff7b72"
MEAN = "#c4a35a"


def _layout(fig: Figure, *, title: str, height: int, ytitle: str) -> Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.0),
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        font=dict(color=TEXT, family="IBM Plex Sans, Source Sans 3, sans-serif"),
        height=height,
        margin=dict(l=56, r=24, t=48, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        hovermode="x unified",
        xaxis=dict(
            title="Step",
            gridcolor=GRID,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            title=ytitle,
            gridcolor=GRID,
            zeroline=False,
            showline=False,
            tickformat=None,
        ),
    )
    return fig


def price_figure(
    result: SimulationResult,
    cfg: DashboardConfig,
) -> Figure:
    steps = result.steps
    prices = result.prices
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=prices,
            name="Price",
            line=dict(color=PRICE, width=1.7),
            hovertemplate="Step %{x}<br>Price %{y:.4f}<extra></extra>",
        )
    )
    if cfg.strategy_name == "Moving Average Crossover":
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=rolling_mean(prices, cfg.fast_window),
                name=f"MA {cfg.fast_window}",
                line=dict(color=FAST, width=1.2),
                hovertemplate="Fast MA %{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=rolling_mean(prices, cfg.slow_window),
                name=f"MA {cfg.slow_window}",
                line=dict(color=SLOW, width=1.2),
                hovertemplate="Slow MA %{y:.4f}<extra></extra>",
            )
        )
    elif cfg.strategy_name == "Mean Reversion":
        mean = rolling_mean(prices, cfg.lookback)
        std = rolling_std(prices, cfg.lookback)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                name=f"Mean {cfg.lookback}",
                line=dict(color=SLOW, width=1.2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean + cfg.entry_z * std,
                name=f"+{cfg.entry_z:.1f}σ",
                line=dict(color=SLOW, width=0.8, dash="dash"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean - cfg.entry_z * std,
                name=f"−{cfg.entry_z:.1f}σ",
                line=dict(color=SLOW, width=0.8, dash="dash"),
                fill="tonexty",
                fillcolor=BAND,
            )
        )

    buys_x, buys_y, sells_x, sells_y = [], [], [], []
    for fill in result.trades:
        if fill.side.value == "BUY":
            buys_x.append(fill.step)
            buys_y.append(fill.market_price)
        elif fill.side.value == "SELL":
            sells_x.append(fill.step)
            sells_y.append(fill.market_price)
    if buys_x:
        fig.add_trace(
            go.Scatter(
                x=buys_x,
                y=buys_y,
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=11, color=BUY, line=dict(width=0)),
                hovertemplate="BUY @ %{y:.4f}<extra></extra>",
            )
        )
    if sells_x:
        fig.add_trace(
            go.Scatter(
                x=sells_x,
                y=sells_y,
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=11, color=SELL, line=dict(width=0)),
                hovertemplate="SELL @ %{y:.4f}<extra></extra>",
            )
        )
    return _layout(fig, title="Simulated price and signals", height=360, ytitle="Price")


def equity_figure(result: SimulationResult) -> Figure:
    fig = go.Figure(
        go.Scatter(
            x=result.steps,
            y=result.equity,
            name="Equity",
            line=dict(color=EQUITY, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(61, 214, 140, 0.08)",
            hovertemplate="Step %{x}<br>Equity %{y:$,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _layout(fig, title="Portfolio equity", height=280, ytitle="Equity")


def drawdown_figure(result: SimulationResult) -> Figure:
    dd = analyze_drawdown(result.equity).series
    fig = go.Figure(
        go.Scatter(
            x=result.steps,
            y=dd,
            name="Drawdown",
            line=dict(color=DRAWDOWN, width=1.4),
            fill="tozeroy",
            fillcolor="rgba(240, 113, 120, 0.12)",
            hovertemplate="Step %{x}<br>Drawdown %{y:.2%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".1%")
    return _layout(fig, title="Drawdown from peak", height=240, ytitle="Drawdown")


def histogram_figure(
    dist: MetricDistribution,
    *,
    title: str,
    x_title: str,
    is_percent: bool = False,
    is_money: bool = False,
) -> Figure:
    values = dist.values[np.isfinite(dist.values)]
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=min(40, max(12, int(np.sqrt(max(values.size, 1))))),
            marker=dict(color="rgba(126, 182, 255, 0.75)", line=dict(width=0)),
            hovertemplate="Count %{y}<extra></extra>",
        )
    )
    for label, loc, color in (
        ("Mean", dist.mean, ACCENT),
        ("Median", dist.median, EQUITY),
    ):
        if np.isfinite(loc):
            fig.add_vline(
                x=loc,
                line_width=1.2,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_font_color=color,
                annotation_font_size=10,
            )
    fig.update_layout(
        bargap=0.04,
        showlegend=False,
    )
    fig = _layout(fig, title=title, height=280, ytitle="Paths")
    fig.update_xaxes(title=x_title)
    if is_percent:
        fig.update_xaxes(tickformat=".1%")
    if is_money:
        fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return fig


def sample_paths_figure(
    paths: np.ndarray,
    *,
    title: str,
    n_show: int,
    seed: int,
    is_money: bool = False,
) -> Figure:
    arr = np.asarray(paths, dtype=float)
    n = min(n_show, arr.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    fig = go.Figure()
    for i, row in enumerate(idx):
        fig.add_trace(
            go.Scatter(
                y=arr[row],
                mode="lines",
                line=dict(color=PRICE if not is_money else EQUITY, width=1.0),
                opacity=0.35,
                name=f"path {int(row)}",
                showlegend=False,
                hovertemplate="Step %{x}<br>%{y:.4f}<extra></extra>",
            )
        )
        if i == 0:
            fig.data[-1].opacity = 0.85
            fig.data[-1].line.width = 1.6
    fig = _layout(fig, title=title, height=300, ytitle="")
    if is_money:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def friction_bar_figure(
    frictionless_equity: float,
    realistic_equity: float,
) -> Figure:
    fig = go.Figure(
        go.Bar(
            x=["Frictionless", "Realistic"],
            y=[frictionless_equity, realistic_equity],
            marker_color=[EQUITY, ACCENT],
            hovertemplate="%{x}<br>%{y:$,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(showlegend=False)
    return _layout(fig, title="Final equity: frictionless vs realistic", height=260, ytitle="Equity")
