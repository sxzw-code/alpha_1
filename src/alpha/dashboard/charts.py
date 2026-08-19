"""Plotly figures for the Alpha dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from alpha.analytics.metrics import analyze_drawdown
from alpha.dashboard.factories import DashboardConfig
from alpha.dashboard.formatting import rolling_mean, rolling_std
from alpha.simulation.engine import SimulationResult
from alpha.simulation.monte_carlo import MetricDistribution

PAPER = "#070b10"
PLOT = "#0d1218"
GRID = "#1c2733"
TEXT = "#d7e0ea"
MUTED = "#7d8b9a"
ACCENT = "#c4a35a"
PRICE = "#6cb6ff"
FAST = "#e0c36e"
SLOW = "#c17cff"
BAND = "rgba(193, 124, 255, 0.10)"
EQUITY = "#3dd68c"
DRAWDOWN = "#f07178"
BUY = "#3dd68c"
SELL = "#ff6b6b"
HOVER_BG = "#151c24"


def _layout(
    fig: Figure,
    *,
    title: str,
    height: int,
    ytitle: str,
    uirevision: str,
) -> Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=12, color=MUTED, family="IBM Plex Sans, sans-serif"),
            x=0.0,
            pad=dict(t=0, b=0),
        ),
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        font=dict(color=TEXT, family="IBM Plex Sans, Source Sans 3, sans-serif", size=11),
        height=height,
        margin=dict(l=58, r=16, t=36, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=1.0,
            xanchor="right",
            font=dict(size=10, color=MUTED),
            bgcolor="rgba(7,11,16,0.0)",
            itemsizing="constant",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor=GRID,
            font=dict(size=11, color=TEXT, family="IBM Plex Mono, ui-monospace, monospace"),
        ),
        uirevision=uirevision,
        xaxis=dict(
            title=dict(text="Step", font=dict(size=10, color=MUTED)),
            gridcolor=GRID,
            gridwidth=1,
            zeroline=False,
            showline=False,
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor=MUTED,
            spikedash="dot",
            tickfont=dict(size=10, color=MUTED),
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(size=10, color=MUTED)),
            gridcolor=GRID,
            zeroline=False,
            showline=False,
            tickfont=dict(size=10, color=MUTED),
            showspikes=True,
            spikemode="toaxis",
            spikethickness=1,
            spikecolor=MUTED,
            spikedash="dot",
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
            line=dict(color=PRICE, width=1.65),
            hovertemplate="Price %{y:.4f}<extra></extra>",
        )
    )
    if cfg.strategy_name == "Moving Average Crossover":
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=rolling_mean(prices, cfg.fast_window),
                name=f"MA {cfg.fast_window}",
                line=dict(color=FAST, width=1.15),
                hovertemplate="MA fast %{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=rolling_mean(prices, cfg.slow_window),
                name=f"MA {cfg.slow_window}",
                line=dict(color=SLOW, width=1.15),
                hovertemplate="MA slow %{y:.4f}<extra></extra>",
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
                line=dict(color=SLOW, width=1.15, dash="dot"),
                hovertemplate="Mean %{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean + cfg.entry_z * std,
                name=f"+{cfg.entry_z:.1f}σ",
                line=dict(color=SLOW, width=0.7, dash="dash"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean - cfg.entry_z * std,
                name=f"−{cfg.entry_z:.1f}σ",
                line=dict(color=SLOW, width=0.7, dash="dash"),
                fill="tonexty",
                fillcolor=BAND,
                hoverinfo="skip",
            )
        )

    buys_x, buys_y, buys_cd = [], [], []
    sells_x, sells_y, sells_cd = [], [], []
    for fill in result.trades:
        row = (
            fill.fill_quantity,
            fill.execution_price,
            fill.commission,
            fill.slippage,
        )
        if fill.side.value == "BUY":
            buys_x.append(fill.step)
            buys_y.append(fill.market_price)
            buys_cd.append(row)
        elif fill.side.value == "SELL":
            sells_x.append(fill.step)
            sells_y.append(fill.market_price)
            sells_cd.append(row)

    hover_trade = (
        "%{fullData.name}<br>Qty %{customdata[0]:.2f}"
        "<br>Mid %{y:.4f}<br>Fill %{customdata[1]:.4f}"
        "<br>Commission $%{customdata[2]:.2f}"
        "<br>Slippage $%{customdata[3]:.2f}<extra></extra>"
    )
    if buys_x:
        fig.add_trace(
            go.Scatter(
                x=buys_x,
                y=buys_y,
                mode="markers",
                name="BUY",
                customdata=np.asarray(buys_cd, dtype=float),
                marker=dict(
                    symbol="triangle-up",
                    size=13,
                    color=BUY,
                    line=dict(width=1.4, color="#04140c"),
                ),
                hovertemplate=hover_trade,
            )
        )
    if sells_x:
        fig.add_trace(
            go.Scatter(
                x=sells_x,
                y=sells_y,
                mode="markers",
                name="SELL",
                customdata=np.asarray(sells_cd, dtype=float),
                marker=dict(
                    symbol="triangle-down",
                    size=13,
                    color=SELL,
                    line=dict(width=1.4, color="#1a0808"),
                ),
                hovertemplate=hover_trade,
            )
        )
    return _layout(
        fig,
        title="PRICE  +  TRADES",
        height=372,
        ytitle="Price",
        uirevision="alpha-price",
    )


def equity_figure(result: SimulationResult) -> Figure:
    eq = np.asarray(result.equity, dtype=float)
    steps = result.steps
    start = float(eq[0]) if eq.size else 0.0
    peak = np.maximum.accumulate(eq) if eq.size else eq

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=peak,
            name="Peak",
            line=dict(color="rgba(240, 113, 120, 0.35)", width=1, dash="dot"),
            hovertemplate="Peak %{y:$,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=eq,
            name="Equity",
            line=dict(color=EQUITY, width=1.8),
            fill="tonexty",
            fillcolor="rgba(240, 113, 120, 0.13)",
            hovertemplate="Equity %{y:$,.2f}<extra></extra>",
        )
    )
    if eq.size:
        fig.add_hline(
            y=start,
            line_width=1,
            line_dash="dash",
            line_color=ACCENT,
            annotation_text="Start",
            annotation_font_color=ACCENT,
            annotation_font_size=10,
            annotation_position="bottom right",
        )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _layout(
        fig,
        title="PORTFOLIO EQUITY",
        height=248,
        ytitle="Equity",
        uirevision="alpha-equity",
    )


def drawdown_figure(result: SimulationResult) -> Figure:
    dd = analyze_drawdown(result.equity).series
    fig = go.Figure(
        go.Scatter(
            x=result.steps,
            y=dd,
            name="Drawdown",
            line=dict(color=DRAWDOWN, width=1.35),
            fill="tozeroy",
            fillcolor="rgba(240, 113, 120, 0.16)",
            hovertemplate="Drawdown %{y:.2%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".1%")
    fig.add_hline(y=0.0, line_width=1, line_color=GRID)
    return _layout(
        fig,
        title="DRAWDOWN FROM PEAK",
        height=250,
        ytitle="Drawdown",
        uirevision="alpha-dd",
    )


def histogram_figure(
    dist: MetricDistribution,
    *,
    title: str,
    x_title: str,
    is_percent: bool = False,
    is_money: bool = False,
    mark_p5: bool = False,
) -> Figure:
    values = dist.values[np.isfinite(dist.values)]
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=min(40, max(12, int(np.sqrt(max(values.size, 1))))),
            marker=dict(
                color="rgba(108, 182, 255, 0.72)",
                line=dict(width=0.4, color=PLOT),
            ),
            hovertemplate="Count %{y}<extra></extra>",
        )
    )
    marks = [("Mean", dist.mean, ACCENT, "dash"), ("Median", dist.median, EQUITY, "dot")]
    if mark_p5:
        marks.append(("5th pct", dist.p5, DRAWDOWN, "dash"))
    for label, loc, color, dash in marks:
        if np.isfinite(loc):
            fig.add_vline(
                x=loc,
                line_width=1.15,
                line_dash=dash,
                line_color=color,
                annotation_text=label,
                annotation_font_color=color,
                annotation_font_size=10,
                annotation_position="top",
            )
    fig.update_layout(bargap=0.05, showlegend=False)
    fig = _layout(fig, title=title, height=246, ytitle="Paths", uirevision=f"alpha-h-{title}")
    fig.update_xaxes(title=dict(text=x_title, font=dict(size=10, color=MUTED)))
    fig.update_layout(hovermode="closest")
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
    color = EQUITY if is_money else PRICE
    for i, row in enumerate(idx):
        fig.add_trace(
            go.Scatter(
                y=arr[row],
                mode="lines",
                line=dict(color=color, width=1.55 if i == 0 else 1.0),
                opacity=0.88 if i == 0 else 0.28,
                name=f"path {int(row)}",
                showlegend=False,
                hovertemplate="Step %{x}<br>%{y:.4f}<extra></extra>",
            )
        )
    fig = _layout(fig, title=title, height=236, ytitle="", uirevision=f"alpha-paths-{title}")
    fig.update_layout(hovermode="closest")
    if is_money:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        fig.update_traces(hovertemplate="Step %{x}<br>%{y:$,.2f}<extra></extra>")
    return fig


def friction_bar_figure(
    frictionless_equity: float,
    realistic_equity: float,
) -> Figure:
    fig = go.Figure(
        go.Bar(
            x=["Frictionless", "With costs"],
            y=[frictionless_equity, realistic_equity],
            marker_color=[EQUITY, ACCENT],
            width=0.45,
            hovertemplate="%{x}<br>%{y:$,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(showlegend=False)
    fig = _layout(
        fig,
        title="FINAL EQUITY",
        height=236,
        ytitle="Equity",
        uirevision="alpha-friction",
    )
    fig.update_xaxes(title=None)
    fig.update_layout(hovermode="closest")
    return fig
