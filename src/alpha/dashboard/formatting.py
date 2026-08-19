"""Display helpers for the dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alpha.execution.orders import Fill
from alpha.simulation.engine import SimulationResult


def fmt_pct(value: float, *, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}%}"


def fmt_sharpe(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if value == float("inf"):
        return "+∞"
    if value == float("-inf"):
        return "−∞"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def fmt_money(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def fmt_int(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{int(round(value)):,}"


def fmt_signed_pct(value: float, *, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}%}"


def fmt_signed_money(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def fmt_signed_num(value: float, *, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if window < 1 or x.size < window:
        return out
    c = np.cumsum(x)
    out[window - 1 :] = (c[window - 1 :] - np.concatenate(([0.0], c[: -window]))) / window
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if window < 2 or x.size < window:
        return out
    mean = rolling_mean(x, window)
    c2 = np.cumsum(x * x)
    mean_sq = np.full(x.shape, np.nan, dtype=float)
    mean_sq[window - 1 :] = (
        c2[window - 1 :] - np.concatenate(([0.0], c2[: -window]))
    ) / window
    var = mean_sq - mean * mean
    var = np.maximum(var, 0.0)
    out[window - 1 :] = np.sqrt(var[window - 1 :])
    return out


def trades_frame(result: SimulationResult) -> pd.DataFrame:
    rows = [_fill_row(fill) for fill in result.trades]
    if not rows:
        return pd.DataFrame(
            columns=[
                "step",
                "timestamp",
                "side",
                "quantity",
                "market_price",
                "execution_price",
                "spread_cost",
                "fixed_slippage_cost",
                "market_impact_cost",
                "market_impact_bps",
                "participation_rate",
                "average_daily_volume",
                "slippage",
                "commission",
                "realized_pnl",
            ]
        )
    return pd.DataFrame(rows)


def impact_trades_frame(result: SimulationResult) -> pd.DataFrame:
    """Trade blotter focused on market-impact audit fields."""
    rows = []
    for fill in result.trades:
        rows.append(
            {
                "step": fill.step,
                "side": fill.side.value,
                "quantity": fill.fill_quantity,
                "market_price": fill.market_price,
                "average_daily_volume": fill.average_daily_volume,
                "participation_rate": fill.participation_rate,
                "market_impact_bps": fill.market_impact_bps,
                "market_impact_cost": fill.market_impact_cost,
                "execution_price": fill.execution_price,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "step",
                "side",
                "quantity",
                "market_price",
                "average_daily_volume",
                "participation_rate",
                "market_impact_bps",
                "market_impact_cost",
                "execution_price",
            ]
        )
    return pd.DataFrame(rows)


def _fill_row(fill: Fill) -> dict[str, object]:
    return {
        "step": fill.step,
        "timestamp": round(fill.timestamp, 6),
        "side": fill.side.value,
        "quantity": fill.fill_quantity,
        "market_price": fill.market_price,
        "execution_price": fill.execution_price,
        "spread_cost": fill.spread_cost,
        "fixed_slippage_cost": fill.fixed_slippage_cost,
        "market_impact_cost": fill.market_impact_cost,
        "market_impact_bps": fill.market_impact_bps,
        "participation_rate": fill.participation_rate,
        "average_daily_volume": fill.average_daily_volume,
        "slippage": fill.slippage,
        "commission": fill.commission,
        "realized_pnl": fill.realized_pnl,
    }
