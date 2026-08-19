"""Performance analytics from equity curves and completed trades.

Formulas (let :math:`E_t` be portfolio equity, :math:`P` periods per year,
:math:`r_f` the annual risk-free rate):

**Period returns**

    r_t = E_t / E_{t-1} - 1

**Total return**

    R = E_T / E_0 - 1

**Annualized return (CAGR)**

    (1 + R)^{P / n} - 1

where ``n`` is the number of return periods (``len(equity) - 1``).
Undefined (NaN) when ``E_T <= 0``.

**Annualized volatility**

    σ_ann = std(r_t, ddof=1) * sqrt(P)

With fewer than two returns the sample standard deviation is undefined;
this module treats that as zero volatility.

**Sharpe ratio** (standard, excess period returns)::

    r_f^{period} = r_f / P
    Sharpe = sqrt(P) * mean(r_t - r_f^{period}) / std(r_t, ddof=1)

This uses the **arithmetic** mean of simple period returns, not CAGR,
so the numerator is consistent with the volatility estimator. ``r_f`` is
an annual rate in the same time unit as ``P`` (default 252).

When ``std(r_t) = 0``:

- excess mean ≈ 0 → Sharpe = 0
- excess mean > 0 → +∞
- excess mean < 0 → −∞

**Drawdown**

    peak_t = max(E_0, …, E_t)
    DD_t = E_t / peak_t - 1
    max DD = min_t DD_t

**Drawdown duration** is the longest consecutive run of strictly negative
drawdown, counted in observation steps.

**Trade statistics** use *completed round trips*: realized P&L accumulated
from the first fill that opens a position until inventory returns to flat.
Open positions at the end of the sample are excluded from win rate / average
trade P&L. ``n_trades`` counts every fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from alpha.execution.orders import Fill

if TYPE_CHECKING:
    from alpha.simulation.engine import SimulationResult

_ZERO_VOL = 1e-15


@dataclass(frozen=True, slots=True)
class DrawdownAnalysis:
    """Full drawdown path plus summary statistics."""

    series: np.ndarray
    max_drawdown: float
    max_drawdown_duration: int


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Scalar strategy statistics for one simulated path."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    n_trades: int
    n_completed_trades: int
    win_rate: float
    average_trade_pnl: float
    total_transaction_costs: float
    final_equity: float
    periods_per_year: float
    risk_free_rate: float


def period_returns(equity: np.ndarray) -> np.ndarray:
    """Simple returns ``E[1:]/E[:-1] - 1``. Empty if ``len(equity) < 2``."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2:
        return np.array([], dtype=float)
    prev = eq[:-1]
    if np.any(prev == 0.0):
        raise ValueError("equity contains zeros; cannot compute simple returns")
    return eq[1:] / prev - 1.0


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Drawdown from the running peak; shape matches ``equity``.

    Supports 1-d curves ``(T,)`` or stacked curves ``(n_paths, T)``.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return eq.copy()
    if eq.ndim == 1:
        peak = np.maximum.accumulate(eq)
        peak = np.where(peak == 0.0, np.nan, peak)
        return eq / peak - 1.0
    if eq.ndim != 2:
        raise ValueError(f"equity must be 1-d or 2-d, got shape {eq.shape}")
    peak = np.maximum.accumulate(eq, axis=1)
    peak = np.where(peak == 0.0, np.nan, peak)
    return eq / peak - 1.0


def max_drawdown(equity: np.ndarray) -> float:
    """Minimum of the drawdown series (a non-positive number)."""
    dd = drawdown_series(equity)
    if dd.size == 0:
        return 0.0
    value = np.nanmin(dd)
    return 0.0 if np.isnan(value) else float(value)


def max_drawdown_duration(drawdown: np.ndarray) -> int:
    """Longest consecutive run of strictly negative drawdown, in steps."""
    dd = np.asarray(drawdown, dtype=float)
    if dd.ndim != 1:
        raise ValueError("max_drawdown_duration expects a 1-d drawdown series")
    in_dd = np.isfinite(dd) & (dd < 0.0)
    if not np.any(in_dd):
        return 0
    padded = np.concatenate([[False], in_dd, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return int((ends - starts).max())


def analyze_drawdown(equity: np.ndarray) -> DrawdownAnalysis:
    """Return the drawdown series, peak-to-trough depth, and duration."""
    eq = np.asarray(equity, dtype=float)
    if eq.ndim != 1:
        raise ValueError("analyze_drawdown expects a 1-d equity curve")
    series = drawdown_series(eq)
    mdd = 0.0 if series.size == 0 else float(np.nanmin(series))
    if np.isnan(mdd):
        mdd = 0.0
    return DrawdownAnalysis(
        series=series,
        max_drawdown=mdd,
        max_drawdown_duration=max_drawdown_duration(series),
    )


def _cagr(total_return: float, n_periods: int, periods_per_year: float) -> float:
    if n_periods <= 0:
        return 0.0
    wealth = 1.0 + total_return
    if wealth <= 0.0:
        return float("nan")
    return float(wealth ** (periods_per_year / n_periods) - 1.0)


def _ratio(numer: float, denom: float) -> float:
    """Sharpe-style ratio with a safe zero-vol convention."""
    if not np.isfinite(numer) or not np.isfinite(denom):
        return float("nan")
    if denom <= _ZERO_VOL:
        if abs(numer) <= _ZERO_VOL:
            return 0.0
        return float("inf") if numer > 0.0 else float("-inf")
    return float(numer / denom)


def _sharpe_from_returns(
    rets: np.ndarray,
    *,
    periods_per_year: float,
    risk_free_rate: float,
) -> tuple[float, float]:
    """Return ``(annualized_volatility, sharpe)`` from 1-d period returns.

    Sharpe = sqrt(P) * mean(r - r_f/P) / std(r, ddof=1)
    """
    n = int(np.asarray(rets).size)
    if n == 0:
        return 0.0, 0.0
    rf_period = risk_free_rate / periods_per_year
    excess = np.asarray(rets, dtype=float) - rf_period
    if n < 2:
        ann_vol = 0.0
        return ann_vol, _ratio(float(np.mean(excess)) * periods_per_year, ann_vol)
    ann_vol = float(np.std(rets, ddof=1)) * float(np.sqrt(periods_per_year))
    ann_excess = float(np.mean(excess)) * periods_per_year
    return ann_vol, _ratio(ann_excess, ann_vol)


def _round_trip_pnls(fills: Sequence[Fill]) -> np.ndarray:
    """Realized P&L of cycles that start and end flat."""
    qty = 0.0
    cycle = 0.0
    completed: list[float] = []
    for fill in fills:
        cycle += float(fill.realized_pnl)
        qty += float(fill.order.signed_quantity())
        if abs(qty) <= 1e-12:
            completed.append(cycle)
            cycle = 0.0
            qty = 0.0
    return np.asarray(completed, dtype=float)


def analyze_equity(
    equity: np.ndarray,
    *,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float, float, DrawdownAnalysis]:
    """Vectorized equity-curve statistics (no trade data).

    Returns
    -------
    total_return, annualized_return, annualized_volatility, sharpe_ratio,
    drawdown analysis
    """
    if periods_per_year <= 0.0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    eq = np.asarray(equity, dtype=float)
    if eq.ndim != 1:
        raise ValueError("analyze_equity expects a 1-d equity curve")
    if eq.size == 0:
        dd = DrawdownAnalysis(
            series=np.array([], dtype=float),
            max_drawdown=0.0,
            max_drawdown_duration=0,
        )
        return 0.0, 0.0, 0.0, 0.0, dd

    total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] != 0.0 else float("nan")
    rets = period_returns(eq)
    n = int(rets.size)
    ann_ret = _cagr(total_return, n, periods_per_year)
    ann_vol, sharpe = _sharpe_from_returns(
        rets, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )
    return total_return, ann_ret, ann_vol, sharpe, analyze_drawdown(eq)


def trade_statistics(fills: Sequence[Fill]) -> tuple[int, int, float, float]:
    """Return ``(n_trades, n_completed, win_rate, average_trade_pnl)``."""
    n_trades = len(fills)
    round_trips = _round_trip_pnls(fills)
    n_completed = int(round_trips.size)
    if n_completed == 0:
        return n_trades, 0, float("nan"), float("nan")
    return (
        n_trades,
        n_completed,
        float(np.mean(round_trips > 0.0)),
        float(np.mean(round_trips)),
    )


def analyze(
    equity: np.ndarray,
    *,
    fills: Sequence[Fill] = (),
    total_transaction_costs: float = 0.0,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute :class:`PerformanceMetrics` from an equity curve and fills."""
    total_return, ann_ret, ann_vol, sharpe, dd = analyze_equity(
        equity,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    eq = np.asarray(equity, dtype=float)
    final_equity = float(eq[-1]) if eq.size else float("nan")
    round_trips_n_trades, n_completed, win_rate, avg_pnl = trade_statistics(fills)
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=dd.max_drawdown,
        max_drawdown_duration=dd.max_drawdown_duration,
        n_trades=round_trips_n_trades,
        n_completed_trades=n_completed,
        win_rate=win_rate,
        average_trade_pnl=avg_pnl,
        total_transaction_costs=float(total_transaction_costs),
        final_equity=final_equity,
        periods_per_year=float(periods_per_year),
        risk_free_rate=float(risk_free_rate),
    )


def analyze_result(
    result: SimulationResult,
    *,
    periods_per_year: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Analyze a :class:`~alpha.simulation.engine.SimulationResult`."""
    ppy = periods_per_year
    if ppy is None:
        if result.n_steps > 0 and result.timestamps.size > 1:
            dt = float(result.timestamps[1] - result.timestamps[0])
            ppy = 1.0 / dt if dt > 0.0 else 252.0
        else:
            ppy = 252.0
    return analyze(
        result.equity,
        fills=result.trades,
        total_transaction_costs=result.total_transaction_costs,
        periods_per_year=ppy,
        risk_free_rate=risk_free_rate,
    )


def stacked_equity_metrics(
    equity_paths: np.ndarray,
    *,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> dict[str, np.ndarray]:
    """Vectorized metrics for many equity curves of equal length.

    Parameters
    ----------
    equity_paths:
        Array of shape ``(n_paths, T)``.

    Returns
    -------
    dict with keys ``final_equity``, ``total_return``, ``annualized_return``,
    ``annualized_volatility``, ``sharpe_ratio``, ``max_drawdown``.
    """
    if periods_per_year <= 0.0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    eq = np.asarray(equity_paths, dtype=float)
    if eq.ndim != 2:
        raise ValueError(f"equity_paths must be 2-d, got shape {eq.shape}")
    n_paths, t = eq.shape
    if n_paths == 0:
        empty = np.array([], dtype=float)
        return {k: empty.copy() for k in (
            "final_equity",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        )}

    start = eq[:, 0]
    final = eq[:, -1]
    total_return = np.where(start != 0.0, final / start - 1.0, np.nan)
    n = t - 1
    if n <= 0:
        ann_ret = np.zeros(n_paths, dtype=float)
        ann_vol = np.zeros(n_paths, dtype=float)
        sharpe = np.zeros(n_paths, dtype=float)
    else:
        wealth = 1.0 + total_return
        with np.errstate(invalid="ignore", divide="ignore"):
            ann_ret = np.where(wealth > 0.0, wealth ** (periods_per_year / n) - 1.0, np.nan)
        prev = eq[:, :-1]
        rets = eq[:, 1:] / prev - 1.0
        rf_period = risk_free_rate / periods_per_year
        excess = rets - rf_period
        if n < 2:
            ann_vol = np.zeros(n_paths, dtype=float)
            ann_excess = np.mean(excess, axis=1) * periods_per_year
            sharpe = np.array(
                [_ratio(float(a), 0.0) for a in ann_excess], dtype=float
            )
        else:
            ann_vol = np.std(rets, axis=1, ddof=1) * np.sqrt(periods_per_year)
            ann_excess = np.mean(excess, axis=1) * periods_per_year
            zero_vol = ann_vol <= _ZERO_VOL
            sharpe = np.empty(n_paths, dtype=float)
            sharpe[~zero_vol] = ann_excess[~zero_vol] / ann_vol[~zero_vol]
            near_zero = np.abs(ann_excess) <= _ZERO_VOL
            sharpe[zero_vol & near_zero] = 0.0
            sharpe[zero_vol & ~near_zero & (ann_excess > 0.0)] = np.inf
            sharpe[zero_vol & ~near_zero & (ann_excess < 0.0)] = -np.inf
        sharpe = np.where(np.isnan(ann_ret), np.nan, sharpe)

    dd = drawdown_series(eq)
    mdd = np.nanmin(dd, axis=1)
    mdd = np.where(np.isnan(mdd), 0.0, mdd)

    return {
        "final_equity": final,
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
    }


@dataclass(frozen=True, slots=True)
class TransactionCostBreakdown:
    """Decomposed execution costs across fills.

    Implicit costs (spread, fixed slippage, market impact) are already
    reflected in execution prices; they are reported here for TCA only.
    Commission is an explicit cash fee.
    """

    commission: float
    spread_cost: float
    fixed_slippage_cost: float
    market_impact_cost: float
    total_implicit: float
    total: float


def decompose_transaction_costs(fills: Sequence[Fill]) -> TransactionCostBreakdown:
    """Sum commission and implicit execution costs from fills.

    Verifies ``total == commission + spread + fixed_slippage + market_impact``
    for each fill's stored components (within floating tolerance).
    """
    commission = 0.0
    spread = 0.0
    fixed_slip = 0.0
    impact = 0.0
    for fill in fills:
        sc = float(fill.spread_cost)
        fc = float(fill.fixed_slippage_cost)
        ic = float(fill.market_impact_cost)
        commission += float(fill.commission)
        spread += sc
        fixed_slip += fc
        impact += ic
        expected_implicit = sc + fc + ic
        if abs(float(fill.slippage) - expected_implicit) > 1e-9 * max(1.0, expected_implicit):
            raise ValueError(
                "Fill slippage field does not match component sum; "
                f"slippage={fill.slippage}, components={expected_implicit}"
            )
        expected_total = float(fill.commission) + expected_implicit
        if abs(float(fill.total_transaction_cost) - expected_total) > 1e-9 * max(
            1.0, expected_total
        ):
            raise ValueError(
                "Fill total_transaction_cost does not match decomposition; "
                f"total={fill.total_transaction_cost}, expected={expected_total}"
            )
    total_implicit = spread + fixed_slip + impact
    return TransactionCostBreakdown(
        commission=float(commission),
        spread_cost=float(spread),
        fixed_slippage_cost=float(fixed_slip),
        market_impact_cost=float(impact),
        total_implicit=float(total_implicit),
        total=float(commission + total_implicit),
    )
