"""Performance analytics."""

from alpha.analytics.metrics import (
    DrawdownAnalysis,
    PerformanceMetrics,
    analyze,
    analyze_drawdown,
    analyze_equity,
    analyze_result,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    period_returns,
    stacked_equity_metrics,
    trade_statistics,
)

__all__ = [
    "DrawdownAnalysis",
    "PerformanceMetrics",
    "analyze",
    "analyze_drawdown",
    "analyze_equity",
    "analyze_result",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "period_returns",
    "stacked_equity_metrics",
    "trade_statistics",
]
