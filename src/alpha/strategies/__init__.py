"""Trading strategy interfaces and implementations."""

from alpha.strategies.base import BuyAndHoldStrategy, HoldStrategy, Strategy
from alpha.strategies.mean_reversion import MeanReversionStrategy
from alpha.strategies.moving_average import MovingAverageCrossover

__all__ = [
    "Strategy",
    "HoldStrategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossover",
    "MeanReversionStrategy",
]
