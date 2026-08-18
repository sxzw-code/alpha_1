"""Abstract price model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True, slots=True)
class MarketState:
    """Snapshot of the market at a single observation.

    Designed so strategies depend on this view rather than on a concrete
    price model. Multi-asset extensions can widen ``price`` to a mapping
    or array without changing the strategy contract shape.
    """

    step: int
    timestamp: float
    price: float
    asset_id: str = "ASSET"


class PriceModel(ABC):
    """Abstract stochastic price (or factor) process.

    Supports:
    - single-path stepping for live / interactive simulation
    - full-path and multi-path generation for Monte Carlo analysis

    RNG is always injected (or created from a seed) so runs are reproducible
    without global NumPy state.
    """

    def __init__(
        self,
        *,
        dt: float,
        asset_id: str = "ASSET",
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if rng is not None and seed is not None:
            raise ValueError("Provide at most one of seed or rng")
        self._dt = float(dt)
        self._asset_id = asset_id
        self._seed = seed
        self._rng: Generator = rng if rng is not None else np.random.default_rng(seed)
        self._step: int = 0

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def step_index(self) -> int:
        """Number of observations advanced from the initial condition."""
        return self._step

    @property
    def timestamp(self) -> float:
        """Elapsed simulated time: ``step_index * dt``."""
        return self._step * self._dt

    @property
    @abstractmethod
    def current_price(self) -> float:
        """Current level of the simulated process."""

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, rng: Optional[Generator] = None) -> None:
        """Reset internal state to the initial condition.

        If ``seed`` or ``rng`` is provided, replace the RNG; otherwise keep
        the existing generator (which continues its stream).
        """

    def _replace_rng(
        self, *, seed: Optional[int] = None, rng: Optional[Generator] = None
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("Provide at most one of seed or rng")
        if rng is not None:
            self._rng = rng
            self._seed = None
        elif seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed

    @abstractmethod
    def step(self, rng: Optional[Generator] = None) -> MarketState:
        """Advance the process by one observation and return the new state."""

    @abstractmethod
    def generate_path(
        self,
        n_steps: int,
        *,
        rng: Optional[Generator] = None,
        include_initial: bool = True,
    ) -> np.ndarray:
        """Generate a single path of length ``n_steps`` (plus optional S0).

        Does not mutate the model's live stepping state.
        """

    @abstractmethod
    def generate_paths(
        self,
        n_paths: int,
        n_steps: int,
        *,
        rng: Optional[Generator] = None,
        include_initial: bool = True,
    ) -> np.ndarray:
        """Generate many paths; shape ``(n_paths, n_steps [+ 1])``.

        Does not mutate the model's live stepping state.
        """

    def state(self) -> MarketState:
        """Current market snapshot without advancing time."""
        return MarketState(
            step=self._step,
            timestamp=self.timestamp,
            price=float(self.current_price),
            asset_id=self._asset_id,
        )

    def _validate_path_args(self, n_paths: int, n_steps: int) -> None:
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
