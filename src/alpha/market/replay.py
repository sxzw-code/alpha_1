"""Replay a pre-generated price path without consuming RNG."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from alpha.market.base import MarketState, PriceModel


class PathReplayModel(PriceModel):
    """Deterministic walker over a 1-d price series that includes t=0.

    Used by Monte Carlo to apply strategies to vectorized GBM/OU paths
    without re-drawing shocks. ``step()`` ignores any passed generator.
    """

    def __init__(
        self,
        prices: np.ndarray,
        dt: float,
        *,
        asset_id: str = "ASSET",
    ) -> None:
        super().__init__(dt=dt, asset_id=asset_id, seed=None, rng=np.random.default_rng(0))
        path = np.asarray(prices, dtype=float).reshape(-1)
        if path.size < 2:
            raise ValueError("prices must include at least an initial point and one step")
        if np.any(path <= 0.0):
            raise ValueError("replay prices must be strictly positive")
        self._prices = path
        self._price = float(path[0])
        self._max_step = int(path.size - 1)

    @property
    def current_price(self) -> float:
        return self._price

    @property
    def n_steps_available(self) -> int:
        return self._max_step

    def reset(self, *, seed: Optional[int] = None, rng: Optional[Generator] = None) -> None:
        del seed, rng
        self._step = 0
        self._price = float(self._prices[0])

    def step(self, rng: Optional[Generator] = None) -> MarketState:
        del rng
        if self._step >= self._max_step:
            raise IndexError(
                f"PathReplayModel exhausted after {self._max_step} steps"
            )
        self._step += 1
        self._price = float(self._prices[self._step])
        return self.state()

    def generate_path(
        self,
        n_steps: int,
        *,
        rng: Optional[Generator] = None,
        include_initial: bool = True,
    ) -> np.ndarray:
        del rng
        if n_steps > self._max_step:
            raise ValueError(
                f"n_steps {n_steps} exceeds stored path length {self._max_step}"
            )
        if include_initial:
            return self._prices[: n_steps + 1].copy()
        return self._prices[1 : n_steps + 1].copy()

    def generate_paths(
        self,
        n_paths: int,
        n_steps: int,
        *,
        rng: Optional[Generator] = None,
        include_initial: bool = True,
    ) -> np.ndarray:
        self._validate_path_args(n_paths, n_steps)
        one = self.generate_path(n_steps, rng=rng, include_initial=include_initial)
        return np.repeat(one[np.newaxis, :], n_paths, axis=0)
