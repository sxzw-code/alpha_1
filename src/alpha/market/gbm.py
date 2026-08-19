"""Geometric Brownian Motion price model."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from alpha.market.base import MarketState, PriceModel
# hello

class GeometricBrownianMotion(PriceModel):
    """Geometric Brownian Motion with exact log-Euler discretization.

    Continuous dynamics::

        dS_t = μ S_t dt + σ S_t dW_t

    Exact discrete solution over Δt::

        S_{t+Δt} = S_t * exp[(μ - ½σ²)Δt + σ√Δt Z],  Z ~ N(0, 1)

    Units
    -----
    ``mu`` and ``sigma`` must be expressed in the **same time unit as**
    ``1 / dt``. With the dashboard default ``dt = 1/252``, both are
    *annualized* (per year). Passing daily μ with annual dt (or the
    reverse) silently produces the wrong distribution.
    """

    def __init__(
        self,
        s0: float,
        mu: float,
        sigma: float,
        dt: float,
        *,
        asset_id: str = "ASSET",
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(dt=dt, asset_id=asset_id, seed=seed, rng=rng)
        if s0 <= 0.0:
            raise ValueError(f"s0 must be positive, got {s0}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self._s0 = float(s0)
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._price = self._s0
        self._drift_term = (self._mu - 0.5 * self._sigma**2) * self._dt
        self._diffusion_scale = self._sigma * np.sqrt(self._dt)

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def current_price(self) -> float:
        return self._price

    def reset(self, *, seed: Optional[int] = None, rng: Optional[Generator] = None) -> None:
        self._replace_rng(seed=seed, rng=rng)
        self._step = 0
        self._price = self._s0

    def step(self, rng: Optional[Generator] = None) -> MarketState:
        gen = rng if rng is not None else self._rng
        z = float(gen.standard_normal())
        self._price = self._price * float(
            np.exp(self._drift_term + self._diffusion_scale * z)
        )
        self._step += 1
        return self.state()

    def generate_path(
        self,
        n_steps: int,
        *,
        rng: Optional[Generator] = None,
        include_initial: bool = True,
    ) -> np.ndarray:
        paths = self.generate_paths(
            1, n_steps, rng=rng, include_initial=include_initial
        )
        return paths[0]

    def generate_paths(
        self,
        n_paths: int,
        n_steps: int,
        *,
        rng: Optional[Generator] = None,
        include_initial: bool = True,
    ) -> np.ndarray:
        self._validate_path_args(n_paths, n_steps)
        gen = rng if rng is not None else self._rng
        z = gen.standard_normal(size=(n_paths, n_steps))
        log_increments = self._drift_term + self._diffusion_scale * z
        log_paths = np.cumsum(log_increments, axis=1)
        prices = self._s0 * np.exp(log_paths)
        if include_initial:
            initial = np.full((n_paths, 1), self._s0, dtype=float)
            return np.concatenate([initial, prices], axis=1)
        return prices
