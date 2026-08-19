"""Ornstein–Uhlenbeck mean-reverting process."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from alpha.market.base import MarketState, PriceModel


class OrnsteinUhlenbeck(PriceModel):
    """Ornstein–Uhlenbeck (Vasicek-style) mean-reverting process.

    Continuous dynamics::

        dX_t = θ(μ - X_t) dt + σ dW_t

    Exact discrete transition (for θ ≠ 0)::

        X_{t+Δt} = X_t e^{-θΔt} + μ(1 - e^{-θΔt})
                   + σ √[(1 - e^{-2θΔt}) / (2θ)] Z,  Z ~ N(0, 1)

    When θ = 0 the process reduces to arithmetic Brownian motion::

        X_{t+Δt} = X_t + σ √Δt Z

    Units
    -----
    ``theta`` has units of 1/time (the same time unit as ``1 / dt``).
    ``sigma`` is the *level* diffusion coefficient of X, **not** a
    return volatility. With ``dt = 1/252``, θ is per year and σ is per
    √year in the same units as X. Unlike GBM, X can become non-positive;
    the trading portfolio requires a positive mark, so GBM is the safer
    choice for a price-like asset.
    """

    def __init__(
        self,
        x0: float,
        mu: float,
        theta: float,
        sigma: float,
        dt: float,
        *,
        asset_id: str = "ASSET",
        seed: Optional[int] = None,
        rng: Optional[Generator] = None,
    ) -> None:
        super().__init__(dt=dt, asset_id=asset_id, seed=seed, rng=rng)
        if theta < 0.0:
            raise ValueError(f"theta (mean-reversion speed) must be >= 0, got {theta}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self._x0 = float(x0)
        self._mu = float(mu)
        self._theta = float(theta)
        self._sigma = float(sigma)
        self._value = self._x0
        self._exp_neg = float(np.exp(-self._theta * self._dt))
        if self._theta > 0.0:
            self._noise_scale = self._sigma * np.sqrt(
                (1.0 - np.exp(-2.0 * self._theta * self._dt)) / (2.0 * self._theta)
            )
        else:
            self._noise_scale = self._sigma * np.sqrt(self._dt)

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def mu(self) -> float:
        """Long-run mean."""
        return self._mu

    @property
    def theta(self) -> float:
        """Mean-reversion speed."""
        return self._theta

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def current_price(self) -> float:
        return self._value

    def reset(self, *, seed: Optional[int] = None, rng: Optional[Generator] = None) -> None:
        self._replace_rng(seed=seed, rng=rng)
        self._step = 0
        self._value = self._x0

    def _transition(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Apply one OU step to arrays ``x`` and Gaussian shocks ``z``."""
        if self._theta > 0.0:
            return (
                x * self._exp_neg
                + self._mu * (1.0 - self._exp_neg)
                + self._noise_scale * z
            )
        return x + self._noise_scale * z

    def step(self, rng: Optional[Generator] = None) -> MarketState:
        gen = rng if rng is not None else self._rng
        z = float(gen.standard_normal())
        self._value = float(self._transition(np.asarray(self._value), np.asarray(z)))
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
        """Vectorized multi-path generation (no Python loop over time when stable).

        For θ = 0 the process is arithmetic Brownian motion (cumsum of shocks).
        For θ > 0 the AR(1) recurrence is unrolled::

            X_t = a^t X_0 + μ(1 - a^t)
                  + σ_ε Σ_{k=1}^{t} a^{t-k} Z_k

        with ``a = exp(-θ Δt)``. The weighted shock sum is ``a^t * cumsum(Z / a^k)``.
        """
        self._validate_path_args(n_paths, n_steps)
        gen = rng if rng is not None else self._rng
        z = gen.standard_normal(size=(n_paths, n_steps))
        paths = self._paths_from_shocks(z)
        if include_initial:
            initial = np.full((n_paths, 1), self._x0, dtype=float)
            return np.concatenate([initial, paths], axis=1)
        return paths

    def _paths_from_shocks(self, z: np.ndarray) -> np.ndarray:
        """Map Gaussian shocks of shape ``(n_paths, n_steps)`` to levels."""
        n_steps = z.shape[1]
        c = self._noise_scale
        if self._theta == 0.0:
            return self._x0 + c * np.cumsum(z, axis=1)

        a = self._exp_neg
        k = np.arange(n_steps, dtype=float)
        a_k = np.power(a, k)
        # Division by a^k is unstable once powers underflow; fall back to
        # a path-vectorized time loop (still no Python loop over paths).
        if a == 0.0 or a_k[-1] == 0.0 or not np.isfinite(1.0 / max(a_k[-1], 1e-300)):
            x = np.full(z.shape[0], self._x0, dtype=float)
            out = np.empty_like(z, dtype=float)
            for t in range(n_steps):
                x = self._transition(x, z[:, t])
                out[:, t] = x
            return out

        weighted = a_k * np.cumsum(z / a_k, axis=1)
        a_t1 = a_k * a
        return a_t1 * self._x0 + self._mu * (1.0 - a_t1) + c * weighted
