"""Kalman filter for a regression with random-walk coefficients.

    y_t = [1, x_t] . b_t + e_t,     e_t ~ N(0, r)
    b_t = b_{t-1} + w_t,            w_t ~ N(0, Q),     Q_ii = delta / (1 - delta) * r / mean(x_i^2)

Scaling Q by the size of each regressor makes delta unit-free: every
coefficient moves the fitted equilibrium by about sqrt(delta) noise standard
deviations per bar, whether x is a 400 bp yield or a 70 dollar crude price.

The filter is causal: `prior` at row t is E[b_t | data through t-1], the hedge
ratio that could actually be held over bar t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanPath:
    prior: np.ndarray  # (T, k+1): intercept then slopes, before seeing bar t
    innovation: np.ndarray  # y_t minus its one-step forecast
    innovation_var: np.ndarray

    def loglik(self, start: int, end: int) -> float:
        e, s = self.innovation[start:end], self.innovation_var[start:end]
        ok = np.isfinite(e) & (s > 0)
        return float(-0.5 * np.sum(np.log(2 * np.pi * s[ok]) + e[ok] ** 2 / s[ok]))


def run(y: np.ndarray, x: np.ndarray, delta: float, burn: int = 120) -> KalmanPath:
    """Filter the whole series. The first `burn` valid bars set the starting state by OLS.

    Rows with NaN inputs carry the state forward unchanged. Nothing after bar t
    enters row t, so one pass can be shared by every walk-forward fold.
    """
    x = x.reshape(len(y), -1)
    n, k = len(y), x.shape[1] + 1
    design = np.column_stack([np.ones(n), x])
    valid = np.isfinite(y) & np.isfinite(design).all(axis=1)
    prior = np.full((n, k), np.nan)
    innovation = np.full(n, np.nan)
    innovation_var = np.full(n, np.nan)

    idx = np.flatnonzero(valid)
    if len(idx) <= burn:
        return KalmanPath(prior, innovation, innovation_var)
    init = idx[:burn]
    b, *_ = np.linalg.lstsq(design[init], y[init], rcond=None)
    resid = y[init] - design[init] @ b
    r = max(float(resid.var()), 1e-12)
    p = np.linalg.inv(design[init].T @ design[init]) * r
    q = np.diag(delta / (1 - delta) * r / (design[init] ** 2).mean(axis=0))

    for t in range(init[-1] + 1, n):
        p = p + q
        prior[t] = b
        if not valid[t]:
            continue
        h = design[t]
        s = float(h @ p @ h + r)
        e = float(y[t] - h @ b)
        gain = p @ h / s
        b = b + gain * e
        p = p - np.outer(gain, h) @ p
        innovation[t], innovation_var[t] = e, s
    return KalmanPath(prior, innovation, innovation_var)
