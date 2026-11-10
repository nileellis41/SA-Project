"""Trend estimators: moving-average momentum and L1 trend filtering.

L1 trend filter (Kim, Koh, Boyd, Gorinevsky 2009):
    minimize 0.5 * ||y - x||^2 + lam * ||D2 x||_1
whose solution is piecewise linear. It is a smoother, so it is only ever fitted
on a trailing window ending at t and only the slope at the right edge is used.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded

from amp_sa import config


def ma_signal(level: pd.Series, change_vol: pd.Series, fast: int = config.MA_FAST, slow: int = config.MA_SLOW) -> pd.Series:
    """EMA crossover scaled by the volatility of the level over the slow horizon."""
    gap = level.ewm(span=fast, min_periods=fast).mean() - level.ewm(span=slow, min_periods=slow).mean()
    return gap / (change_vol * np.sqrt(slow))


def second_difference(n: int) -> np.ndarray:
    d = np.zeros((n - 2, n))
    rows = np.arange(n - 2)
    d[rows, rows], d[rows, rows + 1], d[rows, rows + 2] = 1.0, -2.0, 1.0
    return d


class L1Filter:
    """L1 trend filter for one window length.

    `fit_cvxpy` is the reference solution (one compiled problem; y and lam are
    parameters). `fit` solves the same problem with the primal-dual interior
    point method of Kim et al. on the dual, where every Newton step is a
    pentadiagonal solve. It is about 50x faster, which the walk-forward needs
    for its tens of thousands of refits. tests/test_models.py checks the two agree.
    """

    def __init__(self, n: int):
        self.n = n
        self.d = second_difference(n)
        self._ddt = self.d @ self.d.T
        self._ddt_inv_d = np.linalg.solve(self._ddt, self.d)
        # Upper band storage of D D' (bandwidth 2) for solveh_banded.
        m = n - 2
        self._bands = np.zeros((3, m))
        self._bands[2] = np.diag(self._ddt)
        self._bands[1, 1:] = np.diag(self._ddt, 1)
        self._bands[0, 2:] = np.diag(self._ddt, 2)
        self._y, self._lam = cp.Parameter(n), cp.Parameter(nonneg=True)
        self._x = cp.Variable(n)
        self._problem = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(self._y - self._x) + self._lam * cp.norm1(self.d @ self._x))
        )

    def lambda_max(self, y: np.ndarray) -> float:
        """Smallest lam for which the fit is a straight line."""
        return float(np.abs(self._ddt_inv_d @ y).max())

    def fit_cvxpy(self, y: np.ndarray, lam: float) -> np.ndarray:
        self._y.value, self._lam.value = y, max(lam, 1e-10)
        self._problem.solve(solver=cp.CLARABEL)
        return self._x.value

    def fit(self, y: np.ndarray, lam: float, tol: float = 1e-5, max_iter: int = 60) -> np.ndarray:
        """Interior point on the dual: min 0.5 |D'v|^2 - y'D'v  s.t. |v| <= lam;  x = y - D'v."""
        d, m = self.d, self.n - 2
        dy = d @ y
        z, mu1, mu2 = np.zeros(m), np.ones(m), np.ones(m)
        f1, f2 = z - lam, -z - lam
        t, step = 1e-10, np.inf
        for _ in range(max_iter):
            dtz = d.T @ z
            ddtz = d @ dtz
            gap = dtz @ dtz + lam * np.abs(dy - ddtz).sum() - dy @ z
            if gap <= tol:
                break
            if step >= 0.2:
                t = max(2 * m * 2.0 / gap, 1.2 * t)
            res_dual = ddtz - dy + mu1 - mu2
            bands = self._bands.copy()
            bands[2] -= mu1 / f1 + mu2 / f2
            dz = solveh_banded(bands, dy - ddtz + (1 / t) / f1 - (1 / t) / f2)
            dmu1 = -(mu1 + ((1 / t) + dz * mu1) / f1)
            dmu2 = -(mu2 + ((1 / t) - dz * mu2) / f2)
            res_norm = np.sqrt(res_dual @ res_dual + ((mu1 * f1 + 1 / t) ** 2).sum() + ((mu2 * f2 + 1 / t) ** 2).sum())
            step = 1.0
            for mu, dmu in ((mu1, dmu1), (mu2, dmu2)):
                neg = dmu < 0
                if neg.any():
                    step = min(step, 0.99 * (-mu[neg] / dmu[neg]).min())
            for _ in range(20):
                nz, nmu1, nmu2 = z + step * dz, mu1 + step * dmu1, mu2 + step * dmu2
                nf1, nf2 = nz - lam, -nz - lam
                nres = d @ (d.T @ nz) - dy + nmu1 - nmu2
                nnorm = np.sqrt(nres @ nres + ((nmu1 * nf1 + 1 / t) ** 2).sum() + ((nmu2 * nf2 + 1 / t) ** 2).sum())
                if max(nf1.max(), nf2.max()) < 0 and nnorm <= (1 - 0.01 * step) * res_norm:
                    break
                step *= 0.5
            z, mu1, mu2, f1, f2 = nz, nmu1, nmu2, nf1, nf2
        return y - d.T @ z

    def edge_slopes(self, y: np.ndarray, fracs=config.L1_LAMBDA_FRACS) -> list[float]:
        """Slope of the fitted trend at the last bar, one per lam = frac * lambda_max."""
        scale = y.std()
        if not np.isfinite(scale) or scale == 0:
            return [0.0] * len(fracs)
        z = (y - y.mean()) / scale
        lam_max = self.lambda_max(z)
        out = []
        for frac in fracs:
            x = self.fit(z, frac * lam_max)
            out.append(float((x[-1] - x[-2]) * scale) if x is not None else np.nan)
        return out


def l1_slopes(
    level: pd.Series,
    window: int | None = None,
    step: int | None = None,
    fracs=config.L1_LAMBDA_FRACS,
    filt: L1Filter | None = None,
) -> pd.DataFrame:
    """Causal edge slopes, refitted every `step` bars on the trailing `window` and held in between.

    The refit schedule is anchored on the bar count from the first valid
    observation, so it does not depend on where the series ends.
    """
    window = config.L1_WINDOW if window is None else window
    step = config.L1_STEP if step is None else step
    filt = filt or L1Filter(window)
    values = level.to_numpy(float)
    out = np.full((len(values), len(fracs)), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) >= window:
        first = valid[0]
        for t in range(first + window - 1, len(values), step):
            chunk = values[t - window + 1 : t + 1]
            if np.isfinite(chunk).all():
                out[t] = filt.edge_slopes(chunk, fracs)
    frame = pd.DataFrame(out, index=level.index, columns=[f"frac_{f:g}" for f in fracs])
    return frame.ffill(limit=step - 1)
