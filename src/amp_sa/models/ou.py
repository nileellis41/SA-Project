"""Ornstein-Uhlenbeck fit and band trading rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amp_sa import config


@dataclass(frozen=True)
class OU:
    mu: float
    half_life: float  # bars; inf when the AR(1) root is >= 1
    sigma_eq: float  # stationary standard deviation

    @property
    def mean_reverting(self) -> bool:
        return np.isfinite(self.half_life) and self.half_life > 0


def fit_ou(spread: np.ndarray) -> OU:
    """AR(1) by OLS: s_t = a + b s_{t-1} + e. Exact discretisation of an OU process."""
    s = spread[np.isfinite(spread)]
    if len(s) < 30:
        return OU(np.nan, np.inf, np.nan)
    lag, cur = s[:-1], s[1:]
    b, a = np.polyfit(lag, cur, 1)
    resid = cur - (a + b * lag)
    if not 0 < b < 1:
        return OU(float(s.mean()), np.inf, float(s.std()))
    return OU(float(a / (1 - b)), float(-np.log(2) / np.log(b)), float(resid.std() / np.sqrt(1 - b * b)))


def rolling_half_life(spread: pd.Series, window: int) -> pd.Series:
    """Half-life from a trailing-window AR(1). inf where the window is not mean reverting."""
    lag = spread.shift(1)
    b = spread.rolling(window).cov(lag) / lag.rolling(window).var()
    return (-np.log(2) / np.log(b.where((b > 0) & (b < 1)))).fillna(np.inf)


def band_positions(z: np.ndarray, entry: float, exit_: float, stop: float = config.STOP_Z) -> np.ndarray:
    """Short the spread above +entry, long below -entry, flat once |z| <= exit.

    Beyond `stop` the position is closed and no new one is opened until z is
    back inside the entry band. The position at t depends on z through t only.
    """
    pos = np.zeros(len(z))
    state, stopped = 0.0, False
    for t, zt in enumerate(z):
        if not np.isfinite(zt):
            state, stopped = 0.0, False
        elif abs(zt) > stop:
            state, stopped = 0.0, True
        else:
            if stopped and abs(zt) < entry:
                stopped = False
            if state > 0 and zt >= -exit_:
                state = 0.0
            elif state < 0 and zt <= exit_:
                state = 0.0
            if state == 0 and not stopped:
                if zt > entry:
                    state = -1.0
                elif zt < -entry:
                    state = 1.0
        pos[t] = state
    return pos


def optimize_bands(z: np.ndarray, change: np.ndarray, vol: np.ndarray, unit_cost: np.ndarray) -> tuple[float, float, float]:
    """(entry, exit, score): the bands with the best net per-bar Sharpe on a training window.

    change, vol and unit_cost are aligned with z; P&L uses the next bar's change.
    """
    best, best_score = (config.ENTRY_GRID[len(config.ENTRY_GRID) // 2], config.EXIT_GRID[0]), -np.inf
    safe_vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
    for entry in config.ENTRY_GRID:
        for exit_ in config.EXIT_GRID:
            if exit_ >= entry:
                continue
            pos = np.nan_to_num(band_positions(z, entry, exit_) / safe_vol)
            pnl = pos[:-1] * np.nan_to_num(change[1:]) - np.abs(np.diff(pos, prepend=0.0))[:-1] * unit_cost[:-1]
            sd = pnl.std()
            score = pnl.mean() / sd if sd > 0 else -np.inf
            if score > best_score:
                best, best_score = (entry, exit_), score
    return best[0], best[1], float(best_score)
