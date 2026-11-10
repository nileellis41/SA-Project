"""Spread construction from a hedge ratio. Differences, never log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.config import Spread


def hedged(y: pd.Series, x: pd.DataFrame, beta: np.ndarray) -> pd.Series:
    """y - x . beta. `beta` is one row of ratios or one row per bar (already lagged by the caller)."""
    beta = np.asarray(beta, dtype=float)
    return y - (x * beta).sum(axis=1, skipna=False)


def unit_cost(spread: Spread, beta: np.ndarray, index: pd.Index) -> pd.Series:
    """One-way cost of trading one unit of the spread: every leg pays its own cost."""
    beta = np.abs(np.asarray(beta, dtype=float)).reshape(-1, len(spread.x))
    legs = np.array([config.COST_PER_UNIT[c] for c in spread.x])
    cost = config.COST_PER_UNIT[spread.y] + beta @ legs
    return pd.Series(np.broadcast_to(cost, len(index)), index=index).ffill().bfill()
