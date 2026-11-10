"""Regime features. Every column at row t uses data through the close of t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config

MARKET_FEATURES = ("log_vix", "log_move", "vol_ratio")
HALF_LIFE_CAP = 250.0


def market_features(levels: pd.DataFrame, spread_change: pd.Series) -> pd.DataFrame:
    """Risk-appetite and spread-turbulence features shared by the HMM and the classifier."""
    fast = spread_change.ewm(span=20, min_periods=10).std()
    slow = spread_change.ewm(span=120, min_periods=60).std()
    return pd.DataFrame(
        {
            "log_vix": np.log(levels["vix"]),
            "log_move": np.log(levels["move"]),
            "vol_ratio": fast / slow,
        },
        index=levels.index,
    )


def classifier_features(market: pd.DataFrame, z: pd.Series, half_life: pd.Series) -> pd.DataFrame:
    out = market.copy()
    out["z"] = z
    out["abs_z"] = z.abs()
    out["half_life"] = half_life.clip(upper=HALF_LIFE_CAP)
    return out


def reversion_label(spread: pd.Series, z: pd.Series, horizon: int = config.NN_LABEL_HORIZON) -> pd.Series:
    """1 if the spread moved back toward its mean over the next `horizon` bars, else 0.

    This looks `horizon` bars AHEAD. It is a training label only: the caller
    must drop the last `horizon` rows of the training window (they are NaN here)
    and the walk-forward purge keeps the label horizon out of the test block.
    """
    forward = spread.shift(-horizon) - spread
    label = (-np.sign(z) * forward > 0).astype(float)
    return label.where(forward.notna() & (z.abs() > 0.5))
