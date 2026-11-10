"""Sleeve 2: trend on the same series as Sleeve 1, traded as outrights or levels.

  ma         EMA crossover baseline.
  l1_single  L1 trend filter slope at one regularisation, chosen per fold on the training window.
  l1_multi   average of the L1 slopes across all regularisation scales (nothing to choose).

Position = normalised slope x volatility scaling. The normalisation constant
(the standard deviation of the raw signal) and the l1_single scale are fitted
on the training window of each fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest.engine import account, trailing_vol
from amp_sa.backtest.splits import folds_for
from amp_sa.data.load_bbg import PROCESSED_DIR
from amp_sa.data.panel import adjusted_levels, changes
from amp_sa.models.trend import l1_slopes, ma_signal
from amp_sa.sleeves.rv import StrategyRun

VARIANTS = ("ma", "l1_single", "l1_multi")
SLOPE_CACHE = PROCESSED_DIR / "l1_slopes.parquet"
SIGNAL_CAP = 2.0


def compute_slopes(levels: pd.DataFrame, rolls: pd.DataFrame, series=tuple(config.TREND_SERIES), n_jobs: int = 4,
                   cache=SLOPE_CACHE) -> pd.DataFrame:
    """Causal L1 edge slopes for every series; columns are (series, frac). Cached on disk."""
    series = [s for s in series if s in levels.columns]
    if cache is not None and cache.exists():
        cached = pd.read_parquet(cache)
        if cached.index.equals(levels.index) and set(series) <= set(cached.columns.get_level_values(0)):
            return cached
    from joblib import Parallel, delayed

    adj = adjusted_levels(levels[series], rolls[series])
    frames = Parallel(n_jobs=n_jobs)(delayed(l1_slopes)(adj[s]) for s in series)
    slopes = pd.concat(dict(zip(series, frames)), axis=1)
    if cache is not None:
        slopes.to_parquet(cache)
    return slopes


def _train_sharpe(signal: np.ndarray, vol: np.ndarray, change: np.ndarray, cost: float) -> float:
    pos = np.nan_to_num(signal / vol)
    pnl = pos[:-1] * np.nan_to_num(change[1:]) - np.abs(np.diff(pos, prepend=0.0))[:-1] * cost
    sd = pnl.std()
    return pnl.mean() / sd if sd > 0 else -np.inf


def run_series(name: str, levels: pd.DataFrame, rolls: pd.DataFrame, slopes: pd.DataFrame) -> StrategyRun | None:
    if name not in levels.columns:
        return None
    level = adjusted_levels(levels[[name]], rolls[[name]])[name]
    change = changes(levels[[name]], rolls[[name]])[name]
    vol = trailing_vol(change)
    cost = config.COST_PER_UNIT[name]
    raw = slopes.div(vol, axis=0).copy()  # slope per bar in units of daily volatility
    fracs = list(raw.columns)
    raw["ma"] = ma_signal(level, vol)

    folds = folds_for(levels[name])
    if not folds:
        return None
    signals = {v: pd.Series(np.nan, index=levels.index) for v in VARIANTS}
    diag = []
    for fold in folds:
        train, test = fold.train(raw), fold.test(raw)
        scale = train.std().replace(0, np.nan)
        scaled_train = (train / scale).clip(-SIGNAL_CAP, SIGNAL_CAP) / SIGNAL_CAP
        scaled_test = (test / scale).clip(-SIGNAL_CAP, SIGNAL_CAP) / SIGNAL_CAP
        vol_tr, chg_tr = fold.train(vol).to_numpy(), fold.train(change).to_numpy()
        scores = {f: _train_sharpe(scaled_train[f].to_numpy(), vol_tr, chg_tr, cost) for f in fracs}
        best = max(scores, key=scores.get)
        signals["ma"].loc[test.index] = scaled_test["ma"]
        signals["l1_single"].loc[test.index] = scaled_test[best]
        signals["l1_multi"].loc[test.index] = scaled_test[fracs].mean(axis=1, skipna=False)
        diag.append({"series": name, "fold": fold.k, "test_start": test.index[0].date(), "l1_frac": best})

    is_roll = rolls[name]
    results = {v: account(sig, change, cost, is_roll, vol=vol) for v, sig in signals.items()}
    live = signals["ma"].notna()
    return StrategyRun(name, config.TREND_SERIES[name], results, live, pd.DataFrame(diag), {})


def run_sleeve(levels: pd.DataFrame, rolls: pd.DataFrame, series=tuple(config.TREND_SERIES), n_jobs: int = 4,
               slopes: pd.DataFrame | None = None) -> list[StrategyRun]:
    slopes = compute_slopes(levels, rolls, series, n_jobs) if slopes is None else slopes
    runs = [run_series(s, levels, rolls, slopes[s]) for s in series if s in slopes.columns.get_level_values(0)]
    return [r for r in runs if r is not None]
