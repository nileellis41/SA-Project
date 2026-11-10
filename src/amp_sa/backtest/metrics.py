"""Performance metrics on daily P&L series expressed as a fraction of capital."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from amp_sa.config import ANN

EULER_GAMMA = 0.5772156649


def sharpe(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    sd = pnl.std()
    return float(pnl.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan


def max_drawdown(pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative (additive) P&L."""
    equity = pnl.fillna(0).cumsum()
    return float((equity - equity.cummax()).min())


def deflated_sharpe(pnl: pd.Series, trial_sharpes: list[float]) -> float:
    """Bailey and Lopez de Prado (2014): P(true Sharpe > 0) after selecting among trials.

    `trial_sharpes` are the annualised Sharpes of every variant that was tried.
    """
    pnl = pnl.dropna()
    n, trials = len(pnl), np.asarray([s for s in trial_sharpes if np.isfinite(s)])
    if n < 30 or pnl.std() == 0:
        return np.nan
    sr = pnl.mean() / pnl.std()  # per bar
    n_trials = max(len(trials), 1)
    sd_trials = trials.std(ddof=1) / np.sqrt(ANN) if n_trials > 1 else 0.0
    if n_trials > 1:
        sr0 = sd_trials * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        sr0 = 0.0
    skew, kurt = stats.skew(pnl), stats.kurtosis(pnl, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def summarize(
    net: pd.Series,
    gross: pd.Series | None = None,
    turnover: pd.Series | None = None,
    trial_sharpes: list[float] | None = None,
) -> dict:
    """Headline numbers. `turnover` is |position change| per bar in risk units."""
    net = net.dropna()
    out = {
        "start": net.index.min().date() if len(net) else None,
        "end": net.index.max().date() if len(net) else None,
        "ann_return": net.mean() * ANN,
        "ann_vol": net.std() * np.sqrt(ANN),
        "sharpe": sharpe(net),
        "max_dd": max_drawdown(net),
        "hit_rate": float((net[net != 0] > 0).mean()) if (net != 0).any() else np.nan,
    }
    if trial_sharpes is not None:
        out["deflated_sharpe"] = deflated_sharpe(net, trial_sharpes)
    if gross is not None:
        gross = gross.reindex(net.index)
        out["gross_sharpe"] = sharpe(gross)
        out["cost_drag"] = (gross.mean() - net.mean()) * ANN
    if turnover is not None:
        out["turnover"] = turnover.reindex(net.index).mean() * ANN
    return out


def by_year(net: pd.Series, years) -> pd.DataFrame:
    rows = {}
    for year in years:
        chunk = net[net.index.year == year].dropna()
        if len(chunk):
            rows[year] = {"return": chunk.sum(), "sharpe": sharpe(chunk), "max_dd": max_drawdown(chunk)}
    return pd.DataFrame.from_dict(rows, orient="index")
