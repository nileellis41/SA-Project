"""Sleeve 1, factor version: PCA residual mean reversion on the Treasury curve.

The construction follows Avellaneda and Lee (2010), applied to yields:

  1. On the training window, take the first k principal components of daily
     yield changes across the tenors. With k = 2 these are level and slope;
     k = 3 adds curvature.
  2. Each tenor's residual is its yield change with the factor moves projected
     out. Holding one unit of a residual is a portfolio of all the tenors that
     is neutral to the k factors by construction.
  3. The cumulative residual is modelled as an OU process on a trailing window.
     Its s-score, (level - mean) / stationary sigma, is the signal. Because the
     mean and sigma are re-estimated every bar from the last `window` bars, the
     equilibrium is local and is allowed to drift.
  4. Entry and exit bands are optimised on the training window, as for the
     pairwise spreads.

The loadings and bands are fitted per fold on the training window. The
rolling OU statistics are trailing and therefore causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest.engine import account, trailing_vol
from amp_sa.backtest.splits import folds_for
from amp_sa.models import ou
from amp_sa.models.ou_optimal import OUOptimal
from amp_sa.sleeves.rv import StrategyRun

TENORS = ("ust_2y", "ust_3y", "ust_5y", "ust_7y", "ust_10y", "ust_30y")
BASE_TENORS = ("ust_2y", "ust_5y", "ust_10y", "ust_30y")  # full history; these set the fold calendar
OU_WINDOW = 60  # bars, as in Avellaneda-Lee
VARIANT = "pca"  # bands from the training-window grid search
VARIANT_LL = "pca_ll"  # bands from the Leung-Li optimal stopping levels
LL_RATE = 0.05
GROUP = "rates"


def factor_loadings(changes: pd.DataFrame, k: int) -> tuple[np.ndarray, float]:
    """Top-k eigenvectors of the covariance of yield changes, and the variance share they explain."""
    values, vectors = np.linalg.eigh(changes.cov().to_numpy())
    order = np.argsort(values)[::-1]
    return vectors[:, order[:k]], float(values[order[:k]].sum() / values.sum())


def rolling_ou(cumulative: pd.DataFrame, window: int = OU_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trailing-window OU fit of each cumulative residual: (s-score, half-life in bars, stationary sigma).

    All three are NaN where the window is not mean reverting. AR(1) by rolling
    moments: C_t = a + b C_{t-1} + e. A window is tradeable when 0 < b < 1 and the
    half-life is under half the window (Avellaneda-Lee's speed filter).
    """
    lag = cumulative.shift(1)
    roll, roll_lag = cumulative.rolling(window), lag.rolling(window)
    var_lag = roll_lag.var()
    b = roll.cov(lag) / var_lag
    a = roll.mean() - b * roll_lag.mean()
    resid_var = (roll.var() - b**2 * var_lag).clip(lower=0)
    ok = (b > 0) & (b < 1)
    half_life = -np.log(2) / np.log(b.where(ok))
    ok &= half_life < window / 2
    mean = a / (1 - b)
    sigma_eq = np.sqrt(resid_var / (1 - b**2))
    ok &= sigma_eq > 0
    return ((cumulative - mean) / sigma_eq).where(ok), half_life.where(ok), sigma_eq.where(ok)


def s_scores(cumulative: pd.DataFrame, window: int = OU_WINDOW) -> pd.DataFrame:
    return rolling_ou(cumulative, window)[0]


def leung_li_bands(half_life: float, cost_in_sigma: float) -> tuple[float, float] | None:
    """Leung-Li (2015) entry and exit levels in s-score units, as (entry, exit) for ou.band_positions.

    The OU process is standardised (mean 0, stationary sigma 1), so only the
    reversion speed and the cost in sigma units matter. The paper solves the
    long side: enter at d* below the mean, take profit at b* above it, stop at
    -STOP_Z. The short side is the mirror image. Returns None when the paper's
    no-trade condition holds (one round trip is not worth its costs).

    `exit` comes back negative: the position is held through the mean and closed
    b* sigmas on the far side, which is how the paper makes a trade pay for its costs.
    """
    if not (np.isfinite(half_life) and half_life > 0 and np.isfinite(cost_in_sigma) and cost_in_sigma > 0):
        return None
    speed = np.log(2) / half_life * config.ANN  # per year
    model = OUOptimal(0.0, speed, np.sqrt(2 * speed), r=LL_RATE, r_hat=LL_RATE, c=cost_in_sigma, c_hat=cost_in_sigma)
    try:
        band = model.entry_band(-config.STOP_Z, n=400)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    if band is None or band.get("d") is None:
        return None
    return float(-band["d"]), float(-band["b"])


def sparse_hedges(loadings: np.ndarray) -> np.ndarray:
    """Factor-neutral portfolios that use as few legs as possible.

    Being neutral to k factors needs k hedge legs, so each row holds one unit of
    its own tenor and hedges with the k nearest tenors on the curve (k + 1 legs:
    three for level and slope, four once curvature is hedged too). The hedge
    weights solve  loadings[hedges]' w = -loadings[own]  exactly. A row is NaN
    if its hedge tenors cannot span the factors.
    """
    n, k = loadings.shape
    out = np.zeros((n, n))
    for i in range(n):
        nearest = sorted((j for j in range(n) if j != i), key=lambda j: (abs(j - i), j))[:k]
        block = loadings[nearest].T  # k x k
        if np.linalg.cond(block) > 1e8:
            out[i] = np.nan
            continue
        out[i, i] = 1.0
        out[i, nearest] = np.linalg.solve(block, -loadings[i])
    return out


def run_curve(levels: pd.DataFrame, k: int, tenors=TENORS, window: int = OU_WINDOW,
              sparse: bool = False) -> list[StrategyRun]:
    """One StrategyRun per tenor residual, with variants 'pca' (grid bands) and 'pca_ll' (Leung-Li levels).

    sparse=False: each residual is the full projection, a portfolio of every tenor.
    sparse=True:  each residual is the own tenor hedged with its k nearest neighbours only.
    """
    label = f"pca{k}s" if sparse else f"pca{k}"
    tenors = [t for t in tenors if t in levels.columns]
    base = [t for t in BASE_TENORS if t in tenors]
    d = levels[tenors].diff()
    index = levels.index
    costs = np.array([config.COST_PER_UNIT[t] for t in tenors])
    blank = lambda: pd.Series(np.nan, index=index)  # noqa: E731
    store = {t: {"signal": blank(), "signal_ll": blank(), "change": blank(), "vol": blank(), "cost": blank()}
             for t in tenors}
    legs = {t: pd.DataFrame(np.nan, index=index, columns=tenors) for t in tenors}
    diag = []
    # [diag-logging] analysis/portfolio_diagnostics: per-bar OU state and per-fold factors. Read-only, no effect on signals.
    ou_log = {t: {"z": blank(), "half_life": blank(), "kappa": blank(), "sigma_eq": blank()} for t in tenors}
    factor_scores = pd.DataFrame(np.nan, index=index, columns=[f"pc{j + 1}" for j in range(k)])
    loading_rows = []

    for fold in folds_for(levels[base]):
        w = slice(fold.train_start, fold.test_end)
        n_train = fold.train_end - fold.train_start
        test = slice(fold.test_start - fold.train_start, fold.test_end - fold.train_start)
        test_index = index[fold.test_start : fold.test_end]
        avail = [t for t in tenors if levels[t].iloc[w].notna().all()]
        if len(avail) < k + 1:
            continue
        cols = [tenors.index(t) for t in avail]
        dw = d[avail].iloc[w]
        loadings, explained = factor_loadings(dw.iloc[1:n_train], k)
        # Row i is the factor-neutral portfolio for tenor i.
        project = sparse_hedges(loadings) if sparse else np.eye(len(avail)) - loadings @ loadings.T
        resid = pd.DataFrame(dw.to_numpy() @ project.T, index=dw.index, columns=avail)
        scores, half_lives, sigmas = rolling_ou(resid.cumsum(), window)
        # [diag-logging] fold loadings (fitted on the training window) and test-window factor scores (daily PC moves, bp)
        factor_scores.loc[test_index] = (dw.to_numpy() @ loadings)[test]
        loading_rows += [{"fold": fold.k, "test_start": test_index[0], "tenor": t, "explained": explained,
                          **{f"pc{j + 1}": loadings[m, j] for j in range(k)}} for m, t in enumerate(avail)]

        for i, tenor in enumerate(avail):
            if not np.isfinite(project[i]).all():
                continue
            change = resid[tenor]
            vol = trailing_vol(change)
            unit_cost = float(np.abs(project[i]) @ costs[cols])
            cost = np.full(len(change), unit_cost)
            z = scores[tenor].to_numpy()
            entry, exit_, _ = ou.optimize_bands(z[:n_train], change.to_numpy()[:n_train], vol.to_numpy()[:n_train],
                                                cost[:n_train])
            pos = ou.band_positions(z, entry, exit_)
            # Leung-Li levels from the typical training-window OU: median half-life and cost in sigma units.
            ll_half_life = float(np.nanmedian(half_lives[tenor].to_numpy()[:n_train]))
            ll_cost = unit_cost / float(np.nanmedian(sigmas[tenor].to_numpy()[:n_train]))
            ll = leung_li_bands(ll_half_life, ll_cost)
            pos_ll = ou.band_positions(z, *ll) if ll is not None else np.zeros(len(z))
            for key, val in (("signal", pos), ("signal_ll", pos_ll), ("change", change), ("vol", vol), ("cost", cost)):
                store[tenor][key].loc[test_index] = np.asarray(val, dtype=float)[test]
            legs[tenor].loc[test_index, avail] = project[i]
            # [diag-logging] trailing 60-bar OU state behind the signal; kappa = ln 2 / half-life, per bar
            hl = half_lives[tenor].to_numpy()
            for key, val in (("z", z), ("half_life", hl), ("kappa", np.log(2) / hl), ("sigma_eq", sigmas[tenor].to_numpy())):
                ou_log[tenor][key].loc[test_index] = np.asarray(val, dtype=float)[test]
            diag.append({
                "strategy": f"{label}_{tenor}", "fold": fold.k, "legs": int((np.abs(project[i]) > 1e-12).sum()), "test_start": test_index[0].date(),
                "tenors": len(avail), "explained": explained, "entry": entry, "exit": exit_,
                "cost_over_daily_vol": unit_cost / float(np.nanmedian(vol.to_numpy()[:n_train])),
                "tradeable_share": float(np.isfinite(z[test]).mean()),
                "ll_half_life": ll_half_life, "ll_cost_in_sigma": ll_cost, "ll_trade": ll is not None,
                "ll_entry": ll[0] if ll else np.nan, "ll_take_profit": -ll[1] if ll else np.nan,
            })

    diag = pd.DataFrame(diag)
    runs = []
    for tenor in tenors:
        s = store[tenor]
        live = s["signal"].notna()
        if not live.any():
            continue
        results = {v: account(s[key], s["change"], s["cost"].ffill().bfill(), vol=s["vol"])
                   for v, key in ((VARIANT, "signal"), (VARIANT_LL, "signal_ll"))}
        name = f"{label}_{tenor}"
        extras = {"legs_adaptive": legs[tenor]}
        # [diag-logging] OU state per bar, fold loadings (long: fold x tenor), test-window factor scores
        extras.update({f"ou_{key}": val for key, val in ou_log[tenor].items()})
        extras.update({"fold_loadings": pd.DataFrame(loading_rows), "factor_scores": factor_scores})
        runs.append(StrategyRun(name, GROUP, results, live, diag[diag["strategy"] == name], extras))
    return runs
