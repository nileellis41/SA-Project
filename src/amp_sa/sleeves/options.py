"""Delta-hedged options: trade mean reversion in implied volatility.

The traded instrument is a 30-day at-the-money straddle, delta-hedged daily in the
underlying and rolled daily to keep 30 days to expiry. Long the straddle = long
volatility. With Black-Scholes greeks at the previous close (sigma = implied vol,
T = 30/365, ATM so d1 = sigma sqrt(T) / 2), one day's P&L per $1 of underlying is

    vega part         2 phi(d1) sqrt(T) * (sigma_t - sigma_{t-1})
    gamma-theta part  phi(d1) / (sigma sqrt(T)) * (r_t^2 - sigma^2 dt)

The vega part is the bet the strategy is built on: implied vol moving back to its
mean. The gamma-theta part is realised against implied variance: a long straddle
pays it whenever the market moves less than the option priced. Splitting the two
shows which one a strategy actually earns.

Signals, per fold on the training window (same walk-forward as every other sleeve):

  iv       implied vol itself, in vol points
  vrp      implied vol minus trailing 21-day realised vol (the volatility risk premium)

  static        z against the training-window OU mean; trade only if the OU fit
                reverts (half-life 1 to 60 bars)
  rolling       s-score on a trailing 60-bar OU fit, as in the PCA sleeve
  rolling_hmm   rolling, traded only in HMM states (log VIX, log MOVE, turbulence)
                where it made money on the training window

Above the entry band the strategy sells the straddle (vol is rich), below it buys.
Bands come from the training-window grid, as for the curve. Benchmarks on the
same test windows: always short vol, always long vol, a pure HMM short-vol switch,
and the underlying itself.

Costs: half the bid-ask in vol points, times vega, each time the straddle is
traded. These are assumptions (COST_VOL_PTS), not measured spreads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from amp_sa import config
from amp_sa.backtest.engine import account, trailing_vol
from amp_sa.backtest.splits import folds_for
from amp_sa.features import regime as rf
from amp_sa.models import ou
from amp_sa.sleeves import pca
from amp_sa.sleeves.rv import StrategyRun, _hmm_gate, _tradeable

TENOR = 30 / 365
DT = 1 / config.ANN
RV_WINDOW = 21
# Half bid-ask, one way, in implied-vol points. Assumed: index options are tighter than single names.
COST_VOL_PTS = {"spx": 0.20, "ndx": 0.30, "msft": 0.30}
HEDGE_COST = 0.5e-4  # one-way cost of the underlying, as a fraction of price (buy-and-hold benchmark)
KINDS = ("iv", "vrp")
VARIANTS = ("static", "rolling", "rolling_hmm")
BENCHMARKS = ("short_vol", "long_vol", "hmm_short_vol")
GROUP = "options"


@dataclass
class Underlying:
    name: str
    price: pd.Series  # close
    iv: pd.Series  # 30-day ATM implied vol, decimal (0.20 = 20%)


def straddle_pnl(u: Underlying) -> pd.DataFrame:
    """Daily P&L of one long delta-hedged 30-day ATM straddle per $1 of underlying, split into parts."""
    r = np.log(u.price).diff()
    sig = u.iv.shift(1)
    sqrt_t = np.sqrt(TENOR)
    phi = norm.pdf(sig * sqrt_t / 2)
    vega = 2 * phi * sqrt_t
    out = pd.DataFrame({
        "vega": vega * u.iv.diff(),
        "gamma_theta": phi / (sig * sqrt_t) * (r**2 - sig**2 * DT),
    })
    out["total"] = out["vega"] + out["gamma_theta"]
    out["unit_cost"] = COST_VOL_PTS[u.name] / 100 * 2 * norm.pdf(u.iv * sqrt_t / 2) * sqrt_t
    return out


def signal_series(u: Underlying) -> dict[str, pd.Series]:
    rv = np.log(u.price).diff().rolling(RV_WINDOW).std() * np.sqrt(config.ANN)
    return {"iv": u.iv * 100, "vrp": (u.iv - rv) * 100}


def run_underlying(u: Underlying, levels: pd.DataFrame) -> tuple[list[StrategyRun], pd.DataFrame]:
    """StrategyRuns for every signal kind (variants as results) plus one for the benchmarks.

    Runs on the underlying's own trading days: the panel follows the Treasury calendar,
    which has days when equity markets are shut, and a blank day would break the
    rolling windows and force a needless close-and-reopen.
    """
    valid = u.price.notna() & u.iv.notna()
    u = Underlying(u.name, u.price[valid], u.iv[valid])
    levels = levels.loc[valid[valid].index]
    pnl = straddle_pnl(u)
    change, cost = pnl["total"], pnl["unit_cost"]
    vol = trailing_vol(change)
    market = rf.market_features(levels, change)
    index = levels.index
    blank = lambda: pd.Series(np.nan, index=index)  # noqa: E731
    xs = signal_series(u)
    base_frame = pd.DataFrame({"iv": xs["iv"], "move": levels["move"], "change": change}).dropna()
    folds = folds_for(base_frame)
    sig = {(k, v): blank() for k in KINDS for v in VARIANTS} | {b: blank() for b in BENCHMARKS}
    diag = []

    for fold in folds:
        # folds index into base_frame; map back to the panel index
        w_idx = base_frame.index[fold.train_start : fold.test_end]
        n_train = fold.train_end - fold.train_start
        test = slice(fold.test_start - fold.train_start, fold.test_end - fold.train_start)
        test_index = w_idx[test]
        ch, vo, co = change.loc[w_idx].to_numpy(), vol.loc[w_idx].to_numpy(), cost.loc[w_idx].to_numpy()
        mk = market.loc[w_idx].to_numpy()

        for kind in KINDS:
            x = xs[kind].loc[w_idx]
            row = {"underlying": u.name, "kind": kind, "fold": fold.k, "test_start": test_index[0].date()}
            # static: fixed training-window equilibrium
            fit = ou.fit_ou(x.to_numpy()[:n_train])
            z = ((x - fit.mu) / fit.sigma_eq).to_numpy() if fit.sigma_eq > 0 else np.full(len(x), np.nan)
            e, ex, _ = ou.optimize_bands(z[:n_train], ch[:n_train], vo[:n_train], co[:n_train])
            pos_static = ou.band_positions(z, e, ex) if _tradeable(fit) else np.zeros(len(z))
            # rolling: local equilibrium, as in the PCA sleeve
            s, _, _ = pca.rolling_ou(x.to_frame())
            s = s.iloc[:, 0].to_numpy()
            er, exr, _ = ou.optimize_bands(s[:n_train], ch[:n_train], vo[:n_train], co[:n_train])
            pos_roll = ou.band_positions(s, er, exr)
            gate, states = _hmm_gate(mk, n_train, pos_roll, vo, ch)
            pos_hmm = pos_roll * gate
            for v, p in (("static", pos_static), ("rolling", pos_roll), ("rolling_hmm", pos_hmm)):
                sig[(kind, v)].loc[test_index] = p[test]
            row.update(train_half_life=fit.half_life, static_traded=_tradeable(fit), static_entry=e, static_exit=ex,
                       rolling_entry=er, rolling_exit=exr, rolling_tradeable_share=float(np.isfinite(s[:n_train]).mean()),
                       hmm_states_allowed=states,
                       cost_over_daily_vol=float(np.nanmedian(co[:n_train] / vo[:n_train])),
                       iv_mean_train=float(xs["iv"].loc[w_idx].iloc[:n_train].mean()))
            diag.append(row)

        ones = np.ones(len(w_idx))
        gate, _ = _hmm_gate(mk, n_train, -ones, vo, ch)
        for b, p in (("short_vol", -ones), ("long_vol", ones), ("hmm_short_vol", -ones * gate)):
            sig[b].loc[test_index] = p[test]

    runs = []
    for kind in KINDS:
        live = sig[(kind, "rolling")].notna()
        results = {v: account(sig[(kind, v)], change, cost, vol=vol) for v in VARIANTS}
        runs.append(StrategyRun(f"{u.name}_{kind}", GROUP, results, live, pd.DataFrame(diag),
                                {"vega": pnl["vega"], "gamma_theta": pnl["gamma_theta"]}))
    live = sig["short_vol"].notna()
    results = {b: account(sig[b], change, cost, vol=vol) for b in BENCHMARKS}
    ret = u.price.pct_change()
    results["underlying"] = account(pd.Series(1.0, index=index).where(live), ret, HEDGE_COST)
    runs.append(StrategyRun(f"{u.name}_bench", GROUP, results, live, pd.DataFrame(),
                            {"vega": pnl["vega"], "gamma_theta": pnl["gamma_theta"]}))
    return runs, pd.DataFrame(diag)
