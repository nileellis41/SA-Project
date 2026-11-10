"""Sleeve 1: mean reversion on relative-value spreads.

Three variants are run through the same walk-forward folds:

  static        OLS hedge ratio and OU mean/variance fixed on the training
                window, optimised bands, no regime layer.
  adaptive_hmm  Kalman hedge ratio and intercept, z-score of the Kalman forecast
                error, health monitor, HMM regime gate.
  adaptive_nn   as adaptive_hmm with the neural regime classifier as the gate.

Everything that is estimated (cointegration screen, hedge ratio, Kalman
process noise, OU parameters, bands, scalers, HMM, classifier) is estimated on
the training window of the fold. The Kalman filter, EW statistics, rolling
half-life and HMM forward filter then run causally into the test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest.engine import Result, account, trailing_vol
from amp_sa.backtest.splits import folds_for
from amp_sa.config import Spread
from amp_sa.data.panel import changes
from amp_sa.features import regime as rf
from amp_sa.features.spreads import hedged, unit_cost
from amp_sa.models import cointegration, kalman, ou
from amp_sa.models.regime import HMMRegime, NeuralRegime

VARIANTS = ("static", "adaptive_hmm", "adaptive_nn")
NN_SMOOTH_SPAN = 5


@dataclass
class StrategyRun:
    name: str
    group: str
    results: dict[str, Result]
    live: pd.Series  # True from the first out-of-sample bar
    diagnostics: pd.DataFrame
    extras: dict[str, pd.Series]

    def net(self, variant: str, cost_mult: float = 1.0) -> pd.Series:
        return self.results[variant].net_at(cost_mult).where(self.live)


def _tradeable(fit: ou.OU) -> bool:
    return fit.mean_reverting and config.MIN_HALF_LIFE <= fit.half_life <= config.MAX_HALF_LIFE


def _hmm_gate(market: np.ndarray, n_train: int, base: np.ndarray, vol: np.ndarray, change: np.ndarray):
    """Trade only in HMM states where the ungated rule made money on the training window."""
    model = HMMRegime.fit(market[:n_train])
    if model is None:
        return np.ones(len(market), dtype=bool), "none"
    probs = model.filter(market)
    state = np.where(np.isfinite(probs).all(axis=1), probs.argmax(axis=1), -1)
    pnl = np.nan_to_num(base[: n_train - 1] / vol[: n_train - 1]) * np.nan_to_num(change[1:n_train])
    allowed = np.array(
        [pnl[state[: n_train - 1] == k].mean() > 0 if (state[: n_train - 1] == k).any() else True
         for k in range(probs.shape[1])]
    )
    gate = np.where(state >= 0, allowed[np.clip(state, 0, None)], False)
    return gate, "".join("1" if a else "0" for a in allowed)


def _nn_gate(features: pd.DataFrame, spread: pd.Series, z: pd.Series, n_train: int):
    """Trade only when the classifier's smoothed reversion probability clears the gate."""
    label = rf.reversion_label(spread.iloc[:n_train], z.iloc[:n_train])
    model = NeuralRegime.fit(features.to_numpy()[:n_train], label.to_numpy())
    if model is None:
        return np.ones(len(features), dtype=bool), False
    prob = pd.Series(model.predict(features.to_numpy()), index=features.index)
    smooth = prob.ewm(span=NN_SMOOTH_SPAN, ignore_na=True).mean().where(prob.notna())
    return (smooth >= config.NN_GATE).to_numpy(), True


def run_spread(spread: Spread, levels: pd.DataFrame, rolls: pd.DataFrame) -> StrategyRun | None:
    cols = [spread.y, *spread.x]
    if not set(cols) <= set(levels.columns):
        return None
    y, x = levels[spread.y], levels[list(spread.x)]
    d = changes(levels[cols], rolls[cols])
    dy, dx = d[spread.y], d[list(spread.x)]
    is_roll = rolls[cols].any(axis=1)
    folds = folds_for(levels[cols])
    if not folds:
        return None

    # One causal pass per process-noise setting; folds pick among them on their training window.
    paths = {delta: kalman.run(y.to_numpy(float), x.to_numpy(float), delta) for delta in config.KALMAN_DELTAS}

    index = levels.index
    blank = lambda: pd.Series(np.nan, index=index)  # noqa: E731
    store = {v: {"signal": blank(), "change": blank(), "vol": blank(), "cost": blank()} for v in VARIANTS}
    disabled_all, z_all = blank(), blank()
    legs = {k: pd.DataFrame(np.nan, index=index, columns=cols) for k in ("static", "adaptive")}
    diag = []

    for fold in folds:
        w = slice(fold.train_start, fold.test_end)
        n_train = fold.train_end - fold.train_start
        test = slice(fold.test_start - fold.train_start, fold.test_end - fold.train_start)
        test_index = index[fold.test_start : fold.test_end]
        yw, xw, dyw, dxw, roll_w = y.iloc[w], x.iloc[w], dy.iloc[w], dx.iloc[w], is_roll.iloc[w]

        scr = cointegration.screen(yw.iloc[:n_train], xw.iloc[:n_train])

        # ---- static
        s = hedged(yw, xw, scr.beta)
        ds = hedged(dyw, dxw, scr.beta).mask(roll_w, 0.0)
        vol = trailing_vol(ds)
        cost = unit_cost(spread, scr.beta, s.index)
        fit = ou.fit_ou(s.to_numpy()[:n_train])
        static_on = scr.passed and _tradeable(fit)
        z = (s - fit.mu) / fit.sigma_eq
        entry, exit_, _ = ou.optimize_bands(
            z.to_numpy()[:n_train], ds.to_numpy()[:n_train], vol.to_numpy()[:n_train], cost.to_numpy()[:n_train]
        )
        pos = ou.band_positions(z.to_numpy(), entry, exit_) if static_on else np.zeros(len(z))
        for key, val in (("signal", pos), ("change", ds), ("vol", vol), ("cost", cost)):
            store["static"][key].loc[test_index] = np.asarray(val)[test]

        # ---- adaptive
        # The traded deviation is the one-step forecast error y_t - a_t - b_t x_t:
        # the gap to an equilibrium that drifts at the process noise delta.
        # Likelihood would always pick the fastest filter (it whitens the gap),
        # so delta is chosen with the bands, on training-window net Sharpe.
        best = None
        for delta_k, path in paths.items():
            beta_k = path.prior[w][:, 1:]  # row t is known at the close of t-1
            s_k = pd.Series(path.innovation[w], index=yw.index)
            ds_k = hedged(dyw, dxw, beta_k).mask(roll_w, 0.0)
            vol_k = trailing_vol(ds_k)
            cost_k = unit_cost(spread, np.nan_to_num(beta_k, nan=0.0), s_k.index)
            fit_k = ou.fit_ou(s_k.to_numpy()[:n_train])
            z_k = (s_k - fit_k.mu) / fit_k.sigma_eq
            entry_k, exit_k, score = ou.optimize_bands(
                z_k.to_numpy()[:n_train], ds_k.to_numpy()[:n_train], vol_k.to_numpy()[:n_train],
                cost_k.to_numpy()[:n_train],
            )
            rank = (_tradeable(fit_k), score)
            if best is None or rank > best[0]:
                best = (rank, delta_k, beta_k, s_k, ds_k, vol_k, cost_k, fit_k, z_k, entry_k, exit_k)
        _, delta, beta, sa, dsa, vola, costa, fita, za, entry_a, exit_a = best
        adaptive_on = scr.passed and _tradeable(fita)
        base = ou.band_positions(za.to_numpy(), entry_a, exit_a)

        half_life = ou.rolling_half_life(sa, config.HEALTH_WINDOW)
        disabled = (
            (half_life > config.MAX_HALF_LIFE) | (half_life < config.MIN_HALF_LIFE)
            | (za.abs() > config.STOP_Z) | za.isna()
        ).to_numpy()

        market = rf.market_features(levels.iloc[w], dsa)
        gate_hmm, hmm_states = _hmm_gate(market.to_numpy(), n_train, base, vola.to_numpy(), dsa.to_numpy())
        gate_nn, nn_fitted = _nn_gate(rf.classifier_features(market, za, half_life), sa, za, n_train)

        for variant, gate in (("adaptive_hmm", gate_hmm), ("adaptive_nn", gate_nn)):
            sig = base * gate * ~disabled if adaptive_on else np.zeros(len(base))
            for key, val in (("signal", sig), ("change", dsa), ("vol", vola), ("cost", costa)):
                store[variant][key].loc[test_index] = np.asarray(val, dtype=float)[test]
        disabled_all.loc[test_index] = (disabled | (not adaptive_on))[test].astype(float)
        legs["static"].loc[test_index] = np.concatenate([[1.0], -scr.beta])
        legs["adaptive"].loc[test_index] = np.column_stack([np.ones(len(beta)), -beta])[test]
        z_all.loc[test_index] = za.to_numpy()[test]

        diag.append({
            "spread": spread.name, "fold": fold.k,
            "train_start": index[fold.train_start].date(), "train_end": index[fold.train_end - 1].date(),
            "test_start": test_index[0].date(), "test_end": test_index[-1].date(),
            "eg_pvalue": scr.eg_pvalue, "johansen_pass": scr.johansen_pass, "coint_pass": scr.passed,
            "beta_static": scr.beta[0], "half_life_static": fit.half_life, "entry_static": entry, "exit_static": exit_,
            "kalman_delta": delta, "half_life_adaptive": fita.half_life, "entry_adaptive": entry_a,
            "exit_adaptive": exit_a, "hmm_states_allowed": hmm_states, "nn_fitted": nn_fitted,
            "disabled_share": float(np.mean((disabled | (not adaptive_on))[test])),
        })

    results = {
        v: account(s["signal"], s["change"], s["cost"].ffill().bfill(), is_roll, vol=s["vol"])
        for v, s in store.items()
    }
    live = store["static"]["signal"].notna()
    extras = {"spread_disabled": disabled_all, "z_adaptive": z_all, "legs_static": legs["static"], "legs_adaptive": legs["adaptive"]}
    return StrategyRun(spread.name, spread.group, results, live, pd.DataFrame(diag), extras)


def run_sleeve(levels: pd.DataFrame, rolls: pd.DataFrame, spreads=config.SPREADS, n_jobs: int = 4) -> list[StrategyRun]:
    from joblib import Parallel, delayed

    runs = Parallel(n_jobs=n_jobs)(delayed(run_spread)(s, levels, rolls) for s in spreads)
    return [r for r in runs if r is not None]
