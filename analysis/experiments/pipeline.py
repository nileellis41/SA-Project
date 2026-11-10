"""A parameterised copy of amp_sa.sleeves.pca.run_curve for walk-forward experiments.

With the default Rule and no `fixed`/`loo`, it reproduces the cached PCA runs
exactly (tests/test_pipeline.py checks this). The hooks:

  Rule.entry_fixed   entry band fixed (exit still optimised on the training window)
  Rule.time_stop     close after this many bars held; no re-entry until |z| is back inside the entry band
  Rule.gate          boolean Series known at t; new entries only where True
  Rule.resize_tol    keep the open position unless signal/vol moves more than this fraction from it
  Rule.freeze_size   hold the entry size signal/vol until exit (no intra-trade resizing)
  Rule.vol_gate      new entries only when trailing residual vol >= X; X per member per fold, from training-window
                     vol percentiles VOL_GATE_PCTS, maximising training net P&L per trade (bands stay production)
  fixed             {name: {tenor: weight}}: trade fixed-weight portfolios instead of PCA residuals
  loo                leave-one-out residual: the tenor regressed on factors fitted without it

Every fitted quantity (loadings, LOO betas, bands) uses the training window only.
Bands are optimised with the same rule they are traded with (time stop, gate, entry).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest.engine import Result, account, trailing_vol
from amp_sa.backtest.splits import folds_for
from amp_sa.models import ou
from amp_sa.sleeves import pca
from amp_sa.sleeves.rv import StrategyRun


@dataclass(frozen=True)
class Rule:
    entry_fixed: float | None = None
    time_stop: int | None = None
    gate: pd.Series | None = None
    resize_tol: float | None = None
    freeze_size: bool = False
    vol_gate: bool = False


VOL_GATE_PCTS = (10, 20, 30, 40, 50)  # training-window percentiles of trailing residual vol tried as X


def freeze_positions(target: np.ndarray) -> np.ndarray:
    """Keep the entry size while the side is unchanged; a new size only on entry or flip."""
    pos = np.zeros(len(target))
    for t in range(len(target)):
        prev = pos[t - 1] if t else 0.0
        pos[t] = prev if prev != 0 and np.sign(target[t]) == np.sign(prev) else target[t]
    return pos


def freeze_result(res: Result) -> Result:
    """res with each trade held at its entry size. Works on any engine Result (PCA or pairwise).

    Gross is rescaled by frozen / original held position, so the instrument change is not needed.
    """
    frozen = pd.Series(freeze_positions(res.position.to_numpy()), index=res.position.index)
    held, held_f = res.position.shift(1).fillna(0.0), frozen.shift(1).fillna(0.0)
    ratio = (held_f / held.where(held != 0)).fillna(0.0)
    gross = res.gross * ratio
    cost = (frozen - held_f).abs() * res.unit_cost
    return Result(res.signal, frozen, gross, cost, res.turnover, res.unit_cost)


def fit_vol_gate(z, change, vol, unit_cost, entry, exit_, rule: Rule, allow=None) -> tuple[float, float, float]:
    """(percentile, X, training net per trade): the vol floor that maximises training net P&L per trade."""
    best = (VOL_GATE_PCTS[0], float(np.nanpercentile(vol, VOL_GATE_PCTS[0])), -np.inf)
    safe_vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
    for q in VOL_GATE_PCTS:
        x = float(np.nanpercentile(vol, q))
        ok = np.isfinite(vol) & (vol >= x)
        sig = positions(z, entry, exit_, time_stop=rule.time_stop, allow=ok if allow is None else ok & allow)
        pos = np.nan_to_num(sig / safe_vol)
        if rule.freeze_size:
            pos = freeze_positions(pos)
        side = np.sign(pos)
        n_trades = int(((side != 0) & (side != np.r_[0.0, side[:-1]])).sum())
        if n_trades == 0:
            continue
        pnl = pos[:-1] * np.nan_to_num(change[1:]) - np.abs(np.diff(pos, prepend=0.0))[:-1] * unit_cost[:-1]
        score = pnl.sum() / n_trades
        if score > best[2]:
            best = (q, x, score)
    return best


def positions(z: np.ndarray, entry: float, exit_: float, stop: float = config.STOP_Z,
              time_stop: int | None = None, allow: np.ndarray | None = None) -> np.ndarray:
    """amp_sa.models.ou.band_positions plus an optional time stop and entry gate. Identical when both are None."""
    pos = np.zeros(len(z))
    state, stopped, age = 0.0, False, 0  # age: bars the current state has been set, through t-1
    for t, zt in enumerate(z):
        prev = state
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
            if state != 0 and time_stop is not None and age >= time_stop:
                state, stopped = 0.0, True  # timed out: wait for |z| < entry before a new trade
            if state == 0 and not stopped and (allow is None or allow[t]):
                if zt > entry:
                    state = -1.0
                elif zt < -entry:
                    state = 1.0
        age = age + 1 if (state != 0 and state == prev) else (1 if state != 0 else 0)
        pos[t] = state
    return pos


def optimize(z, change, vol, unit_cost, rule: Rule, allow=None) -> tuple[float, float]:
    """amp_sa.models.ou.optimize_bands, with the rule applied inside the search."""
    entries = (rule.entry_fixed,) if rule.entry_fixed is not None else config.ENTRY_GRID
    best = (entries[len(entries) // 2], config.EXIT_GRID[0])
    best_score = -np.inf
    safe_vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
    for entry in entries:
        for exit_ in config.EXIT_GRID:
            if exit_ >= entry:
                continue
            pos = np.nan_to_num(positions(z, entry, exit_, time_stop=rule.time_stop, allow=allow) / safe_vol)
            pnl = pos[:-1] * np.nan_to_num(change[1:]) - np.abs(np.diff(pos, prepend=0.0))[:-1] * unit_cost[:-1]
            sd = pnl.std()
            score = pnl.mean() / sd if sd > 0 else -np.inf
            if score > best_score:
                best, best_score = (entry, exit_), score
    return best


def account_sticky(signal: pd.Series, change: pd.Series, unit_cost: pd.Series, vol: pd.Series, tol: float) -> Result:
    """engine.account, except an open position is only resized when signal/vol moves more than `tol` from it."""
    signal = signal.reindex(change.index).fillna(0.0)
    target = (signal / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    pos = np.zeros(len(target))
    for t in range(len(target)):
        prev = pos[t - 1] if t else 0.0
        same_side = prev != 0 and np.sign(target[t]) == np.sign(prev)
        pos[t] = prev if same_side and abs(target[t] / prev - 1) <= tol else target[t]
    position = pd.Series(pos, index=change.index)
    held = position.shift(1).fillna(0.0)
    gross = held * change.fillna(0.0)
    cost = (position - held).abs() * unit_cost
    return Result(signal, position, gross, cost, signal.diff().abs().fillna(0.0), unit_cost)


def loo_rows(dw_train: pd.DataFrame, k: int) -> np.ndarray:
    """Row i: one unit of tenor i hedged on the k factors of the other tenors, betas by OLS on training changes."""
    n = dw_train.shape[1]
    out = np.zeros((n, n))
    for i in range(n):
        others = [j for j in range(n) if j != i]
        sub = dw_train.iloc[:, others]
        loadings, _ = pca.factor_loadings(sub, k)
        f = sub.to_numpy() @ loadings
        beta, *_ = np.linalg.lstsq(f, dw_train.iloc[:, i].to_numpy(), rcond=None)
        out[i, i] = 1.0
        out[i, others] = -loadings @ beta
    return out


def run_members(levels: pd.DataFrame, k: int = 2, sparse: bool = False, rule: Rule = Rule(),
                fixed: dict[str, dict[str, float]] | None = None, loo: bool = False,
                label: str | None = None, tenors=pca.TENORS) -> list[StrategyRun]:
    tenors = [t for t in tenors if t in levels.columns]
    base = [t for t in pca.BASE_TENORS if t in tenors]
    d = levels[tenors].diff()
    index = levels.index
    costs = np.array([config.COST_PER_UNIT[t] for t in tenors])
    names = list(fixed) if fixed else tenors
    label = label or (f"pca{k}s" if sparse else f"pca{k}")
    blank = lambda: pd.Series(np.nan, index=index)  # noqa: E731
    store = {m: {"signal": blank(), "change": blank(), "vol": blank(), "cost": blank(), "z": blank()} for m in names}
    legs = {m: pd.DataFrame(np.nan, index=index, columns=tenors) for m in names}
    gate = None if rule.gate is None else rule.gate.reindex(index).fillna(False).to_numpy(bool)
    plain = rule.entry_fixed is None and rule.time_stop is None and gate is None
    diag = []

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
        if fixed:
            rows = {m: np.array([wt.get(t, 0.0) for t in avail]) for m, wt in fixed.items() if set(wt) <= set(avail)}
        else:
            if loo:
                project = loo_rows(dw.iloc[1:n_train], k)
            else:
                loadings, _ = pca.factor_loadings(dw.iloc[1:n_train], k)
                project = pca.sparse_hedges(loadings) if sparse else np.eye(len(avail)) - loadings @ loadings.T
            rows = {t: project[i] for i, t in enumerate(avail)}
        if not rows:
            continue
        # One matrix product over all rows, as production does (row-by-row products round differently).
        resid = pd.DataFrame(dw.to_numpy() @ np.vstack(list(rows.values())).T, index=dw.index, columns=list(rows))
        scores = pca.rolling_ou(resid.cumsum(), pca.OU_WINDOW)[0]
        allow = None if gate is None else gate[w]
        for m, row in rows.items():
            if not np.isfinite(row).all():
                continue
            change = resid[m]
            vol = trailing_vol(change)
            unit_cost = float(np.abs(row) @ costs[cols])
            cost = np.full(len(change), unit_cost)
            z = scores[m].to_numpy()
            if plain:  # the production optimiser, so the baseline is bit-identical
                entry, exit_, _ = ou.optimize_bands(z[:n_train], change.to_numpy()[:n_train], vol.to_numpy()[:n_train],
                                                    cost[:n_train])
            else:
                entry, exit_ = optimize(z[:n_train], change.to_numpy()[:n_train], vol.to_numpy()[:n_train],
                                        cost[:n_train], rule, None if allow is None else allow[:n_train])
            allow_m, gate_fit = allow, (np.nan, np.nan, np.nan)
            if rule.vol_gate:
                v = vol.to_numpy()
                gate_fit = fit_vol_gate(z[:n_train], change.to_numpy()[:n_train], v[:n_train], cost[:n_train],
                                        entry, exit_, rule, None if allow is None else allow[:n_train])
                above = np.isfinite(v) & (v >= gate_fit[1])
                allow_m = above if allow is None else allow & above
            pos = positions(z, entry, exit_, time_stop=rule.time_stop, allow=allow_m)
            for key, val in (("signal", pos), ("change", change), ("vol", vol), ("cost", cost), ("z", z)):
                store[m][key].loc[test_index] = np.asarray(val, dtype=float)[test]
            legs[m].loc[test_index, avail] = row
            diag.append({"strategy": f"{label}_{m}", "fold": fold.k, "test_start": test_index[0].date(),
                         "entry": entry, "exit": exit_, "unit_cost": unit_cost, "vol_gate_pct": gate_fit[0],
                         "vol_gate_x_bp": gate_fit[1], "vol_gate_train_net_per_trade": gate_fit[2]})

    diag = pd.DataFrame(diag)
    runs = []
    for m in names:
        s = store[m]
        live = s["signal"].notna()
        if not live.any():
            continue
        unit = s["cost"].ffill().bfill()
        if rule.resize_tol is None:
            res = account(s["signal"], s["change"], unit, vol=s["vol"])
        else:
            res = account_sticky(s["signal"], s["change"], unit, s["vol"], rule.resize_tol)
        if rule.freeze_size:
            res = freeze_result(res)
        name = f"{label}_{m}"
        runs.append(StrategyRun(name, pca.GROUP, {pca.VARIANT: res}, live, diag[diag["strategy"] == name],
                                {"legs_adaptive": legs[m], "ou_z": s["z"]}))
    return runs
