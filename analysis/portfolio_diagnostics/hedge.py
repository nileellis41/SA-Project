"""Step 3: hedge performance.

Factor exposures use the PCA-3 loadings of the fold in effect on each bar (fitted on that
fold's training window, logged by the [diag-logging] change). Signs are fixed so that
PC1 (level) loads positively on average, PC2 (slope) rises from 2y to 30y, and PC3
(curvature) is wings minus belly (positive on 2y + 30y - 2 x 5y).

Per-trade attribution splits each bar's member change into
    sum_k (w . L_k) f_k        leaked factor P&L, f = L' dy (factor move in bp)
    w . (dy - L f)             residual beyond three factors
for Treasury legs. Non-Treasury legs (swaps, credit) go wholly to `other_legs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest.splits import folds_for
from amp_sa.sleeves import pca
from analysis.portfolio_diagnostics.book import TREASURY, Book

PCS = ("pc1", "pc2", "pc3")


def orient(load: pd.DataFrame) -> pd.DataFrame:
    """Fix eigenvector signs (rows = tenors in curve order, columns pc1..pc3)."""
    load = load.copy()
    t = list(load.index)
    if load["pc1"].sum() < 0:
        load["pc1"] *= -1
    if load["pc2"].iloc[-1] - load["pc2"].iloc[0] < 0:
        load["pc2"] *= -1
    if "pc3" in load:
        belly = "ust_5y" if "ust_5y" in t else t[len(t) // 2]
        if load["pc3"].iloc[0] + load["pc3"].iloc[-1] - 2 * load.loc[belly, "pc3"] < 0:
            load["pc3"] *= -1
    return load


def fold_loadings(pca3_runs) -> dict[pd.Timestamp, pd.DataFrame]:
    """test_start -> oriented 3-factor loadings (tenors x pc1..pc3)."""
    raw = pca3_runs[0].extras["fold_loadings"]
    out = {}
    for start, g in raw.groupby("test_start"):
        load = g.set_index("tenor")[list(PCS)]
        load = load.loc[[t for t in TREASURY if t in load.index]]
        out[pd.Timestamp(start)] = orient(load)
    return out


def trailing_loadings(levels: pd.DataFrame, window: int = config.ANN, step: int = 21) -> dict[pd.Timestamp, pd.DataFrame]:
    """Causal drift check: PCA-3 loadings from the `window` bars of yield changes ending the bar before each
    key date, re-estimated every `step` bars, over the tenors with a full window. Key = first bar they apply to."""
    dy = levels[list(TREASURY)].diff()
    out = {}
    for i in range(window + 1, len(dy), step):
        win = dy.iloc[i - window : i]
        cols = [c for c in TREASURY if win[c].notna().all()]
        if len(cols) < 4:
            continue
        load, _ = pca.factor_loadings(win[cols], 3)
        out[dy.index[i]] = orient(pd.DataFrame(load, index=cols, columns=list(PCS)))
    return out


def loadings_by_bar(loads: dict, index: pd.Index) -> pd.Series:
    """Which fold's loadings are in effect on each bar (the last fold whose test block has started)."""
    starts = pd.Series(list(loads), index=list(loads)).sort_index()
    return starts.reindex(index, method="ffill")


def factor_exposure(book: Book, loads: dict) -> pd.DataFrame:
    """Daily exposure of the netted book to PC1..PC3: fraction of capital per 1 bp factor move."""
    D = book.dv01_net()
    which = loadings_by_bar(loads, D.index)
    out = pd.DataFrame(np.nan, index=D.index, columns=list(PCS))
    for start, rows in D.groupby(which):
        L = loads[start]
        out.loc[rows.index] = rows.reindex(columns=L.index, fill_value=0.0).to_numpy() @ L.to_numpy()
    treas = [c for c in D.columns if c in TREASURY]
    out["gross_dv01"] = D[treas].abs().sum(axis=1)
    out["net_dv01"] = D[treas].sum(axis=1)
    return out


def trade_ids(pos: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(tid, held_tid) as in amp_sa.run_lessons.trades: tid on bars where the trade is set, held_tid where it is held."""
    side = np.sign(pos)
    start = (side != 0) & (side != side.shift(1).fillna(0.0))
    tid = start.cumsum().where(side != 0)
    return tid, tid.shift(1)


def bar_attribution(book: Book, loads: dict, levels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per member, per bar, in risk units: gross split into pc1..pc3 leak, residual and other legs."""
    dy = levels[list(TREASURY)].diff()
    which = loadings_by_bar(loads, dy.index)
    out = {}
    held = book.pos.shift(1).fillna(0.0)
    for m in book.members:
        L = book.legs[m]
        treas = [c for c in L.columns if c in TREASURY]
        parts = pd.DataFrame(0.0, index=L.index, columns=[*PCS, "residual", "other_legs"])
        for start, idx in L.groupby(which).groups.items():
            ld = loads[start]
            w = L.loc[idx].reindex(columns=ld.index, fill_value=0.0).to_numpy()
            d = dy.loc[idx, ld.index].fillna(0.0).to_numpy()
            f = d @ ld.to_numpy()  # factor moves, bp
            expo = w @ ld.to_numpy()  # member exposure per unit, per factor
            leak = expo * f
            parts.loc[idx, list(PCS)] = leak
            parts.loc[idx, "residual"] = (w * d).sum(axis=1) - leak.sum(axis=1)
        other = [c for c in L.columns if c not in treas]
        parts["other_legs"] = book.change[m] - parts[[*PCS, "residual"]].sum(axis=1)
        if not other:
            # Treasury-only member: other_legs is rounding plus bars where a leg is outside the fold's loadings
            parts["residual"] += parts["other_legs"]
            parts["other_legs"] = 0.0
        out[m] = parts.mul(held[m], axis=0)
    return out


def trade_attribution(book: Book, bars: dict[str, pd.DataFrame], hedged_k: int | None) -> pd.DataFrame:
    rows = []
    for (r, v) in book.model.items:
        m = r.name
        res = r.results[v]
        pos = book.pos[m]
        tid, held_tid = trade_ids(pos)
        cost = res.cost.where(book.live[m], 0.0)
        cost_tid = tid.fillna(held_tid)
        g = bars[m].groupby(held_tid).sum()
        t = g.assign(cost=cost.groupby(cost_tid).sum(), bars=held_tid.value_counts(),
                     entry=pos.index.to_series().groupby(tid).first(),
                     exit=pos.index.to_series().groupby(held_tid).last(),
                     entry_z=_entry_z(r, tid)).dropna(subset=["bars"])
        t["gross"] = t[[*PCS, "residual", "other_legs"]].sum(axis=1)
        t["net"] = t["gross"] - t["cost"]
        hedged = list(PCS[:hedged_k]) if hedged_k else list(PCS)
        t["leak_hedged_factors"] = t[hedged].sum(axis=1)
        t.insert(0, "member", m)
        rows.append(t.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def _entry_z(r, tid: pd.Series) -> pd.Series:
    z = r.extras.get("ou_z", r.extras.get("z_adaptive"))
    if z is None:
        return pd.Series(dtype=float)
    return z.reindex(tid.index).groupby(tid).first()


def leg_weights_by_fold(book: Book, starts: pd.DatetimeIndex) -> pd.DataFrame:
    """Mean leg weight of each member over each test block (fold x leg). Constant within a PCA fold;
    the Kalman legs of the pairwise spreads move daily, so theirs is the fold average."""
    rows = []
    fold = pd.Series(np.arange(len(starts)), index=starts).reindex(book.pos.index, method="ffill")
    for m in book.members:
        L = book.legs[m].where(book.live[m], axis=0)
        g = L.groupby(fold).mean().dropna(how="all")
        for k, row in g.iterrows():
            rows += [{"member": m, "date": starts[int(k)], "tenor": c, "weight": row[c]} for c in g.columns]
    return pd.DataFrame(rows)


def fold_starts(levels: pd.DataFrame) -> pd.DatetimeIndex:
    base = [t for t in pca.BASE_TENORS if t in levels.columns]
    return pd.DatetimeIndex([levels.index[f.test_start] for f in folds_for(levels[base])])


def refit_turnover(book: Book, levels: pd.DataFrame, cost_table: pd.DataFrame) -> pd.DataFrame:
    """At each fold boundary b: the unpriced re-weight W_b |p_{b-1}| |w_b - w_{b-1}| . c, and the cost the
    engine charged on bars b-1 and b (band state is re-initialised, so positions can jump)."""
    starts = [d for d in fold_starts(levels) if d in book.pos.index]
    held = book.pos.shift(1).fillna(0.0)
    port_cost = book.model.port.cost
    typical = port_cost.mean()  # the median bar trades nothing
    rows = []
    for b in starts:
        i = book.pos.index.get_loc(b)
        prev = book.pos.index[i - 1]
        rew = 0.0
        for m in book.members:
            L = book.legs[m]
            dw = (L.loc[b] - L.loc[prev]).abs()
            if book.live[m].loc[b] and book.live[m].loc[prev]:
                rew += book.w_member.loc[b] * abs(held[m].loc[b]) * float((dw * cost_table.loc[b, dw.index]).sum())
        rows.append({"fold_start": b.date(), "unpriced_reweight": rew,
                     "charged_boundary_bars": float(port_cost.loc[[prev, b]].sum()),
                     "mean_two_bars": 2 * typical})
    return pd.DataFrame(rows)


def own_leg(member: str) -> str:
    """The tenor a member is about: the PCA residual's tenor, or the pairwise spread's y leg."""
    if member.startswith("pca"):
        return member.split("_", 1)[1]
    return next(s.y for s in config.SPREADS if s.name == member)


def residual_vs_raw(book: Book, levels: pd.DataFrame) -> pd.DataFrame:
    """Member residual change vs its own tenor's raw yield change and the raw 5y change, bp, live bars."""
    dy = levels.diff()
    rows = []
    for m in book.members:
        own = own_leg(m)
        if own not in TREASURY:
            continue  # credit pairs: no raw Treasury comparison
        live = book.live[m] & (book.legs[m].abs().sum(axis=1) > 0)
        res = book.change[m][live]
        rows.append({"member": m, "own_leg": own, "resid_vol_bp": res.std(),
                     "own_leg_vol_bp": dy[own][live].std(), "ust_5y_vol_bp": dy["ust_5y"][live].std(),
                     "variance_removed_vs_own": 1 - res.var() / dy[own][live].var(),
                     "variance_removed_vs_5y": 1 - res.var() / dy["ust_5y"][live].var()})
    return pd.DataFrame(rows)


def rolling_vol(book: Book, levels: pd.DataFrame, window: int = config.ANN) -> pd.DataFrame:
    """Rolling 1y vol: equal-weight average over live members of the residual change, vs the raw 5y change."""
    ch = book.change.where(book.live)
    vols = ch.rolling(window, min_periods=window // 2).std()
    return pd.DataFrame({"residual_avg_member": vols.mean(axis=1),
                         "raw_ust_5y": levels["ust_5y"].diff().rolling(window, min_periods=window // 2).std()})
