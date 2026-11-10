"""Backlog item 1: on-the-run roll contamination in the USGG generic yields (a data check, not a trial).

    uv run python -m analysis.experiments.roll_check [--force]

The design, the placebo and the kill criterion are pre-registered in analysis/experiments/roll_check.md.
In short: on each tenor's new-issue bars (auction date, issue date, or the bar after the auction), replace that tenor's daily change by
the maturity-interpolated change of its neighbours, rebuild the levels, and re-run the production PCA models
walk-forward through pipeline.run_members. Compare with a placebo that does the same on shifted ordinary days.

Writes results/roll_check/*.csv (summary.md is written by hand from them); caches the re-runs in results/roll_check/runs.pkl.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from amp_sa import config
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import metrics
from amp_sa.run_lessons import model_trades
from amp_sa.sleeves import pca
from analysis.experiments.pipeline import run_members
from analysis.portfolio_diagnostics import book as bk
from analysis.portfolio_diagnostics.common import Model, load_models
from analysis.portfolio_diagnostics.external import AUCTION_TENORS, RAW

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "roll_check"
ANN = config.ANN
TENORS = list(pca.TENORS)
MATURITY = {"ust_2y": 2, "ust_3y": 3, "ust_5y": 5, "ust_7y": 7, "ust_10y": 10, "ust_30y": 30}
OFFSETS = range(-3, 6)
QUIET = 5  # bars either side of any event excluded from the "ordinary bars" reference
PLACEBO_SHIFT, PLACEBO_GAP = 6, 2
HORIZON = 10  # forward bars for the IC, as in diagnostics step 5
DATE_TYPES = ("auction", "issue", "switch")  # switch = auction bar + 1: added after the event study (roll_check.md)
CONFIGS = tuple(c for dt in DATE_TYPES for c in (dt, f"{dt}_placebo"))


# ---------- calendar ----------

def auction_calendar() -> pd.DataFrame:
    """Nominal coupon auctions: tenor, auction date, issue date, reopening."""
    frames = []
    for kind in ("Note", "Bond"):
        d = pd.DataFrame(json.load(open(RAW / "treasury" / f"auctions_{kind}.json")))
        d = d[(d["inflationProtected"] != "Yes") & (d["floatingRate"] != "Yes")]
        d = d[d["originalSecurityTerm"].isin(AUCTION_TENORS)]
        frames.append(pd.DataFrame({
            "tenor": d["originalSecurityTerm"].map(AUCTION_TENORS).to_numpy(),
            "auction": pd.to_datetime(d["competitiveAuctionCloseDate"].str[:10]).to_numpy(),
            "issue": pd.to_datetime(d["issueDate"].str[:10]).to_numpy(),
            "reopening": d["reopening"].eq("Yes").to_numpy(),
        }))
    return pd.concat(frames).drop_duplicates().sort_values("auction").reset_index(drop=True)


def to_bars(dates: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Position of the first panel bar on or after each date (dates past the panel end dropped)."""
    pos = index.searchsorted(pd.DatetimeIndex(dates))
    return np.unique(pos[pos < len(index)])


def event_bars(cal: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
    """{(date_type, tenor): bars} for new issues (switch = auction bar + 1), plus ('reopen', tenor) and ('any', tenor) (all real events)."""
    out = {}
    for t in TENORS:
        c = cal[cal["tenor"] == t]
        new = c[~c["reopening"]]
        for dt in ("auction", "issue"):
            out[(dt, t)] = to_bars(new[dt], index)
        nxt = out[("auction", t)] + 1
        out[("switch", t)] = nxt[nxt < len(index)]
        out[("reopen", t)] = to_bars(c.loc[c["reopening"], "auction"], index)
        a = to_bars(c["auction"], index)
        out[("any", t)] = np.unique(np.concatenate([a, a + 1, to_bars(c["issue"], index)]))
    return out


def placebo_bars(real: np.ndarray, any_real: np.ndarray, n: int) -> np.ndarray:
    p = real + PLACEBO_SHIFT
    p = p[p < n]
    if len(any_real):
        dist = np.abs(p[:, None] - any_real[None, :]).min(axis=1)
        p = p[dist > PLACEBO_GAP]
    return np.unique(p)


# ---------- idiosyncratic move ----------

def neighbour_change(dy: pd.DataFrame) -> pd.DataFrame:
    """Per tenor: change interpolated in maturity from the nearest available shorter and longer tenor (flat at the edges)."""
    x = dy[TENORS].to_numpy()
    mats = np.array([MATURITY[t] for t in TENORS], dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(len(TENORS)):
        for r in range(x.shape[0]):
            ok = np.isfinite(x[r]) & (np.arange(len(TENORS)) != i)
            lo = np.flatnonzero(ok & (mats < mats[i]))
            hi = np.flatnonzero(ok & (mats > mats[i]))
            if len(lo) and len(hi):
                a, b = lo[-1], hi[0]
                wgt = (mats[i] - mats[a]) / (mats[b] - mats[a])
                out[r, i] = (1 - wgt) * x[r, a] + wgt * x[r, b]
            elif len(lo):
                out[r, i] = x[r, lo[-1]]
            elif len(hi):
                out[r, i] = x[r, hi[0]]
    return pd.DataFrame(out, index=dy.index, columns=TENORS)


def event_study(e: pd.DataFrame, ev: dict) -> pd.DataFrame:
    n, rows = len(e), []
    for t in TENORS:
        x = e[t].to_numpy()
        quiet = np.ones(n, bool)
        for b in ev[("any", t)]:
            quiet[max(0, b - QUIET): b + QUIET + 1] = False
        ref = np.nanmean(np.abs(x[quiet & np.isfinite(x)]))
        for dt in (*DATE_TYPES, "reopen"):
            bars = ev[(dt, t)]
            for o in OFFSETS if dt != "reopen" else (0,):
                p = bars + o
                v = x[p[(p >= 0) & (p < n)]]
                v = v[np.isfinite(v)]
                if len(v) < 5:
                    continue
                rows.append({"tenor": t, "date_type": dt, "offset": o, "events": len(v),
                             "mean_abs_e_bp": np.mean(np.abs(v)), "ordinary_mean_abs_e_bp": ref,
                             "ratio": np.mean(np.abs(v)) / ref, "mean_e_bp": v.mean(),
                             "t_stat": v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))})
    return pd.DataFrame(rows)


# ---------- treatment ----------

def cleaned_levels(levels: pd.DataFrame, dy: pd.DataFrame, nb: pd.DataFrame, bars: dict) -> pd.DataFrame:
    """Levels with dy_T replaced by the neighbour value on `bars[T]`; every other change unchanged."""
    clean = dy.copy()
    for t, b in bars.items():
        col = clean.columns.get_loc(t)
        repl = nb[t].to_numpy()[b]
        keep = np.isfinite(repl) & np.isfinite(clean.iloc[b, col].to_numpy())
        clean.iloc[b[keep], col] = repl[keep]
    out = levels.copy()
    out[TENORS] = levels[TENORS] + (clean - dy).fillna(0.0).cumsum()
    return out


def config_bars(name: str, ev: dict, n: int) -> dict:
    dt = name.split("_")[0]
    if name.endswith("placebo"):
        return {t: placebo_bars(ev[(dt, t)], ev[("any", t)], n) for t in TENORS}
    return {t: ev[(dt, t)] for t in TENORS}


def _run(lev, name, k, sparse):
    return name, k, run_members(lev, k=k, sparse=sparse)


def load_runs(levels, dy, nb, ev, force: bool) -> dict:
    cache = OUT / "runs.pkl"
    if cache.exists() and not force:
        return pickle.load(open(cache, "rb"))
    jobs = []
    for name in CONFIGS:
        lev = cleaned_levels(levels, dy, nb, config_bars(name, ev, len(levels)))
        jobs += [(lev, name, 2, False), (lev, name, 3, True)]
    done = Parallel(n_jobs=4)(delayed(_run)(*j) for j in jobs)
    runs = {(name, k): r for name, k, r in done}
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(runs, open(cache, "wb"))
    return runs


# ---------- metrics ----------

def ic(items, levels, rolls, model: Model) -> dict:
    """Spearman of -z with the forward HORIZON-bar member change, PCA members only, all bars and entry bars."""
    b = bk.build_book(model, levels, rolls)
    az, af, ez, ef = [], [], [], []
    for r, v in items:
        z = r.extras.get("ou_z")
        if z is None:
            continue
        c = b.change[r.name].where(b.live[r.name]).cumsum()
        fwd = c.shift(-HORIZON) - c
        ok = z.notna() & fwd.notna() & b.live[r.name]
        az.append(z[ok]); af.append(fwd[ok])
        side = np.sign(r.results[v].position)
        entry = (side != 0) & (side != side.shift(1).fillna(0.0)) & ok
        ez.append(z[entry]); ef.append(fwd[entry])
    out = {}
    for scope, zz, ff in (("all", az, af), ("entry", ez, ef)):
        zz, ff = pd.concat(zz), pd.concat(ff)
        out[f"ic_{scope}"] = stats.spearmanr(-zz, ff)[0]
        out[f"n_{scope}"] = len(zz)
    return out


def summarize(config_name: str, label: str, items, levels, rolls, with_ic: bool) -> dict:
    port, _, _ = s1.sleeve(items)
    m = Model(label, label, items, port)
    live = port.net.notna()
    tv = bk.reprice(m, bk.time_varying_cost)
    row = {"config": config_name, "model": label,
           "sharpe_gross": metrics.sharpe(port.gross[live]), "sharpe_net": metrics.sharpe(port.net[live]),
           "sharpe_net_tv": metrics.sharpe(tv.net[tv.net.notna()]),
           "gross_pa": port.gross[live].mean() * ANN, "trades_per_year": len(model_trades(items)) / (live.sum() / ANN)}
    if with_ic:
        row.update(ic(items, levels, rolls, m))
    return row


def verdict(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = res[res["config"] == "baseline"].set_index("model")["sharpe_gross"]
    for dt in DATE_TYPES:
        tr = res[res["config"] == dt].set_index("model")
        pl = res[res["config"] == f"{dt}_placebo"].set_index("model")
        for model in tr.index:
            g, gp, gb = tr.at[model, "sharpe_gross"], pl.at[model, "sharpe_gross"], base[model]
            rows.append({"date_type": dt, "model": model, "gross_baseline": gb, "gross_placebo": gp, "gross_treated": g,
                         "drop_vs_placebo": 1 - g / gp, "drop_vs_baseline": 1 - g / gb,
                         "KILL (treated < 2/3 placebo)": g < 2 / 3 * gp,
                         "flag (treated < 2/3 baseline)": g < 2 / 3 * gb})
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    levels, rolls, models = load_models()
    cal = auction_calendar()
    ev = event_bars(cal, levels.index)
    dy = levels[TENORS].diff()
    nb = neighbour_change(dy)
    e = dy - nb

    counts = pd.DataFrame([{"tenor": t, **{k: len(ev[(k, t)]) for k in (*DATE_TYPES, "reopen")},
                            **{f"{k}_placebo": len(config_bars(f"{k}_placebo", ev, len(levels))[t]) for k in DATE_TYPES}}
                           for t in TENORS])
    es = event_study(e, ev)
    counts.to_csv(OUT / "event_counts.csv", index=False)
    es.to_csv(OUT / "event_study.csv", index=False)

    runs = load_runs(levels, dy, nb, ev, force)
    credit = [(r, v) for r, v in models["headline"].items if not r.name.startswith("pca")]
    rows = []
    base = {"Headline": models["headline"].items, "PCA-2 full": models["pca2"].items,
            "PCA-3 sparse": models["pca3_sparse"].items}
    for label, items in base.items():
        rows.append(summarize("baseline", label, items, levels, rolls, label != "Headline"))
    for name in CONFIGS:
        lev = cleaned_levels(levels, dy, nb, config_bars(name, ev, len(levels)))
        p2 = [(r, pca.VARIANT) for r in runs[(name, 2)]]
        p3 = [(r, pca.VARIANT) for r in runs[(name, 3)]]
        rows.append(summarize(name, "Headline", p2 + credit, lev, rolls, False))
        rows.append(summarize(name, "PCA-2 full", p2, lev, rolls, True))
        rows.append(summarize(name, "PCA-3 sparse", p3, lev, rolls, True))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    v = verdict(res)
    v.to_csv(OUT / "verdict.csv", index=False)

    fmt = {"display.width": 250, "display.max_columns": 30, "display.float_format": "{:.3f}".format}
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print(counts.to_string(index=False))
        print(es[es["offset"].isin([-1, 0, 1, 2])].to_string(index=False))
        print(res.to_string(index=False))
        print(v.to_string(index=False))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
