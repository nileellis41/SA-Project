"""Cost-reduction round 1: freeze size at entry, then a per-fold vol floor on new entries.

    uv run python -m analysis.experiments.cost_round1 [--force]

Configs, on the headline (PCA-2 full + adaptive credit pairs) and PCA-3 sparse:
  baseline              the cached production runs, sized by run_sleeve1.sleeve
  freeze_size           each trade keeps its entry size (member: signal/vol at entry; sleeve: the sleeve
                        weight of its first held bar, known at the entry close). No intra-trade resizing and
                        no leverage re-scaling of open trades. New entries are sized as before.
  freeze_size+vol_gate  as freeze_size, and PCA members take new entries only when trailing residual vol
                        >= X, X fitted per member per fold on the training window (pipeline.fit_vol_gate)

Bands, folds, PCA, OU window and the cost table are unchanged. Bands and X are fitted at 1x cost;
0.5x and 2x re-price the same positions. The drawdown overlay stays off, as in production run_sleeve1.sleeve.
The headline's credit pairs come from amp_sa.sleeves.rv and are frozen but not vol-gated.

Writes results/cost_round1/*.csv and summary.md; caches the gated runs in results/cost_round1/runs.pkl.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import metrics
from amp_sa.portfolio.combine import combine
from amp_sa.run_lessons import model_trades
from analysis.experiments.pipeline import Rule, freeze_result, run_members
from analysis.portfolio_diagnostics import signals_regimes as sr
from analysis.portfolio_diagnostics.common import load_models
from analysis.portfolio_diagnostics.run_all import TRIALS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cost_round1"
ANN = config.ANN
COST_MULTS = (0.5, 1.0, 2.0)
CONFIGS = ("baseline", "freeze_size", "freeze_size+vol_gate")
MODELS = {"Headline": "headline", "PCA-3 sparse": "pca3_sparse"}
# This round's 4 trials (2 configs x 2 models; cost multiples are not trials) are in run_all.TRIALS.
STEP8_TRIALS = 12  # analysis/experiments/run_experiments.py: 6 configs x 2 constructions, not in TRIALS


@dataclass
class Book:
    gross: pd.Series  # fraction of capital per bar, NaN before the sleeve is sized
    cost: pd.Series

    @property
    def net(self) -> pd.Series:
        return self.gross - self.cost

    def net_at(self, m: float) -> pd.Series:
        return self.gross - m * self.cost


def distinct_weights(items) -> tuple[pd.DataFrame, float]:
    """Member weights per bar with each distinct portfolio counted once (vs run_sleeve1.sleeve's 1/n_live).

    Members whose leg vectors are equal up to scale on a bar are one portfolio (sparse PCA-3: 2y = 3y = 5y
    before the 7y enters). Each distinct portfolio gets 1/n_distinct, split equally over its copies.
    Also returns the largest gross P&L gap between copies, which should be zero.
    """
    names = [r.name for r, _ in items]
    live = pd.DataFrame({r.name: r.live for r, _ in items}).fillna(False).astype(bool)
    cols = sorted({c for r, _ in items for c in r.extras["legs_adaptive"].columns})
    legs = np.stack([r.extras["legs_adaptive"].reindex(index=live.index, columns=cols).fillna(0.0).to_numpy()
                     for r, _ in items], axis=1)
    big = np.abs(legs).argmax(axis=2)
    pivot = np.take_along_axis(legs, big[..., None], axis=2)
    norm = np.round(legs / np.where(pivot == 0, 1.0, pivot), 6)
    gross = pd.DataFrame({r.name: r.results[v].gross for r, v in items}).reindex(live.index).to_numpy()
    omega, gap = np.full(live.shape, np.nan), 0.0
    lv = live.to_numpy()
    for t in range(len(live)):
        on = np.flatnonzero(lv[t])
        if not len(on):
            continue
        groups = {}
        for j in on:
            groups.setdefault(norm[t, j].tobytes(), []).append(j)
        for g in groups.values():
            omega[t, g] = 1.0 / (len(g) * len(groups))
            gap = max(gap, float(np.ptp(gross[t, g])))
    return pd.DataFrame(omega, index=live.index, columns=names), gap


def weighted_sleeve(items, omega: pd.DataFrame) -> Book:
    """run_sleeve1.sleeve with member weights omega instead of the equal-weight mean (same combine, same hold cost)."""
    def wsum(get):
        return (pd.DataFrame({r.name: get(r.results[v]).where(r.live) for r, v in items}) * omega).sum(axis=1, min_count=1)
    port = combine({"s": (wsum(lambda x: x.gross), wsum(lambda x: x.cost))}, budget={"s": 1.0}, use_overlay=False,
                   hold_cost={"s": wsum(lambda x: x.position.abs() * x.unit_cost)})
    return Book(port.gross, port.cost)


def frozen_sleeve(items, freeze_weights: bool = True, omega: pd.DataFrame | None = None) -> Book:
    """run_sleeve1.sleeve with each trade's sleeve weight fixed at its first held bar.

    The weight path w_t is production's: combine() on the equal-weight risk-unit sleeve, which depends
    on member P&L through t-1 only. Member m's exposure over bar t is F * held_t, F = w/n_live of the
    trade's first held bar; cost is the change in that exposure times the unit cost.
    freeze_weights=False uses w_t/n_t every bar (a plumbing check: gross then equals production's).
    omega: member weights per bar in place of 1/n_live (distinct_weights for the merged basis).
    """
    live = pd.DataFrame({r.name: r.live for r, _ in items}).fillna(False).astype(bool)
    if omega is None:
        omega = live.where(live).div(live.sum(axis=1).replace(0, np.nan), axis=0).astype(float)
    g = (pd.DataFrame({r.name: r.results[v].gross.where(r.live) for r, v in items}) * omega).sum(axis=1, min_count=1)
    c = (pd.DataFrame({r.name: r.results[v].cost.where(r.live) for r, v in items}) * omega).sum(axis=1, min_count=1)
    w = combine({"s": (g, c)}, budget={"s": 1.0}, use_overlay=False).weights["s"]
    gross, cost = [], []
    for r, v in items:
        res, lv, f = r.results[v], live[r.name], w * omega[r.name]
        held = res.position.shift(1).fillna(0.0).where(lv, 0.0)
        if freeze_weights:
            side = np.sign(held)
            tid = ((side != 0) & (side != side.shift(1).fillna(0.0))).cumsum().where(side != 0)
            F = f.groupby(tid).transform("first").fillna(0.0)  # first non-NaN weight of the trade
        else:
            F = f.where(held != 0, 0.0).fillna(0.0)
        H = F * held
        nxt = H.shift(-1).fillna(H)
        gross.append(F * res.gross.where(lv, 0.0))
        cost.append((nxt - H).abs() * res.unit_cost)
    on = w.notna()
    return Book(sum(gross).where(on), sum(cost).where(on))


def frozen_items(items):
    return [(replace(r, results={**r.results, v: freeze_result(r.results[v])}), v) for r, v in items]


def load_gated(levels, force: bool) -> dict:
    cache = OUT / "runs.pkl"
    if cache.exists() and not force:
        return pickle.load(open(cache, "rb"))
    rule = Rule(freeze_size=True, vol_gate=True)
    runs = {"PCA-2 full": run_members(levels, k=2, sparse=False, rule=rule),
            "PCA-3 sparse": run_members(levels, k=3, sparse=True, rule=rule)}
    pickle.dump(runs, open(cache, "wb"))
    return runs


def entry_vol(trades: pd.DataFrame, items) -> pd.Series:
    """Residual vol at entry in bp/day: |signal| / |position| on the entry bar (as signals_regimes.entry_context)."""
    res = {r.name: r.results[v] for r, v in items}
    first = trades["instrument"].str.split(" = ").str[0]
    return pd.Series([abs(res[m].signal.at[e]) / abs(res[m].position.at[e]) if res[m].position.at[e] != 0 else np.nan
                      for m, e in zip(first, trades["entry"])], index=trades.index)


def rows_for(model: str, cfg: str, book: Book, items) -> tuple[list[dict], pd.DataFrame]:
    live = book.net.notna()
    years = live.sum() / ANN
    t = model_trades(items)
    t["entry_vol_bp"] = entry_vol(t, items)
    g, c = book.gross[live], book.cost[live]
    out = []
    for m in COST_MULTS:
        tnet = t["gross"] - m * t["cost"]
        out.append({"model": model, "config": cfg, "cost_mult": m,
                    "sharpe_gross": metrics.sharpe(g), "sharpe_net": metrics.sharpe(g - m * c),
                    "trades_per_year": len(t) / years, "hit_rate_net": float((tnet > 0).mean()),
                    "mean_net_per_trade": float(tnet.mean()), "mean_gross_per_trade": float(t["gross"].mean()),
                    "net_pa_pct": (g - m * c).mean() * ANN * 100, "cost_pa_pct": m * c.mean() * ANN * 100,
                    "breakeven_cost_mult": g.mean() / c.mean()})
    return out, t.assign(model=model, config=cfg)


def terciles(trades: pd.DataFrame) -> pd.DataFrame:
    """Mean net per trade (1x) by entry-vol tercile. Cut points are the baseline's, so both configs share buckets."""
    out = []
    for model, t in trades.groupby("model", sort=False):
        cuts = t.loc[t["config"] == "baseline", "entry_vol_bp"].quantile([1 / 3, 2 / 3]).to_numpy()
        bins = np.r_[-np.inf, cuts, np.inf]
        for cfg in ("baseline", "freeze_size+vol_gate"):
            x = t[t["config"] == cfg].copy()
            x["tercile"] = pd.cut(x["entry_vol_bp"], bins, labels=["low vol", "mid vol", "high vol"])
            gr = x.groupby("tercile", observed=False)
            out.append(pd.DataFrame({"model": model, "config": cfg, "trades": gr.size(),
                                     "entry_vol_bp_max": [cuts[0], cuts[1], np.inf],
                                     "median_entry_vol_bp": gr["entry_vol_bp"].median(),
                                     "mean_gross": gr["gross"].mean(), "mean_cost": gr["cost"].mean(),
                                     "mean_net": gr["net"].mean()}).reset_index())
    return pd.concat(out, ignore_index=True)


def deflated(books: dict, results: pd.DataFrame) -> pd.DataFrame:
    """DSR of the best new config (net Sharpe at 1x), at the documented trial count plus this round."""
    one = results[(results["cost_mult"] == 1.0) & (results["config"] != "baseline")]
    best = one.loc[one["sharpe_net"].idxmax()]
    b = books[(best["model"], best["config"])]
    live = b.net.notna()
    model_sr = pd.read_csv(s1.REPORT_DIR / "tables" / "sleeve1_models.csv")["sharpe"]
    cf_sr = pd.read_csv(s1.REPORT_DIR / "tables" / "cost_filter_effect.csv").query("filter != 'no filter'")["sharpe_net"]
    cv = pd.read_csv(s1.REPORT_DIR / "tables" / "curve_full.csv")
    step8 = pd.read_csv(Path(__file__).resolve().parent / "results.csv")
    step8 = step8[(step8["experiment"] != "baseline") & (step8["model"] != "Headline")]["sharpe_net"]
    broad = pd.concat([model_sr, cf_sr, cv[~cv["strategy"].str.contains("PCA")]["sharpe"], one["sharpe_net"]])
    rows = []
    for name, n, sds in (("documented, incl. this round", sum(TRIALS.values()), broad),
                         ("documented + step-8 experiments + this round",
                          sum(TRIALS.values()) + STEP8_TRIALS, pd.concat([broad, step8]))):
        sd = sds.std(ddof=1)
        rows.append({"trial_set": name, "n_trials": n, "trial_sharpe_sd": sd, "best": f"{best['model']} / {best['config']}",
                     "sharpe_net": metrics.sharpe(b.net[live]), "dsr_net": sr.deflated(b.net[live], n, sd),
                     "sharpe_gross": metrics.sharpe(b.gross[live]), "dsr_gross": sr.deflated(b.gross[live], n, sd)})
    return pd.DataFrame(rows)


def check_baseline(models, levels) -> pd.DataFrame:
    """Baseline must equal production: the sleeve table, and the pipeline re-run of the PCA members."""
    prod = pd.read_csv(s1.REPORT_DIR / "tables" / "sleeve1_models.csv").set_index("model")
    rows = []
    for label, key in MODELS.items():
        m = models[key]
        live = m.port.net.notna()
        rows.append({"check": f"{label} sleeve net Sharpe vs reports/tables/sleeve1_models.csv",
                     "value": metrics.sharpe(m.port.net[live]), "production": prod.at[m.name, "sharpe"]})
        plumb = frozen_sleeve(m.items, freeze_weights=False)
        rows.append({"check": f"{label} frozen_sleeve(freeze_weights=False) gross == production gross (max abs diff)",
                     "value": float((plumb.gross - m.port.gross).abs().max()), "production": 0.0})
    return pd.DataFrame(rows)


def merged_basis(books: dict, built: dict) -> pd.DataFrame:
    """Net Sharpe with each distinct portfolio counted once per bar, next to the production (equal-weight) basis."""
    rows = []
    for (label, cfg), items in built.items():
        omega, gap = distinct_weights(items)
        b = weighted_sleeve(items, omega) if cfg == "baseline" else frozen_sleeve(items, omega=omega)
        prod, lv, lp = books[(label, cfg)], b.net.notna(), books[(label, cfg)].net.notna()
        rows.append({"model": label, "config": cfg,
                     "sharpe_net_book": metrics.sharpe(prod.net[lp]), "sharpe_net_distinct": metrics.sharpe(b.net[lv]),
                     "sharpe_gross_book": metrics.sharpe(prod.gross[lp]), "sharpe_gross_distinct": metrics.sharpe(b.gross[lv]),
                     "net_pa_pct_distinct": b.net[lv].mean() * ANN * 100,
                     "max_gross_gap_between_copies": gap})
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    levels, _, models = load_models()
    gated = load_gated(levels, force)
    credit = [(r, v) for r, v in models["headline"].items if not r.name.startswith("pca")]
    gated_items = {"Headline": [(r, "pca") for r in gated["PCA-2 full"]] + frozen_items(credit),
                   "PCA-3 sparse": [(r, "pca") for r in gated["PCA-3 sparse"]]}

    books, built, rows, trades = {}, {}, [], []
    for label, key in MODELS.items():
        m = models[key]
        live_items = {"baseline": m.items, "freeze_size": frozen_items(m.items), "freeze_size+vol_gate": gated_items[label]}
        for cfg in CONFIGS:
            items = live_items[cfg]
            book = Book(m.port.gross, m.port.cost) if cfg == "baseline" else frozen_sleeve(items)
            books[(label, cfg)] = book
            built[(label, cfg)] = items
            r, t = rows_for(label, cfg, book, items)
            rows += r
            trades.append(t)
    results = pd.DataFrame(rows)
    trades = pd.concat(trades, ignore_index=True)
    terc = terciles(trades)
    dsr = deflated(books, results)
    checks = check_baseline(models, levels)
    merged = merged_basis(books, built)
    merged.to_csv(OUT / "merged_basis.csv", index=False)
    xlog = pd.concat([pd.DataFrame(pd.concat([r.diagnostics for r in runs], ignore_index=True)).assign(model=k)
                      for k, runs in gated.items()], ignore_index=True)
    xlog = xlog[["model", "strategy", "fold", "test_start", "entry", "exit", "unit_cost", "vol_gate_pct",
                 "vol_gate_x_bp", "vol_gate_train_net_per_trade"]]

    results.to_csv(OUT / "results.csv", index=False)
    terc.to_csv(OUT / "terciles.csv", index=False)
    dsr.to_csv(OUT / "deflated_sharpe.csv", index=False)
    checks.to_csv(OUT / "baseline_checks.csv", index=False)
    xlog.to_csv(OUT / "vol_gate_x_by_fold.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(results.to_string(index=False))
        print(terc.to_string(index=False))
        print(dsr.to_string(index=False))
        print(checks.to_string(index=False))
        print(merged.to_string(index=False))
        print(xlog.groupby(["model", "strategy"])[["vol_gate_pct", "vol_gate_x_bp"]].median().to_string())


if __name__ == "__main__":
    main(force="--force" in sys.argv)
