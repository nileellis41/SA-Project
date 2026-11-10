"""Step 8: walk-forward experiments, each against the unchanged baseline.

    python -m analysis.experiments.run_experiments

Experiments 1-5 re-run the PCA curve members through analysis/experiments/pipeline.py
(bit-identical to production with no hook) for the PCA-2 full and PCA-3 sparse
constructions; the headline is the PCA-2 result plus the unchanged adaptive credit pairs.
Experiment 6 (leg netting) keeps every position and re-prices cost on the netted book.

Writes analysis/experiments/results.csv and netting.csv; caches runs in analysis/experiments/runs.pkl.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from amp_sa import config
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import metrics
from amp_sa.run_lessons import model_trades
from analysis.experiments.pipeline import Rule, run_members
from analysis.portfolio_diagnostics import book as bk
from analysis.portfolio_diagnostics.common import load_models
from analysis.portfolio_diagnostics.cost_audit import split_cost

OUT = Path(__file__).resolve().parent
ANN = config.ANN
MOVE_WINDOW = 5 * ANN  # trailing window for the MOVE tercile cutoff

# Where a parameter came from. "after results" = proposed after full-sample results were seen
# (lessons.md, trades.md or this review), so a good number is partly in-sample by choice.
PROVENANCE = {
    "baseline": "unchanged production model",
    "time_stop_20": "after results: lessons.md looked at trades held past 20 bars on the full sample",
    "resize_25pct": "after results: 25% chosen in this review's brief; not tuned here",
    "entry_2.5": "after results: lessons.md/trades.md bucketed trades by |z| on the full sample",
    "entry_3.0": "after results: as entry_2.5",
    "move_top_tercile": "after results: lessons.md found high-MOVE trades did better on the full sample; cutoff itself is trailing",
    "loo_residual": "no tuned parameter; construction change only",
}


def move_gate(levels: pd.DataFrame) -> pd.Series:
    move = levels["move"]
    cut = move.rolling(MOVE_WINDOW, min_periods=ANN).quantile(2 / 3)
    return (move >= cut) & cut.notna()


def configs(levels) -> dict[str, dict]:
    return {
        "time_stop_20": {"rule": Rule(time_stop=20)},
        "resize_25pct": {"rule": Rule(resize_tol=0.25)},
        "entry_2.5": {"rule": Rule(entry_fixed=2.5)},
        "entry_3.0": {"rule": Rule(entry_fixed=3.0)},
        "move_top_tercile": {"rule": Rule(gate=move_gate(levels))},
        "loo_residual": {"loo": True},
    }


def _run(levels, name, k, sparse, kw):
    if kw.get("loo"):
        sparse = False  # LOO hedges with every other tenor
    return name, k, run_members(levels, k=k, sparse=sparse, **kw)


def load_runs(levels, force=False) -> dict:
    cache = OUT / "runs.pkl"
    if cache.exists() and not force:
        return pickle.load(open(cache, "rb"))
    jobs = [(name, k, sparse, kw) for name, kw in configs(levels).items() for k, sparse in ((2, False), (3, True))]
    done = Parallel(n_jobs=4)(delayed(_run)(levels, *j) for j in jobs)
    runs = {(name, k): r for name, k, r in done}
    pickle.dump(runs, open(cache, "wb"))
    return runs


def summarize(name: str, model: str, items) -> dict:
    port, _, _ = s1.sleeve(items)
    live = port.net.notna()
    net, gross, cost = port.net[live], port.gross[live], port.cost[live]
    t = model_trades(items)
    return {"experiment": name, "model": model, "trades": len(t), "trades_per_year": len(t) / (live.sum() / ANN),
            "gross_pa": gross.mean() * ANN, "cost_pa": cost.mean() * ANN, "net_pa": net.mean() * ANN,
            "sharpe_net": metrics.sharpe(net), "sharpe_gross": metrics.sharpe(gross),
            "max_dd": metrics.max_drawdown(net), "provenance": PROVENANCE[name]}


def netting(models, levels, rolls) -> pd.DataFrame:
    rows = []
    for label, m in models.items():
        b = bk.build_book(m, levels, rolls)
        live = m.port.net.notna()
        roll = sum(split_cost(r.results[v].position.where(r.live, 0.0), r.results[v].cost.where(r.live, 0.0),
                              r.results[v].unit_cost)["roll"] * b.w_member for r, v in m.items)
        unnetted = bk.leg_cost(b, netted=False).sum(axis=1) + roll
        netted = bk.leg_cost(b, netted=True).sum(axis=1) + roll
        g = m.port.gross[live]
        rows.append({"model": label, "cost_engine_pa": m.port.cost[live].mean() * ANN,
                     "cost_leg_unnetted_pa": unnetted[live].mean() * ANN, "cost_leg_netted_pa": netted[live].mean() * ANN,
                     "saved_vs_engine_pa": (m.port.cost[live] - netted[live]).mean() * ANN,
                     "saved_vs_unnetted_share": 1 - netted[live].sum() / unnetted[live].sum(),
                     "sharpe_net_engine": metrics.sharpe(m.port.net[live]),
                     "sharpe_net_netted": metrics.sharpe(g - netted[live]),
                     "net_pa_netted": (g - netted[live]).mean() * ANN})
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    levels, rolls, models = load_models()
    runs = load_runs(levels, force)
    credit = [(r, v) for r, v in models["headline"].items if not r.name.startswith("pca")]
    base = {"PCA-2 full": models["pca2"].items, "Headline": models["headline"].items,
            "PCA-3 sparse": models["pca3_sparse"].items}
    rows = [summarize("baseline", k, v) for k, v in base.items()]
    for name in configs(levels):
        p2 = [(r, "pca") for r in runs[(name, 2)]]
        p3 = [(r, "pca") for r in runs[(name, 3)]]
        rows += [summarize(name, "PCA-2 full", p2), summarize(name, "Headline", p2 + credit),
                 summarize(name, "PCA-3 sparse", p3)]
    res = pd.DataFrame(rows)
    basemap = res[res["experiment"] == "baseline"].set_index("model")
    res["d_sharpe_net"] = res["sharpe_net"] - res["model"].map(basemap["sharpe_net"])
    res["d_net_pa"] = res["net_pa"] - res["model"].map(basemap["net_pa"])
    res.to_csv(OUT / "results.csv", index=False)
    net = netting(models, levels, rolls)
    net.to_csv(OUT / "netting.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(res.drop(columns="provenance").to_string(index=False))
        print(net.to_string(index=False))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
