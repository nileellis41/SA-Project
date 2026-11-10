"""The deliverable: one statistical-arbitrage sleeve, three ways of building it.

    python -m amp_sa.run_sleeve1            # reuse cached runs
    python -m amp_sa.run_sleeve1 --force    # recompute the PCA runs (pairwise runs come from amp_sa.run)

Writes reports/report.md, its figures and tables, and signals/signals.parquet.
Everything outside the sleeve (trend, the regime-switching book, the wider
universe, intraday FX, the AMP plan) is written by amp_sa.run to reports/appendix.md.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

from amp_sa import config, run
from amp_sa.backtest import attribution, metrics
from amp_sa.data.load_bbg import PROCESSED_DIR
from amp_sa.portfolio.combine import Portfolio, combine
from amp_sa.sleeves import pca, rv

REPORT_DIR = run.REPORT_DIR
CORE = config.CORE_GROUPS

# Declared before the PCA model was run on real data: the headline sleeve is the
# level-and-slope-neutral curve model (k = 2) plus the adaptive pairwise credit
# spreads, since two or three credit series are too few for a factor model.
HEADLINE = "PCA-2 curve + adaptive credit pairs"
# Net Sharpe is reported at these multiples of the cost table (1x = 0.25 bp per Treasury leg, one way).
COST_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def load_pca_runs(levels, force: bool = False) -> dict:
    cache = PROCESSED_DIR / "pca_runs.pkl"
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            return pickle.load(f)
    runs = {k: pca.run_curve(levels, k) for k in (2, 3)}
    runs.update({f"{k}s": pca.run_curve(levels, k, sparse=True) for k in (2, 3)})
    with open(cache, "wb") as f:
        pickle.dump(runs, f)
    return runs


def sleeve(items: list[tuple], use_overlay: bool = False) -> tuple[Portfolio, pd.Series, pd.Series]:
    """Equal-weight sleeve of (run, variant) pairs at the volatility target. Returns (portfolio, turnover, in_market)."""
    def avg(get):
        return pd.DataFrame({r.name: get(r.results[v]).where(r.live) for r, v in items}).mean(axis=1)

    gross, cost, turnover = avg(lambda x: x.gross), avg(lambda x: x.cost), avg(lambda x: x.turnover)
    hold = avg(lambda x: x.position.abs() * x.unit_cost)
    active = pd.DataFrame({r.name: (r.results[v].signal != 0).where(r.live) for r, v in items})
    in_market = active.fillna(False).any(axis=1).where(active.notna().any(axis=1))
    port = combine({"s": (gross, cost)}, budget={"s": 1.0}, use_overlay=use_overlay, hold_cost={"s": hold})
    return port, turnover, in_market


def models(rv_runs, pca_runs) -> dict[str, list[tuple]]:
    core = [r for r in rv_runs if r.group in CORE]
    rates = [r for r in core if r.group == "rates"]
    credit = [r for r in core if r.group == "credit"]
    return {
        "pairwise static": [(r, "static") for r in core],
        "pairwise adaptive, HMM gate": [(r, "adaptive_hmm") for r in core],
        "pairwise adaptive, neural gate": [(r, "adaptive_nn") for r in core],
        "pairwise adaptive, rates only": [(r, "adaptive_nn") for r in rates],
        "PCA-2 curve (level, slope hedged)": [(r, pca.VARIANT) for r in pca_runs[2]],
        "PCA-3 curve (+ curvature hedged)": [(r, pca.VARIANT) for r in pca_runs[3]],
        HEADLINE: [(r, pca.VARIANT) for r in pca_runs[2]] + [(r, "adaptive_nn") for r in credit],
        "PCA-2 sparse (3 legs per residual)": [(r, pca.VARIANT) for r in pca_runs["2s"]],
        "PCA-3 sparse (4 legs per residual)": [(r, pca.VARIANT) for r in pca_runs["3s"]],
        "PCA-2 curve, Leung-Li levels": [(r, pca.VARIANT_LL) for r in pca_runs[2]],
        "PCA-3 curve, Leung-Li levels": [(r, pca.VARIANT_LL) for r in pca_runs[3]],
    }


def build(levels, rolls, rv_runs, pca_runs) -> tuple[dict, dict]:
    built = {name: sleeve(items) for name, items in models(rv_runs, pca_runs).items()}
    trials = [metrics.sharpe(p.net) for p, _, _ in built.values()]
    rows = []
    for name, (port, turnover, in_market) in built.items():
        rows.append({"model": name, **run.summary_row(name, port, trials, turnover),
                     "in_market": float(in_market.mean())})
    summary = pd.DataFrame(rows).drop(columns=["book", "end", "hit_rate"])

    cost_curve = pd.DataFrame(
        [{"model": name, **{f"{m:g}x": metrics.sharpe(p.net_at(m)) for m in COST_GRID}}
         for name, (p, _, _) in built.items()])

    regime = pd.concat(
        [metrics.by_year(p.net, config.REGIME_YEARS).assign(model=name).rename_axis("year").reset_index()
         for name, (p, _, _) in built.items()], ignore_index=True)
    alpha = attribution.attribution_table({name: p.net for name, (p, _, _) in built.items()}, levels, rolls)

    pca_rows = []
    for k, runs in pca_runs.items():
        for r in runs:
            res = r.results[pca.VARIANT]
            pca_rows.append({
                "strategy": r.name, "legs": int(r.diagnostics["legs"].max()),
                "start": r.net(pca.VARIANT).dropna().index.min().date(),
                "sharpe_net": metrics.sharpe(r.net(pca.VARIANT)),
                "sharpe_gross": metrics.sharpe(res.gross.where(r.live)),
                "sharpe_net_2x": metrics.sharpe(r.net(pca.VARIANT, 2.0)),
                "in_market": float((res.signal != 0).where(r.live).mean()),
                "turnover_per_year": float(res.turnover.where(r.live).mean() * config.ANN),
                "cost_over_daily_vol": float(r.diagnostics["cost_over_daily_vol"].median()),
                "variance_explained_by_factors": float(r.diagnostics["explained"].median()),
            })
    ll_rows = []
    for k, runs in pca_runs.items():
        if not isinstance(k, int):
            continue
        for r in runs:
            d, res = r.diagnostics, r.results[pca.VARIANT_LL]
            ll_rows.append({
                "strategy": r.name, "half_life_bars": d["ll_half_life"].median(),
                "cost_in_sigma": d["ll_cost_in_sigma"].median(), "folds_traded": float(d["ll_trade"].mean()),
                "entry_sigma": d["ll_entry"].median(), "take_profit_sigma_past_mean": d["ll_take_profit"].median(),
                "grid_entry": d["entry"].median(), "grid_exit": d["exit"].median(),
                "sharpe_net": metrics.sharpe(r.net(pca.VARIANT_LL)),
                "sharpe_gross": metrics.sharpe(res.gross.where(r.live)),
                "in_market": float((res.signal != 0).where(r.live).mean()),
                "turnover_per_year": float(res.turnover.where(r.live).mean() * config.ANN),
            })
    pairwise = run.strategy_table([r for r in rv_runs if r.group in CORE], rv.VARIANTS)
    folds = pd.concat([r.diagnostics for r in rv_runs if r.group in CORE], ignore_index=True)
    screen = folds.groupby("spread", sort=False).agg(
        folds=("fold", "count"), eg_pass=("eg_pvalue", lambda p: float((p < config.COINT_PVALUE).mean())),
        johansen_pass=("johansen_pass", "mean"), both_pass=("coint_pass", "mean"),
        median_half_life=("half_life_static", "median"), health_disabled_share=("disabled_share", "mean"),
    ).reset_index()
    tables = {"sleeve1_models": summary, "sleeve1_cost_curve": cost_curve, "sleeve1_stress_years": regime, "sleeve1_alpha": alpha,
              "sleeve1_pca_by_tenor": pd.DataFrame(pca_rows), "sleeve1_leung_li": pd.DataFrame(ll_rows), "sleeve1_pairwise_by_spread": pairwise,
              "sleeve1_screen": screen}
    return built, tables


def write_signals(items, port: Portfolio) -> pd.DataFrame:
    """date, instrument, sleeve, signal, target_vol_weight for the headline sleeve."""
    live = pd.DataFrame({r.name: r.live for r, _ in items})
    weights = live.astype(float).div(live.sum(axis=1).replace(0, np.nan), axis=0).mul(port.weights["s"], axis=0)
    frames = []
    for r, v in items:
        signal = r.results[v].signal.where(r.live)
        frame = pd.DataFrame({"signal": signal, "target_vol_weight": signal * weights[r.name]}).dropna()
        frames.append(frame.assign(instrument=r.name, sleeve="rv").reset_index())
    out = pd.concat(frames, ignore_index=True)[["date", "instrument", "sleeve", "signal", "target_vol_weight"]]
    out.to_parquet(run.SIGNALS_PATH, index=False)
    return out


NOTES = [
    "## Method notes and caveats", "",
    "- **Walk-forward.** Every parameter (hedge ratios, factor loadings, OU fits, bands, scalers, regime models) is "
    "fitted on the training window of its fold. Rolling statistics are trailing. The leakage suite rewrites all "
    "data after a cut-off date and checks that nothing decided before it changes.",
    "- **Timing.** A signal uses data through the close of bar t and is held over bar t+1.",
    "- **Costs.** One-way, per leg: 0.25 bp for Treasury yields, 2 bp HY OAS, 0.5 bp IG OAS, 1 bp CDX HY. A PCA "
    "residual trades every tenor, so its cost is the sum over its legs. Changes in sleeve leverage are also charged.",
    "- **Sizing.** The sleeve is scaled to a 10% volatility target on the larger of recent and long-run realised "
    "volatility, so realised volatility sits below target.",
    "- **Yields, not futures.** The curve is traded in cash-yield basis points. Carry and roll-down are not in the "
    "P&L. The nearest tradeable instruments are CME micro yield futures at about $10 per bp.",
    "- **Credit is a research proxy.** HY and IG OAS are index spreads and cannot be traded as modelled.",
    "- **Model selection.** The headline model was named before the PCA model was run on real data. The deflated "
    "Sharpe counts the eleven rows of the model table as trials. It does not count earlier design revisions to the "
    "pairwise model, so treat it as optimistic.",
    "- **PCA settings are the published defaults**: a 60-bar OU window and the half-life filter of Avellaneda and "
    "Lee (2010). Neither was tuned.",
    "",
]


def write_report(levels, tables) -> None:
    summary = tables["sleeve1_models"]
    alpha = tables["sleeve1_alpha"]
    lines = [
        "# Statistical arbitrage on the Treasury curve and credit spreads",
        "",
        f"One sleeve, mean reversion, built three ways. Sample ends {levels.index[-1].date()}. All results are out "
        f"of sample from walk-forward folds ({config.TRAIN_BARS}-bar training, {config.TEST_BARS}-bar test, "
        f"{config.PURGE_BARS}-bar purge, {config.EMBARGO_BARS}-bar embargo), net of costs, at a "
        f"{config.TARGET_VOL:.0%} volatility target.",
        "",
        "- **Pairwise static.** Engle-Granger and Johansen screen, OLS hedge ratio, OU bands.",
        "- **Pairwise adaptive.** Kalman hedge ratio, health monitor, and a regime gate (HMM or neural classifier).",
        "- **PCA residual.** Hedge each tenor against the first two or three principal components of the curve and "
        "trade the residual's s-score (Avellaneda-Lee).",
        "",
        f"Headline model, named in advance: **{HEADLINE}**.",
        "",
        "## Model comparison", "",
        run._md(summary), "",
        "![models](figures/sleeve1_models.png)", "",
        "![headline](figures/sleeve1_headline.png)", "",
        "## How much cost each model can bear", "",
        "Net Sharpe at multiples of the cost table. 1x is 0.25 bp per Treasury leg, one way; 0.4x is 0.1 bp. "
        "Bands were optimised at 1x, so the lower-cost columns understate what re-optimised bands would do.", "",
        run._md(tables["sleeve1_cost_curve"]), "",
        "## Alpha and beta", "",
        "Daily P&L on unit-volatility market factors, Newey-West t-statistics. `rates` uses duration and the 2s10s "
        "steepener over the full sample; `macro` adds credit, crude and equity from 2005.", "",
        run._md(alpha[["factors", "series", "start", "ann_return", "alpha", "alpha_t", "appraisal_ratio",
                       "r_squared", "t_duration", "t_steepener"]], 3), "",
        "## Stress years", "", run._md(tables["sleeve1_stress_years"][["model", "year", "return", "sharpe", "max_dd"]]), "",
        "## PCA residuals by tenor", "",
        "`cost_over_daily_vol` is the one-way cost of the residual portfolio relative to its daily volatility.", "",
        run._md(tables["sleeve1_pca_by_tenor"]), "",
        "Rows named `pca2s` and `pca3s` are the sparse construction: the own tenor hedged with only its nearest "
        "two (or three) tenors, which is the fewest legs that can be neutral to two (or three) factors.", "",
        "## Leung-Li levels on the PCA residuals", "",
        "The same residuals and s-scores, with entry and exit set by the Leung-Li optimal stopping solution "
        "instead of the grid search. Levels are in sigmas from the mean; the take-profit sits past the mean. "
        "Inputs are the median half-life and cost on each training window.", "",
        run._md(tables["sleeve1_leung_li"]), "",
        "## Pairwise spreads", "", run._md(tables["sleeve1_pairwise_by_spread"]), "",
        "## Cointegration screen, pairwise spreads", "", run._md(tables["sleeve1_screen"]), "",
        *NOTES,
        "Trend, the regime-switching two-sleeve book, the energy, grain and FX extension, the intraday FX study, "
        "the Leung-Li cost check and the AMP paper-trading plan are in [appendix.md](appendix.md). Trade-level and "
        "period-level analysis is in [trades.md](trades.md) and [periods.md](periods.md), with a summary in "
        "[lessons.md](lessons.md).", "",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(force: bool = False) -> None:
    levels, rolls, rv_runs, _ = run.load_runs()
    pca_runs = load_pca_runs(levels, force)
    built, tables = build(levels, rolls, rv_runs, pca_runs)
    (REPORT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / "tables" / f"{name}.csv", index=False)

    pct = lambda s: s.fillna(0).cumsum().where(s.notna()) * 100  # noqa: E731
    fig = REPORT_DIR / "figures"
    run._line_chart({k: pct(built[k][0].net) for k in ("pairwise static", "pairwise adaptive, neural gate",
                                                       "PCA-2 curve (level, slope hedged)")},
                    "Sleeve 1: three ways to build the mean-reversion trade, net of costs",
                    "cumulative return, % of capital", fig / "sleeve1_models.png")
    run._line_chart({HEADLINE: pct(built[HEADLINE][0].net)},
                    f"Headline sleeve: {HEADLINE}, net of costs", "cumulative return, % of capital",
                    fig / "sleeve1_headline.png")

    signals = write_signals(models(rv_runs, pca_runs)[HEADLINE], built[HEADLINE][0])
    write_report(levels, tables)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(tables["sleeve1_models"].to_string(index=False))
        print(tables["sleeve1_cost_curve"].to_string(index=False))
        print(tables["sleeve1_pca_by_tenor"].to_string(index=False))
        print(tables["sleeve1_alpha"][["factors", "series", "alpha", "alpha_t", "appraisal_ratio"]].to_string(index=False))
        print(tables["sleeve1_stress_years"].to_string(index=False))
        print(f"signals: {len(signals)} rows")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
