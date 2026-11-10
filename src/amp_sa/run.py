"""Run the whole book and write the report.

    python -m amp_sa.run            # reuse cached sleeve runs if present
    python -m amp_sa.run --force    # recompute everything

Outputs: reports/report.md, reports/figures/*.png, reports/tables/*.csv and
signals/signals.parquet.
"""

from __future__ import annotations

import pickle
import sys
from itertools import product

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest import attribution, metrics
from amp_sa.data.load_bbg import PROCESSED_DIR, PROJECT_ROOT
from amp_sa.data.panel import load_research_panel
from amp_sa.execution import limits as lim
from amp_sa.execution import plan
from amp_sa.portfolio.combine import Portfolio, combine, sleeve_hold_cost, sleeve_parts
from amp_sa.portfolio.regime_switch import regime_budget, regime_features
from amp_sa.sleeves import intraday_fx, rv, trend

REPORT_DIR = PROJECT_ROOT / "reports"
SIGNALS_PATH = PROJECT_ROOT / "signals" / "signals.parquet"
AMP_SPECS = PROJECT_ROOT / "data" / "raw" / "amp" / "specs.parquet"

# Declared before any result was looked at: the adaptive RV model with the
# neural gate, and the multi-scale L1 trend (the variant with nothing to select).
BOOK = {"rv": "adaptive_nn", "trend": "l1_multi"}
SCOPES = {"core": config.CORE_GROUPS, "all": None}
COST_MULTS = (1.0, 2.0, 3.0)
PAPER_CAPITAL = 3000.0

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # validated categorical slots 1-3, fixed order
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"


def load_runs(force: bool = False):
    levels, rolls = load_research_panel()
    cache = PROCESSED_DIR / "sleeve_runs.pkl"
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            rv_runs, trend_runs = pickle.load(f)
    else:
        rv_runs = rv.run_sleeve(levels, rolls)
        trend_runs = trend.run_sleeve(levels, rolls)
        with open(cache, "wb") as f:
            pickle.dump((rv_runs, trend_runs), f)
    return levels, rolls, rv_runs, trend_runs


def standalone(runs, variant: str, groups) -> Portfolio:
    """One sleeve on its own, scaled to the volatility target with the same causal rule. No overlay."""
    gross, cost, _ = sleeve_parts(runs, variant, groups)
    hold = {"s": sleeve_hold_cost(runs, variant, groups)}
    return combine({"s": (gross, cost)}, budget={"s": 1.0}, use_overlay=False, hold_cost=hold)


def book(rv_runs, trend_runs, rv_variant: str, trend_variant: str, groups, use_overlay: bool = True,
         budget=None) -> Portfolio:
    """Two-sleeve book. `budget` is a fixed dict or a lagged per-bar frame; default is the fixed 50/50."""
    g1, c1, _ = sleeve_parts(rv_runs, rv_variant, groups)
    g2, c2, _ = sleeve_parts(trend_runs, trend_variant, groups)
    hold = {"rv": sleeve_hold_cost(rv_runs, rv_variant, groups),
            "trend": sleeve_hold_cost(trend_runs, trend_variant, groups)}
    return combine({"rv": (g1, c1), "trend": (g2, c2)}, budget=config.SLEEVE_RISK_BUDGET if budget is None else budget,
                   use_overlay=use_overlay, hold_cost=hold)


def regime_book(levels, rv_runs, trend_runs, groups, use_overlay: bool = True):
    """The declared sleeves with the risk budget switched by regime. Returns (portfolio, budget, diagnostics)."""
    g1, c1, _ = sleeve_parts(rv_runs, BOOK["rv"], groups)
    g2, c2, _ = sleeve_parts(trend_runs, BOOK["trend"], groups)
    budget, diag = regime_budget(pd.DataFrame({"rv": g1 - c1, "trend": g2 - c2}), regime_features(levels))
    return book(rv_runs, trend_runs, BOOK["rv"], BOOK["trend"], groups, use_overlay, budget), budget, diag


def regime_tables(levels, rv_runs, trend_runs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regime switching against fixed allocations, all over the window where the switch is live."""
    rows, diags = [], []
    for scope, groups in SCOPES.items():
        switched, budget, diag = regime_book(levels, rv_runs, trend_runs, groups)
        window = switched.net.dropna().index
        fixed = lambda b: book(rv_runs, trend_runs, BOOK["rv"], BOOK["trend"], groups, budget=b)  # noqa: E731
        books = {
            "regime switching": switched,
            "fixed 50/50": fixed(None),
            "trend only": fixed({"rv": 0.0, "trend": 1.0}),
            "mean reversion only": fixed({"rv": 1.0, "trend": 0.0}),
        }
        trials = [metrics.sharpe(p.net.reindex(window)) for p in books.values()]
        for label, p in books.items():
            cut = Portfolio(p.weights.reindex(window), p.gross.reindex(window), p.cost.reindex(window),
                            p.sleeve_net.reindex(window), p.overlay.reindex(window))
            rows.append({"scope": scope, **summary_row(label, cut, trials)})
        rows[-4]["avg_rv_budget"] = float(budget.reindex(window)["rv"].mean())
        rows[-4]["budget_switches_per_year"] = float(
            (budget.reindex(window)["rv"].round().diff().abs() > 0).mean() * config.ANN)
        diags.append(diag.assign(scope=scope))
    return pd.DataFrame(rows), pd.concat(diags, ignore_index=True)


def strategy_table(runs, variants) -> pd.DataFrame:
    rows = []
    for r in runs:
        for v in variants:
            res = r.results[v]
            net = r.net(v)
            rows.append({
                "strategy": r.name, "group": r.group, "variant": v,
                "start": net.dropna().index.min().date(),
                "sharpe_net": metrics.sharpe(net),
                "sharpe_gross": metrics.sharpe(res.gross.where(r.live)),
                "sharpe_net_2x": metrics.sharpe(r.net(v, 2.0)),
                "sharpe_net_3x": metrics.sharpe(r.net(v, 3.0)),
                "in_market": float((res.signal != 0).where(r.live).mean()),
                "turnover_per_year": float(res.turnover.where(r.live).mean() * config.ANN),
            })
    return pd.DataFrame(rows)


def summary_row(label: str, port: Portfolio, trial_sharpes, turnover: pd.Series | None = None) -> dict:
    out = {"book": label, **metrics.summarize(port.net, port.gross, turnover, trial_sharpes)}
    for m in COST_MULTS[1:]:
        out[f"sharpe_{m:g}x_cost"] = metrics.sharpe(port.net_at(m))
    return out


def alpha_table(levels, rolls, rv_runs, trend_runs) -> pd.DataFrame:
    frames = []
    for scope, groups in SCOPES.items():
        series = {
            "mean reversion": standalone(rv_runs, BOOK["rv"], groups).net,
            "trend": standalone(trend_runs, BOOK["trend"], groups).net,
            "fixed 50/50 book": book(rv_runs, trend_runs, BOOK["rv"], BOOK["trend"], groups).net,
            "regime-switching book": regime_book(levels, rv_runs, trend_runs, groups)[0].net,
        }
        frames.append(attribution.attribution_table(series, levels, rolls).assign(scope=scope))
    return pd.concat(frames, ignore_index=True)


def build_tables(rv_runs, trend_runs, levels) -> dict[str, pd.DataFrame]:
    t = {
        "rv_by_spread": strategy_table(rv_runs, rv.VARIANTS),
        "trend_by_series": strategy_table(trend_runs, trend.VARIANTS),
        "rv_folds": pd.concat([r.diagnostics for r in rv_runs], ignore_index=True),
        "trend_folds": pd.concat([r.diagnostics for r in trend_runs], ignore_index=True),
    }
    folds = t["rv_folds"]
    t["rv_screen"] = folds.groupby("spread", sort=False).agg(
        folds=("fold", "count"), eg_pass=("eg_pvalue", lambda p: float((p < config.COINT_PVALUE).mean())),
        johansen_pass=("johansen_pass", "mean"), both_pass=("coint_pass", "mean"),
        median_half_life_static=("half_life_static", "median"),
        median_half_life_adaptive=("half_life_adaptive", "median"),
        health_disabled_share=("disabled_share", "mean"),
    ).reset_index()

    sleeve_rows, book_rows, regime_rows, corr_rows = [], [], [], []
    for scope, groups in SCOPES.items():
        for name, runs, variants in (("rv", rv_runs, rv.VARIANTS), ("trend", trend_runs, trend.VARIANTS)):
            ports = {v: standalone(runs, v, groups) for v in variants}
            trials = [metrics.sharpe(p.net) for p in ports.values()]
            for v, p in ports.items():
                turnover = sleeve_parts(runs, v, groups)[2]
                sleeve_rows.append({"scope": scope, "sleeve": name, **summary_row(v, p, trials, turnover)})

        combos = {
            (a, b): book(rv_runs, trend_runs, a, b, groups) for a, b in product(rv.VARIANTS, trend.VARIANTS)
        }
        trials = [metrics.sharpe(p.net) for p in combos.values()]
        for (a, b), p in combos.items():
            row = {"scope": scope, "rv": a, "trend": b, "declared": (a, b) == (BOOK["rv"], BOOK["trend"])}
            book_rows.append({**row, **summary_row(f"{a} + {b}", p, trials)})
        chosen = combos[(BOOK["rv"], BOOK["trend"])]
        no_overlay = book(rv_runs, trend_runs, BOOK["rv"], BOOK["trend"], groups, use_overlay=False)
        book_rows.append({"scope": scope, "rv": BOOK["rv"], "trend": BOOK["trend"], "declared": False,
                          **summary_row("declared book, no drawdown overlay", no_overlay, trials)})

        parts = {"rv": standalone(rv_runs, BOOK["rv"], groups).net,
                 "trend": standalone(trend_runs, BOOK["trend"], groups).net, "book": chosen.net}
        both = pd.DataFrame(parts).dropna()
        corr_rows.append({"scope": scope, "rv_trend_correlation": both["rv"].corr(both["trend"]),
                          "monthly_correlation": both.resample("ME").sum()["rv"].corr(both.resample("ME").sum()["trend"]),
                          "bars": len(both)})
        for label, series in parts.items():
            for year, row in metrics.by_year(series, config.REGIME_YEARS).iterrows():
                regime_rows.append({"scope": scope, "series": label, "year": year, **row.to_dict()})

    t["sleeves"] = pd.DataFrame(sleeve_rows)
    t["books"] = pd.DataFrame(book_rows)
    t["regimes"] = pd.DataFrame(regime_rows)
    t["sleeve_correlation"] = pd.DataFrame(corr_rows)
    t["regime_switch"], t["regime_folds"] = regime_tables(levels, rv_runs, trend_runs)
    return t


def strategy_weights(runs, port: Portfolio, sleeve: str, groups) -> pd.DataFrame:
    """Capital weight on one risk unit of each strategy: sleeve weight / live strategies."""
    pick = [r for r in runs if groups is None or r.group in groups]
    live = pd.DataFrame({r.name: r.live for r in pick})
    return live.astype(float).div(live.sum(axis=1).replace(0, np.nan), axis=0).mul(port.weights[sleeve], axis=0)


def write_signals(rv_runs, trend_runs, port: Portfolio) -> pd.DataFrame:
    """The signal contract: date, instrument, sleeve, signal, target_vol_weight."""
    frames = []
    for sleeve, runs in (("rv", rv_runs), ("trend", trend_runs)):
        weights = strategy_weights(runs, port, sleeve, None)
        for r in runs:
            signal = r.results[BOOK[sleeve]].signal.where(r.live)
            frame = pd.DataFrame({"signal": signal, "target_vol_weight": signal * weights[r.name]}).dropna()
            frames.append(frame.assign(instrument=r.name, sleeve=sleeve).reset_index())
    out = pd.concat(frames, ignore_index=True)[["date", "instrument", "sleeve", "signal", "target_vol_weight"]]
    SIGNALS_PATH.parent.mkdir(exist_ok=True)
    out.to_parquet(SIGNALS_PATH, index=False)
    return out


def execution_plan(levels, rolls, rv_runs, trend_runs, port: Portfolio, capital: float = PAPER_CAPITAL):
    """Latest target positions as AMP micro contracts, with the limit check. Dry run."""
    if not AMP_SPECS.exists():
        return None, ["no AMP spec snapshot; run python -m amp_sa.execution.mt5_client"], None
    specs = pd.read_parquet(AMP_SPECS)
    asof = levels.index[-1]
    legs = pd.Series(0.0, index=levels.columns)
    for r in rv_runs:
        w = strategy_weights(rv_runs, port, "rv", None)[r.name].iloc[-1]
        pos = r.results[BOOK["rv"]].position.iloc[-1]
        legs = legs.add(r.extras["legs_adaptive"].iloc[-1].fillna(0.0) * w * pos, fill_value=0.0)
    for r in trend_runs:
        w = strategy_weights(trend_runs, port, "trend", None)[r.name].iloc[-1]
        legs[r.name] += w * r.results[BOOK["trend"]].position.iloc[-1]
    legs = legs[legs.abs() > 0]
    table = plan.target_contracts(legs, specs, capital, specs["snapshot_time"].max())

    from amp_sa.backtest.engine import trailing_vol
    from amp_sa.data.panel import changes
    vol = changes(levels, rolls).apply(trailing_vol).iloc[-1]
    table["dollar_vol_per_contract"] = [
        vol[s] * plan.CONTRACT_MAP[s][1] if s in plan.CONTRACT_MAP else np.nan for s in table["series"]
    ]
    table["one_lot_needs_capital"] = (capital * 0.5 / table["exact"].abs()).where(table["symbol"].notna())
    mapped = table[table["symbol"].notna()]
    targets = mapped.groupby("symbol")["contracts"].sum().astype(float)
    dollar_vol = mapped.groupby("symbol")["dollar_vol_per_contract"].first()
    violations = lim.check(targets, pd.Series(dtype=float), specs, dollar_vol, capital)
    # What the smallest tradeable book (one lot of each mapped leg) would do to the limits.
    one_lot = pd.Series(1.0, index=dollar_vol.index)
    one_lot_check = lim.check(one_lot, pd.Series(dtype=float), specs, dollar_vol, capital)
    return table, violations, one_lot_check


def _style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")


def _line_chart(series: dict[str, pd.Series], title: str, ylabel: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ends = [s.dropna().iloc[-1] for s in series.values()]
    span = max(max(s.max() for s in series.values()) - min(s.min() for s in series.values()), 1e-9)
    labelled: list[float] = []
    for (label, s), color in zip(series.items(), SERIES_COLORS):
        s = s.dropna()
        ax.plot(s.index, s.to_numpy(), color=color, linewidth=1.8, label=label)
        # Direct label only where it will not sit on top of another; the legend always carries identity.
        if len(ends) > 1 and all(abs(s.iloc[-1] - y) > 0.05 * span for y in labelled):
            ax.annotate(label, (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color="#52514e", va="center")
            labelled.append(s.iloc[-1])
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    _style(ax, title, ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor="#52514e")
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.86)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figures(rv_runs, trend_runs, tables, levels) -> None:
    out = REPORT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pct = lambda s: s.fillna(0).cumsum().where(s.notna()) * 100  # noqa: E731
    for scope, groups in SCOPES.items():
        _line_chart({v: pct(standalone(rv_runs, v, groups).net) for v in rv.VARIANTS},
                    f"Sleeve 1, mean reversion ({scope} universe): static vs adaptive, net of costs",
                    "cumulative return, % of capital", out / f"rv_variants_{scope}.png")
        _line_chart({v: pct(standalone(trend_runs, v, groups).net) for v in trend.VARIANTS},
                    f"Sleeve 2, trend ({scope} universe): MA baseline vs L1 filters, net of costs",
                    "cumulative return, % of capital", out / f"trend_variants_{scope}.png")
        parts = {
            "mean reversion": standalone(rv_runs, BOOK["rv"], groups).net,
            "trend": standalone(trend_runs, BOOK["trend"], groups).net,
            "combined book": book(rv_runs, trend_runs, BOOK["rv"], BOOK["trend"], groups).net,
        }
        _line_chart({k: pct(v) for k, v in parts.items()},
                    f"Declared book ({scope} universe): sleeves and combination at a 10% volatility target",
                    "cumulative return, % of capital", out / f"book_{scope}.png")
        switched = regime_book(levels, rv_runs, trend_runs, groups)[0].net
        window = switched.dropna().index
        _line_chart({"regime switching": pct(switched.reindex(window)),
                     "fixed 50/50": pct(parts["combined book"].reindex(window)),
                     "trend only": pct(book(rv_runs, trend_runs, BOOK["rv"], BOOK["trend"], groups,
                                            budget={"rv": 0.0, "trend": 1.0}).net.reindex(window))},
                    f"Regime-switching allocation vs fixed ({scope} universe), net of costs incl. switching",
                    "cumulative return, % of capital", out / f"regime_switch_{scope}.png")
        eq = parts["combined book"].fillna(0).cumsum()
        _line_chart({"combined book": ((eq - eq.cummax()) * 100).where(parts["combined book"].notna())},
                    f"Declared book ({scope} universe): drawdown", "drawdown, % of capital",
                    out / f"drawdown_{scope}.png")


NOTES = [
    "## Method notes and caveats", "",
    "- **Timing.** A signal uses data through the close of bar t and is held over bar t+1. Fills are assumed at "
    "the close of t; the cost model is the only allowance for slippage.",
    "- **Costs.** One-way cost per leg: 0.25 bp for Treasury yields, 2 bp HY OAS, 0.5 bp IG OAS, about one tick "
    "plus commission for futures. The 1x rates figure is an institutional cash-market cost. AMP's micro yield "
    "futures quoted about 1.4 bp wide in the snapshot, which is closer to the 3x case.",
    "- **Risk scaling.** Sleeves and the book are sized on the larger of recent and long-run realised volatility, "
    "so realised volatility sits below the 10% target. The mean-reversion sleeve holds a position on only 17% of "
    "bars in the core universe (31% in the full one) and concentrates its risk when it is on. Its worst days are "
    "far outside a normal 10%-volatility book: -13.6% of capital on 2008-09-15 (core, stand-alone sleeve), and "
    "-47% for the stand-alone sleeve and -26% for the full book on 2020-04-20, when front-month WTI settled at "
    "-37.63 with CL spread positions open. The full-universe book is not fit to trade as sized here.",
    "- **Credit is not tradeable as modelled.** HY and IG OAS are index spreads, used as a research proxy.",
    "- **Futures data.** Bloomberg generics and the Yahoo grain chains are unadjusted front-month series. "
    "The change on the bar where a generic switches contract is set to zero and a held position pays a roll cost. "
    "Spread levels across a roll are those of the new contract pair.",
    "- **Wheat and corn come from Yahoo Finance**, because the Bloomberg exports have no CBOT grain history and the "
    "AMP terminal only serves the live contracts. Yahoo KC wheat matches Bloomberg KW1 on 99.7% of days, and Yahoo "
    "ZW/ZC are within 1.75 cents of AMP's December 2026 contracts on every day since the September roll.",
    "- **AMP data is used for contract specs, margins and quotes only**, never as a research input.",
    "- **Deflated Sharpe** counts the nine sleeve combinations as the trial set (three per sleeve for the sleeve "
    "rows). It does not count design revisions: during development the adaptive model was changed twice after "
    "out-of-sample diagnostics showed it almost never traded, the cointegration screen was tightened from either "
    "test to both, and the volatility estimate was made more conservative after a first full run. Treat the "
    "deflated figures as optimistic.",
    "- **Universe history.** The first full run used 10 spreads and 11 trend series. It was then widened to 27 "
    "spreads and 16 series (more curve points, swap spreads, CDX, extra energy and grain spreads), with the list "
    "fixed before running. The wider mean-reversion sleeve did worse, not better; the earlier tables are kept in "
    "`tables_before_breadth/`. SOFR swap rates before 2018 are Bloomberg backfill. The CDX generic is flagged as "
    "rolling from the 20th to month end each March and September.",
    "- **Hedge-ratio rebalancing** of the Kalman legs is not costed. Changes in sleeve weights (volatility "
    "re-scaling, the drawdown overlay and regime switches) are charged at the cost of the positions they re-trade.",
    "",
]


def _md(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def write_report(tables, plan_table, violations, one_lot_check, levels) -> None:
    books = tables["books"]
    keep = ["book", "start", "ann_return", "ann_vol", "sharpe", "deflated_sharpe", "max_dd", "gross_sharpe",
            "cost_drag", "sharpe_2x_cost", "sharpe_3x_cost"]
    lines = [
        "# Appendix: trend, the two-sleeve book and the wider universe",
        "",
        "Supporting material. The deliverable is [report.md](report.md), which covers the mean-reversion sleeve only.",
        "",
        f"Sample ends {levels.index[-1].date()}. All results are out of sample: walk-forward folds with "
        f"{config.TRAIN_BARS}-bar training windows, {config.TEST_BARS}-bar test blocks, a {config.PURGE_BARS}-bar purge "
        f"and a {config.EMBARGO_BARS}-bar embargo. Returns are net of costs and additive, on a "
        f"{config.TARGET_VOL:.0%} volatility target.",
        "",
        "The declared book was fixed before results were inspected: "
        f"Sleeve 1 = `{BOOK['rv']}`, Sleeve 2 = `{BOOK['trend']}`. "
        "`core` is the graded universe (Treasury curve and HY/IG OAS); `all` adds the energy and grain extension.",
        "",
    ]
    for scope in SCOPES:
        sl = tables["sleeves"][tables["sleeves"]["scope"] == scope]
        bk = books[books["scope"] == scope]
        declared = bk[bk["declared"] | bk["book"].str.startswith("declared")]
        lines += [
            f"## {scope.capitalize()} universe", "",
            "### Declared book", "", _md(declared[keep]), "",
            f"![book](figures/book_{scope}.png)", "", f"![drawdown](figures/drawdown_{scope}.png)", "",
            "### Regime-switching allocation between the sleeves", "",
            "A two-state HMM on log VIX and log MOVE defines the regime. On each expanding training window the "
            "regime is assigned to whichever sleeve had the higher Sharpe in it, and out of sample the risk budget "
            "is the filtered-probability mix of those assignments. All four rows cover the same window and include "
            "the cost of moving risk between sleeves.", "",
            _md(tables["regime_switch"][tables["regime_switch"]["scope"] == scope].drop(columns=["scope", "end", "hit_rate"])), "",
            f"![regime](figures/regime_switch_{scope}.png)", "",
            "Per-fold regime fit (training-window Sharpe of each sleeve by state, and the assignment):", "",
            _md(tables["regime_folds"][tables["regime_folds"]["scope"] == scope].drop(columns="scope")), "",
            "### Alpha and beta", "",
            "Daily P&L regressed on unit-volatility market factors (long 10y Treasury, 2s10s steepener, long HY "
            "credit, long WTI), Newey-West t-statistics. `alpha` is annualised, in the same units as `ann_return`. "
            "`appraisal_ratio` is alpha over residual volatility: the Sharpe of what the factors do not explain. "
            "Betas are annualised return per unit of factor Sharpe. The `macro` rows start in 2005.", "",
            _md(tables["alpha"][tables["alpha"]["scope"] == scope].drop(columns=["scope", "bars"]), 3), "",
            "### Sleeve 1: mean reversion, static vs adaptive", "",
            _md(sl[sl["sleeve"] == "rv"][keep + ["turnover"]]), "", f"![rv](figures/rv_variants_{scope}.png)", "",
            "### Sleeve 2: trend, MA vs L1", "",
            _md(sl[sl["sleeve"] == "trend"][keep + ["turnover"]]), "", f"![trend](figures/trend_variants_{scope}.png)", "",
            "### All nine sleeve combinations (the trial set for the deflated Sharpe)", "",
            _md(bk[~bk["book"].str.startswith("declared")][["book", "declared", "sharpe", "max_dd", "sharpe_2x_cost", "sharpe_3x_cost"]]), "",
            "### Sleeve correlation", "", _md(tables["sleeve_correlation"][tables["sleeve_correlation"]["scope"] == scope]), "",
            "### Stress years", "",
            _md(tables["regimes"][tables["regimes"]["scope"] == scope].drop(columns="scope")), "",
        ]
    lines += [
        "## Sleeve 1 by spread", "", _md(tables["rv_by_spread"]), "",
        "## Cointegration screen and health monitor, by spread", "",
        "Share of folds passing each test on the training window. A fold trades only if both pass and the "
        f"training half-life is between {config.MIN_HALF_LIFE:g} and {config.MAX_HALF_LIFE:g} bars.", "",
        _md(tables["rv_screen"]), "",
        "## Sleeve 2 by series", "", _md(tables["trend_by_series"]), "",
        "## Intraday FX study (not part of the book)", "",
        "The Sleeve 1 model on five-minute EURUSD, USDJPY and USDCAD bars, each pair against the other two. "
        f"Folds train on {intraday_fx.TRAIN_DAYS} days and test on the next {intraday_fx.TEST_DAYS}. Costs are "
        "half a pip one way. Sharpe is annualised from P&L summed to days. The sample is about six months, so "
        "these are indicative only.", "",
        _md(tables["intraday_fx"]) if len(tables.get("intraday_fx", [])) else "Intraday file not found.", "",
        "## AMP paper-trading plan (dry run, nothing sent)", "",
        f"Latest target positions of the declared `all` book at ${PAPER_CAPITAL:,.0f} of capital. "
        "`exact` is the unrounded contract count; `one_lot_needs_capital` is the account size at which "
        "that leg would round to one micro contract.", "",
        _md(plan_table) if plan_table is not None else "No AMP spec snapshot available.", "",
        "Limit check on the rounded targets: " + ("pass" if not violations else "; ".join(violations)) + ".", "",
        "Limit check on one micro lot of every mapped leg: "
        + ("pass" if not one_lot_check else "; ".join(one_lot_check or [])) + ".", "",
    ]
    lines += NOTES
    (REPORT_DIR / "appendix.md").write_text("\n".join(lines), encoding="utf-8")


def main(force: bool = False) -> None:
    levels, rolls, rv_runs, trend_runs = load_runs(force)
    tables = build_tables(rv_runs, trend_runs, levels)
    tables["alpha"] = alpha_table(levels, rolls, rv_runs, trend_runs)
    intraday_cache = PROCESSED_DIR / "intraday_fx.parquet"
    if intraday_cache.exists() and not force:
        tables["intraday_fx"] = pd.read_parquet(intraday_cache)
    else:
        tables["intraday_fx"] = intraday_fx.run(levels)[1]
        tables["intraday_fx"].to_parquet(intraday_cache)
    (REPORT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / "tables" / f"{name}.csv", index=False)
    port = regime_book(levels, rv_runs, trend_runs, None)[0]  # signals and the AMP plan follow the regime-switching book
    # signals/signals.parquet belongs to the single-sleeve deliverable (amp_sa.run_sleeve1); not written here.
    plan_table, violations, one_lot_check = execution_plan(levels, rolls, rv_runs, trend_runs, port)
    if plan_table is not None:
        plan_table.to_csv(REPORT_DIR / "tables" / "amp_plan.csv", index=False)
    figures(rv_runs, trend_runs, tables, levels)
    write_report(tables, plan_table, violations, one_lot_check, levels)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(tables["sleeves"].drop(columns=["start", "end"]).to_string(index=False))
        print(tables["books"].drop(columns=["start", "end"]).to_string(index=False))
        print(tables["sleeve_correlation"].to_string(index=False))
        print(tables["regime_switch"].drop(columns=["end", "hit_rate"]).to_string(index=False))
        print(tables["alpha"].to_string(index=False))
        print(tables["intraday_fx"].to_string(index=False))

        print("limit check:", violations or "pass")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
