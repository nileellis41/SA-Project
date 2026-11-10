"""Implied-volatility mean reversion with delta-hedged options, against the PCA sleeve.

    python -m amp_sa.run_options

Writes reports/options.md with its tables and figures.

Underlyings:
  spx   S&P 500: Bloomberg VIX (panel) as implied vol, Yahoo ^GSPC as the underlying.
  ndx   Nasdaq-100: Yahoo ^VXN as implied vol, ^NDX as the underlying. The closest
        free proxy for a mega-cap tech name.
  msft  Microsoft: only if data/raw/msft_options.csv exists (Bloomberg export, Layout A,
        see MSFT_REQUEST). There is no free history of single-stock implied vol.

Declared before the first run and not changed after:
  BOOK_OPTION   the options sleeve that enters the PCA + options book.
  The cost assumptions in sleeves/options.COST_VOL_PTS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config, run
from amp_sa import run_lessons as rl
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import attribution, metrics
from amp_sa.data.load_bbg import RAW_DIR, read_layout_a
from amp_sa.data.load_yahoo import fetch
from amp_sa.portfolio.combine import combine
from amp_sa.portfolio.regime_switch import regime_budget, regime_features
from amp_sa.sleeves import options as opt

REPORT_DIR = run.REPORT_DIR
MSFT_FILE = RAW_DIR / "msft_options.csv"
BOOK_OPTION = ("spx_iv", "rolling")  # declared before the first run
PCA_HEADLINE = "PCA headline (PCA-2 + credit)"
COST_MULTS = (0.0, 0.5, 1.0, 2.0, 3.0)
STRESS_YEARS = (1998, 2001, 2008, 2011, 2018, 2020, 2022)
VARIANT_LABELS = {"static": "static", "rolling": "rolling", "rolling_hmm": "rolling + HMM gate"}
BENCH_LABELS = {"short_vol": "always short vol", "long_vol": "always long vol",
                "hmm_short_vol": "HMM switch: short vol in calm states", "underlying": "underlying, long"}
MSFT_REQUEST = [
    ("MSFT US Equity", "PX_LAST", "close; the hedge instrument"),
    ("MSFT US Equity", "30DAY_IMPVOL_100.0%MNY_DF", "30-day ATM implied vol: the traded signal"),
    ("MSFT US Equity", "60DAY_IMPVOL_100.0%MNY_DF", "60-day ATM implied vol: term structure view"),
    ("MSFT US Equity", "30DAY_IMPVOL_90.0%MNY_DF", "30-day 90% put-side vol: skew view"),
    ("MSFT US Equity", "VOLATILITY_30D", "30-day realised vol, to check our own estimate"),
    ("MSFT US Equity", "EARN_ANN_DT_TIME_HIST_WITH_EPS", "earnings dates: IV jumps before and collapses after"),
]


# Written by hand from the run of 2026-10-14 (sample to 2026-09-25, SPX and NDX proxies, no MSFT). The tables
# are regenerated on every run; this section is not, so re-check its numbers after any change.
FINDINGS = """## Findings (written from the 2026-10-14 run, index proxies only)

**Headline: implied-vol mean reversion beats the PCA sleeve after costs, but it does not beat simply selling
volatility, and all of its profit comes from the short-vol side.**

1. **It beats PCA because it is cheap to trade.** Net Sharpe: SPX IV level, rolling 0.18; NDX IV level,
   static 0.44; against -0.60 for the PCA headline and -0.09 for sparse PCA-3. A straddle costs 0.14-0.21
   daily sigmas one way against 0.4-1.4 for a PCA residual, so cost drag is 0.4-2% a year instead of 10%.
   Before costs PCA is still the better signal: gross Sharpe 1.21 for sparse PCA-3 against 0.39-0.55 here.
2. **The mean-reversion bet itself works.** The IV-level strategies earn in the vega column (SPX rolling
   +3.7% a year, NDX static +4.9%), which is the "implied vol returns to its mean" profit the trade is
   built on.
3. **But buying cheap vol loses money.** Every long-vol leg bleeds the gamma-theta cost of owning options:
   SPX rolling long-vol trades average -0.58 sigmas net (45 trades) against +1.05 for short-vol trades (108).
   NDX long-vol trades average -4.1 (7 trades). The textbook trade "buy the underpriced option" is the losing
   half. The strategy works as a way of timing when to sell vol.
4. **Always short vol beats every timing rule.** SPX: net Sharpe 1.15, +16% a year from gamma-theta, -4.5%
   from vega. NDX: 0.64. This number is inflated: VIX sits above at-the-money vol because of skew, so it
   overstates the premium a straddle seller collects (see caveats). Even halved, it would match the best
   timing rule.
5. **The regime layer adds nothing, again.** The HMM gate moved SPX from 0.18 to 0.16 and NDX from 0.27 to
   0.31. It allowed trading in both states in 28 of 32 SPX folds. The HMM short-vol switch never switched
   off, so it is identical to always short vol. This matches the curve, where the regime gate also did not
   help.
6. **Mean reversion earns when VIX is high.** SPX rolling: +3.6% a year in the top VIX tercile, -0.6% in the
   bottom. Short vol too earns most there (+17% a year), because the premium is largest when fear is. The PCA
   headline does the opposite: -10% a year when VIX is high.
7. **The tail is the danger, and it arrives after calm.** The worst days come when VIX was low the day
   before: SPX rolling lost 17.7% on 2018-10-10 (VIX from 16 to 23), always short vol lost 14.8% on
   2007-02-27 (VIX from 11) and 7.3% on 2018-02-05. Volatility targeting makes this worse: a short straddle's
   P&L is quiet in calm markets, so the position is largest just before a shock. 2018 was the worst year for
   both SPX always short vol (-13.5%) and SPX IV level, rolling (-23.5%).
8. **IV minus realised vol is the weaker signal.** The rolling version loses (SPX 0.00, NDX -0.25), mostly
   on long-vol trades when implied vol is below realised: -0.99 sigmas per trade over 94 trades. The static
   SPX version (0.28) is the most cost-robust timing rule, still 0.15 at twice the cost.
9. **It diversifies PCA but not equities.** Daily correlation with the PCA headline is 0.00-0.05. With the
   S&P 500 it is 0.14 for the timing rules and 0.57 for always short vol, which also has an equity t-stat of
   10.8 in the factor regression.
10. **In a book, regime switching mostly picks the options sleeve.** PCA headline plus SPX IV level, rolling
    (declared in advance): PCA alone -0.68, fixed 50/50 -0.38, regime switching 0.00 with 94% of risk on
    options on average, and 0.36 worst drawdown with the drawdown overlay. The allocator gave both states to
    options in 28 of 30 folds. It is avoiding PCA rather than switching between the two.
11. **Few decisions.** SPX IV level, static made 46 trades in 30 years; NDX static 35. Results of this size
    rest on a handful of episodes.
"""


def load_underlyings(levels: pd.DataFrame) -> list[opt.Underlying]:
    idx = levels.index
    close = lambda t: fetch(t)["close"].reindex(idx)  # noqa: E731
    out = [
        opt.Underlying("spx", close("^GSPC"), levels["vix"] / 100),
        opt.Underlying("ndx", close("^NDX"), close("^VXN") / 100),
    ]
    if MSFT_FILE.exists():
        long = read_layout_a(MSFT_FILE)
        wide = long.pivot_table(index="date", columns="field", values="value").reindex(idx)
        out.append(opt.Underlying("msft", wide["PX_LAST"], wide["30DAY_IMPVOL_100.0%MNY_DF"] / 100))
    return out


def label(run_name: str, variant: str) -> str:
    und, kind = run_name.split("_", 1)
    if kind == "bench":
        return f"{und.upper()} {BENCH_LABELS[variant]}"
    return f"{und.upper()} {'IV level' if kind == 'iv' else 'IV minus realised'}, {VARIANT_LABELS[variant]}"


def hold_cost(items) -> pd.Series:
    """Cost of re-establishing one risk unit of the sleeve's positions (for charging weight changes)."""
    return pd.DataFrame({r.name: (r.results[v].position.abs() * r.results[v].unit_cost).where(r.live)
                         for r, v in items}).mean(axis=1)


def decomposition(r, v, port) -> dict:
    """Annualised % of capital from the vega part, the gamma-theta part and costs."""
    res = r.results[v]
    held = res.position.shift(1).where(r.live)
    w = port.weights["s"]
    parts = {k: (w * held * r.extras[k]).mean() * config.ANN * 100 for k in ("vega", "gamma_theta")}
    return {"vega_pct_per_year": parts["vega"], "gamma_theta_pct_per_year": parts["gamma_theta"],
            "cost_pct_per_year": port.cost.mean() * config.ANN * 100,
            "net_pct_per_year": port.net.mean() * config.ANN * 100}


def regime_view(ports: dict, by: pd.Series, order: list[str]) -> pd.DataFrame:
    rows = []
    for name, p in ports.items():
        net = p.net.dropna()
        g = net.groupby(by.reindex(net.index))
        rows.append({"strategy": name, **{k: g.get_group(k).mean() * config.ANN * 100 if k in g.groups else np.nan
                                          for k in order}})
    return pd.DataFrame(rows)


def main() -> None:
    levels, rolls, rv_runs, _ = run.load_runs()
    pca_runs = s1.load_pca_runs(levels)
    pca_models = s1.models(rv_runs, pca_runs)

    unds = load_underlyings(levels)
    runs, diags = [], []
    for u in unds:
        r, d = opt.run_underlying(u, levels)
        runs += r
        diags.append(d)
    diag = pd.concat(diags, ignore_index=True)

    items = {}
    for r in runs:
        variants = list(opt.BENCHMARKS) + ["underlying"] if r.name.endswith("bench") else list(opt.VARIANTS)
        for v in variants:
            items[label(r.name, v)] = [(r, v)]
    built = {k: s1.sleeve(v) for k, v in items.items()}
    built[PCA_HEADLINE] = s1.sleeve(pca_models[s1.HEADLINE])
    built["PCA-3 sparse"] = s1.sleeve(pca_models["PCA-3 sparse (4 legs per residual)"])
    ports = {k: v[0] for k, v in built.items()}
    names = list(ports)

    # 1. Performance, per underlying, from each strategy's own start
    trials = [metrics.sharpe(p.net) for p in ports.values()]
    perf = []
    for name, (p, turnover, in_market) in built.items():
        row = metrics.summarize(p.net, p.gross, turnover, trials)
        row.update(sharpe_2x_cost=metrics.sharpe(p.net_at(2.0)), in_market=float(in_market.mean()))
        perf.append({"strategy": name, **row})
    perf = pd.DataFrame(perf)

    # 2. Costs
    cost_curve = pd.DataFrame([{"strategy": n, **{f"{m:g}x": metrics.sharpe(p.net_at(m)) for m in COST_MULTS}}
                               for n, p in ports.items()])

    # 3. Where the money comes from
    decomp = pd.DataFrame([{"strategy": n, **decomposition(*items[n][0], ports[n])}
                           for n in items if not n.endswith("underlying, long")])

    # 4. Conditions
    vix_prev = levels["vix"].shift(1)
    vix_bucket = pd.qcut(vix_prev, 3, labels=["low VIX", "mid VIX", "high VIX"]).astype(str).where(vix_prev.notna())
    spx = fetch("^GSPC")["close"].reindex(levels.index)
    trend = np.sign(spx.pct_change(126)).shift(1).map({1.0: "equities up 6m", -1.0: "equities down 6m"})
    two = levels["ust_2y"].diff(126)
    fed = pd.Series(np.select([two > 50, two < -50], ["hiking", "easing"], "on hold"), index=levels.index).where(two.notna())
    by_vix = regime_view(ports, vix_bucket, ["low VIX", "mid VIX", "high VIX"])
    by_trend = regime_view(ports, trend, ["equities up 6m", "equities down 6m"])
    by_fed = regime_view(ports, fed, ["hiking", "on hold", "easing"])
    vix_ranges = vix_prev.groupby(vix_bucket).agg(["min", "max"]).round(1)

    stress = pd.concat([metrics.by_year(p.net, STRESS_YEARS).assign(strategy=n).rename_axis("year").reset_index()
                        for n, p in ports.items()], ignore_index=True)
    stress = (stress.pivot(index="strategy", columns="year", values="return").reindex(names) * 100).reset_index()
    stress.columns = [str(c) for c in stress.columns]

    key = [label("spx_iv", "rolling"), label("spx_iv", "rolling_hmm"), label("spx_vrp", "rolling"),
           label("spx_bench", "short_vol"), label("ndx_iv", "rolling"), PCA_HEADLINE]
    yearly = pd.DataFrame({k: ports[k].net.groupby(ports[k].net.index.year).sum(min_count=1) * 100 for k in key})
    yearly = yearly.dropna(how="all").rename_axis("year").reset_index()

    # 5. Trades
    trade_rows, hold_rows, standout = [], [], []
    for n, its in items.items():
        if "short vol" in n or "long vol" in n or "underlying" in n:
            continue  # always-on benchmarks are one trade per fold
        t = rl.model_trades(its)
        if t.empty:
            continue
        years = its[0][0].live.sum() / config.ANN
        trade_rows.append({"strategy": n, **rl.trade_summary(t, years)})
        hold_rows.append(rl.grouped(t, pd.cut(t["bars"], rl.HOLD_BUCKETS, labels=rl.HOLD_LABELS).rename("holding"))
                         .assign(strategy=n))
        if n in key:
            side = t["side"].map({1.0: "long vol", -1.0: "short vol"})
            pick = pd.concat([t.nsmallest(5, "net").assign(kind="worst"), t.nlargest(5, "net").assign(kind="best")])
            pick = pick.assign(strategy=n, side=side.loc[pick.index])
            standout.append(pick[["strategy", "kind", "side", "entry", "exit", "bars", "gross", "cost", "net"]])
    trade_tbl = pd.DataFrame(trade_rows)
    hold_tbl = pd.concat(hold_rows, ignore_index=True)
    standout = pd.concat(standout, ignore_index=True)
    for c in ("entry", "exit"):
        standout[c] = pd.to_datetime(standout[c]).dt.date
    vol_by_side = pd.concat([rl.grouped(rl.model_trades(items[n]), rl.model_trades(items[n])["side"]
                                        .map({1.0: "long vol", -1.0: "short vol"}).rename("side")).assign(strategy=n)
                             for n in key if n in items and "short vol" not in n], ignore_index=True)

    # 6. Worst and best days, with what the market did
    days = []
    spx_ret = spx.dropna().pct_change()  # from the last day the market was open
    for n in key[:4]:
        net = ports[n].net.dropna()
        for kind, d in (("worst", net.nsmallest(5).index), ("best", net.nlargest(5).index)):
            for day in d:
                i = levels.index.get_loc(day)
                days.append({"strategy": n, "kind": kind, "date": day.date(), "net_pct": net.loc[day] * 100,
                             "spx_return_pct": spx_ret.get(day, np.nan) * 100,
                             "vix_change": levels["vix"].iloc[i] - levels["vix"].iloc[i - 1],
                             "vix_prev_close": levels["vix"].iloc[i - 1]})
    days = pd.DataFrame(days)

    # 7. Correlation and factor exposure
    nets = pd.DataFrame({n: p.net for n, p in ports.items()})
    nets["S&P 500 daily return"] = spx.dropna().pct_change()
    corr = nets.corr(min_periods=250)[[PCA_HEADLINE, "PCA-3 sparse", "S&P 500 daily return"]].loc[
        [n for n in names if n not in (PCA_HEADLINE, "PCA-3 sparse")]].round(2).rename_axis("strategy").reset_index()
    alpha = attribution.attribution_table({n: ports[n].net for n in key}, levels, rolls)
    alpha = alpha[alpha["factors"] == "macro"][["series", "start", "ann_return", "alpha", "alpha_t", "r_squared",
                                                 "t_duration", "t_credit", "t_equity"]]

    # 8. Fold diagnostics
    fold_tbl = diag.groupby(["underlying", "kind"]).agg(
        folds=("fold", "count"), train_half_life_median=("train_half_life", "median"),
        static_traded_share=("static_traded", "mean"), rolling_tradeable_share=("rolling_tradeable_share", "mean"),
        rolling_entry=("rolling_entry", "median"),
        rolling_exit=("rolling_exit", "median"), cost_over_daily_vol=("cost_over_daily_vol", "median"),
        hmm_states_allowed=("hmm_states_allowed", lambda s: ", ".join(f"{k}: {v}" for k, v in s.value_counts().items())),
    ).reset_index()

    # 9. The book: PCA headline + options sleeve, fixed and regime-switched
    opt_name = label(*BOOK_OPTION)
    opt_items = items[opt_name]
    pca_items = pca_models[s1.HEADLINE]

    def parts(its):
        g = pd.DataFrame({r.name: r.results[v].gross.where(r.live) for r, v in its}).mean(axis=1)
        c = pd.DataFrame({r.name: r.results[v].cost.where(r.live) for r, v in its}).mean(axis=1)
        return g, c

    (g1, c1), (g2, c2) = parts(pca_items), parts(opt_items)
    # The options sleeve runs on equity trading days; on Treasury-only days it earns and pays nothing.
    span = (g1.index >= g2.first_valid_index()) & (g1.index <= g2.last_valid_index())
    g2 = g2.reindex(g1.index).where(~span | g2.reindex(g1.index).notna(), 0.0)
    c2 = c2.reindex(g1.index).where(~span | c2.reindex(g1.index).notna(), 0.0)
    sleeves = {"pca": (g1, c1), "options": (g2, c2)}
    hold = {"pca": hold_cost(pca_items), "options": hold_cost(opt_items)}
    budget, switch_diag = regime_budget(pd.DataFrame({"pca": g1 - c1, "options": g2 - c2}), regime_features(levels))
    books = {
        "PCA headline only": combine(sleeves, {"pca": 1.0, "options": 0.0}, use_overlay=False, hold_cost=hold),
        f"{opt_name} only": combine(sleeves, {"pca": 0.0, "options": 1.0}, use_overlay=False, hold_cost=hold),
        "fixed 50/50": combine(sleeves, {"pca": 0.5, "options": 0.5}, use_overlay=False, hold_cost=hold),
        "regime switching": combine(sleeves, budget, use_overlay=False, hold_cost=hold),
        "regime switching + drawdown overlay": combine(sleeves, budget, use_overlay=True, hold_cost=hold),
    }
    window = books["regime switching"].net.dropna().index
    btrials = [metrics.sharpe(b.net.reindex(window)) for b in books.values()]
    book_tbl = pd.DataFrame([{"book": n, **metrics.summarize(b.net.reindex(window), b.gross.reindex(window), None, btrials),
                              "sharpe_2x_cost": metrics.sharpe(b.net_at(2.0).reindex(window))}
                             for n, b in books.items()])
    book_tbl["avg_options_budget"] = [np.nan, np.nan, 0.5, budget["options"].reindex(window).mean(),
                                      budget["options"].reindex(window).mean()]
    book_stress = pd.concat([metrics.by_year(b.net.reindex(window), STRESS_YEARS).assign(book=n)
                             .rename_axis("year").reset_index() for n, b in books.items()], ignore_index=True)
    book_stress = (book_stress.pivot(index="book", columns="year", values="return").reindex(list(books)) * 100).reset_index()
    book_stress.columns = [str(c) for c in book_stress.columns]
    sleeve_corr = float((g1 - c1).corr(g2 - c2))

    tables = {"options_performance": perf, "options_cost_curve": cost_curve, "options_decomposition": decomp,
              "options_by_vix": by_vix, "options_by_equity_trend": by_trend, "options_by_fed": by_fed,
              "options_stress": stress, "options_yearly": yearly, "options_trades": trade_tbl,
              "options_trades_by_holding": hold_tbl, "options_trades_by_side": vol_by_side,
              "options_standout_trades": standout, "options_days": days, "options_corr": corr,
              "options_alpha": alpha, "options_folds": fold_tbl, "options_book": book_tbl,
              "options_book_stress": book_stress, "options_switch_folds": switch_diag}
    for k, t in tables.items():
        t.to_csv(REPORT_DIR / "tables" / f"{k}.csv", index=False)

    pct = lambda s: s.dropna().cumsum() * 100  # noqa: E731
    fig = REPORT_DIR / "figures"
    run._line_chart({label("spx_iv", "rolling"): pct(ports[label("spx_iv", "rolling")].net),
                     label("spx_bench", "short_vol"): pct(ports[label("spx_bench", "short_vol")].net),
                     PCA_HEADLINE: pct(ports[PCA_HEADLINE].net)},
                    "Implied-vol mean reversion (S&P 500) against short vol and the PCA sleeve, net",
                    "cumulative return, % of capital", fig / "options_vs_pca.png")
    run._line_chart({n: pct(books[n].net.reindex(window)) for n in ("PCA headline only", "fixed 50/50", "regime switching")},
                    "PCA headline plus the options sleeve: fixed and regime-switched risk budgets",
                    "cumulative return, % of capital", fig / "options_book.png")

    write_report(levels, unds, tables, opt_name, window, sleeve_corr, vix_ranges)
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for k, t in tables.items():
            print(f"\n== {k}")
            print(t.to_string(index=False))
        print("sleeve corr", sleeve_corr)


def write_report(levels, unds, t, opt_name, window, sleeve_corr, vix_ranges) -> None:
    md = run._md
    perf_cols = ["strategy", "start", "ann_return", "ann_vol", "sharpe", "deflated_sharpe", "gross_sharpe",
                 "cost_drag", "turnover", "max_dd", "sharpe_2x_cost", "in_market"]
    have_msft = any(u.name == "msft" for u in unds)
    lines = [
        "# Implied-volatility mean reversion with delta-hedged options", "",
        f"Sample ends {levels.index[-1].date()}. Out of sample from walk-forward folds (1,260-bar training, "
        "252-bar test, 10-bar purge, 5-bar embargo), net of costs, at a 10% volatility target, the same rig as "
        "the PCA sleeve.", "",
        FINDINGS, "",
        "## The trade", "",
        "Buy a 30-day at-the-money straddle when implied volatility is cheap against its mean, sell it when it "
        "is rich, and hedge the delta every day in the underlying. Daily P&L per $1 of underlying, with "
        "Black-Scholes greeks at the previous close, has two parts:", "",
        "- **vega part**: vega x change in implied vol. This is the mean-reversion bet.",
        "- **gamma-theta part**: (gamma / 2) x (realised move squared - implied variance x dt). A long straddle "
        "loses this every day the market moves less than the option priced. On average, index options price more "
        "variance than is realised (the volatility risk premium), so a seller earns it.", "",
        "Signals, fitted per fold on the training window:", "",
        "| signal | static | rolling | rolling + HMM gate |",
        "|:--|:--|:--|:--|",
        "| IV level | z against the training OU mean; trades only if the fit reverts (half-life 1-60 bars) | "
        "s-score on a trailing 60-bar OU fit, as in the PCA sleeve | rolling, only in HMM states where it made "
        "money in training |",
        "| IV minus 21-day realised vol | same | same | same |", "",
        "Above the entry band the straddle is sold (vol rich); below, bought. Bands from the training-window grid. "
        "Benchmarks on the same test windows: always short vol, always long vol, an HMM switch that sells vol "
        "only in calm states, and the underlying itself.", "",
        "## Data: proxies now, MSFT next", "",
        "There is no free history of single-stock implied volatility, so this run uses two index proxies:", "",
        "| underlying | implied vol | price | from |",
        "|:--|:--|:--|:--|",
        "| S&P 500 (`SPX`) | VIX, Bloomberg (panel) | ^GSPC, Yahoo | 1990 |",
        "| Nasdaq-100 (`NDX`) | VXN, Yahoo | ^NDX, Yahoo | 2001 |", "",
        ("MSFT is included below from `data/raw/msft_options.csv`." if have_msft else
         "**MSFT is not in this run.** To add it, export the fields below from the Bloomberg terminal as "
         "`data/raw/msft_options.csv` in the same layout as `rates_1.csv` (Layout A: one date/value column pair "
         "per field) and rerun `python -m amp_sa.run_options`. Check the field names with FLDS first."), "",
        "| ticker | field | why |", "|:--|:--|:--|",
        *[f"| {a} | `{b}` | {c} |" for a, b, c in MSFT_REQUEST], "",
        "Single names differ from indices in ways that matter for this trade: implied vol rises into every "
        "earnings date and drops the day after, so an IV-level signal on MSFT will mostly be an earnings calendar "
        "unless earnings windows are handled. Single-stock options also trade wider, and the volatility risk "
        "premium is smaller than on indices.", "",
        "## 1. Performance", "",
        md(t["options_performance"][perf_cols]), "",
        "![options vs pca](figures/options_vs_pca.png)", "",
        "## 2. Where the money comes from", "",
        "Annualised % of capital from each part of the straddle P&L. A strategy that earns in the vega column is "
        "trading implied-vol mean reversion. One that earns in the gamma-theta column is harvesting (or paying) the "
        "volatility risk premium.", "",
        md(t["options_decomposition"]), "",
        "## 3. Cost sensitivity", "",
        f"Net Sharpe at multiples of the assumed cost: half the bid-ask, {opt.COST_VOL_PTS['spx']} vol points for "
        f"SPX and {opt.COST_VOL_PTS['ndx']} for NDX, one way. Option spreads are the least certain input here, so "
        "read the 2x and 3x columns.", "",
        md(t["options_cost_curve"]), "",
        "## 4. Under different conditions", "",
        "Annualised net return, % of capital. Conditions are known at the previous close.", "",
        "### By VIX level", "",
        "VIX terciles over the whole panel: " + ", ".join(f"{k} {r['min']:.1f}-{r['max']:.1f}" for k, r in vix_ranges.iterrows()) + ".", "",
        md(t["options_by_vix"], 1), "",
        "### By equity trend (S&P 500 over the past 126 bars)", "",
        md(t["options_by_equity_trend"], 1), "",
        "### By Fed cycle (2-year yield over the past 126 bars, +/-50 bp)", "",
        md(t["options_by_fed"], 1), "",
        "### Stress years, % return", "",
        md(t["options_stress"], 1), "",
        "### Year by year, % net", "",
        md(t["options_yearly"], 1), "",
        "## 5. Trades", "",
        "Trades in daily sigmas of the straddle's P&L, as in [trades.md](trades.md).", "",
        md(t["options_trades"]), "",
        "### By holding period", "",
        md(rl.first_cols(t["options_trades_by_holding"], ["strategy", "holding"])), "",
        "### Long vol against short vol", "",
        md(rl.first_cols(t["options_trades_by_side"], ["strategy", "side"])), "",
        "### Best and worst trades", "",
        md(t["options_standout_trades"]), "",
        "### Best and worst days", "",
        md(t["options_days"]), "",
        "## 6. Fold diagnostics", "",
        "Median across folds. `train_half_life_median` is the OU half-life of the signal on the training window, "
        "in bars; `cost_over_daily_vol` is the one-way cost of a straddle over the daily volatility of its P&L.", "",
        md(t["options_folds"]), "",
        "## 7. Diversification and factor exposure", "",
        "Correlation of daily net P&L with the PCA sleeves and with the S&P 500:", "",
        md(t["options_corr"]), "",
        "Factor regression (macro factors from 2005: duration, steepener, credit, crude, equity; Newey-West t):", "",
        md(t["options_alpha"], 3), "",
        "## 8. A book: PCA headline plus the options sleeve", "",
        f"The options sleeve for the book was declared before the first run: **{opt_name}**. Daily correlation of "
        f"the two sleeves: {sleeve_corr:.2f}. `regime switching` is the appendix's allocator: a two-state HMM on "
        "log VIX and log MOVE, each state assigned on the training window to the sleeve with the better Sharpe in "
        f"it, budget = filtered-probability mix. All rows cover the same window ({window[0].date()} on) and pay "
        "for re-weighting.", "",
        md(t["options_book"]), "",
        "![book](figures/options_book.png)", "",
        md(t["options_book_stress"], 1), "",
        "## Caveats", "",
        "- **VIX is not an ATM implied vol.** It is a model-free 30-day variance price and sits above ATM vol "
        "because of skew. The vega and gamma-theta parts are priced with it as if it were ATM vol, which "
        "overstates the premium a straddle seller collects.",
        "- **Constant maturity.** The straddle is rolled every day to keep 30 days to expiry. A real position is "
        "rolled monthly; its greeks drift between rolls and each roll costs a spread.",
        "- **Costs are assumed**, not measured: half the bid-ask in vol points times vega. Delta-hedging costs "
        "are not charged.",
        "- **Black-Scholes greeks, daily hedging.** Jumps between closes are in the gamma-theta part (a squared "
        "return), so crash days are captured, but intraday hedging is not.",
        f"- **Trials.** The deflated Sharpe counts the {len(t['options_performance'])} rows of the performance table.",
        "- **Index proxies, not MSFT.** See the data section.", "",
    ]
    (REPORT_DIR / "options.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

