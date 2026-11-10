"""Directional 2s10s curve trades (steepeners and flatteners) against the PCA residual sleeve.

    python -m amp_sa.run_curve

Writes reports/curve_trades.md with its tables and figure.

A steepener is long the 2-year and short the 10-year, DV01-neutral; a flattener is
the reverse. In yield space both are a position in the 2s10s slope (10y - 2y, in bp):
the steepener earns when the slope rises. The rules below turn the textbook story
("steepen when growth picks up or the Fed eases, flatten when the Fed hikes") into
signals that use data through the close of t only. Their parameters are fixed here,
declared before the first run, and not tuned:

  always steepener   +1 every bar; the unconditional slope bet. Always flattener is -1.
  policy             2y yield up more than 25 bp over 126 bars (hiking priced) -> flattener;
                     down more than 25 bp (easing) -> steepener; otherwise flat.
  growth             HY OAS tighter than 126 bars ago (growth improving) -> steepener;
                     wider -> flattener. From 2000-08, when daily OAS starts.
  policy + growth    sign of the sum of the two: trades when they agree or one is flat.
  slope momentum     sign of the 63-bar change in 2s10s: follow the curve's own trend.

Every rule uses the shared engine: position = signal / trailing vol, held over t+1,
charged 0.25 bp per leg one way (0.5 bp per unit of slope). Each is scaled to the 10%
volatility target exactly as the PCA sleeve is. Carry and roll-down are not in the
P&L of either, which matters more here: a steepener usually earns carry and a
flattener pays it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config, run
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import attribution, metrics
from amp_sa.backtest.engine import account
from amp_sa.sleeves.rv import StrategyRun

REPORT_DIR = run.REPORT_DIR
POLICY_BARS, POLICY_BP = 126, 25.0
GROWTH_BARS = 126
MOMENTUM_BARS = 63
REGIME_BARS, REGIME_BP = 126, 50.0  # for labelling periods only, not a trading rule
PCA_ROWS = {"PCA-2 curve + adaptive credit pairs": "PCA headline (PCA-2 + credit)",
            "PCA-3 sparse (4 legs per residual)": "PCA-3 sparse"}
MR_ROWS = {"static": "2s10s mean reversion, static", "adaptive_nn": "2s10s mean reversion, adaptive"}

# Written by hand from the run of 2026-10-14 (sample to 2026-09-25). The tables below it are regenerated on
# every run; this section is not, so re-check its numbers after any model or data change.
FINDINGS = """## Findings (written from the 2026-10-14 run)

1. **The policy rule is the best strategy in this comparison, PCA included.** Net Sharpe 0.45 since 1995
   (0.52 since 2001), deflated Sharpe 0.53 across the ten rows, 0.22 even at twice the cost table, and a
   worst drawdown of 0.36 against 1.79 for the PCA headline. Its alpha t-statistic is 2.5 with an R-squared
   of 0.01 on the duration and steepener factors: it earns by timing the slope, not by holding it.
2. **The textbook story holds when it is read off the 2-year yield.** The policy rule earned about 10% a
   year while hikes were being priced (flattener) and about 12% a year while cuts were being priced
   (steepener), and nothing in between. Holding a steepener through hiking periods lost 11% a year.
3. **Unconditional slope bets are a factor, not a strategy.** Always steepener: net Sharpe -0.18 in yield
   terms, R-squared 0.81 on the steepener factor, -21% in 2022. Always flattener: +0.05. Carry is missing
   from both and would move them toward the steepener.
4. **The growth story does not work as stated.** Using HY spreads as the growth signal loses (-0.09), and
   loses most when the Fed is easing (-7% a year): credit widens in a slowdown, the rule goes flattener, and
   the curve bull-steepens as the Fed cuts. Adding it to the policy rule cut the Sharpe from 0.45 to 0.26.
5. **Slope momentum has a gross edge (0.48) but trades most often of any curve rule** and nets 0.09.
6. **The real difference from PCA is cost, not signal.** Sparse PCA-3 has the best gross Sharpe here (1.21
   against 0.68 for the policy rule). But a slope trade pays about 0.17 daily sigmas to enter against a
   median 0.53 for a PCA-2 residual. Curve rules lose 1-4% a year to cost; the PCA sleeves lose 10-11%.
7. **Curve trades have their own bad episodes.** The policy rule lost 6.4% in 2008: it held a steepener
   January to June, switched to a flattener in July to September after the spring rise in 2-year yields,
   just as 2s10s widened from 144 to 240 bp, then went back to a steepener as the curve flattened to 145 bp.
   Rules on a 126-bar lookback are slow at turning points.
8. **They would diversify the PCA sleeve.** Daily correlation with the PCA headline is between -0.07 and
   +0.01 for every curve rule.
9. **Hold the result lightly.** The policy rule changes sign a few times a year, so 30 years hold a few
   dozen independent calls. Its 25 bp / 126-bar settings were chosen once and not tuned, but they are one
   choice among many. The next test is a grid of thresholds and lookbacks, declared in advance and counted
   as trials, to see whether 0.45 is typical or lucky.
10. **This is a directional macro trade, not stat arb.** It belongs next to the trend sleeve (net Sharpe
   0.51 in the appendix), which is also a directional rates trade, rather than in the graded mean-reversion
   deliverable.
"""


def slope(levels: pd.DataFrame) -> pd.Series:
    return levels["ust_10y"] - levels["ust_2y"]


def signals(levels: pd.DataFrame) -> dict[str, pd.Series]:
    s = slope(levels)
    two = levels["ust_2y"].diff(POLICY_BARS)
    policy = pd.Series(np.select([two > POLICY_BP, two < -POLICY_BP], [-1.0, 1.0], 0.0), index=levels.index)
    policy = policy.where(two.notna())
    growth = -np.sign(levels["hy_oas"].diff(GROWTH_BARS))
    both = np.sign(policy + growth)
    return {
        "always steepener": pd.Series(1.0, index=levels.index).where(s.notna()),
        "always flattener": pd.Series(-1.0, index=levels.index).where(s.notna()),
        "policy rule": policy,
        "growth rule": growth,
        "policy + growth": both,
        "slope momentum": np.sign(s.diff(MOMENTUM_BARS)),
    }


def curve_runs(levels: pd.DataFrame, start: pd.Timestamp) -> list[StrategyRun]:
    change = slope(levels).diff()
    unit_cost = config.COST_PER_UNIT["ust_2y"] + config.COST_PER_UNIT["ust_10y"]
    runs = []
    for name, sig in signals(levels).items():
        live = sig.notna() & (levels.index >= start)
        res = account(sig.where(live), change, unit_cost)
        runs.append(StrategyRun(name, "curve", {"dir": res}, live, pd.DataFrame(), {}))
    return runs


def fed_regime(levels: pd.DataFrame) -> pd.Series:
    """Hiking / easing / on hold from the trailing 126-bar change in the 2y yield (known at t)."""
    two = levels["ust_2y"].diff(REGIME_BARS)
    return pd.Series(np.select([two > REGIME_BP, two < -REGIME_BP], ["hiking", "easing"], "on hold"),
                     index=levels.index).where(two.notna())


def main() -> None:
    levels, rolls, rv_runs, _ = run.load_runs()
    pca_runs = s1.load_pca_runs(levels)
    pca_models = s1.models(rv_runs, pca_runs)
    pca_ports = {label: s1.sleeve(pca_models[name]) for name, label in PCA_ROWS.items()}
    start = pca_ports["PCA headline (PCA-2 + credit)"][0].net.first_valid_index()

    crv = curve_runs(levels, start)
    ports = {r.name: s1.sleeve([(r, "dir")]) for r in crv}
    mr = next(r for r in rv_runs if r.name == "ust_2s10s")
    ports.update({label: s1.sleeve([(mr, v)]) for v, label in MR_ROWS.items()})
    ports.update(pca_ports)
    names = list(ports)
    curve_names = [r.name for r in crv]

    def table(window_start=None) -> pd.DataFrame:
        cut = lambda s: s.loc[window_start:] if window_start is not None else s  # noqa: E731
        trials = [metrics.sharpe(cut(p.net)) for p, _, _ in ports.values()]
        rows = []
        for name, (p, turnover, in_market) in ports.items():
            row = metrics.summarize(cut(p.net), cut(p.gross), cut(turnover), trials)
            row["sharpe_2x_cost"] = metrics.sharpe(cut(p.net_at(2.0)))
            row["in_market"] = float(cut(in_market).mean())
            rows.append({"strategy": name, **row})
        return pd.DataFrame(rows)

    full = table()
    # The 2s10s mean-reversion model is in the market about 3% of the time, so its sleeve only starts
    # producing P&L in 2009; leave it out when choosing the common window.
    common_start = max(ports[n][0].net.first_valid_index() for n in names if n not in MR_ROWS.values())
    slope_vol = slope(levels).diff().loc[start:].ewm(span=config.VOL_SPAN).std().median()
    slope_cost = config.COST_PER_UNIT["ust_2y"] + config.COST_PER_UNIT["ust_10y"]
    pca_cost_ratio = pd.concat([r.diagnostics for r in pca_runs[2]])["cost_over_daily_vol"].median()
    common = table(common_start)

    # Correlation of daily net P&L with the PCA headline, and with each other.
    nets = pd.DataFrame({n: ports[n][0].net for n in names})
    corr = nets.corr(min_periods=250)[["PCA headline (PCA-2 + credit)", "PCA-3 sparse"]].loc[
        curve_names + list(MR_ROWS.values())].round(2).rename_axis("strategy").reset_index()

    # Does the textbook story hold? Net return by Fed regime.
    regime = fed_regime(levels)
    reg_rows = []
    for n in names:
        net = ports[n][0].net.dropna()
        g = net.groupby(regime.reindex(net.index))
        for lab, chunk in g:
            reg_rows.append({"strategy": n, "regime": lab, "share_of_bars": len(chunk) / len(net),
                             "net_pct_per_year": chunk.mean() * config.ANN * 100, "net_sharpe": metrics.sharpe(chunk)})
    by_regime = pd.DataFrame(reg_rows).pivot(index="strategy", columns="regime", values="net_pct_per_year")
    by_regime = by_regime.loc[names, ["hiking", "on hold", "easing"]].reset_index()
    by_regime.columns.name = None
    regime_share = (regime.loc[start:].value_counts(normalize=True) * 100).round(0)

    stress = pd.concat([metrics.by_year(ports[n][0].net, config.REGIME_YEARS).assign(strategy=n)
                        .rename_axis("year").reset_index() for n in names], ignore_index=True)
    stress = stress.pivot(index="strategy", columns="year", values="return").loc[names] * 100
    stress.columns = [f"{c} return %" for c in stress.columns]
    stress = stress.reset_index()

    cost_curve = pd.DataFrame([{"strategy": n, **{f"{m:g}x": metrics.sharpe(p.net_at(m)) for m in s1.COST_GRID}}
                               for n, (p, _, _) in ports.items()])
    alpha = attribution.attribution_table({n: ports[n][0].net for n in names}, levels, rolls)
    alpha = alpha[alpha["factors"] == "rates"][["series", "ann_return", "alpha", "alpha_t", "r_squared",
                                                 "t_duration", "t_steepener"]]

    tables = {"curve_full": full, "curve_common": common, "curve_corr": corr, "curve_by_regime": by_regime,
              "curve_stress": stress, "curve_cost_curve": cost_curve, "curve_alpha": alpha}
    (REPORT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(REPORT_DIR / "tables" / f"{k}.csv", index=False)

    pct = lambda s: s.dropna().cumsum() * 100  # noqa: E731
    run._line_chart({"PCA headline": pct(ports["PCA headline (PCA-2 + credit)"][0].net),
                     "policy + growth": pct(ports["policy + growth"][0].net),
                     "slope momentum": pct(ports["slope momentum"][0].net)},
                    "Directional curve trades against the PCA sleeve, net of costs, 10% vol target",
                    "cumulative return, % of capital", REPORT_DIR / "figures" / "curve_vs_pca.png")

    keep = ["strategy", "start", "ann_return", "ann_vol", "sharpe", "gross_sharpe", "cost_drag", "turnover",
            "max_dd", "sharpe_2x_cost", "in_market"]
    md = run._md
    lines = [
        "# Directional curve trades vs the PCA sleeve", "",
        f"Sample ends {levels.index[-1].date()}. Steepeners and flatteners on the 2s10s slope, run through the same "
        "engine, cost model and 10% volatility target as the PCA sleeve, from the PCA sleeve's first out-of-sample "
        f"day ({start.date()}). Net of costs unless marked gross.", "",
        "**The two are different kinds of trade.** A steepener or flattener is a bet on the slope itself: the second "
        "principal component of the curve. The PCA sleeve hedges level and slope out and trades what is left. So "
        "this is a comparison between taking slope risk on purpose and removing it.", "",
        FINDINGS,
        "## The rules", "",
        "Each rule uses data through the close of the day it trades on. Parameters were fixed before the first run "
        "and not tuned.", "",
        "| strategy | goes steepener when | goes flattener when |",
        "|:--|:--|:--|",
        "| always steepener / flattener | always / never | never / always |",
        f"| policy rule | 2y yield fell more than {POLICY_BP:g} bp over {POLICY_BARS} bars (easing priced) | "
        f"2y yield rose more than {POLICY_BP:g} bp (hikes priced) |",
        f"| growth rule | HY OAS tighter than {GROWTH_BARS} bars ago (growth improving) | HY OAS wider |",
        "| policy + growth | the two agree on steepener, or one says steepener and the other is flat | the mirror image |",
        f"| slope momentum | 2s10s rose over the past {MOMENTUM_BARS} bars | 2s10s fell |",
        "| 2s10s mean reversion | the existing pairwise model: slope is below its fitted mean | slope is above it |", "",
        "A steepener is long the 2-year and short the 10-year, weighted to equal DV01. Cost: 0.25 bp per leg one way.", "",
        "## Performance, each from its own start", "",
        md(full[keep]), "",
        f"## Performance on the common window from {common_start.date()}", "",
        "The growth rule needs daily HY OAS, so this window starts later. Same columns. The 2s10s "
        "mean-reversion rows trade so rarely (about 3% of days) that the sleeve only starts in 2009 and their "
        "volatility is far below target; they are kept for reference.", "",
        md(common[keep]), "",
        "![curve vs pca](figures/curve_vs_pca.png)", "",
        "## Does the textbook story hold? Net return by Fed regime", "",
        f"Annualised net return, in % of capital, by the trailing {REGIME_BARS}-bar change in the 2-year yield: "
        f"`hiking` when it rose more than {REGIME_BP:g} bp, `easing` when it fell more than {REGIME_BP:g} bp. Share "
        f"of days since {start.date()}: "
        + ", ".join(f"{k} {v:.0f}%" for k, v in regime_share.items())
        + ". The policy rule uses the same variable with a lower threshold, so its row is partly by construction.", "",
        md(by_regime, 1), "",
        "## Stress years", "",
        md(stress, 1), "",
        "## Correlation with the PCA sleeve", "",
        "Correlation of daily net P&L. Low or negative means the curve trade would diversify the PCA sleeve.", "",
        md(corr), "",
        "## What drives the P&L: factor regression", "",
        "Daily net P&L on unit-volatility duration and 2s10s-steepener factors, Newey-West t-statistics (same as "
        "report.md). A directional curve trade should load on the steepener factor; the PCA sleeve should not.", "",
        md(alpha, 3), "",
        "## Cost sensitivity", "",
        f"One unit of slope costs {slope_cost:.2f} bp one way and the slope moves about {slope_vol:.1f} bp on a "
        f"typical day, so a curve trade pays about {slope_cost / slope_vol:.2f} daily sigmas to enter. A PCA-2 "
        f"residual pays a median {pca_cost_ratio:.2f}. Net Sharpe at multiples of the cost table "
        "(1x = 0.25 bp per leg, one way):", "",
        md(cost_curve), "",
        "## Caveats", "",
        "- **No carry or roll-down.** Both sides are measured in yield changes only. On a positively sloped curve a "
        "DV01-neutral steepener earns carry and roll-down and a flattener pays it, so the true gap between the "
        "always-steepener and always-flattener rows is wider than shown.",
        "- **Hindsight in the story, not in the rules.** The rules are trailing and untuned, but the idea that "
        "hikes flatten the curve is itself known from history. Treat good results in the `hiking` column as "
        "expected, not discovered.",
        f"- **Trials.** The deflated Sharpe counts the {len(names)} rows of each table as trials.",
        "- **Few, long bets.** A directional rule changes sign a few times a year, so a 30-year sample holds a "
        "few dozen independent decisions. Sharpe estimates here are much noisier than for the PCA sleeve, which "
        "makes hundreds of short trades.", "",
    ]
    (REPORT_DIR / "curve_trades.md").write_text("\n".join(lines), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        for k, t in tables.items():
            print(f"\n== {k}")
            print(t.to_string(index=False))
        print(regime_share)


if __name__ == "__main__":
    main()
