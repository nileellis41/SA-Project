"""Write analysis/portfolio_diagnostics/report.md and dashboard.html from the computed tables.

    python -m analysis.portfolio_diagnostics.cost_audit        # step 1 (also run by run_all)
    python -m analysis.experiments.run_experiments             # step 8
    python -m analysis.portfolio_diagnostics.run_all           # steps 1-7
    python -m analysis.portfolio_diagnostics.report            # this file

The findings below are written by hand from the tables; every number in them is in tables/.
"""

from __future__ import annotations

import json
import pickle

import pandas as pd

from analysis.experiments.run_experiments import OUT as EXP_DIR
from analysis.portfolio_diagnostics.common import OUT_DIR
from analysis.portfolio_diagnostics.run_all import LABELS

FINDINGS = [
    "**No model makes money after cost, and the cost table is the reason.** Net Sharpe is −0.60 for the headline, −0.09 "
    "for PCA-3 sparse, −0.67 for PCA-2 full and −0.26 for pairwise adaptive. Gross Sharpe is 0.55, 1.21, 0.45 and 0.46. "
    "The PCA models pay 9.6–10.8% of capital a year in cost, against gross P&L of 3.8–10.1%.",
    "**The signal is real.** On entry bars, the rank correlation between −z and the next 10 bars of residual change is "
    "0.33 for PCA-3 sparse, 0.17 for the headline and 0.13 for pairwise (all p < 0.05). "
    "The fixed 2s5s10s and 3s5s7s flies, traded with the same rules, have gross Sharpe of only 0.11 and 0.38.",
    "**Breakeven cost is below the 1x table for every model.** The cost multiple at which mean net P&L is zero is 0.93x "
    "for PCA-3 sparse, 0.48x for the headline, 0.40x for PCA-2 full and 0.64x for pairwise. A more realistic table "
    "(0.5 bp before 2008, 0.25 bp after, 30y at 1.5x) moves net Sharpe to −1.12 for the headline and −0.63 for PCA-3 sparse.",
    "**Cost comes from entries and exits, not from maintenance.** Entries and exits are 75–87% of cost. Daily vol "
    "resizing is 8–10%, sleeve leverage changes 4–8%, and re-weighting legs at refits is at most 0.035% of capital a year. "
    "Netting legs across members saves only 0.5–5.5%, because members rarely hold offsetting legs at the same time.",
    "**Low-volatility entries are where the money is lost.** For the headline, the lowest-vol third of trades was entered "
    "when the residual moved 0.39 bp a day. A round trip cost 1.8x that daily vol, and those trades averaged −1.25 sigma "
    "net. In the highest-vol third, round trip was 0.6x vol. PCA-3 sparse makes +0.49 sigma per trade in its high-vol third "
    "and loses −0.48 in its low-vol third. A cost-aware entry rule at k = 1 would not remove them: 92% of PCA-2 entries "
    "already pass it (median forecast reversion 1.67 bp against a 0.73 bp round trip), because the 60-bar OU forecasts "
    "far more than arrives (mean realised move 0.45 bp a trade). k has to be fitted, and halves the entries only at about 2.4.",
    "**The hedge does its job; the trade it isolates is small.** PCA-2 residuals move 0.5–1.1 bp a day against 5–6 bp for "
    "raw yields, so 96–99% of the variance is removed. Against causal trailing-1y loadings, leakage to the hedged "
    "factors is 2–7% of gross DV01. PCA-2 deliberately keeps curvature: 27% of its gross DV01, and 73 of its 237 risk "
    "units of gross P&L.",
    "**PCA-3 sparse has a construction defect in 2010–2013, and it does fewer bets than it appears.** Before the 7y has a "
    "full training window, the 2y/3y/5y residual is hedged with 2y, 5y and 10y, which nearly span the same factors. Leg "
    "weights reach −121 on the 5y per unit of 3y. Trades entered in those four folds lost 67 risk units net, against +9 for "
    "the whole model. The 2y, 3y and 5y members are the same portfolio, so its 6 members are 2.3 effective bets.",
    "**The OU fit the signal uses is much faster than the training-window fit, and that is mostly small-sample bias.** "
    "On each 5-year training window, the cumulative residual's AR(1) half-life has a median of 103 bars for PCA-2 and 32 "
    "for PCA-3 sparse. The trailing 60-bar fit behind the s-score has a median of 4.5–6.4 bars, and trades are held "
    "10 bars. An AR(1) fitted on 60 bars is biased towards fast reversion: simulating the training-window process and "
    "fitting it on 60 bars gives 8.3 bars (PCA-2) and 7.1 (PCA-3 sparse) before any screen. The speed screen passes "
    "88–92% of simulated windows and moves this only to 7.6 and 6.7. Selection at entry does the rest (3.0–3.5 bars).",
    "**None of the six experiments turns net P&L positive.** Holding positions unless the target moves more than 25% helps "
    "every model (+0.06 to +0.09 Sharpe; PCA-3 sparse reaches −0.00). Entry at 2.5 or 3.0 cuts cost but leaves 0.6–6 "
    "trades a year. The MOVE gate helps a little, the 20-bar time stop hurts PCA-3 sparse (−0.26), and the leave-one-out "
    "residual is worse everywhere. Every parameter except leave-one-out was chosen after full-sample results were seen.",
    "**Overfitting check: PCA-3 sparse's deflated Sharpe is 0.016 net with 11 trials and 0.000 with all 31 documented "
    "trials.** The gross P&L clears the bar (deflated Sharpe 1.00); the net does not. Results are not driven by "
    "auction weeks (p > 0.09) and correlate weakly with rates, MOVE and HY OAS (|ρ| < 0.1). The worst windows are the "
    "GFC for the headline (−17%) and February 2021 for PCA-3 sparse (−10%).",
]

NOT_MODELED = [
    "## What the P&L does not include", "",
    "- **Financing (repo), carry, roll-down and specialness are not modeled.** The curve is traded in cash-yield bp; "
    "the funding rate is loaded into the panel but never enters P&L. A residual position that is long high-carry "
    "tenors earns carry that is not counted, and vice versa.",
    "- **The alpha attribution will show carry as alpha.** `amp_sa/backtest/attribution.py` regresses P&L on price-move "
    "factors only, so any return that comes from collecting carry appears in the intercept.",
    "- **Futures roll** is modeled only for the credit pairs (CDX series roll, 2x cost). The Treasury curve has no roll "
    "because it is traded as cash yields.",
    "- **Costs are per package, not netted** (experiment 6 measures the gap), and the same 0.25 bp per Treasury leg is used "
    "for every tenor and every date (step 2 shows a time-varying alternative).",
    "",
]


def why(text: str) -> str:
    """Plain-language reading of the tables that follow."""
    return f"> **What this tells you:** {text}"


def md(df: pd.DataFrame, digits: int = 3) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def pct_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (out[c] * 100).round(2)
    return out


def build_md(T: dict, exp: pd.DataFrame, net: pd.DataFrame) -> str:
    k = T["s2_kpis"]
    knet = k[k["basis"] == "net"].drop(columns="basis")
    kgross = k[k["basis"] == "gross"][["model", "ann_return", "sharpe", "max_dd"]].rename(
        columns={"ann_return": "gross_return", "sharpe": "gross_sharpe", "max_dd": "gross_max_dd"})
    headline = knet.merge(kgross, on="model")[["model", "start", "ann_return", "ann_vol", "sharpe", "gross_sharpe", "sortino",
                                              "calmar", "max_dd", "longest_dd_bars", "worst_day", "worst_week", "var_99", "es_99"]]
    L = ["# Sleeve 1 portfolio diagnostics", "",
         "Out-of-sample walk-forward results only, from the cached production runs. P&L is in fractions of capital at the "
         "sleeve's 10% volatility target (additive, no compounding) unless marked as risk units (1 = one trailing daily "
         "standard deviation of the member). Dollar figures are for a notional $1m book. Models: "
         + ", ".join(f"`{a}` = {b}" for a, b in LABELS.items()) + ".", "",
         "## Findings", "", *[f"{i}. {f}" for i, f in enumerate(FINDINGS, 1)], "",
         *NOT_MODELED,
         "## Key numbers (net of 1x cost)", "",
         why("every model loses money after cost, and every model makes money before it. The four models' net Sharpe of −0.09 to −0.67 "
             "against 0.45–1.21 gross means the signal exists but is smaller than what it costs to trade. Drawdowns above "
             "100% of capital appear because P&L is added up, not compounded, over 31 years of losing; a real book would "
             "have stopped long before."), "",
         "Returns, vol, drawdowns and VaR/ES are fractions of capital (0.05 = 5%). `longest_dd_bars` is the longest spell "
         "below a previous peak.", "",
         md(headline), "",
         "### Breakeven cost", "",
         why("how cheap trading would have to be. PCA-3 sparse breaks even at 0.93x the cost table, so it is close; the "
             "headline needs costs cut by more than half (0.48x). Gross Sharpe is the ceiling: no cost cut can do better."), "",
         "Net Sharpe at multiples of the 1x cost table with positions and sleeve weights held at the 1x run. "
         "`breakeven_mult` is where mean net P&L is zero.", "",
         md(T["s2_breakeven"]), "",
         "### Flat vs time-varying cost", "",
         why("the flat 0.25 bp table is generous. Charging the wider pre-2008 spreads and a dearer 30y makes every model "
             "worse, mostly before 2008."), "",
         "Time-varying: Treasury legs 0.5 bp one way before 2008 and 0.25 bp after; the 30y at 1.5x the other tenors. "
         "The sleeve is rebuilt with re-priced cost; positions and bands are unchanged (bands were optimised at the flat table).", "",
         md(T["s2_cost_sensitivity"][["model", "gross_sharpe", "net_sharpe_flat", "net_sharpe_time_varying", "cost_pa_flat",
                                      "cost_pa_time_varying", "net_sharpe_time_varying_pre2008", "net_sharpe_time_varying_2008on"]]), "",
         "### P&L by year (net, % of capital)", "",
         why("losses are spread over many years, not one bad episode. No model has a run of positive years long enough "
             "to suggest a regime where it works after cost."), "",
         md(T["s2_yearly"].pivot(index="year", columns="model", values="net").mul(100).round(1).reset_index(), 1), "",
         "Benchmarks: the fixed-weight flies (belly +1, wings −0.5 each, in bp) traded with the PCA rules (60-bar OU "
         "s-score, training-window bands, same folds and costs). Trades: "
         + ", ".join(f"{r.benchmark} {r.trades}" for r in T["s2_benchmark_trades"].itertuples()) + ".", "",
         "## Step 1. Cost model audit", "",
         why("what you pay and when. 75–87% of cost is opening and closing trades. Keeping a position the right size "
             "(resizing, leverage changes) is the rest, and re-hedging at refits is free in practice. So cutting cost "
             "means fewer, better round trips, not tidier maintenance."), "",
         "- **How cost is charged** (`amp_sa/backtest/engine.py`): per member and bar, `|p_t − p_{t−1}| × Σ_legs |w_leg| × c_leg`, "
         "with `p = signal / trailing vol` in risk units and `c` = 0.25 bp per Treasury tenor one way (2 bp HY OAS, 0.5 bp IG "
         "OAS, 1 bp CDX HY). It is per package: each member pays for all its legs on its own position change. Members are not netted.",
         "- **Entry / exit / resize**: an entry is opening from flat or the new side of a flip, an exit is closing to flat or the "
         "old side of a flip, and a resize is the same side held through a change in `signal / vol`. Resizes are charged every "
         "bar at the same rate. The sleeve also pays `|Δ sleeve weight| × mean |p| × unit cost` when leverage changes "
         "(`amp_sa/portfolio/combine.py`).",
         "- **Does cost scale with DV01?** Not directly: the charge is bp per unit of residual. Because the position is "
         "1 / vol units, a quiet market means more DV01 per risk unit and the same bp cost is a larger share of the edge.",
         "- **Not modeled**: financing, carry, roll-down, specialness; futures roll only on the credit pairs (see above).",
         "- **Unpriced**: when leg weights change under a held position (PCA refit, Kalman β), the engine does not charge the "
         "re-hedge. Priced leg by leg it is 0.002–0.04% of capital a year: negligible.", "",
         md(pct_cols(T["s1_cost_split_by_model"], ["pct_cap_total"]).assign(
             pct_cap_per_year=lambda d: d["pct_cap_per_year"].round(3)).drop(columns="pct_cap_total")), "",
         "Per member (risk units and % of capital, with the median one-way unit cost in bp): `tables/s1_cost_split_by_member.csv`.", "",
         "Headline members: one-way and round-trip cost of one unit of the package (bp), cost by type summed over the "
         "sample (risk units), and the charged total in % of capital per year of the member's own live period.", "",
         md(member_costs(T["s1_cost_split_by_member"], "headline"), 2), "",
         why("the two spread packages (hy_ig_oas, cdx_cash_basis) cost 6-10x a curve residual per unit (2.4-3.6 bp "
             "against about 0.4 bp one way), and cdx_cash_basis also pays the CDX roll. Credit trades rarely, so the curve "
             "members still pay most of the bill."), "",
         "## Step 2. Portfolio KPIs", "",
         why("the same models gross and net side by side. Gross drawdowns of the four models are 20–37%, and three of "
             "them recover; no net drawdown recovers (`recovery_bars` blank), because cost is a steady drag that an edge this small cannot outrun."),
         "", "Gross and net, all models and the two benchmarks: `tables/s2_kpis.csv`; by year: `tables/s2_yearly.csv`. "
         "Headline numbers above.", "",
         md(k[["model", "basis", "ann_return", "ann_vol", "sharpe", "sortino", "calmar", "max_dd", "recovery_bars", "worst_week", "es_99"]]), "",
         "## Step 3. Hedge performance", "",
         why("whether the book really trades residuals. It does: with drifting loadings, level and slope leak about "
             "7% of gross DV01 for PCA-2 and the three hedged factors about 12% for PCA-3 sparse, and the residuals move a fraction of a raw yield. The exceptions are by design (PCA-2 "
             "keeps curvature) or a defect (PCA-3 sparse 2010–2013, below)."), "",
         "Factor exposure of the netted book, measured two ways. Against the fold's own loadings (the specified test), the "
         "factors a PCA model hedges are zero by construction, so that number cannot reveal a failure. Against PCA-3 "
         "loadings re-estimated every 21 bars from the prior 252 bars (causal), the exposure shows how much the factors drift "
         "away from the hedge.", "",
         md(T["s3_factor_exposure_summary"][["model", "loadings", "mean_abs_pc1_per_gross_dv01", "mean_abs_pc2_per_gross_dv01",
                                             "mean_abs_pc3_per_gross_dv01", "mean_gross_dv01_usd_per_bp", "mean_abs_net_dv01_usd_per_bp"]]), "",
         "Per-trade attribution, summed over trades, risk units. Treasury-leg changes split into leaked factor P&L "
         "(PC1–3) and the residual beyond three factors; non-Treasury legs (credit pairs, swaps) are `other_legs`.", "",
         "Fold loadings:", "", md(T["s3_trade_attribution_summary"][["model", "trades", "pc1", "pc2", "pc3", "residual", "other_legs", "gross", "cost", "net"]], 1), "",
         "Trailing loadings:", "", md(T["s3_trade_attribution_trailing_summary"][["model", "trades", "pc1", "pc2", "pc3", "residual", "other_legs", "gross", "cost", "net"]], 1), "",
         "Refit turnover. `unpriced_reweight_pa` is the leg trade forced by new weights under a held position, which the "
         "engine does not charge. `charged_on_boundary_bars_pa` is the cost the engine charged on the last bar of each fold and the first bar of the next "
         "(band state restarts each fold, so positions can jump), and `boundary_bars_vs_mean` compares those two bars with an average pair.", "",
         md(T["s3_refit_summary"], 4), "",
         "Residual package vs raw yields (bp per day, live bars):", "",
         md(pd.concat([T[f"s3_residual_vs_raw_{m}"].assign(model=m) for m in ("pca2", "pca3_sparse")])[
             ["model", "member", "resid_vol_bp", "own_leg_vol_bp", "ust_5y_vol_bp", "variance_removed_vs_own"]], 2), "",
         "In PCA-3 sparse the 2y and 3y members come out with negative variance removed. The package is mostly hedge legs: "
         "in 2010–2013 the 3y unit carried weights of +68 (2y), −121 (5y) and +69 (10y), from a nearly singular 3x3 solve "
         "that passes the `cond < 1e8` check in `sparse_hedges`. Leg weights by fold: `tables/s3_leg_weights_<model>.csv`. "
         "Daily gross and net DV01 by leg: `tables/s3_dv01_daily_<model>.csv`.", "",
         "DV01 by leg, averaged over live days: bp of capital per 1 bp move, and dollars per bp on a $1m book. Gross sums "
         "members' |exposure|; net is the book after members offset. Daily PC1–3 exposure in both units: "
         "`tables/s3_factor_exposure_daily_<model>.csv`.", "",
         md(T["s3_dv01_by_leg_summary"].query("model in ['headline', 'pca3_sparse']").assign(
             mean_gross_frac_per_bp=lambda d: d["mean_gross_frac_per_bp"] * 1e4,
             mean_abs_net_frac_per_bp=lambda d: d["mean_abs_net_frac_per_bp"] * 1e4).rename(columns={
             "mean_gross_frac_per_bp": "mean_gross_bp_of_cap_per_bp", "mean_abs_net_frac_per_bp": "mean_abs_net_bp_of_cap_per_bp"}), 2), "",
         why("a 1 bp move in one Treasury leg moves the headline by roughly 0.07–0.15 bp of capital ($700–1,500 on $1m). "
             "Net is 80–95% of gross for the headline and 96–100% for PCA-3 sparse, so members seldom offset each other, which is why netting saves little (experiment 6)."), "",
         "## Step 4. Cost attribution", "",
         why("why low-volatility trades lose. Cost is fixed in bp, but the position is sized to 1 / vol, so in a quiet "
             "market the same risk needs more DV01 and the round trip eats more of the edge. In the low-vol third a round "
             "trip costs 1.3–2.3x a day's move, and those trades lose for every model."), "",
         "Cost by leg (% of capital a year): the engine charge allocated to legs by |w| × c, and the leg-level cost of the netted book.", "",
         md(pd.concat([T[f"s4_cost_by_leg_{m}"].assign(model=m) for m in ("headline", "pca3_sparse")])[
             ["model", "leg", "engine_allocated_pa", "netted_leg_pa"]], 4), "",
         "Cost per trade against the residual's volatility at entry (risk units; vol in bp a day, read off the position as "
         "`|signal| / |p|`). Terciles are over each model's trades.", "",
         md(pd.concat([T[f"s4_cost_vs_vol_{m}"].assign(model=m) for m in LABELS])[
             ["model", "vol_tercile", "trades", "median_entry_vol_bp", "mean_round_trip_over_vol", "mean_cost", "mean_gross", "mean_net"]], 2), "",
         "## Step 5. Signal diagnostics", "",
         why("the signal points the right way (positive IC at every horizon), but the OU model behind it promises bigger, "
             "faster reversions than arrive. That is why a cost filter built on the OU forecast barely filters."), "",
         "Information coefficient: Spearman correlation of −z with the forward h-bar change of the member (bp). Positive means "
         "the signal points the right way.", "",
         md(pd.concat([T[f"s5_ic_{m}"].assign(model=m) for m in LABELS])[["model", "horizon", "scope", "n", "ic", "p_value"]]), "",
         "Win rate and mean P&L by |z| at entry (risk units). Entry bands are optimised per fold from 1.0 to 2.5, so many "
         "trades open below 2.", "",
         md(pd.concat([T[f"s5_z_buckets_{m}"].assign(model=m) for m in LABELS])[["model", "entry_abs_z", "trades", "win_rate_net", "mean_gross", "mean_cost", "mean_net"]], 2), "",
         "OU half-life: AR(1) on each training window's cumulative residual (that fold's weights), the median trailing "
         "60-bar half-life the signal used in the test block, and the actual median holding. Per fold: `tables/s5_ou_by_fold_<model>.csv`. "
         "For pairwise, the half-life is the adaptive spread's from the backtest's fold diagnostics.", "",
         md(T["s5_ou_summary"], 1), "",
         "Why the 60-bar half-life is so short. Each fold's training-window AR(1) is taken as the true process, "
         "simulated 500 times and fitted on 60 bars exactly as `pca.rolling_ou` does, before and after its screen "
         "(0 < b < 1, half-life under 30 bars). The training b is itself biased slightly fast, so this understates the bias. "
         "`kendall_implied` is the textbook first-order correction b − (1 + 3b) / 60.", "",
         md(T["s5_ou_bias"], 2), "",
         why("most of the gap from 103 to about 6 bars is the estimator, not the market and not the screen. A 60-bar "
             "AR(1) fit on a slowly reverting series reads it as fast: 8.3 bars before the screen, 7.6 after it. The "
             "screen passes about 9 in 10 windows, so it selects little. Entries happen when the fit looks fastest "
             "(3.0–3.5 bars), yet trades last 10. Treat the 60-bar half-life and σ_eq as noisy, optimistic estimates."), "",
         "Fix A (cost-aware entry) checked on the entries that happened: enter only if (|z| − exit band) × σ_eq > k × "
         "round trip. All inputs are known on the entry bar. `realised_move_bp` is what the residual then did, signed so "
         "that reversion is positive. `k_that_halves_entries` is the median of forecast / round trip. Per entry: "
         "`tables/s5_cost_aware_entry_<model>.csv`.", "",
         md(T["s5_cost_aware_entry_summary"], 3), "",
         why("at k = 1 the rule keeps 92% of PCA-2 entries and 86% of PCA-3 sparse's, so it would change little. For "
             "PCA-2 the OU forecast (1.67 bp median) is nearly 4x the mean realised move (0.45 bp). The entries it would "
             "drop do lose more, so the idea is sound, but k must be fitted per fold and will be well above 1."), "",
         "Residual paths of the ten worst trades per model: `tables/s5_worst_paths_<model>.csv` and the dashboard.", "",
         "## Step 6. Regimes and robustness", "",
         why("the losses are not one regime's fault. Every Fed regime loses net, auction weeks are no different, and the "
             "P&L barely correlates with rates, MOVE or credit. Stress windows matter at the margin: the GFC hurt the "
             "headline and February 2021 hurt PCA-3 sparse. The curve sat out March–December 2022."), "",
         "Stress windows (net, gross and cost in fractions of capital):", "", md(T["s6_windows"]), "",
         "Fed regime: hiking / cutting / on hold from the change in the fed funds target (DFEDTAR to 2008-12-15, then the "
         "upper bound DFEDTARU) over the trailing 126 bars.", "", md(T["s6_fed_regime"]), "",
         "Calendar eras, as the brief asked (1995–2004 vs 2009+). These are data eras: the 3y data starts in 2005 and the "
         "7y in 2009, but neither trades until it has a full 5-year training window (next tables).", "", md(T["s6_era"]), "",
         "When each tenor's data starts and when its member first trades:", "",
         md(T["s6_tenor_start"].query("model == 'pca2'")), "",
         "The same split by the tenors the PCA actually used:", "", md(T["s6_era_traded"]), "",
         why("the 3y trades from 2010 and the 7y from 2014, so the six-tenor curve covers only the last 12 years. That "
             "era is the worst for the headline and PCA-2 (net Sharpe −0.88 and −1.10), so adding tenors did not "
             "help them. PCA-3 sparse's weak 2010–2014 era is the construction defect in Step 3."), "",
         "5y auction weeks (TreasuryDirect auction dates, reopenings included) vs other weeks:", "", md(T["s6_auction_weeks"]), "",
         "Correlation of daily net P&L with market moves:", "",
         md(T["s6_market_corr"].pivot(index="model", columns="factor", values="corr").reset_index()), "",
         "Headline: curve members vs credit members, daily net contribution. Stress = MOVE at or above its trailing "
         "5-year 80th percentile.", "", md(T["s6_curve_vs_credit_headline"]), "",
         "Headline: curve vs credit performance. Daily capital contribution at the sleeve weight, fractions of capital a "
         "year; the sleeve-rescale cost is not split and is left out. Credit-live days start 2005-07-07 (hy_ig_oas); the "
         "CDX pairs start in 2016-17. `var_share_credit_live` is each part's share of book variance on those days.", "",
         md(T["s6_curve_credit_parts_headline"]), "",
         "By member, each over its own live days. `in_market_share` is the share of live days with a position.", "",
         md(T["s6_curve_credit_members_headline"]), "",
         "Net by stress window (fractions of capital):", "", md(T["s6_curve_credit_windows_headline"]), "",
         "PCA-2 curve members in 2022 (all one fold, starting 2021-12-23): the entry band, the largest |z|, the share "
         "of live bars where the 60-bar screen blanked the signal, and days holding a position.", "",
         md(T["s6_curve_2022"], 3), "",
         "Credit is a small diversifier: its gross Sharpe is lower than the curve's, each pair holds a position on 8-13% "
         "of its live days, and cdx_cash_basis is the only member of the headline with positive net P&L (48 trades since "
         "2017, too few to separate from luck). `cost_over_gross` is blank where gross P&L is not positive. Credit and "
         "curve lost together in the GFC but not in 2020 or 2022. The curve is flat from March to December 2022: the "
         "fold starting 2021-12-23 set entry at 2.5 for every tenor but the 5y (2.0), no residual's |z| passed 2.11 from March "
         "(the 5y peaked at 1.99), and the 60-bar speed screen blanked 17–49% of bars. Only the 5y traded in 2022, for 13 "
         "days in January–February.", "",
         "## Step 7. Overfitting", "",
         why("how much of the best result is luck from trying many things. PCA-3 sparse is the best of 31 documented "
             "trials. Its gross P&L survives that adjustment; its net P&L does not. Its six members behave like 2.3 "
             "independent bets, so its sample is thinner than the member count suggests."), "",
         "Variants tried before and around PCA-3 sparse. The repo has no git history, so the count comes from the tables "
         "and the appendix's own list of design revisions. Per-fold band and Kalman choices are fitted in-sample within "
         "each fold and are not counted as trials.", "",
         md(T["s7_trial_count"]), "",
         md(T["s7_deflated_sharpe"]), "",
         "The deflated Sharpe is the probability that the true Sharpe exceeds zero after selecting the best of N trials "
         "(Bailey and López de Prado), using the dispersion of the trials' Sharpe ratios. The 4 revisions have no saved "
         "Sharpe, so the broad set uses the dispersion of the 27 that do.", "",
         "Effective number of independent bets, from the correlation of members' daily P&L (reproduces trades.md):", "",
         md(T["s7_effective_bets"], 2), "",
         "## Step 8. Experiments", "",
         why("what simple rule changes do out of sample. Trading less (higher entry band, rebalancing only on big "
             "changes, a MOVE gate) cuts cost faster than it cuts gross, so net improves, but nothing turns positive. "
             "Most parameters were picked after seeing results, so treat the improvements as upper bounds."), "",
         "Walk-forward, each against the unchanged baseline, built with `analysis/experiments/pipeline.py`. With no hook set "
         "it reproduces the production runs bit for bit (tested). Bands are re-optimised on each training window under the "
         "experiment's rule. Headline = the experiment's PCA-2 curve plus the unchanged credit pairs. Returns in fractions of "
         "capital a year; `trades` merges identical trades across members, as trades.md does.", "",
         md(exp.drop(columns="provenance")), "",
         "Where each parameter came from. Anything marked *after results* was proposed after full-sample results were "
         "seen, so even a walk-forward improvement is partly an in-sample choice:", "",
         md(exp[["experiment", "provenance"]].drop_duplicates()), "",
         "LOO residual: each tenor is regressed (training window) on the k factors of the other tenors; it hedges with every "
         "other tenor, so PCA-3 sparse's comparison is against a full, not sparse, construction.", "",
         "### Experiment 6: leg netting", "",
         why("charging cost on the whole book's net trade instead of member by member saves 0.5–5.5%. Members rarely "
             "trade opposite sides of the same tenor on the same day."), "",
         "Positions unchanged. Each tenor's exposure is netted across members daily and cost is charged only on the netted "
         "leg change; CDX roll charges are kept. `cost_leg_unnetted_pa` is the same leg-level pricing without netting "
         "(the engine-equivalent); the engine itself also charges sleeve rescaling slightly differently.", "",
         md(net), "",
         "## Changes made to existing code", "",
         "- `src/amp_sa/sleeves/pca.py`: added `# [diag-logging]` lines that save per-bar OU state (z, half-life, kappa, "
         "σ_eq), fold loadings and test-window factor scores to `extras`. After rerunning with `--force`, all 360 P&L, cost, position, "
         "signal, turnover and unit-cost series of the 24 PCA runs were bit-identical to the previous cache "
         "(`analysis/portfolio_diagnostics/check_logging_identical.py`), as were `reports/` and `signals/signals.parquet`.",
         "- External data cached (gitignored): `data/raw/fred/DFEDTAR.csv`, `DFEDTARU.csv`, `DFEDTARL.csv`; "
         "`data/raw/treasury/auctions_Note.json`, `auctions_Bond.json`, `coupon_auctions.csv`.",
         "- Note: the repo's `.gitignore` ignores `*.csv`, so `tables/` will not be committed unless that rule is changed.", ""]
    return "\n".join(L)


def member_costs(by_member: pd.DataFrame, model: str) -> pd.DataFrame:
    m = by_member[by_member["model"] == model]
    return pd.DataFrame({"member": m["member"], "unit_cost_bp": m["unit_cost_bp_median"],
                         "round_trip_bp": m["round_trip_bp_median"], "entry_risk": m["entry_risk"],
                         "exit_risk": m["exit_risk"], "resize_risk": m["resize_risk"], "roll_risk": m["roll_risk"],
                         "charged_pct_cap_pa": m["charged_pct_cap"] / m["years_live"]})


def main() -> None:
    T = pickle.load(open(OUT_DIR / "tables.pkl", "rb"))
    exp = pd.read_csv(EXP_DIR / "results.csv")
    net = pd.read_csv(EXP_DIR / "netting.csv")
    (OUT_DIR / "report.md").write_text(build_md(T, exp, net), encoding="utf-8")
    exp.to_csv(OUT_DIR / "tables" / "s8_experiments.csv", index=False)
    net.to_csv(OUT_DIR / "tables" / "s8_netting.csv", index=False)

    data = json.loads((OUT_DIR / "dashboard_data.json").read_text(encoding="utf-8"))
    tr = T["s3_trade_attribution_trailing_summary"]
    data["attr_trailing"] = json.loads(tr.round(3).to_json(orient="records"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (OUT_DIR / "dashboard_template.html").read_text(encoding="utf-8").replace("/*DATA*/", payload)
    (OUT_DIR / "dashboard.html").write_text(html, encoding="utf-8")
    print("wrote report.md and dashboard.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
