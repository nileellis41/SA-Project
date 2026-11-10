"""Steps 1-7 of the portfolio diagnostics: every table as CSV, chart data for the dashboard.

    python -m analysis.portfolio_diagnostics.run_all

Reads the cached walk-forward runs; run analysis.experiments.run_experiments for step 8,
then analysis.portfolio_diagnostics.report to write report.md and dashboard.html.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import metrics
from analysis.portfolio_diagnostics import book as bk
from analysis.portfolio_diagnostics import cost_audit, external, hedge, kpis
from analysis.portfolio_diagnostics import signals_regimes as sr
from amp_sa.run_lessons import LABELS as RL_LABELS
from analysis.portfolio_diagnostics.common import CAPITAL, OUT_DIR, TABLE_DIR, load_models

ANN = config.ANN
HEDGED_K = {"headline": 2, "pca3_sparse": 3, "pca2": 2, "pairwise_nn": None}
LABELS = {"headline": "Headline (PCA-2 + credit)", "pca3_sparse": "PCA-3 sparse", "pca2": "PCA-2 full",
          "pairwise_nn": "Pairwise adaptive NN"}
# Step 7 trial counts, from the model table, cost-filter table, curve rules and appendix.md's own list of revisions.
TRIALS = {"model table rows (reports/tables/sleeve1_models.csv)": 11,
          "cost-filter variants not already in the model table (cost_filter_effect.csv, 12 rows - 4 duplicates)": 8,
          "2s10s curve rules other than PCA (curve_full.csv, 10 rows - 2 PCA rows)": 8,
          "design revisions listed in appendix.md (adaptive x2, cointegration screen, vol estimate)": 4,
          "cost round 1 (results/cost_round1): freeze_size, freeze_size+vol_gate x headline, PCA-3 sparse": 4}


def weekly_last(s: pd.Series) -> pd.Series:
    return s.resample("W-FRI").last()


def to_records(df: pd.DataFrame, digits: int = 6) -> list:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(str)
    out = out.replace([np.inf, -np.inf], np.nan).round(digits)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def series_json(df: pd.DataFrame, digits: int = 6) -> dict:
    df = df.replace([np.inf, -np.inf], np.nan).round(digits)
    return {"dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "series": {c: [None if pd.isna(v) else float(v) for v in df[c]] for c in df.columns}}


def main() -> None:
    levels, rolls, models = load_models()
    pca_runs = s1.load_pca_runs(levels)
    loads = hedge.fold_loadings(pca_runs[3])
    loads_tr = hedge.trailing_loadings(levels)  # causal drift check, re-estimated every 21 bars
    books = {k: bk.build_book(m, levels, rolls) for k, m in models.items()}
    ports = {k: m.port for k, m in models.items()}
    T, chart = {}, {"labels": LABELS, "models": {}}

    # ---- step 1
    by_model, by_member, daily_cost = cost_audit.audit(models)
    T["s1_cost_split_by_model"], T["s1_cost_split_by_member"] = by_model, by_member

    # ---- step 2
    bench = kpis.benchmark_ports(levels)
    allports = {**ports, **{k: p for k, (p, _) in bench.items()}}
    T["s2_kpis"] = kpis.kpi_table(allports)
    T["s2_yearly"] = kpis.yearly(allports)
    T["s2_breakeven"] = kpis.breakeven(allports)
    T["s2_cost_sensitivity"] = kpis.cost_sensitivity(models)
    curve = kpis.breakeven_curve(allports)
    bench_trades = {k: len(hedge.trade_ids(r.results["pca"].position.where(r.live, 0.0))[0].dropna().unique())
                    for k, (_, r) in bench.items()}
    T["s2_benchmark_trades"] = pd.DataFrame([{"benchmark": k, "trades": v} for k, v in bench_trades.items()])

    # ---- step 3
    trades_all, attr_rows, expo_rows, attr_tr_rows, dv01_leg_rows = {}, [], [], [], []
    for k, b in books.items():
        live = models[k].port.net.notna()
        daily_ex = {}
        for basis, ld in (("fold loadings (spec; zero by construction for hedged PCA factors)", loads),
                          ("trailing 1y loadings (causal drift check)", loads_tr)):
            ex = hedge.factor_exposure(b, ld)[live]
            daily_ex[basis.split()[0]] = ex
            g = ex["gross_dv01"].replace(0, np.nan)
            expo_rows.append({"model": k, "loadings": basis,
                              **{f"mean_abs_{p}_per_gross_dv01": (ex[p].abs() / g).mean() for p in hedge.PCS},
                              **{f"mean_abs_{p}_usd_per_bp": CAPITAL * ex[p].abs().mean() for p in hedge.PCS},
                              "mean_gross_dv01_usd_per_bp": CAPITAL * ex["gross_dv01"].mean(),
                              "mean_abs_net_dv01_usd_per_bp": CAPITAL * ex["net_dv01"].abs().mean()})
        ex_tr = ex  # the trailing version feeds the chart
        # Daily PC1-3 exposure and DV01 of the netted book: fraction of capital per bp, and $ per bp at $1m.
        exd = pd.concat({f"{bs}_{c}": daily_ex[bs][c] for bs in daily_ex for c in daily_ex[bs].columns}, axis=1)
        T[f"s3_factor_exposure_daily_{k}"] = pd.concat([exd, (exd * CAPITAL).add_suffix("_usd")], axis=1).rename_axis("date").reset_index()
        t_tr = hedge.trade_attribution(b, hedge.bar_attribution(b, loads_tr, levels), HEDGED_K[k])
        tot_tr = t_tr[[*hedge.PCS, "residual", "other_legs", "cost", "gross", "net", "leak_hedged_factors"]].sum()
        attr_tr_rows.append({"model": k, "trades": len(t_tr), **tot_tr.to_dict()})
        T[f"s3_trades_trailing_attr_{k}"] = t_tr
        bars = hedge.bar_attribution(b, loads, levels)
        t = hedge.trade_attribution(b, bars, HEDGED_K[k])
        t = sr.entry_context(t, b, levels)
        trades_all[k] = t
        tot = t[[*hedge.PCS, "residual", "other_legs", "cost", "gross", "net", "leak_hedged_factors"]].sum()
        attr_rows.append({"model": k, "trades": len(t), **tot.to_dict(),
                          **{f"mean_{c}": t[c].mean() for c in ("residual", "leak_hedged_factors", "cost", "net")}})
        dv_net, dv_gross = b.dv01_net()[live], b.dv01_gross()[live]
        dv01_leg_rows += [{"model": k, "leg": c, "mean_gross_frac_per_bp": dv_gross[c].mean(),
                           "mean_abs_net_frac_per_bp": dv_net[c].abs().mean(),
                           "mean_gross_usd_per_bp": CAPITAL * dv_gross[c].mean(),
                           "mean_abs_net_usd_per_bp": CAPITAL * dv_net[c].abs().mean(),
                           "net_over_gross": dv_net[c].abs().mean() / dv_gross[c].mean() if dv_gross[c].mean() > 0 else np.nan}
                          for c in dv_net.columns]
        treas = [c for c in dv_net.columns if c in bk.TREASURY]
        lw = hedge.leg_weights_by_fold(b, hedge.fold_starts(levels))
        T[f"s3_leg_weights_{k}"] = lw
        T[f"s3_refit_turnover_{k}"] = hedge.refit_turnover(b, levels, bk.flat_cost(b.series(), levels.index))
        T[f"s3_residual_vs_raw_{k}"] = hedge.residual_vs_raw(b, levels)
        T[f"s3_dv01_daily_{k}"] = pd.concat({"net": dv_net * CAPITAL, "gross": dv_gross * CAPITAL}, axis=1).rename_axis("date").reset_index()
        T[f"s3_trades_{k}"] = t
        chart["models"][k] = {
            "exposure": series_json((ex_tr[list(hedge.PCS) + ["gross_dv01", "net_dv01"]] * CAPITAL).resample("W-FRI").mean(), 2),
            "dv01_net": series_json((dv_net[treas] * CAPITAL).resample("W-FRI").mean(), 2),
            "dv01_gross": series_json((dv_gross[treas] * CAPITAL).resample("W-FRI").mean(), 2),
            "rolling_vol": series_json(weekly_last(hedge.rolling_vol(b, levels)["residual_avg_member"]).to_frame()
                                       .join(weekly_last(hedge.rolling_vol(b, levels)["raw_ust_5y"])), 3),
            "leg_weights": to_records(lw.assign(date=lw["date"].dt.strftime("%Y-%m-%d")) if len(lw) else lw, 4),
        }
    T["s3_factor_exposure_summary"] = pd.DataFrame(expo_rows)
    T["s3_trade_attribution_summary"] = pd.DataFrame(attr_rows)
    T["s3_trade_attribution_trailing_summary"] = pd.DataFrame(attr_tr_rows)
    refit = pd.DataFrame([{"model": k, "unpriced_reweight_pa": T[f"s3_refit_turnover_{k}"]["unpriced_reweight"].sum() / (models[k].port.net.notna().sum() / ANN),
                           "charged_on_boundary_bars_pa": T[f"s3_refit_turnover_{k}"]["charged_boundary_bars"].sum() / (models[k].port.net.notna().sum() / ANN),
                           "boundary_bars_vs_mean": T[f"s3_refit_turnover_{k}"]["charged_boundary_bars"].mean() / T[f"s3_refit_turnover_{k}"]["mean_two_bars"].mean()}
                          for k in models])
    T["s3_refit_summary"] = refit
    T["s3_dv01_by_leg_summary"] = pd.DataFrame(dv01_leg_rows)

    # ---- step 4
    for k, b in books.items():
        live = models[k].port.net.notna()
        eng = sr.cost_by_leg(b)[live]
        net_leg = bk.leg_cost(b, netted=True)[live]
        T[f"s4_cost_by_leg_{k}"] = pd.DataFrame({"engine_allocated_pa": eng.mean() * ANN,
                                                 "netted_leg_pa": net_leg.mean() * ANN}).fillna(0).rename_axis("leg").reset_index()
        d = daily_cost[daily_cost["model"] == k].set_index("date")
        cum = d[[*cost_audit.TYPES, "sleeve_rescale"]].fillna(0).cumsum()
        chart["models"][k]["cum_cost"] = series_json(weekly_last_frame(cum), 4)
        chart["models"][k]["cost_by_leg"] = to_records(T[f"s4_cost_by_leg_{k}"], 6)
        t = trades_all[k]
        t["vol_tercile"] = pd.qcut(t["entry_vol_bp"], 3, labels=["low vol", "mid vol", "high vol"])
        g = t.groupby("vol_tercile", observed=True)
        T[f"s4_cost_vs_vol_{k}"] = pd.DataFrame({"trades": g.size(), "median_entry_vol_bp": g["entry_vol_bp"].median(),
                                                 "mean_round_trip_over_vol": g["round_trip_over_vol"].mean(),
                                                 "mean_cost": g["cost"].mean(), "mean_gross": g["gross"].mean(),
                                                 "mean_net": g["net"].mean()}).reset_index()

    # ---- step 5
    for k, b in books.items():
        t = trades_all[k]
        T[f"s5_ic_{k}"] = sr.information_coefficient(b, t)
        T[f"s5_z_buckets_{k}"] = sr.z_buckets(t)
        ou = sr.ou_by_fold(b, levels, t)
        T[f"s5_ou_by_fold_{k}"] = ou
        T[f"s5_worst_paths_{k}"] = sr.worst_paths(b, t)
        chart["models"][k]["trades"] = to_records(t[["member", "entry", "exit", "bars", "entry_z", "gross", "cost", "net",
                                                     "residual", "leak_hedged_factors", "entry_vol_bp"]], 4)
        chart["models"][k]["worst_paths"] = to_records(T[f"s5_worst_paths_{k}"], 4)
    T["s5_ou_summary"] = pd.DataFrame([{"model": k, "members": T[f"s5_ou_by_fold_{k}"]["member"].nunique() if len(T[f"s5_ou_by_fold_{k}"]) else 0,
                                        "median_train_ar1_half_life": T[f"s5_ou_by_fold_{k}"]["half_life"].median() if len(T[f"s5_ou_by_fold_{k}"]) else np.nan,
                                        "share_folds_not_mean_reverting": T[f"s5_ou_by_fold_{k}"]["half_life"].isna().mean() if len(T[f"s5_ou_by_fold_{k}"]) else np.nan,
                                        "median_rolling60_half_life": T[f"s5_ou_by_fold_{k}"]["rolling60_half_life_test_median"].median() if len(T[f"s5_ou_by_fold_{k}"]) else np.nan,
                                        "median_holding_bars": trades_all[k]["bars"].median()} for k in models])
    # Why the 60-bar half-life is short (small-sample AR(1) bias vs the screen), and fix A (cost-aware entry) at k = 1.
    T["s5_ou_bias"] = pd.DataFrame([{"model": k, **sr.ou_bias(books[k], T[f"s5_ou_by_fold_{k}"], trades_all[k])}
                                    for k in ("pca2", "pca3_sparse")])
    ca_rows = []
    for k in ("pca2", "pca3_sparse"):
        T[f"s5_cost_aware_entry_{k}"] = sr.cost_aware_entry(books[k], trades_all[k])
        ca_rows.append({"model": k, **sr.cost_aware_summary(T[f"s5_cost_aware_entry_{k}"])})
    T["s5_cost_aware_entry_summary"] = pd.DataFrame(ca_rows)
    # Pairwise members: the adaptive spread's OU half-life per fold, from the backtest's own fold diagnostics.
    pw = pd.concat([r.diagnostics for r, _ in models["pairwise_nn"].items])
    T["s5_ou_by_fold_pairwise_nn"] = pw[["spread", "fold", "test_start", "half_life_adaptive", "coint_pass", "disabled_share"]]
    T["s5_ou_summary"].loc[T["s5_ou_summary"]["model"] == "pairwise_nn", ["members", "median_train_ar1_half_life"]] = (
        pw["spread"].nunique(), pw["half_life_adaptive"].replace(np.inf, np.nan).median())

    # ---- step 6
    T["s6_windows"] = sr.window_table(allports)
    T["s6_fed_regime"] = sr.by_label(ports, external.fed_regime, "fed_regime")
    T["s6_era"] = sr.by_label(ports, sr.era, "era")
    lv = books["pca2"].live
    T["s6_tenor_start"] = pd.DataFrame([{"model": k, "member": m, "data_start": levels[hedge.own_leg(m)].first_valid_index().date(),
                                         "first_traded": books[k].live[m].idxmax().date()}
                                        for k in ("pca2", "pca3_sparse") for m in books[k].members])
    T["s6_era_traded"] = sr.by_label(ports, lambda idx: sr.era_traded(idx, lv["pca2_ust_3y"].idxmax(), lv["pca2_ust_7y"].idxmax()), "era")
    T["s6_curve_2022"] = sr.curve_idle(books["pca2"], {"Jan-Feb 2022": ("2022-01-01", "2022-02-28"),
                                                       "Mar-Dec 2022": ("2022-03-01", "2022-12-31")})
    T["s6_auction_weeks"] = sr.auction_test(ports)
    T["s6_market_corr"] = sr.market_correlations(ports, levels)
    T["s6_curve_vs_credit_headline"] = sr.sleeve_split_corr(books["headline"], levels)
    for name, df in sr.curve_credit_split(books["headline"], ports["headline"].net).items():
        T[f"s6_curve_credit_{name}_headline"] = df

    # ---- step 7
    model_sr = pd.read_csv(s1.REPORT_DIR / "tables" / "sleeve1_models.csv")["sharpe"]
    cf_sr = pd.read_csv(s1.REPORT_DIR / "tables" / "cost_filter_effect.csv").query("filter != 'no filter'")["sharpe_net"]
    cv = pd.read_csv(s1.REPORT_DIR / "tables" / "curve_full.csv")
    cv_sr = cv[~cv["strategy"].str.contains("PCA")]["sharpe"]
    broad_sr = pd.concat([model_sr, cf_sr, cv_sr])
    p3 = ports["pca3_sparse"]
    live = p3.net.notna()
    rows = []
    for name, n, sd in (("narrow: model table", 11, model_sr.std(ddof=1)),
                        ("broad: all documented trials", sum(TRIALS.values()), broad_sr.std(ddof=1))):
        rows.append({"trial_set": name, "n_trials": n, "trial_sharpe_sd": sd,
                     "pca3_sparse_sharpe_net": metrics.sharpe(p3.net[live]),
                     "dsr_net": sr.deflated(p3.net[live], n, sd),
                     "pca3_sparse_sharpe_gross": metrics.sharpe(p3.gross[live]),
                     "dsr_gross": sr.deflated(p3.gross[live], n, sd)})
    T["s7_deflated_sharpe"] = pd.DataFrame(rows)
    T["s7_trial_count"] = pd.DataFrame([{"source": k, "trials": v} for k, v in TRIALS.items()])
    prior = pd.read_csv(s1.REPORT_DIR / "tables" / "lessons_breadth.csv")
    T["s7_effective_bets"] = pd.DataFrame([{"model": k, **{f"{kk}_gross": vv for kk, vv in sr.effective_bets(b, "gross").items()},
                                            "effective_bets_net": sr.effective_bets(b, "net")["effective_bets"]}
                                           for k, b in books.items()])
    T["s7_effective_bets"]["trades_md_effective_bets"] = T["s7_effective_bets"]["model"].map(RL_LABELS).map(
        prior.set_index("model")["effective_bets"])

    # ---- portfolio-level chart data
    for k, m in models.items():
        b = books[k]
        live = m.port.net.notna()
        held = b.pos.shift(1).fillna(0.0)
        cost = pd.DataFrame({r.name: r.results[v].cost.where(b.live[r.name], 0.0) for r, v in m.items})
        contrib = ((held * b.change) - cost).mul(b.w_member, axis=0)[live]
        contrib["sleeve_rescale"] = m.port.net[live] - contrib.sum(axis=1)
        chart["models"][k]["contrib_cum"] = series_json(weekly_last_frame(contrib.fillna(0.0).cumsum()), 5)
        yr = contrib.groupby(contrib.index.year).sum()
        chart["models"][k]["year_member"] = {"years": [int(y) for y in yr.index], "members": list(yr.columns),
                                             "values": [[None if pd.isna(v) else round(float(v), 5) for v in row] for row in yr.to_numpy()]}
        T[f"s2_year_member_{k}"] = yr.rename_axis("year").reset_index()
    net = pd.DataFrame({k: p.net for k, p in allports.items()})
    eq = net.fillna(0).cumsum().where(net.notna())
    chart["underwater"] = series_json(weekly_last_frame(eq - eq.cummax()), 5)
    chart["cum_net"] = series_json(weekly_last_frame(eq), 5)
    roll = net.rolling(ANN, min_periods=ANN // 2)
    chart["rolling_sharpe"] = series_json(weekly_last_frame(roll.mean() / roll.std() * np.sqrt(ANN)), 3)
    chart["breakeven"] = to_records(curve, 4)
    chart["kpis"] = to_records(T["s2_kpis"], 5)
    chart["breakeven_table"] = to_records(T["s2_breakeven"], 4)
    chart["attr_summary"] = to_records(T["s3_trade_attribution_summary"], 3)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in T.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (OUT_DIR / "dashboard_data.json").write_text(json.dumps(chart, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    with open(OUT_DIR / "tables.pkl", "wb") as f:
        pickle.dump(T, f)
    print(f"{len(T)} tables written to {TABLE_DIR}")


def weekly_last_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-FRI").last().ffill()  # weeks with no live bar keep the last value


if __name__ == "__main__":
    main()
