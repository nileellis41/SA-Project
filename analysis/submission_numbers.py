"""Numbers for submission/deck_numbers.md that no existing table logs. Read-only: no model is refitted.

    uv run python -m analysis.submission_numbers

Writes submission/tables/*.csv. Everything here is marked [RECOMPUTED] in deck_numbers.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amp_sa import run
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest.splits import folds_for
from amp_sa.run_lessons import model_trades, trades
from amp_sa.sleeves import pca
from analysis.experiments.pipeline import run_members
from analysis.portfolio_diagnostics import book as bk
from analysis.portfolio_diagnostics.common import CAPITAL, load_models
from analysis.portfolio_diagnostics.kpis import BENCHMARKS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "tables"
DIAG = ROOT / "analysis" / "portfolio_diagnostics" / "tables"
PCA_FOLD = pd.Timestamp("2022-12-12")  # test window 2022-12-12 onwards; trains on 2017-11 to 2022-11
SVB = ("2023-02-15", "2023-04-28")


def pca_fold(levels, pca_runs) -> dict[str, pd.DataFrame]:
    """Covariance, eigenvalues, loadings and W for the fold whose test window starts PCA_FOLD."""
    tenors = [t for t in pca.TENORS if t in levels.columns]
    base = [t for t in pca.BASE_TENORS if t in tenors]
    fold = next(f for f in folds_for(levels[base]) if levels.index[f.test_start] == PCA_FOLD)
    d = levels[tenors].diff()
    w = slice(fold.train_start, fold.test_end)
    n_train = fold.train_end - fold.train_start
    avail = [t for t in tenors if levels[t].iloc[w].notna().all()]
    train = d[avail].iloc[w].iloc[1:n_train]  # exactly what run_curve passes to factor_loadings
    cov = train.cov()
    vals, vecs = np.linalg.eigh(cov.to_numpy())
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    logged = {k: pd.DataFrame(pca_runs[k][0].extras["fold_loadings"]).query("test_start == @PCA_FOLD").set_index("tenor")
              for k in (2, 3)}
    L3 = logged[3][["pc1", "pc2", "pc3"]].reindex(avail)
    # sign of each recomputed eigenvector aligned to the logged one, then check they agree
    for j in range(3):
        if np.dot(vecs[:, j], L3.iloc[:, j]) < 0:
            vecs[:, j] *= -1
    gap = float(np.abs(vecs[:, :3] - L3.to_numpy()).max())
    eig = pd.DataFrame({"pc": [f"pc{j + 1}" for j in range(len(vals))], "eigenvalue_bp2": vals,
                        "share": vals / vals.sum(), "cum_share": np.cumsum(vals) / vals.sum()})
    L2 = logged[2][["pc1", "pc2"]].reindex(avail).to_numpy()
    W2 = np.eye(len(avail)) - L2 @ L2.T
    W3 = np.eye(len(avail)) - L3.to_numpy() @ L3.to_numpy().T
    wdiag = pd.DataFrame({"tenor": avail, "W_diag_pca2": np.diag(W2), "W_diag_pca3": np.diag(W3)})
    wdiag.loc[len(wdiag)] = ["trace", np.trace(W2), np.trace(W3)]
    legs = {}
    for k, label in ((2, "pca2_full"), ("3s", "pca3_sparse")):
        r = next(r for r in pca_runs[k] if r.name.endswith("ust_5y"))
        legs[label] = r.extras["legs_adaptive"].loc[PCA_FOLD, avail]
    pkg = pd.DataFrame(legs).rename_axis("leg").reset_index()
    pkg["pca2_full_from_W_row"] = W2[avail.index("ust_5y")]
    load = L3.rename(columns={"pc1": "level", "pc2": "slope", "pc3": "curvature"}).rename_axis("tenor").reset_index()
    meta = pd.DataFrame([{"test_start": PCA_FOLD.date(), "train_start": levels.index[fold.train_start].date(),
                          "train_end": levels.index[fold.train_end - 1].date(), "train_bars": len(train),
                          "tenors": " ".join(avail), "max_abs_gap_recomputed_vs_logged_loadings": gap}])
    return {"b_pca_fold": meta, "b_covariance_bp2": cov.rename_axis("tenor").reset_index(), "b_eigen": eig,
            "b_loadings": load, "b_W_diag": wdiag, "b_5y_package": pkg}


def ou_by_tenor(pca_runs) -> pd.DataFrame:
    rows = []
    for k, label in ((2, "PCA-2 full (= headline curve members)"), ("3s", "PCA-3 sparse")):
        for r in pca_runs[k]:
            live = r.live.reindex(r.extras["ou_half_life"].index).fillna(False).astype(bool)
            row = {"model": label, "member": r.name}
            for key in ("half_life", "kappa", "sigma_eq"):
                s = r.extras[f"ou_{key}"][live].dropna()
                row |= {f"{key}_median": s.median(), f"{key}_q25": s.quantile(0.25), f"{key}_q75": s.quantile(0.75)}
            row["screen_pass_share"] = float(r.extras["ou_z"][live].notna().mean())
            rows.append(row)
    return pd.DataFrame(rows)


def bands(pca_runs, rv_runs) -> pd.DataFrame:
    rows = []
    for k, label in ((2, "PCA-2 full / headline"), ("3s", "PCA-3 sparse")):
        for r in pca_runs[k]:
            d = r.diagnostics
            rows.append({"model": label, "member": r.name, "folds": len(d), "entry_median": d["entry"].median(),
                         "entry_mode": d["entry"].mode().iloc[0], "entry_share_at_grid_top_2.5": (d["entry"] == 2.5).mean(),
                         "exit_median": d["exit"].median(), "exit_mode": d["exit"].mode().iloc[0],
                         "exit_share_0": (d["exit"] == 0).mean()})
    for r in rv_runs:
        if r.group == "credit" and r.name in {"hy_ig_oas", "cdx_cash_basis", "hyg_cdx"}:
            d = r.diagnostics
            rows.append({"model": "headline credit pair (adaptive)", "member": r.name, "folds": len(d),
                         "entry_median": d["entry_adaptive"].median(), "entry_mode": d["entry_adaptive"].mode().iloc[0],
                         "entry_share_at_grid_top_2.5": (d["entry_adaptive"] == 2.5).mean(),
                         "exit_median": d["exit_adaptive"].median(), "exit_mode": d["exit_adaptive"].mode().iloc[0],
                         "exit_share_0": (d["exit_adaptive"] == 0).mean()})
    return pd.DataFrame(rows)


def cg_cl(t: pd.DataFrame, model: str, basis: str) -> dict:
    win, loss = t[t["net"] > 0]["net"], t[t["net"] <= 0]["net"]
    p = len(win) / len(t)
    W, L = win.mean(), -loss.mean()
    return {"model": model, "basis": basis, "trades": len(t), "hit_gross": (t["gross"] > 0).mean(), "hit_net": p,
            "mean_gross": t["gross"].mean(), "mean_cost": t["cost"].mean(), "mean_net": t["net"].mean(),
            "W_mean_net_gain_of_winners": W, "L_mean_net_loss_of_losers": L, "win_loss_ratio": W / L,
            "expectancy_pW_minus_qL": p * W - (1 - p) * L, "breakeven_hit_rate_L_over_W_plus_L": L / (W + L)}


def trade_tables(models, levels):
    """Hit ratio / conditional gain-loss, all models, plus by |z| at entry. Units: member risk units."""
    rows, zrows = [], []
    diag = {"headline": "Headline", "pca3_sparse": "PCA-3 sparse", "pca2": "PCA-2 full", "pairwise_nn": "Pairwise NN"}
    for key, label in diag.items():
        t = pd.read_csv(DIAG / f"s3_trades_{key}.csv")
        rows.append(cg_cl(t, label, "member trades (s3_trades table)"))
        t["z_bucket"] = pd.cut(t["entry_z"].abs(), [0, 2, 2.5, 3, np.inf], right=False, labels=["<2", "2-2.5", "2.5-3", "3+"])
        for b, g in t.groupby("z_bucket", observed=True):
            zrows.append({**cg_cl(g, label, "member trades"), "entry_abs_z": b})
    rows.append(cg_cl(model_trades(models["pca3_sparse"].items), "PCA-3 sparse", "merged (identical copies once)"))
    bench = run_members(levels, fixed=BENCHMARKS, label="bench")
    for r in bench:
        rows.append(cg_cl(trades(r, pca.VARIANT), r.name.replace("bench_", ""), "member trades"))
    return pd.DataFrame(rows), pd.DataFrame(zrows)


def tercile_cuts() -> pd.DataFrame:
    rows = []
    for key in ("headline", "pca3_sparse", "pca2", "pairwise_nn"):
        v = pd.read_csv(DIAG / f"s3_trades_{key}.csv")["entry_vol_bp"]
        rows.append({"model": key, "trades": v.notna().sum(), "low_mid_cut_bp": v.quantile(1 / 3),
                     "mid_high_cut_bp": v.quantile(2 / 3)})
    return pd.DataFrame(rows)


def svb_trade(models, levels, rolls) -> pd.DataFrame:
    """Every 5y-member trade open during the SVB window, end to end, in risk units and $ at $1m capital."""
    rows = []
    for key in ("headline", "pca3_sparse"):
        m = models[key]
        b = bk.build_book(m, levels, rolls)
        for r, v in m.items:
            if not r.name.endswith("ust_5y"):
                continue
            t = trades(r, v)
            t = t[(t["exit"] >= SVB[0]) & (t["entry"] <= SVB[1])]
            res = r.results[v]
            for _, x in t.iterrows():
                e, xdt = x["entry"], x["exit"]
                held = slice(res.position.index[res.position.index.get_loc(e) + 1], xdt)
                legs = r.extras["legs_adaptive"].loc[e]
                legs = legs[legs.abs() > 1e-12]
                usd_dv01 = CAPITAL * b.w_member.shift(-1).loc[e] * res.position.loc[e] * legs  # held over e+1
                z, sig = r.extras["ou_z"].loc[e], r.extras["ou_sigma_eq"].loc[e]
                trade_cost = res.cost.loc[e:xdt]  # entry cost at e, exit cost at xdt, resizes between
                rows.append({
                    "model": key, "member": r.name, "entry": e.date(), "exit": xdt.date(), "bars": x["bars"],
                    "side_yield_units": x["side"], "entry_z": z, "sigma_eq_bp": sig, "gap_bp_z_times_sigma_eq": z * sig,
                    "half_life_bars": r.extras["ou_half_life"].loc[e], "entry_vol_bp_per_day": abs(res.signal.loc[e] / res.position.loc[e]),
                    "position_risk_units": res.position.loc[e],
                    **{f"w_{k}": val for k, val in legs.items()},
                    **{f"usd_per_bp_{k}": val for k, val in usd_dv01.items()},
                    "gross_risk_units": x["gross"], "cost_risk_units": x["cost"], "net_risk_units": x["net"],
                    "gross_usd_at_1m": CAPITAL * (b.w_member * res.gross).loc[held].sum(),
                    "cost_usd_at_1m": CAPITAL * (b.w_member * trade_cost).sum(),
                    "residual_move_bp_entry_to_exit": res.gross.loc[held].sum() / res.position.loc[e] if res.position.loc[e] else np.nan,
                })
    out = pd.DataFrame(rows)
    out["net_usd_at_1m"] = out["gross_usd_at_1m"] - out["cost_usd_at_1m"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    levels, rolls, models = load_models()
    _, _, rv_runs, _ = run.load_runs()
    pca_runs = s1.load_pca_runs(levels)
    tables = pca_fold(levels, pca_runs)
    tables["c_ou_by_tenor"] = ou_by_tenor(pca_runs)
    tables["a_bands_by_member"] = bands(pca_runs, rv_runs)
    tables["d_cg_cl"], tables["d_cg_cl_by_z"] = trade_tables(models, levels)
    tables["f_tercile_cuts"] = tercile_cuts()
    tables["k_svb_5y_trades"] = svb_trade(models, levels, rolls)
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4g}".format):
        for name, df in tables.items():
            print(f"\n== {name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
