"""Step 1: cost model audit.

    python -m analysis.portfolio_diagnostics.cost_audit

What the backtest charges (amp_sa/backtest/engine.py `account`), per member and bar:

    cost_t = |p_t - p_{t-1}| * u_t        p_t = signal_t / vol_t (risk units per unit of the member)
                                          u_t = sum_legs |w_leg| * c_leg   (bp, one way)

plus, at the sleeve level (portfolio/combine.py), |delta sleeve weight| x mean |p| x u.

This module splits that charge into entry, exit, resize, roll and sleeve rescaling,
and computes what a leg-by-leg charge would have been:

    true_t = sum_legs c_leg * |p_t * w_{t+1,leg} - p_{t-1} * w_{t,leg}|

The engine assumes a held position switches to new leg weights for free (PCA
loadings at a fold boundary, the Kalman beta every bar). The gap true - charged is
that unpriced re-hedging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.portfolio_diagnostics.common import CAPITAL, TABLE_DIR, leg_cost_vector, legs_for, load_models
from amp_sa import config

TYPES = ("entry", "exit", "resize", "roll")


def split_cost(position: pd.Series, cost: pd.Series, unit_cost: pd.Series) -> pd.DataFrame:
    """Per-bar cost split into entry / exit / resize / roll. Columns sum to `cost`.

    entry:  opening from flat, or the new side of a flip
    exit:   closing to flat, or the old side of a flip
    resize: same side held in and out (p = signal / vol moves daily)
    roll:   whatever the engine charged beyond |delta p| * u (ROLL_COST_MULT on roll bars)
    """
    p = position.fillna(0.0)
    h = p.shift(1).fillna(0.0)
    sp, sh = np.sign(p), np.sign(h)
    base = (p - h).abs() * unit_cost
    flip = (sp != 0) & (sh != 0) & (sp != sh)
    same = (sp != 0) & (sp == sh)
    out = pd.DataFrame(0.0, index=p.index, columns=list(TYPES))
    out["entry"] = np.where((sh == 0) & (sp != 0), base, 0.0) + np.where(flip, p.abs() * unit_cost, 0.0)
    out["exit"] = np.where((sp == 0) & (sh != 0), base, 0.0) + np.where(flip, h.abs() * unit_cost, 0.0)
    out["resize"] = np.where(same, base, 0.0)
    out["roll"] = cost.fillna(0.0) - base
    return out


def leg_level_cost(position: pd.Series, legs: pd.DataFrame, leg_cost: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Cost if every leg were traded separately; aligned with the engine's cost bar.

    Holding over bar t is p_{t-1} * w_t (the engine prices the change on bar t with
    the leg weights of bar t). The trade at the close of t moves it to p_t * w_{t+1}.
    Returns (cost per bar, traded amount per bar and leg).
    """
    p = position.fillna(0.0)
    w = legs.reindex(p.index).fillna(0.0)
    w_next = w.shift(-1).fillna(w)
    traded = (w_next.mul(p, axis=0) - w.mul(p.shift(1).fillna(0.0), axis=0)).abs()
    return traded.mul(leg_cost[traded.columns], axis=1).sum(axis=1), traded


def member_frames(model) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Per member: split cost, leg-level cost, and its portfolio weight (sleeve weight / live members)."""
    live = pd.DataFrame({r.name: r.live for r, _ in model.items}).fillna(False)
    cost_live = pd.DataFrame({r.name: r.results[v].cost.where(r.live) for r, v in model.items})
    n_live = cost_live.notna().sum(axis=1).replace(0, np.nan)  # mirrors run_sleeve1.sleeve (mean over live)
    w_member = model.port.weights["s"] / n_live
    frames = {}
    for r, v in model.items:
        res = r.results[v]
        lv = live[r.name]
        pos = res.position.where(lv, 0.0)
        split = split_cost(pos, res.cost.where(lv, 0.0), res.unit_cost)
        legs = legs_for(r, v)
        true, _ = leg_level_cost(pos, legs, leg_cost_vector(legs.columns))
        f = split.assign(charged=res.cost.where(lv, 0.0), leg_level=true.where(lv, 0.0),
                         unit_cost=res.unit_cost, weight=w_member.where(lv))
        frames[r.name] = f[lv]
    return frames, model.port.cost


def audit(models) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_model, by_member, daily = [], [], []
    for label, m in models.items():
        frames, port_cost = member_frames(m)
        contrib = pd.DataFrame(index=port_cost.index)
        for name, f in frames.items():
            # Same member traded under several names (identical sparse hedges) is counted as the backtest counts it.
            cap = f[list(TYPES) + ["charged", "leg_level"]].mul(f["weight"], axis=0)
            for t in TYPES:
                contrib[f"{name}|{t}"] = cap[t]
            years = len(f) / config.ANN
            row = {"model": label, "member": name, "years_live": years,
                   "unit_cost_bp_median": f["unit_cost"].median(),
                   "round_trip_bp_median": 2 * f["unit_cost"].median()}
            for t in TYPES:
                row[f"{t}_risk"] = f[t].sum()
            row["charged_risk"] = f["charged"].sum()
            row["leg_level_risk"] = f["leg_level"].sum()
            row["unpriced_rehedge_risk"] = row["leg_level_risk"] - (row["charged_risk"] - row["roll_risk"])  # roll is charged separately
            for t in TYPES:
                row[f"{t}_pct_cap"] = 100 * cap[t].sum()
            row["charged_pct_cap"] = 100 * cap["charged"].sum()
            row["unpriced_rehedge_pct_cap"] = 100 * (cap["leg_level"].sum() - cap["charged"].sum() + cap["roll"].sum())
            by_member.append(row)
        per_type = {t: contrib.filter(like=f"|{t}").sum(axis=1, min_count=1) for t in TYPES}
        member_total = sum(s.fillna(0.0) for s in per_type.values())
        rescale = (port_cost - member_total).where(port_cost.notna())
        d = pd.DataFrame({**per_type, "sleeve_rescale": rescale, "total": port_cost}).dropna(subset=["total"])
        check = (d[list(TYPES)].fillna(0).sum(axis=1) + d["sleeve_rescale"] - d["total"]).abs().max()
        assert check < 1e-12, f"{label}: split does not add up ({check})"
        years = len(d) / config.ANN
        tot = d["total"].sum()
        for t in [*TYPES, "sleeve_rescale"]:
            s = d[t].fillna(0.0).sum()
            by_model.append({"model": label, "type": t, "pct_cap_total": 100 * s, "pct_cap_per_year": 100 * s / years,
                             "usd_per_year_at_1m": CAPITAL * s / years, "share_of_cost": s / tot})
        mem = pd.DataFrame([r for r in by_member if r["model"] == label])
        unpriced = mem["unpriced_rehedge_pct_cap"].sum()
        by_model.append({"model": label, "type": "unpriced_rehedge (not charged)", "pct_cap_total": unpriced,
                         "pct_cap_per_year": unpriced / years, "usd_per_year_at_1m": CAPITAL * unpriced / 100 / years,
                         "share_of_cost": unpriced / 100 / tot})
        daily.append(d.assign(model=label))
    return pd.DataFrame(by_model), pd.DataFrame(by_member), pd.concat(daily).rename_axis("date").reset_index()


def main() -> None:
    _, _, models = load_models()
    by_model, by_member, daily = audit(models)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    by_model.to_csv(TABLE_DIR / "cost_split_by_model.csv", index=False)
    by_member.to_csv(TABLE_DIR / "cost_split_by_member.csv", index=False)
    daily.to_csv(TABLE_DIR / "cost_daily_by_type.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(by_model.to_string(index=False))
        cols = ["model", "member", "round_trip_bp_median", "entry_risk", "exit_risk", "resize_risk", "roll_risk",
                "charged_risk", "unpriced_rehedge_risk", "charged_pct_cap", "unpriced_rehedge_pct_cap"]
        print(by_member[cols].to_string(index=False))


if __name__ == "__main__":
    main()
