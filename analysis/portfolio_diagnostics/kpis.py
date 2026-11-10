"""Step 2: portfolio KPIs, gross and net, benchmarks, breakeven cost, time-varying cost.

P&L is additive, in fractions of capital at the sleeve's 10% volatility target (no compounding).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import metrics
from analysis.experiments.pipeline import run_members
from analysis.portfolio_diagnostics import book as bk

ANN = config.ANN
COST_MULTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BENCHMARKS = {  # fixed weights in bp of yield: one unit of belly against half a unit of each wing
    "fly_2s5s10s": {"ust_2y": -0.5, "ust_5y": 1.0, "ust_10y": -0.5},
    "fly_3s5s7s": {"ust_3y": -0.5, "ust_5y": 1.0, "ust_7y": -0.5},
}


def drawdown_stats(pnl: pd.Series) -> dict:
    """Max drawdown, longest peak-to-recovery spell, and bars from the max-drawdown trough to a new high."""
    eq = pnl.fillna(0).cumsum()
    peak = eq.cummax()
    dd = eq - peak
    underwater = dd < 0
    spell = underwater.groupby((~underwater).cumsum()).cumsum()
    trough = dd.idxmin()
    after = eq.loc[trough:]
    recovered = after[after >= peak.loc[trough]]
    return {
        "max_dd": float(dd.min()),
        "max_dd_trough": trough.date(),
        "longest_dd_bars": int(spell.max()),
        "recovery_bars": int(len(eq.loc[trough:recovered.index[0]]) - 1) if len(recovered) else np.nan,
    }


def kpi(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    sd = pnl.std()
    downside = np.sqrt((np.minimum(pnl, 0) ** 2).mean())
    week = pnl.resample("W-FRI").sum()
    q = pnl.quantile(0.01)
    dd = drawdown_stats(pnl)
    ann_ret = pnl.mean() * ANN
    return {
        "start": pnl.index[0].date(), "ann_return": ann_ret, "ann_vol": sd * np.sqrt(ANN),
        "sharpe": metrics.sharpe(pnl), "sortino": ann_ret / (downside * np.sqrt(ANN)) if downside > 0 else np.nan,
        "calmar": ann_ret / -dd["max_dd"] if dd["max_dd"] < 0 else np.nan, **dd,
        "worst_day": float(pnl.min()), "worst_week": float(week.min()),
        "var_99": float(-q), "es_99": float(-pnl[pnl <= q].mean()),
    }


def kpi_table(ports: dict) -> pd.DataFrame:
    rows = []
    for label, port in ports.items():
        live = port.net.notna()
        for basis, s in (("gross", port.gross), ("net", port.net)):
            rows.append({"model": label, "basis": basis, **kpi(s[live])})
    return pd.DataFrame(rows)


def yearly(ports: dict) -> pd.DataFrame:
    rows = []
    for label, port in ports.items():
        df = pd.DataFrame({"gross": port.gross, "cost": port.cost, "net": port.net}).dropna()
        y = df.groupby(df.index.year).sum()
        rows.append(y.assign(model=label).rename_axis("year").reset_index())
    return pd.concat(rows, ignore_index=True)


def breakeven(ports: dict) -> pd.DataFrame:
    """Net Sharpe at multiples of the 1x cost table with positions and weights fixed; m* sets mean net to 0."""
    rows = []
    for label, port in ports.items():
        live = port.net.notna()
        g, c = port.gross[live], port.cost[live]
        row = {"model": label, "gross_sharpe": metrics.sharpe(g)}
        row.update({f"net_sharpe_{m:g}x": metrics.sharpe(g - m * c) for m in COST_MULTS})
        row["breakeven_mult"] = g.mean() / c.mean()
        row["gross_return_pa"], row["cost_pa_1x"] = g.mean() * ANN, c.mean() * ANN
        rows.append(row)
    return pd.DataFrame(rows)


def breakeven_curve(ports: dict, grid=np.linspace(0, 1.5, 31)) -> pd.DataFrame:
    rows = []
    for label, port in ports.items():
        live = port.net.notna()
        rows += [{"model": label, "mult": m, "net_sharpe": metrics.sharpe(port.gross[live] - m * port.cost[live])}
                 for m in grid]
    return pd.DataFrame(rows)


def cost_sensitivity(models: dict) -> pd.DataFrame:
    """Flat 1x table vs time-varying bid/ask, sleeve rebuilt with re-priced cost (positions unchanged)."""
    rows = []
    for label, m in models.items():
        out = {"model": label}
        for name, fn in (("flat", bk.flat_cost), ("time_varying", bk.time_varying_cost)):
            p = bk.reprice(m, fn)
            live = p.net.notna()
            out[f"net_sharpe_{name}"] = metrics.sharpe(p.net[live])
            out[f"cost_pa_{name}"] = p.cost[live].mean() * ANN
            out[f"net_pa_{name}"] = p.net[live].mean() * ANN
            for era, mask in (("pre2008", p.net.index < bk.TV_SWITCH), ("2008on", p.net.index >= bk.TV_SWITCH)):
                out[f"net_sharpe_{name}_{era}"] = metrics.sharpe(p.net[live & mask])
        out["gross_sharpe"] = metrics.sharpe(m.port.gross[m.port.net.notna()])
        rows.append(out)
    return pd.DataFrame(rows)


def benchmark_ports(levels: pd.DataFrame) -> dict:
    """Fixed-weight flies traded with the PCA rules: 60-bar OU s-score, training-window bands, same folds and costs."""
    runs = run_members(levels, fixed=BENCHMARKS, label="bench")
    out = {}
    for r in runs:
        port, _, _ = s1.sleeve([(r, "pca")])
        out[r.name.replace("bench_", "")] = (port, r)
    return out
