"""What to learn from Sleeve 1: individual trades, standout periods, and where the cost goes.

    python -m amp_sa.run_lessons

Reads the cached runs (run amp_sa.run and amp_sa.run_sleeve1 first) and writes
reports/trades.md and reports/periods.md with their tables and figures.

Descriptive only. Nothing here feeds back into a model, so ex-post groupings
(calendar years, volatility terciles over the whole sample) are allowed. Trade
P&L is in the strategy's risk units: one unit is one trailing daily standard
deviation of the instrument, so a trade's gross, cost and net read as "daily
sigmas". Period P&L is in % of capital at the sleeve's 10% volatility target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config, run
from amp_sa import run_sleeve1 as s1
from amp_sa.backtest import metrics

REPORT_DIR = run.REPORT_DIR
MODELS = {  # short label -> model name in run_sleeve1.models
    "headline": s1.HEADLINE,
    "pca3_sparse": "PCA-3 sparse (4 legs per residual)",
    "pca2": "PCA-2 curve (level, slope hedged)",
    "pairwise_nn": "pairwise adaptive, neural gate",
}
LABELS = {
    "headline": "Headline (PCA-2 + credit)",
    "pca3_sparse": "PCA-3 sparse",
    "pca2": "PCA-2 full",
    "pairwise_nn": "Pairwise adaptive, neural",
}
HOLD_BUCKETS = [0, 2, 5, 10, 20, np.inf]
HOLD_LABELS = ["1-2 bars", "3-5", "6-10", "11-20", "21+"]
ERAS = [1990, 2004, 2012, 2019, 2030]
ERA_LABELS = ["1995-2004", "2005-2012", "2013-2019", "2020-2026"]
WINDOW = 126  # bars in a "period" when ranking the best and worst stretches
TIME_STOP = 20  # bars; for the question "what did trades held longer than this earn after it?"


# ---------- trades ----------

def trades(r, variant: str) -> pd.DataFrame:
    """One row per trade: a run of bars with the same non-zero side in the post-close position.

    Gross on bar t belongs to the position held over t (set at t-1). Cost on bar t
    belongs to the position set at t, or, on an exit bar, to the trade being closed.
    A flip from long to short is two trades; the flip's cost goes to the new one.
    """
    res = r.results[variant]
    live = r.live.reindex(res.position.index).fillna(False).astype(bool)
    pos = res.position.where(live, 0.0)
    side = np.sign(pos)
    start = (side != 0) & (side != side.shift(1).fillna(0.0))
    tid = start.cumsum().where(side != 0)
    held_tid = tid.shift(1)
    cost_tid = tid.fillna(held_tid)
    gross = res.gross.where(live)
    cost = res.cost.where(live)
    # Re-sizing: cost on bars where the same trade is held in and out (position = signal / vol moves daily).
    resize = cost.where(tid == held_tid)
    age = held_tid.groupby(held_tid).cumcount()  # 0 on the first bar held

    out = pd.DataFrame({
        "gross": gross.groupby(held_tid).sum(),
        "gross_first_20": gross.where(age < TIME_STOP).groupby(held_tid).sum(),
        "cost": cost.groupby(cost_tid).sum(),
        "resize_cost": resize.groupby(held_tid).sum(),
        "bars": held_tid.value_counts(),
        "side": side.groupby(tid).first(),
        "entry": pos.index.to_series().groupby(tid).first(),
        "exit": pos.index.to_series().groupby(held_tid).last(),
    }).dropna(subset=["bars"])
    out["net"] = out["gross"] - out["cost"]
    out["open_at_end"] = out["exit"] == pos.index[-1]
    out.insert(0, "instrument", r.name)
    return out.reset_index(drop=True)


def model_trades(items) -> pd.DataFrame:
    """All trades of a model, with identical trades in different instruments merged into one row.

    Sparse hedges can give several tenors the same legs (in the PCA-3 sparse model the
    2y, 3y and 5y residuals are all the one portfolio of 2y, 3y, 5y and 7y), so the
    same trade appears under several names. `copies` counts them.
    """
    t = pd.concat([trades(r, v) for r, v in items], ignore_index=True)
    key = t["entry"].astype(str) + "|" + t["exit"].astype(str) + "|" + t["gross"].abs().round(9).astype(str)
    g = t.groupby(key.rename("key").to_numpy(), sort=False)
    out = g.first().reset_index(drop=True)
    out["instrument"] = g["instrument"].apply(" = ".join).to_numpy()
    out["copies"] = g.size().to_numpy()
    return out


def breadth(items) -> dict:
    """How many independent bets the sleeve really holds.

    effective_bets = (sum of eigenvalues)^2 / sum of squared eigenvalues of the
    correlation matrix of the members' daily gross P&L: 1 if all members move as
    one, N if they are uncorrelated. Members that never overlap get correlation 0.
    """
    gross = pd.DataFrame({r.name: r.results[v].gross.where(r.live) for r, v in items})
    gross = gross.loc[:, gross.abs().sum() > 0]
    corr = gross.corr(min_periods=250).fillna(0.0).to_numpy().copy()
    np.fill_diagonal(corr, 1.0)
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    off = corr[~np.eye(len(corr), dtype=bool)]
    return {"members": gross.shape[1], "identical_pairs": int((off > 0.999).sum() // 2),
            "effective_bets": eig.sum() ** 2 / (eig**2).sum(), "mean_pairwise_corr": off.mean()}


def trade_summary(t: pd.DataFrame, years: float) -> dict:
    wins, losses = t.loc[t["net"] > 0, "net"], t.loc[t["net"] <= 0, "net"]
    top = t["gross"].nlargest(max(1, len(t) // 10)).sum()
    return {
        "trades": len(t),
        "trades_per_year": len(t) / years,
        "median_bars": t["bars"].median(),
        "hit_rate_gross": (t["gross"] > 0).mean(),
        "hit_rate_net": (t["net"] > 0).mean(),
        "mean_gross": t["gross"].mean(),
        "mean_cost": t["cost"].mean(),
        "mean_net": t["net"].mean(),
        "cost_share_of_gross": t["cost"].sum() / t["gross"].sum() if t["gross"].sum() > 0 else np.nan,
        "resize_share_of_cost": t["resize_cost"].sum() / t["cost"].sum(),
        "avg_win_over_avg_loss": wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan,
        "top_decile_share_of_gross": top / t["gross"].sum() if t["gross"].sum() > 0 else np.nan,
    }


def long_trades(t: pd.DataFrame) -> dict:
    """Trades held past TIME_STOP bars: what they earned up to that bar and after it.

    The split shows what a time stop at TIME_STOP bars would have changed, before the
    extra exit cost and before any effect on later entries. It is a pointer for a
    walk-forward test, not a result.
    """
    long = t[t["bars"] > TIME_STOP]
    return {
        "trades_held_past_stop": len(long), "share_of_trades": len(long) / len(t),
        "gross_first_bars": long["gross_first_20"].sum(), "gross_after": (long["gross"] - long["gross_first_20"]).sum(),
        "cost_total": long["cost"].sum(), "net_total": long["net"].sum(),
        "net_all_other_trades": t.loc[t["bars"] <= TIME_STOP, "net"].sum(),
    }


def grouped(t: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    g = t.groupby(by, observed=True)
    return pd.DataFrame({
        "trades": g.size(), "mean_gross": g["gross"].mean(), "mean_cost": g["cost"].mean(),
        "mean_net": g["net"].mean(), "hit_rate_net": g["net"].apply(lambda s: (s > 0).mean()),
        "total_net": g["net"].sum(),
    }).reset_index()


# ---------- periods ----------

def contributions(items, port) -> pd.DataFrame:
    """Each instrument's share of the sleeve's net P&L on each bar, in fractions of capital.

    Mirrors run_sleeve1.sleeve: members are equal-weighted over those live on the bar,
    then the sleeve weight applies. The column 'rescaling' is the cost of changes in
    the sleeve weight, which no single instrument owns.
    """
    nets = pd.DataFrame({r.name: r.results[v].net.where(r.live) for r, v in items})
    n = nets.notna().sum(axis=1).replace(0, np.nan)
    out = nets.div(n, axis=0).mul(port.weights["s"], axis=0)
    out["rescaling"] = port.net - out.sum(axis=1, min_count=1)
    return out


def extreme_windows(net: pd.Series, window: int, n: int, worst: bool) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The n non-overlapping `window`-bar stretches with the lowest (or highest) summed net P&L."""
    roll = net.fillna(0).rolling(window).sum().where(net.notna())
    roll = roll if not worst else -roll
    picks = []
    taken = pd.Series(False, index=net.index)
    for end in roll.sort_values(ascending=False).dropna().index:
        i = net.index.get_loc(end)
        if i < window or taken.iloc[i - window + 1 : i + 1].any():
            continue
        taken.iloc[max(0, i - window + 1) : i + window] = True  # keep the next pick a full window away
        picks.append((net.index[i - window + 1], end))
        if len(picks) == n:
            break
    return picks


def market_context(levels: pd.DataFrame, start, end) -> dict:
    """Yields in the panel are already in bp."""
    lv = levels.loc[start:end]
    first, last = lv.iloc[0], lv.iloc[-1]
    slope = lv["ust_10y"] - lv["ust_2y"]
    return {
        "chg_10y_bp": last["ust_10y"] - first["ust_10y"],
        "chg_2s10s_bp": slope.iloc[-1] - slope.iloc[0],
        "avg_move": lv["move"].mean(),
    }


def window_rows(label, port, contrib, levels, picks, kind) -> list[dict]:
    rows = []
    for start, end in picks:
        c = contrib.loc[start:end].sum()
        names = c.drop("rescaling").sort_values()
        lead = names.head(2) if kind == "worst" else names.tail(2)[::-1]
        rows.append({
            "model": label, "kind": kind, "start": start.date(), "end": end.date(),
            "net_pct": port.net.loc[start:end].sum() * 100, "gross_pct": port.gross.loc[start:end].sum() * 100,
            "cost_pct": port.cost.loc[start:end].sum() * 100,
            "main_contributors": ", ".join(f"{k} ({v * 100:+.1f})" for k, v in lead.items()),
            **market_context(levels, start, end),
        })
    return rows


def day_rows(label, port, contrib, levels, n: int) -> list[dict]:
    rows = []
    net = port.net.dropna()
    slope = levels["ust_10y"] - levels["ust_2y"]
    for kind, days in (("worst", net.nsmallest(n).index), ("best", net.nlargest(n).index)):
        for d in days:
            c = contrib.loc[d].drop("rescaling").dropna()
            lead = c.idxmin() if kind == "worst" else c.idxmax()
            i = levels.index.get_loc(d)
            rows.append({
                "model": label, "kind": kind, "date": d.date(), "net_pct": net.loc[d] * 100,
                "main_contributor": f"{lead} ({c[lead] * 100:+.2f})",
                "chg_10y_bp": levels["ust_10y"].iloc[i] - levels["ust_10y"].iloc[i - 1],
                "chg_2s10s_bp": slope.iloc[i] - slope.iloc[i - 1],
                "move_prev_close": levels["move"].iloc[i - 1],
            })
    return rows


def yearly(port) -> pd.DataFrame:
    df = pd.DataFrame({"gross": port.gross, "cost": port.cost, "net": port.net}).dropna()
    return (df.groupby(df.index.year).sum() * 100).rename_axis("year")


def headline_cost_split(items, port) -> pd.DataFrame:
    """Headline cost per year, split into curve residuals, credit pairs and sleeve re-scaling (% of capital)."""
    costs = pd.DataFrame({r.name: r.results[v].cost.where(r.live) for r, v in items})
    n = costs.notna().sum(axis=1).replace(0, np.nan)
    member = costs.div(n, axis=0).mul(port.weights["s"], axis=0)
    curve = member[[c for c in member if c.startswith("pca")]].sum(axis=1)
    credit = member[[c for c in member if not c.startswith("pca")]].sum(axis=1)
    df = pd.DataFrame({"curve_residuals": curve, "credit_pairs": credit,
                       "rescaling": port.cost - curve - credit, "total": port.cost}).where(port.cost.notna())
    out = (df.groupby(df.index.year).sum() * 100).rename_axis("year")
    return out[out["total"] > 0].reset_index()


def by_era(port) -> pd.DataFrame:
    df = pd.DataFrame({"gross": port.gross, "net": port.net, "cost": port.cost}).dropna()
    era = pd.cut(df.index.year, ERAS, labels=ERA_LABELS)
    g = df.groupby(era, observed=True)
    return pd.DataFrame({
        "gross_sharpe": g["gross"].apply(metrics.sharpe), "net_sharpe": g["net"].apply(metrics.sharpe),
        "gross_pct_per_year": g["gross"].mean() * config.ANN * 100,
        "cost_pct_per_year": g["cost"].mean() * config.ANN * 100,
    }).rename_axis("era").reset_index()


def by_move(port, levels) -> pd.DataFrame:
    """P&L by MOVE tercile, conditioning on the previous close (what was known when the position was held)."""
    move = levels["move"].shift(1).reindex(port.net.index)
    df = pd.DataFrame({"gross": port.gross, "cost": port.cost, "net": port.net, "move": move}).dropna()
    df["bucket"] = pd.qcut(df["move"], 3, labels=["low MOVE", "mid MOVE", "high MOVE"])
    g = df.groupby("bucket", observed=True)
    return pd.DataFrame({
        "move_range": g["move"].apply(lambda s: f"{s.min():.0f}-{s.max():.0f}"),
        "gross_pct_per_year": g["gross"].mean() * config.ANN * 100,
        "cost_pct_per_year": g["cost"].mean() * config.ANN * 100,
        "net_sharpe": g["net"].apply(metrics.sharpe), "gross_sharpe": g["gross"].apply(metrics.sharpe),
    }).rename_axis("regime").reset_index()


def cost_ratio_by_fold(pca_runs, levels) -> pd.DataFrame:
    """Median one-way cost over daily residual volatility across tenors, per fold, against the rate level."""
    rows = []
    for key, label in ((2, "pca2"), ("3s", "pca3_sparse")):
        d = pd.concat([r.diagnostics for r in pca_runs[key]], ignore_index=True)
        per_fold = d.groupby("test_start")["cost_over_daily_vol"].median()
        for start, ratio in per_fold.items():
            rows.append({"model": label, "test_start": start, "cost_over_daily_vol": ratio})
    out = pd.DataFrame(rows).pivot(index="test_start", columns="model", values="cost_over_daily_vol")
    out.columns.name = None
    # The ratio is set on the training window, so pair it with that window's rates environment.
    for start in out.index:
        train = levels.loc[: pd.Timestamp(start)].iloc[-config.TRAIN_BARS :]
        out.loc[start, "train_avg_2y_yield"] = train["ust_2y"].mean()
        out.loc[start, "train_2y_daily_vol_bp"] = train["ust_2y"].diff().std()
    return out.reset_index()


# ---------- figures ----------

def figures(ports, cost_folds) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPORT_DIR / "figures"
    p = ports["pca3_sparse"]
    roll = lambda s: s.rolling(config.ANN).sum() * 100  # noqa: E731
    run._line_chart({"gross, trailing 12 months": roll(p.gross), "cost, trailing 12 months": roll(p.cost)},
                    "PCA-3 sparse: what the signal earned and what trading it cost",
                    "% of capital over the past year", fig_dir / "lessons_gross_vs_cost.png")

    folds = cost_folds.set_index(pd.to_datetime(cost_folds["test_start"]))
    run._line_chart({"PCA-2 full": folds["pca2"], "PCA-3 sparse": folds["pca3_sparse"]},
                    "One-way cost of a residual trade, in daily volatilities of that residual",
                    "cost / daily vol (median across tenors)", fig_dir / "lessons_cost_ratio.png")

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=run.SURFACE)
    net = ports["headline"].net.dropna().cumsum() * 100
    ax.plot(net.index, net.to_numpy(), color=run.SERIES_COLORS[0], linewidth=1.8)
    dd = net - net.cummax()
    ax.fill_between(dd.index, net.to_numpy(), net.cummax().to_numpy(), color=run.SERIES_COLORS[1], alpha=0.25,
                    linewidth=0, label="below the previous high")
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    run._style(ax, "Headline sleeve: cumulative net P&L and time spent under water", "% of capital")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor="#52514e")
    fig.savefig(fig_dir / "lessons_underwater.png", dpi=150, facecolor=run.SURFACE)
    plt.close(fig)


# ---------- report ----------

def _md(df: pd.DataFrame, digits: int = 2) -> str:
    return run._md(df, digits)


def build():
    levels, rolls, rv_runs, _ = run.load_runs()
    pca_runs = s1.load_pca_runs(levels)
    all_models = s1.models(rv_runs, pca_runs)
    items = {k: all_models[name] for k, name in MODELS.items()}
    ports = {k: s1.sleeve(v)[0] for k, v in items.items()}

    # Trades
    trade_sets = {k: model_trades(v) for k, v in items.items()}
    rows = []
    for k, t in trade_sets.items():
        live_years = pd.concat([r.live.where(r.live) for r, _ in items[k]], axis=1).any(axis=1).sum() / config.ANN
        rows.append({"model": LABELS[k], **trade_summary(t, live_years)})
    summary = pd.DataFrame(rows)
    long_tbl = pd.DataFrame([{"model": LABELS[k], **long_trades(t)} for k, t in trade_sets.items()])

    hold = pd.concat([grouped(t, pd.cut(t["bars"], HOLD_BUCKETS, labels=HOLD_LABELS)).rename(columns={"bars": "holding"})
                      .assign(model=LABELS[k]) for k, t in trade_sets.items()], ignore_index=True)
    move_at_entry = levels["move"]
    move_rows = []
    for k, t in trade_sets.items():
        m = move_at_entry.reindex(pd.to_datetime(t["entry"])).to_numpy()
        bucket = pd.Series(pd.qcut(m, 3, labels=["low MOVE", "mid MOVE", "high MOVE"]), index=t.index)
        move_rows.append(grouped(t, bucket.rename("move_at_entry")).assign(model=LABELS[k]))
    move_trades = pd.concat(move_rows, ignore_index=True)
    side = pd.concat([grouped(t, t["side"].map({1.0: "long residual", -1.0: "short residual"}).rename("side"))
                      .assign(model=LABELS[k]) for k, t in trade_sets.items()], ignore_index=True)
    per_instrument = pd.concat([grouped(t, t["instrument"]).assign(model=LABELS[k])
                                for k, t in trade_sets.items() if k in ("headline", "pca3_sparse")], ignore_index=True)
    cols = ["instrument", "copies", "side", "entry", "exit", "bars", "gross", "cost", "net"]
    standout_trades = pd.concat([
        pd.concat([t.nsmallest(8, "net").assign(kind="worst"), t.nlargest(8, "net").assign(kind="best")])
        [["kind", *cols]].assign(model=LABELS[k])
        for k, t in trade_sets.items() if k in ("headline", "pca3_sparse")], ignore_index=True)
    for c in ("entry", "exit"):
        standout_trades[c] = pd.to_datetime(standout_trades[c]).dt.date

    # Periods
    contribs = {k: contributions(items[k], ports[k]) for k in items}
    windows, days = [], []
    for k in ("headline", "pca3_sparse"):
        net = ports[k].net
        for kind in ("worst", "best"):
            windows += window_rows(LABELS[k], ports[k], contribs[k], levels,
                                   extreme_windows(net, WINDOW, 5, worst=kind == "worst"), kind)
        days += day_rows(LABELS[k], ports[k], contribs[k], levels, 8)
    yearly_tbl = pd.concat({LABELS[k]: yearly(ports[k]) for k in ("headline", "pca3_sparse", "pairwise_nn")}, axis=1)
    yearly_tbl.columns = [f"{m}: {c}" for m, c in yearly_tbl.columns]
    yearly_tbl = yearly_tbl.rename_axis("year")
    breadth_tbl = pd.DataFrame([{"model": LABELS[k], **breadth(items[k])} for k in items])
    cost_split = headline_cost_split(items["headline"], ports["headline"])
    eras = pd.concat([by_era(ports[k]).assign(model=LABELS[k]) for k in items], ignore_index=True)
    move_tbl = pd.concat([by_move(ports[k], levels).assign(model=LABELS[k]) for k in items], ignore_index=True)
    cost_folds = cost_ratio_by_fold(pca_runs, levels)

    tables = {
        "lessons_trade_summary": summary, "lessons_long_trades": long_tbl, "lessons_trades_by_holding": hold, "lessons_trades_by_move": move_trades,
        "lessons_trades_by_side": side, "lessons_trades_by_instrument": per_instrument,
        "lessons_standout_trades": standout_trades, "lessons_windows": pd.DataFrame(windows),
        "lessons_days": pd.DataFrame(days), "lessons_yearly": yearly_tbl.reset_index(),
        "lessons_breadth": breadth_tbl, "lessons_headline_cost_split": cost_split,
        "lessons_eras": eras, "lessons_move_regimes": move_tbl, "lessons_cost_ratio_by_fold": cost_folds,
    }
    all_trades = pd.concat([t.assign(model=LABELS[k]) for k, t in trade_sets.items()], ignore_index=True)
    return levels, ports, tables, all_trades, cost_folds


def first_cols(df: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    return df[first + [c for c in df.columns if c not in first]]


def write_reports(levels, tables) -> None:
    t = tables
    end = levels.index[-1].date()
    trades_md = [
        "# Sleeve 1, trade by trade", "",
        f"Out-of-sample trades from the walk-forward runs, sample ending {end}. A trade is a run of bars with the "
        "same non-zero position in one instrument (a PCA residual or a pairwise spread). P&L is in risk units: "
        "one unit is one trailing daily standard deviation of that instrument, so `mean_gross` of 0.5 means the "
        "average trade earned half a day's typical move before costs. `cost` is entry, exit and any re-sizing "
        "in between.", "",
        "The four models: the headline (PCA-2 curve plus adaptive credit pairs, named in advance), the best "
        "variant found afterwards (sparse PCA-3), the full PCA-2 projection, and the best pairwise model.", "",
        "## The average trade", "",
        _md(t["lessons_trade_summary"]), "",
        "`cost_share_of_gross` is total cost over total gross: above 1 means costs took more than the signal "
        "earned. `resize_share_of_cost` is the part of cost paid while a trade is open, because the position is "
        "signal divided by trailing volatility and is re-sized every bar. `top_decile_share_of_gross` is the share of gross P&L made by the best tenth of trades.", "",
        "## By holding period", "",
        _md(first_cols(t["lessons_trades_by_holding"], ["model", "holding"])), "",
        f"### Trades held longer than {TIME_STOP} bars", "",
        f"Gross P&L of those trades up to bar {TIME_STOP} and after it. The `after` column is roughly what a time "
        f"stop at {TIME_STOP} bars would have avoided (before its own exit cost); it is a pointer for a "
        "walk-forward test, not a result.", "",
        _md(t["lessons_long_trades"]), "",
        "## By rates volatility at entry", "",
        "MOVE index on the entry date, split into terciles over each model's own trades.", "",
        _md(first_cols(t["lessons_trades_by_move"], ["model", "move_at_entry"])), "",
        "## Long and short residuals", "",
        "Long means the position profits if the residual (or spread) rises back toward its mean.", "",
        _md(first_cols(t["lessons_trades_by_side"], ["model", "side"])), "",
        "## By instrument", "",
        _md(first_cols(t["lessons_trades_by_instrument"], ["model", "instrument"])), "",
        "## The best and worst trades", "",
        _md(first_cols(t["lessons_standout_trades"], ["model", "kind"])), "",
        "## How many independent bets", "",
        "Sparse hedges can give two tenors the same legs, so the same trade shows up under several names. Trades "
        "above are merged when entry, exit and P&L match (`copies`). `effective_bets` uses the correlation of the "
        "members' daily gross P&L: 1 if they all move as one, the member count if they are uncorrelated.", "",
        _md(t["lessons_breadth"]), "",
        "Full trade list: `tables/lessons_all_trades.csv`.", "",
    ]
    (REPORT_DIR / "trades.md").write_text("\n".join(trades_md), encoding="utf-8")

    periods_md = [
        "# Sleeve 1, period by period", "",
        f"Sample ending {end}. P&L in % of capital at the 10% volatility target, out of sample and additive. "
        "Contributions split the sleeve's net P&L across its instruments; `rescaling` is the cost of changing "
        "the sleeve's overall weight, which no instrument owns.", "",
        "![under water](figures/lessons_underwater.png)", "",
        "## Gross, cost and net by year", "",
        _md(t["lessons_yearly"], 1), "",
        "## By era", "",
        _md(first_cols(t["lessons_eras"], ["model", "era"])), "",
        "![gross vs cost](figures/lessons_gross_vs_cost.png)", "",
        "## Where the headline's cost comes from", "",
        "Cost per year in % of capital: trading the curve residuals, trading the credit pairs, and re-trading "
        "every position when the sleeve's volatility scaling changes its weight.", "",
        _md(t["lessons_headline_cost_split"], 1), "",
        "## Why cost moves: the cost of a trade relative to the residual's volatility", "",
        "Median across tenors of the one-way cost of a residual divided by its daily volatility, fitted on each "
        "fold's training window, next to the average 2-year yield and the 2-year's daily volatility in bp over "
        "that window.", "",
        _md(t["lessons_cost_ratio_by_fold"]), "",
        "![cost ratio](figures/lessons_cost_ratio.png)", "",
        f"## The worst and best {WINDOW}-bar stretches", "",
        "Non-overlapping six-month windows ranked by net P&L. Market context: change in the 10-year yield and "
        "the 2s10s slope over the window, in bp, and the average MOVE.", "",
        _md(first_cols(t["lessons_windows"], ["model", "kind", "start", "end"])), "",
        "## The worst and best days", "",
        "Context is the same day's change in the 10-year yield and 2s10s, in bp, and MOVE at the previous close.", "",
        _md(first_cols(t["lessons_days"], ["model", "kind", "date"])), "",
        "## By rates-volatility regime", "",
        "MOVE at the previous close, split into terciles over the whole sample.", "",
        _md(first_cols(t["lessons_move_regimes"], ["model", "regime"])), "",
    ]
    (REPORT_DIR / "periods.md").write_text("\n".join(periods_md), encoding="utf-8")


def main() -> None:
    levels, ports, tables, all_trades, cost_folds = build()
    (REPORT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / "tables" / f"{name}.csv", index=False)
    all_trades.to_csv(REPORT_DIR / "tables" / "lessons_all_trades.csv", index=False)
    figures(ports, cost_folds)
    write_reports(levels, tables)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        for name, table in tables.items():
            print(f"\n== {name}")
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
