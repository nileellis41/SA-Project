"""Steps 4-7: cost attribution, signal diagnostics, regimes and robustness, overfitting.

Descriptive only: forward residual changes and full-sample groupings are used to
describe what happened, never to set a position. Where a grouping could be read as a
rule (MOVE stress), the cutoff is trailing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from amp_sa import config
from amp_sa.backtest import metrics
from amp_sa.backtest.splits import folds_for
from amp_sa.sleeves import pca
from analysis.portfolio_diagnostics import external
from analysis.portfolio_diagnostics.book import Book, flat_cost
from analysis.portfolio_diagnostics.hedge import trade_ids

ANN = config.ANN
Z_BUCKETS = [0, 2.0, 2.5, 3.0, np.inf]
Z_LABELS = ["<2", "2-2.5", "2.5-3", "3+"]
HORIZONS = (5, 10, 20)
WINDOWS = {
    "GFC Sep08-Mar09": ("2008-09-01", "2009-03-31"),
    "Taper May-Sep13": ("2013-05-01", "2013-09-30"),
    "Covid Feb-Apr20": ("2020-02-15", "2020-04-30"),
    "Feb 2021 selloff": ("2021-02-01", "2021-03-15"),
    "2022 hikes": ("2022-03-01", "2022-12-31"),
}
STRESS_WINDOW, STRESS_Q = 5 * ANN, 0.8


# ---------- step 4: cost attribution ----------

def cost_by_leg(book: Book) -> pd.DataFrame:
    """Engine cost allocated to legs in proportion to |w| x c, in fraction of capital per bar."""
    out = []
    for (r, v) in book.model.items:
        m = r.name
        L = book.legs[m]
        share = L.abs() * flat_cost(list(L.columns), L.index)
        share = share.div(share.sum(axis=1).replace(0, np.nan), axis=0)
        cost = r.results[v].cost.where(book.live[m], 0.0) * book.w_member
        out.append(share.mul(cost, axis=0).fillna(0.0))
    cols = sorted({c for d in out for c in d.columns})
    return sum(d.reindex(columns=cols, fill_value=0.0) for d in out)


def entry_context(trades: pd.DataFrame, book: Book, levels: pd.DataFrame) -> pd.DataFrame:
    """Adds residual vol at entry (bp; vol = |signal| / |p| on the entry bar) and MOVE at entry."""
    t = trades.copy()
    p = book.pos
    t["entry_vol_bp"] = [1.0 / abs(p.at[e, m]) if p.at[e, m] != 0 else np.nan for m, e in zip(t["member"], t["entry"])]
    t["entry_move"] = levels["move"].reindex(pd.DatetimeIndex(t["entry"])).to_numpy()
    unit = {r.name: r.results[v].unit_cost for r, v in book.model.items}
    t["unit_cost_bp"] = [unit[m].get(e, np.nan) for m, e in zip(t["member"], t["entry"])]
    t["round_trip_over_vol"] = 2 * t["unit_cost_bp"] / t["entry_vol_bp"]
    return t


# ---------- step 5: signal diagnostics ----------

def information_coefficient(book: Book, trades: pd.DataFrame) -> pd.DataFrame:
    """Spearman corr of -z with the forward h-bar member change (bp). Positive = the signal points the right way."""
    rows = []
    zs = {r.name: r.extras.get("ou_z", r.extras.get("z_adaptive")) for r, _ in book.model.items}
    for h in HORIZONS:
        allz, allf, ez, ef = [], [], [], []
        for m in book.members:
            z = zs[m]
            if z is None:
                continue
            c = book.change[m].where(book.live[m]).cumsum()
            fwd = c.shift(-h) - c
            ok = z.notna() & fwd.notna() & book.live[m]
            allz.append(z[ok]); allf.append(fwd[ok])
            entries = pd.DatetimeIndex(trades.loc[trades["member"] == m, "entry"])
            e = entries[ok.reindex(entries).fillna(False).to_numpy()]
            ez.append(z[e]); ef.append(fwd[e])
        for scope, zz, ff in (("all bars", allz, allf), ("entry bars", ez, ef)):
            zz, ff = pd.concat(zz), pd.concat(ff)
            rho, pval = stats.spearmanr(-zz, ff)
            rows.append({"horizon": h, "scope": scope, "n": len(zz), "ic": rho, "p_value": pval})
    return pd.DataFrame(rows)


def z_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.dropna(subset=["entry_z"])
    b = pd.cut(t["entry_z"].abs(), Z_BUCKETS, labels=Z_LABELS, right=False)
    g = t.groupby(b, observed=True)
    return pd.DataFrame({"trades": g.size(), "win_rate_net": g["net"].apply(lambda s: (s > 0).mean()),
                         "mean_gross": g["gross"].mean(), "mean_cost": g["cost"].mean(),
                         "mean_net": g["net"].mean()}).rename_axis("entry_abs_z").reset_index()


def ar1(x: np.ndarray) -> dict:
    """OLS AR(1) on a level series: x_t = a + b x_{t-1} + e. kappa = -ln b per bar; NaN unless 0 < b < 1."""
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"b": np.nan, "kappa": np.nan, "half_life": np.nan, "sigma_eq": np.nan}
    y, lag = x[1:], x[:-1]
    b, a = np.polyfit(lag, y, 1)
    e = y - (a + b * lag)
    ok = 0 < b < 1
    return {"b": b, "kappa": -np.log(b) if ok else np.nan, "half_life": np.log(2) / -np.log(b) if ok else np.nan,
            "sigma_eq": e.std(ddof=2) / np.sqrt(1 - b**2) if ok else np.nan}


def ou_by_fold(book: Book, levels: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """PCA members: AR(1) of the training-window cumulative residual (that fold's weights), the median
    trailing 60-bar half-life the signal saw in the test block, and the median holding of trades entered there."""
    base = [t for t in pca.BASE_TENORS if t in levels.columns]
    dy = levels[list(pca.TENORS)].diff()
    rows = []
    for (r, v) in book.model.items:
        if not r.name.startswith("pca"):
            continue
        L = book.legs[r.name]
        hl_test = r.extras.get("ou_half_life")
        tm = trades[trades["member"] == r.name]
        for f in folds_for(levels[base]):
            ti = levels.index[f.test_start : f.test_end]
            w = L.loc[ti[0]]
            if not book.live[r.name].loc[ti[0]] or (w == 0).all():
                continue
            train = (dy.iloc[f.train_start + 1 : f.train_end].fillna(0.0) * w.reindex(dy.columns, fill_value=0.0)).sum(axis=1)
            fit = ar1(train.cumsum().to_numpy())
            held = tm[(pd.to_datetime(tm["entry"]) >= ti[0]) & (pd.to_datetime(tm["entry"]) <= ti[-1])]
            rows.append({"member": r.name, "test_start": ti[0].date(), **fit,
                         "rolling60_half_life_test_median": float(hl_test.loc[ti].median()) if hl_test is not None else np.nan,
                         "median_holding_bars": held["bars"].median(), "trades": len(held)})
    return pd.DataFrame(rows)


def half_life(b) -> np.ndarray:
    """AR(1) half-life in bars, NaN unless 0 < b < 1."""
    b = np.asarray(b, dtype=float)
    ok = (b > 0) & (b < 1)
    return np.where(ok, np.log(2) / -np.log(np.where(ok, b, 0.5)), np.nan)


def kendall_b(b, n: int) -> np.ndarray:
    """First-order expected OLS estimate of an AR(1) coefficient with intercept on n pairs (Kendall 1954)."""
    return np.asarray(b, dtype=float) - (1 + 3 * np.asarray(b, dtype=float)) / n


def simulate_window_fits(b: float, window: int, n_sims: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit pca.rolling_ou's AR(1) to `n_sims` stationary AR(1) paths with true coefficient b, `window` pairs each.

    Returns (b_hat, half_life, passes) where `passes` is rolling_ou's screen: 0 < b_hat < 1,
    half-life under window / 2 and a positive stationary sigma.
    """
    x = np.empty((n_sims, window + 1))
    x[:, 0] = rng.normal(0.0, 1.0 / np.sqrt(1 - b**2), n_sims)
    e = rng.normal(size=(n_sims, window))
    for t in range(window):
        x[:, t + 1] = b * x[:, t] + e[:, t]
    y, lag = x[:, 1:], x[:, :-1]
    yc, lc = y - y.mean(axis=1, keepdims=True), lag - lag.mean(axis=1, keepdims=True)
    bh = (yc * lc).sum(axis=1) / (lc**2).sum(axis=1)
    hl = half_life(bh)
    resid_var = y.var(axis=1, ddof=1) - bh**2 * lag.var(axis=1, ddof=1)
    passes = (bh > 0) & (bh < 1) & (hl < window / 2) & (resid_var > 0)
    return bh, hl, passes


def ou_bias(book: Book, ou_fold: pd.DataFrame, trades: pd.DataFrame, window: int = pca.OU_WINDOW,
            n_sims: int = 500, seed: int = 0) -> dict:
    """Why the 60-bar half-life is so much shorter than the training-window one: small-sample bias vs the screen.

    For every fold x member, simulate the training-window AR(1) (its b, taken as the truth) and fit it on
    `window` bars as the signal does. If the simulated 60-bar half-life is close to the observed one, the gap
    is estimation bias, not selection. The training b is itself biased slightly down, so this understates the bias.
    """
    rng = np.random.default_rng(seed)
    ok = ou_fold[(ou_fold["b"] > 0) & (ou_fold["b"] < 1)]
    sim_hl, sim_pass = [], []
    for b in ok["b"]:
        _, hl, passes = simulate_window_fits(float(b), window, n_sims, rng)
        sim_hl.append(hl)
        sim_pass.append(passes)
    sim_hl, sim_pass = np.concatenate(sim_hl), np.concatenate(sim_pass)
    obs_hl = pd.concat([r.extras["ou_half_life"][book.live[r.name]] for r, _ in book.model.items if "ou_half_life" in r.extras])
    hl_at_entry = pd.concat([r.extras["ou_half_life"].reindex(pd.DatetimeIndex(trades.loc[trades["member"] == r.name, "entry"]))
                             for r, _ in book.model.items if "ou_half_life" in r.extras])
    return {"fold_member_fits": len(ok),
            "median_train_half_life": ok["half_life"].median(),
            "kendall_implied_60bar_half_life": float(np.nanmedian(half_life(kendall_b(ok["b"], window)))),
            "sim_60bar_half_life_before_screen": float(np.nanmedian(sim_hl)),
            "sim_screen_pass_share": float(sim_pass.mean()),
            "sim_60bar_half_life_after_screen": float(np.nanmedian(sim_hl[sim_pass])),
            "observed_screen_pass_share": float(obs_hl.notna().mean()),
            "observed_60bar_half_life": float(obs_hl.median()),
            "observed_half_life_at_entry": float(hl_at_entry.median()),
            "median_holding_bars": trades["bars"].median()}


def realised_moves(pos: pd.Series, change: pd.Series) -> pd.Series:
    """Per trade, the member's change over the bars it was held, signed by the side (bp per unit; reversion > 0).
    Indexed by entry date; a flip ends one trade and starts the next on the same bar."""
    tid, held = trade_ids(pos)
    move = (np.sign(pos.shift(1).fillna(0.0)) * change).groupby(held).sum()
    entry = pos.index.to_series().groupby(tid).first()
    return pd.Series(move.reindex(entry.index).to_numpy(), index=pd.DatetimeIndex(entry.to_numpy()))


def cost_aware_entry(book: Book, trades: pd.DataFrame) -> pd.DataFrame:
    """Fix A checked on the entries that happened: would (|z| - exit band) x sigma_eq > k x round-trip cost hold at k = 1?

    All inputs are known on the entry bar: z and sigma_eq from the [diag-logging] OU state, the exit band of the
    fold in effect, and the package's one-way cost. `realised_move_bp` (per unit of residual, signed so that
    reversion is positive) is the outcome, used here only to compare with the forecast.
    """
    rows = []
    for (r, v) in book.model.items:
        if "ou_sigma_eq" not in r.extras:
            continue
        m = r.name
        realised = realised_moves(book.pos[m], book.change[m])
        d = r.diagnostics.assign(test_start=pd.to_datetime(r.diagnostics["test_start"])).sort_values("test_start")
        for t in trades[trades["member"] == m].itertuples():
            e = pd.Timestamp(t.entry)
            z, s = r.extras["ou_z"].get(e, np.nan), r.extras["ou_sigma_eq"].get(e, np.nan)
            exit_band = d.loc[d["test_start"] <= e, "exit"].iloc[-1]
            exp_rev = (abs(z) - exit_band) * s
            rows.append({"member": m, "entry": e.date(), "abs_z": abs(z), "exit_band": exit_band, "sigma_eq_bp": s,
                         "expected_reversion_bp": exp_rev, "round_trip_bp": 2 * t.unit_cost_bp,
                         "expected_over_round_trip": exp_rev / (2 * t.unit_cost_bp),
                         "realised_move_bp": realised.get(e, np.nan), "gross": t.gross, "net": t.net})
    return pd.DataFrame(rows)


def cost_aware_summary(entries: pd.DataFrame) -> dict:
    e = entries.dropna(subset=["expected_reversion_bp"])
    passes = e["expected_over_round_trip"] > 1
    return {"entries": len(e), "pass_share_k1": passes.mean(),
            "median_expected_reversion_bp": e["expected_reversion_bp"].median(),
            "median_round_trip_bp": e["round_trip_bp"].median(),
            "mean_realised_move_bp": e["realised_move_bp"].mean(),
            "median_realised_move_bp": e["realised_move_bp"].median(),
            "k_that_halves_entries": e["expected_over_round_trip"].median(),
            "mean_net_passing": e.loc[passes, "net"].mean(), "mean_net_failing": e.loc[~passes, "net"].mean()}


def worst_paths(book: Book, trades: pd.DataFrame, n: int = 10, pad: tuple[int, int] = (20, 10)) -> pd.DataFrame:
    """Cumulative member change (bp, rebased to 0 at entry) and z from `pad[0]` bars before entry to `pad[1]` after exit."""
    zs = {r.name: r.extras.get("ou_z", r.extras.get("z_adaptive")) for r, _ in book.model.items}
    rows = []
    for rank, (_, t) in enumerate(trades.nsmallest(n, "net").iterrows(), 1):
        m, idx = t["member"], book.change.index
        i0, i1 = idx.get_loc(pd.Timestamp(t["entry"])), idx.get_loc(pd.Timestamp(t["exit"]))
        span = idx[max(0, i0 - pad[0]) : i1 + pad[1] + 1]
        c = book.change[m].reindex(span).fillna(0.0).cumsum()
        c = c - c.loc[pd.Timestamp(t["entry"])]
        z = zs[m].reindex(span) if zs[m] is not None else pd.Series(np.nan, index=span)
        for k, d in enumerate(span):
            rows.append({"rank": rank, "member": m, "entry": t["entry"], "exit": t["exit"], "side": np.sign(book.pos.at[pd.Timestamp(t["entry"]), m]),
                         "net": t["net"], "bar": k - (i0 - max(0, i0 - pad[0])), "date": d.date(),
                         "cum_change_bp": c.loc[d], "z": z.loc[d]})
    return pd.DataFrame(rows)


# ---------- step 6: regimes ----------

def window_table(ports: dict) -> pd.DataFrame:
    rows = []
    for label, port in ports.items():
        for name, (a, b) in WINDOWS.items():
            n, g = port.net.loc[a:b].dropna(), port.gross.loc[a:b].dropna()
            rows.append({"model": label, "window": name, "net": n.sum(), "gross": g.sum(), "cost": (g - n).sum(),
                         "max_dd": metrics.max_drawdown(n), "days": len(n)})
    return pd.DataFrame(rows)


def by_label(ports: dict, label_fn, name: str) -> pd.DataFrame:
    rows = []
    for model, port in ports.items():
        net = port.net.dropna()
        lab = label_fn(net.index)
        for k, s in net.groupby(lab):
            rows.append({"model": model, name: k, "days": len(s), "net_pa": s.mean() * ANN,
                         "gross_pa": port.gross.reindex(s.index).mean() * ANN, "sharpe": metrics.sharpe(s)})
    return pd.DataFrame(rows)


def era(index: pd.DatetimeIndex) -> pd.Series:
    """Calendar split from the brief. These are data eras (3y data from 2005, 7y from 2009), not trading eras."""
    return pd.Series(np.select([index.year <= 2004, index.year <= 2008], ["1995-2004", "2005-2008"], "2009+"), index=index)


def era_traded(index: pd.DatetimeIndex, start_3y: pd.Timestamp, start_7y: pd.Timestamp) -> pd.Series:
    """Split by the tenors the PCA actually uses: a tenor enters only once it has a full training window."""
    return pd.Series(np.select([index < start_3y, index < start_7y],
                               [f"to {start_3y:%Y-%m}: 2y 5y 10y 30y", f"{start_3y:%Y-%m} to {start_7y:%Y-%m}: + 3y"],
                               f"{start_7y:%Y-%m} on: all six"), index=index)


def curve_idle(book: Book, periods: dict) -> pd.DataFrame:
    """PCA curve members over given periods: the fold's entry band, the largest |z| seen, how often the
    60-bar screen blanked the signal, and days with a position."""
    rows = []
    for (r, v) in book.model.items:
        if "ou_z" not in r.extras:
            continue
        m = r.name
        d = r.diagnostics.assign(test_start=pd.to_datetime(r.diagnostics["test_start"])).sort_values("test_start")
        for name, (a, b) in periods.items():
            lv = book.live[m].loc[a:b]
            z = r.extras["ou_z"].loc[a:b][lv]
            fold = d[d["test_start"] <= pd.Timestamp(a)].iloc[-1]
            rows.append({"member": m, "period": name, "fold_start": fold["test_start"].date(), "entry_band": fold["entry"],
                         "max_abs_z": z.abs().max(), "screen_blank_share": z.isna().mean(),
                         "days_in_market": int((book.pos[m].shift(1).loc[a:b][lv] != 0).sum()), "live_days": int(lv.sum())})
    return pd.DataFrame(rows)


def auction_test(ports: dict) -> pd.DataFrame:
    rows = []
    for model, port in ports.items():
        net = port.net.dropna()
        wk = external.auction_week(net.index, "ust_5y")
        a, b = net[wk], net[~wk]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"model": model, "auction_week_days": len(a), "other_days": len(b),
                     "auction_week_bp_per_day": 1e4 * a.mean(), "other_bp_per_day": 1e4 * b.mean(),
                     "auction_week_sharpe": metrics.sharpe(a), "other_sharpe": metrics.sharpe(b),
                     "welch_t": t, "p_value": p})
    return pd.DataFrame(rows)


def stress_flag(levels: pd.DataFrame) -> pd.Series:
    move = levels["move"]
    cut = move.rolling(STRESS_WINDOW, min_periods=ANN).quantile(STRESS_Q)
    return (move >= cut).where(cut.notna())


def market_correlations(ports: dict, levels: pd.DataFrame) -> pd.DataFrame:
    tenors = ["ust_2y", "ust_5y", "ust_10y", "ust_30y"]
    mk = pd.DataFrame({"rates_level_change": levels[tenors].diff().mean(axis=1), "move_level": levels["move"],
                       "move_change": levels["move"].diff(), "hy_oas_change": levels["hy_oas"].diff()})
    rows = []
    for model, port in ports.items():
        net = port.net.dropna()
        for c in mk:
            x = mk[c].reindex(net.index)
            ok = x.notna()
            rows.append({"model": model, "factor": c, "corr": net[ok].corr(x[ok]), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def sleeve_split_corr(book: Book, levels: pd.DataFrame) -> pd.DataFrame:
    """Headline: curve members vs credit members, daily capital P&L contribution, normal vs stress days."""
    held = book.pos.shift(1).fillna(0.0)
    contrib = (held * book.change).mul(book.w_member, axis=0)
    cost = pd.DataFrame({r.name: r.results[v].cost.where(book.live[r.name], 0.0) for r, v in book.model.items})
    contrib = contrib - cost.mul(book.w_member, axis=0)
    curve = contrib[[m for m in book.members if m.startswith("pca")]].sum(axis=1)
    credit = contrib[[m for m in book.members if not m.startswith("pca")]].sum(axis=1)
    both = book.live[[m for m in book.members if not m.startswith("pca")]].any(axis=1)
    stress = stress_flag(levels).reindex(curve.index)
    rows = []
    for name, mask in (("all", stress.notna()), ("normal", stress == False), ("stress", stress == True)):  # noqa: E712
        m = mask & both
        rows.append({"days": name, "n": int(m.sum()), "corr_curve_credit": curve[m].corr(credit[m])})
    return pd.DataFrame(rows)


def curve_credit_split(book: Book, port_net: pd.Series) -> dict[str, pd.DataFrame]:
    """Headline: curve vs credit members, daily capital contribution at the sleeve weight (sleeve rescale cost excluded).

    Returns 'parts' (curve vs credit, full sample and credit-live days), 'members' (each over its own live days)
    and 'windows' (net in the stress windows of window_table).
    """
    live = port_net.notna()
    held = book.pos.shift(1).fillna(0.0)
    gross = (held * book.change).mul(book.w_member, axis=0)[live]
    cost = pd.DataFrame({r.name: r.results[v].cost.where(book.live[r.name], 0.0)
                         for r, v in book.model.items}).mul(book.w_member, axis=0)[live]
    groups = {"curve": [m for m in gross.columns if m.startswith("pca")],
              "credit": [m for m in gross.columns if not m.startswith("pca")]}
    credit_live = book.live[groups["credit"]].any(axis=1)[live]

    def row(g: pd.Series, c: pd.Series) -> dict:
        n = g - c
        return {"days": len(n), "gross_pa": g.mean() * ANN, "cost_pa": c.mean() * ANN, "net_pa": n.mean() * ANN,
                "net_vol_pa": n.std() * np.sqrt(ANN), "gross_sharpe": metrics.sharpe(g), "net_sharpe": metrics.sharpe(n),
                "cost_over_gross": c.sum() / g.sum() if g.sum() > 0 else np.nan}  # undefined without gross edge

    parts, net = [], {}
    for part, cols in groups.items():
        g, c = gross[cols].sum(axis=1), cost[cols].sum(axis=1)
        net[part] = g - c
        for scope, mask in (("full sample", slice(None)), ("credit-live days", credit_live)):
            parts.append({"part": part, "scope": scope, **row(g[mask], c[mask])})
    total = net["curve"][credit_live] + net["credit"][credit_live]
    var_share = {p: float(np.cov(net[p][credit_live], total)[0, 1] / total.var()) for p in net}
    for r in parts:
        r["var_share_credit_live"] = var_share[r["part"]] if r["scope"] == "credit-live days" else np.nan

    members = []
    for part, cols in groups.items():
        for m in cols:
            lv = book.live[m][live]
            members.append({"part": part, "member": m, "first_live": lv[lv].index[0].date(),
                            "in_market_share": float((held[m][live][lv] != 0).mean()),
                            **row(gross[m][lv], cost[m][lv])})
    windows = [{"window": name, "curve_net": net["curve"].loc[a:b].sum(), "credit_net": net["credit"].loc[a:b].sum()}
               for name, (a, b) in WINDOWS.items()]
    return {"parts": pd.DataFrame(parts), "members": pd.DataFrame(members).drop(columns="days"),
            "windows": pd.DataFrame(windows)}


# ---------- step 7: overfitting ----------

def effective_bets(book: Book, basis: str = "gross") -> dict:
    """(sum eig)^2 / sum eig^2 of the correlation of members' daily P&L, as in run_lessons.breadth."""
    held = book.pos.shift(1).fillna(0.0)
    pnl = (held * book.change).where(book.live)
    if basis == "net":
        cost = pd.DataFrame({r.name: r.results[v].cost for r, v in book.model.items}).where(book.live)
        pnl = pnl - cost
    pnl = pnl.loc[:, pnl.abs().sum() > 0]
    corr = pnl.corr(min_periods=250).fillna(0.0).to_numpy().copy()
    np.fill_diagonal(corr, 1.0)
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    return {"members": pnl.shape[1], "effective_bets": eig.sum() ** 2 / (eig**2).sum()}


def deflated(pnl: pd.Series, n_trials: int, trial_sd_ann: float) -> float:
    """Bailey-Lopez de Prado DSR with an explicit trial count and cross-trial Sharpe dispersion (annualised)."""
    fake = list(np.linspace(-1, 1, max(n_trials, 2)))
    fake = list(np.asarray(fake) / np.std(fake, ddof=1) * trial_sd_ann)  # only N and the s.d. enter the formula
    return metrics.deflated_sharpe(pnl, fake)
