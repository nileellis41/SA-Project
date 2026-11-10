"""Capital-level view of a model: exposure per leg (DV01), leg-level trades, cost re-pricing.

Notation, per member m and bar t (engine convention: the position set at the close of
t-1 is held over bar t):

    p_t        member position in risk units (signal / vol), saved by the backtest
    w_t        leg weights of one unit of the member on bar t (saved `legs_adaptive`)
    W_t        member's share of capital: sleeve weight on bar t / members live on bar t
    D_t        exposure held over bar t, per leg = W_t * p_{t-1} * w_t
               (fraction of capital per bp for yield and OAS legs; per unit for $/point legs)

Then the sleeve's gross P&L on bar t is sum_m sum_legs D_t * change_t, which `check_gross`
verifies against the backtest. A leg trade booked on bar t is D_{t+1} - D_t: every input
of D_{t+1} (W_{t+1} uses P&L through t, p_t, next-bar legs fitted before t+1) is known at t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.data.panel import changes
from amp_sa.portfolio.combine import Portfolio, combine
from analysis.portfolio_diagnostics.common import Model, legs_for

TREASURY = ("ust_2y", "ust_3y", "ust_5y", "ust_7y", "ust_10y", "ust_30y")
TV_SWITCH = pd.Timestamp("2008-01-01")


@dataclass
class Book:
    model: Model
    live: pd.DataFrame  # bool, members
    pos: pd.DataFrame  # p_t, 0 when not live
    w_member: pd.Series  # W_t
    legs: dict[str, pd.DataFrame]  # w_t per member, 0-filled, live bars only
    change: pd.DataFrame  # member change on bar t (units of the member), 0 when not live
    dv01: dict[str, pd.DataFrame]  # D_t per member

    @property
    def members(self) -> list[str]:
        return list(self.pos.columns)

    def series(self) -> list[str]:
        cols = []
        for d in self.dv01.values():
            cols += [c for c in d.columns if c not in cols]
        return cols

    def dv01_net(self) -> pd.DataFrame:
        """Netted book: exposure per leg summed over members."""
        cols = self.series()
        return sum(d.reindex(columns=cols, fill_value=0.0) for d in self.dv01.values())

    def dv01_gross(self) -> pd.DataFrame:
        """Sum over members of |exposure| per leg (before netting)."""
        cols = self.series()
        return sum(d.reindex(columns=cols, fill_value=0.0).abs() for d in self.dv01.values())


def build_book(model: Model, levels: pd.DataFrame, rolls: pd.DataFrame) -> Book:
    names = [r.name for r, _ in model.items]
    live = pd.DataFrame({r.name: r.live for r, _ in model.items}).fillna(False).astype(bool)
    n_live = live.sum(axis=1).replace(0, np.nan)
    w_member = (model.port.weights["s"] / n_live).fillna(0.0)
    pos = pd.DataFrame({r.name: r.results[v].position for r, v in model.items}).where(live, 0.0).fillna(0.0)
    held = pos.shift(1).fillna(0.0)
    legs, change, dv01 = {}, {}, {}
    for r, v in model.items:
        L = legs_for(r, v)
        cols = list(L.columns)
        Lw = L.where(live[r.name], axis=0).fillna(0.0)
        dlev = changes(levels[cols], rolls[cols])
        is_roll = rolls[cols].any(axis=1)  # the engine zeroes the member's change on any leg's roll bar
        ch = (Lw * dlev.fillna(0.0)).sum(axis=1).mask(is_roll, 0.0).where(live[r.name], 0.0)
        legs[r.name], change[r.name] = Lw, ch
        dv01[r.name] = Lw.mul(held[r.name] * w_member, axis=0)
    return Book(model, live[names], pos[names], w_member, legs, pd.DataFrame(change)[names], dv01)


def check_gross(book: Book) -> float:
    """Max |sum_m W_t p_{t-1} change_t - sleeve gross_t|; should be ~1e-15."""
    held = book.pos.shift(1).fillna(0.0)
    mine = (held * book.change).sum(axis=1).mul(book.w_member)
    port = book.model.port.gross
    ok = port.notna()
    return float((mine[ok] - port[ok]).abs().max())


def flat_cost(series: list[str], index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({c: config.COST_PER_UNIT[c] for c in series}, index=index)


def time_varying_cost(series: list[str], index: pd.Index) -> pd.DataFrame:
    """Treasury legs 0.5 bp before 2008 and 0.25 bp after; the 30y at 1.5x the other tenors. Other legs flat."""
    out = flat_cost(series, index)
    base = pd.Series(np.where(index < TV_SWITCH, 0.5, 0.25), index=index)
    for c in series:
        if c in TREASURY:
            out[c] = base * (1.5 if c == "ust_30y" else 1.0)
    return out


def leg_trades(book: Book, netted: bool) -> pd.DataFrame:
    """|D_{t+1} - D_t| per leg booked on bar t: netted across members, or summed member by member."""
    def step(d):
        return (d.shift(-1).fillna(d) - d).abs()
    cols = book.series()
    if netted:
        return step(book.dv01_net())
    return sum(step(d.reindex(columns=cols, fill_value=0.0)) for d in book.dv01.values())


def leg_cost(book: Book, netted: bool, cost_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cost per leg per bar, fraction of capital."""
    t = leg_trades(book, netted)
    c = flat_cost(list(t.columns), t.index) if cost_table is None else cost_table.reindex(columns=t.columns)
    return t * c


def reprice(model: Model, cost_fn) -> Portfolio:
    """Rebuild the sleeve with every member's cost re-priced leg by leg; positions unchanged.

    The engine charges cost_t = traded_t * u_t with u_t = sum_legs |w_t| c_leg, so a new
    cost table scales each bar's cost by u_new / u_old. The sleeve is then rebuilt exactly
    as run_sleeve1.sleeve does (vol target from net P&L, rescaling charged at hold cost).
    Bands were optimised at the flat 1x table and are not re-optimised.
    """
    gross, cost, hold = {}, {}, {}
    for r, v in model.items:
        res = r.results[v]
        L = legs_for(r, v).reindex(res.cost.index)
        cols = list(L.columns)
        u_flat = (L.abs() * flat_cost(cols, L.index)).sum(axis=1, min_count=1)
        u_new = (L.abs() * cost_fn(cols, L.index)).sum(axis=1, min_count=1)
        ratio = (u_new / u_flat).where(u_flat > 0).ffill().bfill()
        gross[r.name] = res.gross.where(r.live)
        cost[r.name] = (res.cost * ratio).where(r.live)
        hold[r.name] = (res.position.abs() * res.unit_cost * ratio).where(r.live)
    avg = lambda d: pd.DataFrame(d).mean(axis=1)  # noqa: E731
    return combine({"s": (avg(gross), avg(cost))}, budget={"s": 1.0}, use_overlay=False, hold_cost={"s": avg(hold)})
