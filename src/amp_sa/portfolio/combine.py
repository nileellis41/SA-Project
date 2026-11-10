"""Portfolio layer: sleeve aggregation, risk budget, volatility target, drawdown overlay.

Every scaling factor applied to bar t is computed from P&L through bar t-1.
Returns are additive fractions of a fixed capital base (no compounding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.config import ANN

SLEEVE_VOL_SPAN = 120
MIN_VOL_BARS = 60


def risk_estimate(pnl: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Volatility used to size bar t, from P&L through t-1: the larger of recent and long-run.

    A mean-reversion sleeve is flat much of the time, so its recent volatility
    collapses and a recent-only estimate would lever it up several times over
    just as it re-enters. Taking the larger of the two means exposure is cut in
    turbulent spells but never raised above what long-run volatility supports.
    """
    recent = pnl.ewm(span=SLEEVE_VOL_SPAN, min_periods=MIN_VOL_BARS).std()
    long_run = pnl.expanding(min_periods=MIN_VOL_BARS).std()
    return np.maximum(recent, long_run).shift(1)


def sleeve_pnl(runs, variant: str, cost_mult: float = 1.0, groups=None) -> pd.Series:
    """Equal-weight average of the live strategies' P&L, in risk units. NaN before any is live."""
    cols = {r.name: r.net(variant, cost_mult) for r in runs if groups is None or r.group in groups}
    return pd.DataFrame(cols).mean(axis=1, skipna=True)


def sleeve_parts(runs, variant: str, groups=None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(gross, cost, turnover), each the equal-weight average over live strategies."""
    pick = [r for r in runs if groups is None or r.group in groups]
    avg = lambda attr: pd.DataFrame({r.name: getattr(r.results[variant], attr).where(r.live) for r in pick}).mean(axis=1)  # noqa: E731
    return avg("gross"), avg("cost"), avg("turnover")


def sleeve_hold_cost(runs, variant: str, groups=None) -> pd.Series:
    """Cost of putting on one risk unit of the sleeve as it stands: mean |position| x unit cost."""
    pick = [r for r in runs if groups is None or r.group in groups]
    cols = {r.name: (r.results[variant].position.abs() * r.results[variant].unit_cost).where(r.live) for r in pick}
    return pd.DataFrame(cols).mean(axis=1)


def drawdown_multiplier(pnl: pd.Series, steps=config.DD_STEPS) -> pd.Series:
    """Exposure multiplier for bar t from the drawdown of cumulative P&L through t-1."""
    equity = pnl.fillna(0).cumsum()
    dd = (equity.cummax() - equity).shift(1).fillna(0.0)
    mult = pd.Series(1.0, index=pnl.index)
    for threshold, m in sorted(steps):
        mult = mult.where(dd < threshold, m)
    return mult


@dataclass
class Portfolio:
    weights: pd.DataFrame  # multiplier applied to each sleeve's risk-unit P&L at bar t (already lagged)
    gross: pd.Series
    cost: pd.Series
    sleeve_net: pd.DataFrame  # each sleeve's contribution, net at 1x cost
    overlay: pd.Series

    @property
    def net(self) -> pd.Series:
        return self.gross - self.cost

    def net_at(self, cost_mult: float) -> pd.Series:
        return self.gross - cost_mult * self.cost


def combine(
    sleeves: dict[str, tuple[pd.Series, pd.Series]],
    budget: dict[str, float] | pd.DataFrame = config.SLEEVE_RISK_BUDGET,
    target_vol: float = config.TARGET_VOL,
    use_overlay: bool = True,
    hold_cost: dict[str, pd.Series] | None = None,
) -> Portfolio:
    """sleeves: name -> (gross, cost) in risk units.

    1. Each sleeve is scaled to its share of risk with its own trailing volatility.
    2. The sum is scaled to `target_vol` with the trailing volatility of the mix.
    3. The drawdown overlay cuts exposure after losses.

    budget:    fixed shares, or a frame of shares per bar. Row t of a frame must
               already be lagged: it sizes bar t and may use data through t-1 only.
    hold_cost: name -> cost of establishing one risk unit of the sleeve's current
               positions. When given, every change in a sleeve's weight is charged
               at it, so re-scaling and regime switches are not free.
    """
    names = list(sleeves)
    gross = pd.DataFrame({k: v[0] for k, v in sleeves.items()})
    cost = pd.DataFrame({k: v[1] for k, v in sleeves.items()})
    net = gross - cost

    sleeve_vol = risk_estimate(net)
    if isinstance(budget, pd.DataFrame):
        shares = budget.reindex(index=net.index, columns=names)
    else:
        shares = pd.DataFrame({k: budget[k] for k in names}, index=net.index)
    scale = (shares / sleeve_vol).replace([np.inf, -np.inf], np.nan)
    mix = (scale * net).sum(axis=1, min_count=1)

    mix_vol = risk_estimate(mix)
    # Cap: never lever as if the mix were quieter than 1/MAX_LEVERAGE_SCALE of a unit-risk sleeve.
    lever = (target_vol / np.sqrt(ANN) / mix_vol).clip(upper=config.MAX_LEVERAGE_SCALE * target_vol / np.sqrt(ANN))
    lever = lever.replace([np.inf, -np.inf], np.nan)
    pre_overlay = lever * mix

    overlay = drawdown_multiplier(pre_overlay) if use_overlay else pd.Series(1.0, index=mix.index)
    weights = scale.mul(lever * overlay, axis=0)
    live = weights.notna().any(axis=1)
    port_gross = (weights * gross).sum(axis=1, min_count=1).where(live)
    port_cost = (weights * cost).sum(axis=1, min_count=1).where(live)
    if hold_cost is not None:
        # Positions held into bar t were sized at t-1; a weight change re-trades that fraction of them.
        held = pd.DataFrame({k: hold_cost[k] for k in names}).shift(1)
        port_cost = port_cost + (weights.fillna(0.0).diff().abs() * held).sum(axis=1).where(live, 0.0)
    return Portfolio(weights, port_gross, port_cost, weights * net, overlay)
