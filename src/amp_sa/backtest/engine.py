"""Shared P&L accounting and cost model.

Timing convention for every sleeve: a signal computed from data through the
close of bar t sets the position held over bar t+1. P&L and cost are in
"risk units": positions are divided by a trailing volatility estimate, so a
position of 1 has roughly unit daily standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amp_sa import config


def trailing_vol(change: pd.Series, span: int = config.VOL_SPAN) -> pd.Series:
    """EW standard deviation of bar changes, floored at 25% of its own expanding median.

    Uses data through t only. The floor stops a quiet spell from levering a position up.
    """
    vol = change.ewm(span=span, min_periods=span // 2).std()
    return vol.clip(lower=0.25 * vol.expanding(min_periods=1).median())


@dataclass
class Result:
    """Per-bar accounting for one strategy, in risk units."""

    signal: pd.Series  # in [-1, 1], known at the close of t
    position: pd.Series  # units of the instrument held over t+1 (signal / vol)
    gross: pd.Series
    cost: pd.Series
    turnover: pd.Series  # |change in signal|, i.e. traded risk per bar
    unit_cost: pd.Series  # one-way cost per unit of the instrument

    @property
    def net(self) -> pd.Series:
        return self.gross - self.cost

    def net_at(self, cost_mult: float) -> pd.Series:
        return self.gross - cost_mult * self.cost


def account(
    signal: pd.Series,
    change: pd.Series,
    unit_cost: pd.Series | float,
    is_roll: pd.Series | None = None,
    vol: pd.Series | None = None,
) -> Result:
    """Turn a signal into P&L.

    signal:    desired exposure in [-1, 1] at the close of t; NaN means flat.
    change:    roll-free change of the instrument or spread over bar t.
    unit_cost: one-way cost per unit of the instrument, same units as `change`.
    is_roll:   bars where a leg rolled; a held position pays ROLL_COST_MULT x unit_cost.
    """
    vol = trailing_vol(change) if vol is None else vol
    signal = signal.reindex(change.index).fillna(0.0)
    position = (signal / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    held = position.shift(1).fillna(0.0)
    gross = held * change.fillna(0.0)
    traded = (position - held).abs()
    if is_roll is not None:
        traded = traded + config.ROLL_COST_MULT * held.abs() * is_roll.reindex(change.index).fillna(False)
    cost = traded * unit_cost
    unit_cost = unit_cost if isinstance(unit_cost, pd.Series) else pd.Series(float(unit_cost), index=change.index)
    return Result(signal, position, gross, cost, signal.diff().abs().fillna(0.0), unit_cost)
