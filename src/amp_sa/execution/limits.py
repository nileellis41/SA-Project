"""Pre-trade guardrails for the AMP account. Pure functions, no MetaTrader5 import.

AMP's published account rule is a forced liquidation (with a per-contract
fee) once the day's loss reaches 80% of net liquidity, plus exchange margin
and per-symbol position limits. Every limit below sits far inside those.

The margins in the MT5 symbol table are AMP's day-trade margins. This book
holds positions overnight, which needs full exchange margin, so day margins
are multiplied by OVERNIGHT_MARGIN_MULT. That multiplier is an assumption
until the overnight figures are confirmed with AMP.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

AMP_DAILY_LOSS_LIQUIDATION = 0.80
OVERNIGHT_MARGIN_MULT = 4.0


@dataclass(frozen=True)
class Limits:
    max_margin_frac: float = 0.25  # overnight margin used / equity
    max_daily_loss_frac: float = 0.10  # stop opening risk for the day; AMP liquidates at 0.80
    max_stress_loss_frac: float = 0.10  # loss if every position moves 3 daily sigmas against it
    max_contracts_per_symbol: int = 3
    max_orders_per_run: int = 10


def check(
    targets: pd.Series,
    current: pd.Series,
    specs: pd.DataFrame,
    daily_dollar_vol: pd.Series,
    equity: float,
    day_pnl: float = 0.0,
    limits: Limits = Limits(),
) -> list[str]:
    """Violations of the limits for moving from `current` to `targets` (contracts by symbol).

    specs: indexed by symbol with a `margin_initial` column (AMP day margin, $ per contract).
    daily_dollar_vol: one-sigma daily P&L of ONE contract, in $.
    An empty list means the order set may be sent.
    """
    targets = targets[targets != 0]
    orders = targets.sub(current, fill_value=0.0)
    orders = orders[orders != 0]
    out = []

    unknown = [s for s in targets.index if s not in specs.index]
    if unknown:
        out.append(f"no AMP spec for {unknown}")
        return out
    if equity <= 0:
        return ["equity is not positive"]

    if day_pnl <= -limits.max_daily_loss_frac * equity and (orders.abs() > 0).any():
        opening = [s for s, q in orders.items() if abs(targets.get(s, 0.0)) > abs(current.get(s, 0.0))]
        if opening:
            out.append(
                f"day loss {day_pnl:.0f} is past {limits.max_daily_loss_frac:.0%} of equity; "
                f"no opening orders ({opening})"
            )

    margin = float((targets.abs() * specs.loc[targets.index, "margin_initial"] * OVERNIGHT_MARGIN_MULT).sum())
    if margin > limits.max_margin_frac * equity:
        out.append(f"overnight margin {margin:.0f} exceeds {limits.max_margin_frac:.0%} of equity {equity:.0f}")

    stress = float((targets.abs() * daily_dollar_vol.reindex(targets.index) * 3).sum())
    if stress > limits.max_stress_loss_frac * equity:
        out.append(f"3-sigma stress loss {stress:.0f} exceeds {limits.max_stress_loss_frac:.0%} of equity")

    big = targets[targets.abs() > limits.max_contracts_per_symbol]
    if len(big):
        out.append(f"more than {limits.max_contracts_per_symbol} contracts in {list(big.index)}")
    if len(orders) > limits.max_orders_per_run:
        out.append(f"{len(orders)} orders exceeds {limits.max_orders_per_run} per run")
    return out
