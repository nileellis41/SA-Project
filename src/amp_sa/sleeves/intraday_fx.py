"""Intraday FX study: the Sleeve 1 model on five-minute bars.

The three pairs share the dollar leg, so each is regressed on the other two
and the residual is traded with exactly the same walk-forward machinery as the
daily spreads (cointegration screen, Kalman hedge, OU bands, regime gates).
Only the bar calendar, the fold lengths and the costs differ.

It is reported on its own and is not part of the book: the sample is about six
months, and the bars are not on the daily calendar the other sleeves share.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest import metrics
from amp_sa.config import Spread
from amp_sa.data.load_extra import INTRADAY_FILE, load_intraday
from amp_sa.sleeves import rv

SPREADS = (
    Spread("fx5m_eurusd", "fx_intraday", "eurusd", ("usdjpy", "usdcad")),
    Spread("fx5m_usdjpy", "fx_intraday", "usdjpy", ("eurusd", "usdcad")),
    Spread("fx5m_usdcad", "fx_intraday", "usdcad", ("eurusd", "usdjpy")),
)
# One-way cost per unit of price: about half a pip, an institutional-size spot spread.
# The daily export's composite quotes are several times wider and are not used here.
INTRADAY_COST = {"eurusd": 0.00005, "usdjpy": 0.007, "usdcad": 0.00007}
BARS_PER_DAY = 243  # observed average of five-minute bars per trading day
TRAIN_DAYS, TEST_DAYS = 10, 5


@contextmanager
def intraday_schedule():
    """Swap in the intraday fold lengths and costs, and restore the daily ones afterwards."""
    saved = (config.TRAIN_BARS, config.TEST_BARS, dict(config.COST_PER_UNIT))
    config.TRAIN_BARS, config.TEST_BARS = TRAIN_DAYS * BARS_PER_DAY, TEST_DAYS * BARS_PER_DAY
    config.COST_PER_UNIT.update(INTRADAY_COST)
    try:
        yield
    finally:
        config.TRAIN_BARS, config.TEST_BARS = saved[0], saved[1]
        config.COST_PER_UNIT.clear()
        config.COST_PER_UNIT.update(saved[2])


def intraday_panel(daily_levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five-minute FX closes plus the PREVIOUS day's VIX and MOVE as regime features."""
    bars = load_intraday().dropna()
    prior_day = daily_levels[["vix", "move"]].shift(1)  # yesterday's close is all that is known intraday
    day = bars.index.normalize()
    for col in ("vix", "move"):
        bars[col] = prior_day[col].reindex(prior_day.index.union(day.unique())).ffill().reindex(day).to_numpy()
    rolls = pd.DataFrame(False, index=bars.index, columns=bars.columns)
    return bars, rolls


def run(daily_levels: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Returns (runs, table). Sharpe is computed on P&L summed to calendar days."""
    if not INTRADAY_FILE.exists():
        return [], pd.DataFrame()
    bars, rolls = intraday_panel(daily_levels)
    with intraday_schedule():
        runs = [r for r in (rv.run_spread(s, bars, rolls) for s in SPREADS) if r is not None]
    rows = []
    for r in runs:
        for variant in rv.VARIANTS:
            res = r.results[variant]
            daily = lambda s: s.where(r.live).groupby(s.index.normalize()).sum(min_count=1).dropna()  # noqa: E731
            net, gross = daily(res.net), daily(res.gross)
            rows.append({
                "spread": r.name, "variant": variant, "oos_days": len(net),
                "sharpe_net": metrics.sharpe(net), "sharpe_gross": metrics.sharpe(gross),
                "in_market": float((res.signal != 0).where(r.live).mean()),
                "round_trips_per_day": float(res.turnover.where(r.live).mean() * BARS_PER_DAY / 2),
                "coint_pass": float(r.diagnostics["coint_pass"].mean()),
                "median_half_life_bars": float(np.nanmedian(r.diagnostics["half_life_adaptive"])),
            })
    return runs, pd.DataFrame(rows)
