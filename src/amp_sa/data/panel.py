"""Research panel: Bloomberg panel plus the Yahoo grain proxy, in trading units.

Also holds the one roll rule. Generic futures are unadjusted, so the change on
the bar where a generic switches contract is not a tradeable P&L. Those bars
are flagged and every sleeve sets the change to zero there (the position is
assumed rolled at a cost, see backtest/engine.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amp_sa.data.load_bbg import MAX_FFILL_BARS, PROCESSED_DIR
from amp_sa.data.load_extra import QUOTES_FILE, es_roll_bars, load_quotes
from amp_sa.data.load_yahoo import load_grains

# A generic switched contract when front open interest jumps by this factor.
OI_JUMP = 2.5

BP_COLUMNS = (
    "ust_2y", "ust_3y", "ust_5y", "ust_7y", "ust_10y", "ust_30y", "hy_oas", "ig_oas",
    "sofr_swap_2y", "sofr_swap_5y", "sofr_swap_10y",  # pre-2018 values are Bloomberg backfill
)
# Already in trading units: CDX HY spread in bp, CDX HY price in points, HYG in dollars.
AS_IS_COLUMNS = {"cdx_hy_5y_spread": "cdx_hy_spread", "cdx_hy_5y_price": "cdx_hy_price", "hyg_px": "hyg"}
# The generic CDX index switches to the new series around the 20th-27th of March and
# September. The exact bar is not in the data, so the whole window to month end is flagged.
CDX_ROLL_MONTHS = (3, 9)
CDX_ROLL_FROM_DAY = 20
OI_ROLLED = ("cl1", "cl2", "cl3", "co1", "co2", "ho1", "xb1", "kw1")
# Back generics switch contract on the same bar as the front one.
ROLLS_WITH = {"cl2": "cl1", "cl3": "cl1", "co2": "co1"}
# CBOT and KC grains share contract months and the last trade day, so the Yahoo
# chains roll with Bloomberg KW1 (Yahoo KE matches KW1 on 99.7% of days).
GRAIN_ROLL_SOURCE = "kw1"
GRAINS = ("zw1", "zc1", "ke1")
GALLONS_PER_BBL = 42


def detect_rolls(oi: pd.Series, jump: float = OI_JUMP) -> pd.Series:
    """True on bars where the generic starts quoting the next contract.

    Uses open interest at t and t-1 only, so the flag is known at the close of t.
    """
    oi = oi.where(oi > 0)
    return (oi / oi.shift(1) > jump).fillna(False)


def load_research_panel(
    bbg_path: str | Path = PROCESSED_DIR / "panel_daily.parquet",
    grains: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (levels, rolls). `rolls` is boolean with the same columns as `levels`."""
    bbg = pd.read_parquet(bbg_path)
    levels = pd.DataFrame(index=bbg.index)
    rolls = pd.DataFrame(index=bbg.index)

    for col in BP_COLUMNS:
        if col in bbg:
            levels[col] = bbg[col] * 100.0  # percent -> bp
    for col in ("vix", "move", "funding_rate"):
        if col in bbg:
            levels[col] = bbg[col]

    for root in OI_ROLLED:
        if f"{root}_settle" not in bbg:
            continue
        levels[root] = bbg[f"{root}_settle"]
        rolls[root] = detect_rolls(bbg[f"{root}_oi"])
    for back, front in ROLLS_WITH.items():
        if back in rolls and front in rolls:
            rolls[back] = rolls[front]

    cdx_window = levels.index.month.isin(CDX_ROLL_MONTHS) & (levels.index.day >= CDX_ROLL_FROM_DAY)
    for raw, name in AS_IS_COLUMNS.items():
        if raw in bbg:
            levels[name] = bbg[raw]
            if name.startswith("cdx"):
                rolls[name] = cdx_window

    # Daily ES and FX mids from the bid/ask export.
    if QUOTES_FILE.exists():
        mid, _ = load_quotes()
        for name in mid.columns:
            levels[name] = mid[name].reindex(levels.index).ffill(limit=MAX_FFILL_BARS)
        if "es1" in levels:
            rolls["es1"] = es_roll_bars(levels.index)

    if {"xb1", "ho1"} <= set(levels.columns):
        # XB and HO are quoted in cents per gallon; the basket is in $ per barrel of crude.
        levels["products_321"] = GALLONS_PER_BBL * (2 * levels["xb1"] + levels["ho1"]) / 3 / 100
        rolls["products_321"] = rolls["xb1"] | rolls["ho1"]

    if grains is None:
        try:
            grains = load_grains()
        except Exception:  # no cache and no network: the panel runs without grains
            grains = pd.DataFrame()
    if GRAIN_ROLL_SOURCE in rolls:
        base = rolls[GRAIN_ROLL_SOURCE]
        # Yahoo may switch a bar later than Bloomberg: flag the following bar too.
        # (Flagging the bar before would use the next bar's open interest.)
        grain_roll = base | base.shift(1, fill_value=False)
        first_flaggable = bbg[f"{GRAIN_ROLL_SOURCE}_oi"].first_valid_index()
        for root in GRAINS:
            col = f"{root}_settle"
            if col in grains:
                series = grains[col].reindex(levels.index).ffill(limit=MAX_FFILL_BARS)
                levels[root] = series.where(levels.index >= first_flaggable)  # no roll flags before this
                rolls[root] = grain_roll

    rolls = rolls.reindex(columns=levels.columns, fill_value=False).astype(bool)
    return levels, rolls


def changes(levels: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
    """Bar-to-bar differences with roll bars set to zero. Differences, never log returns."""
    return levels.diff().mask(rolls, 0.0)


def adjusted_levels(levels: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
    """Difference-adjusted continuous series: the cumulative sum of roll-free changes."""
    d = changes(levels, rolls)
    return d.cumsum().where(levels.notna())
