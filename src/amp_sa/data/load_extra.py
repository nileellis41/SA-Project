"""The two Bloomberg exports that sit outside data/raw: daily ES and FX quotes, and intraday FX.

pxbid.csv       Layout B. Daily PX_BID / PX_ASK for ES1 and three FX pairs from 2000.
                Used as mid prices; the bid/ask also sets each instrument's cost.
intraminfx.csv  One-minute closes for the same FX pairs, about six months.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amp_sa.data.load_bbg import PROJECT_ROOT, CleaningLog, read_layout_b

QUOTES_FILE = PROJECT_ROOT / "pxbid.csv"
INTRADAY_FILE = PROJECT_ROOT / "intraminfx.csv"

QUOTE_NAMES = {"ES1 Index": "es1", "EURUSD Curncy": "eurusd", "USDJPY Curncy": "usdjpy", "USDCAD Curncy": "usdcad"}
# A quote wider than this fraction of the mid is stale or indicative, not a market.
MAX_RELATIVE_SPREAD = 0.005
INTRADAY_BAR = "5min"


def load_quotes(path: str | Path = QUOTES_FILE, log: CleaningLog | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (mid, half_spread), daily, one column per instrument. Bad quotes are NaN."""
    log = log if log is not None else CleaningLog()
    long = read_layout_b(path, log)
    wide = long.pivot_table(index="date", columns=["ticker", "field"], values="value")
    mid, half = {}, {}
    for ticker, name in QUOTE_NAMES.items():
        if ticker not in wide.columns.get_level_values(0):
            continue
        bid, ask = wide[(ticker, "PX_BID")], wide[(ticker, "PX_ASK")]
        m = (bid + ask) / 2
        bad = (ask < bid) | ((ask - bid) / m > MAX_RELATIVE_SPREAD) | (m <= 0)
        if bad.any():
            log.add(Path(path).name, "coerce_to_nan", ticker, f"{int(bad.sum())} crossed, non-positive or wide quotes")
        mid[name] = m.mask(bad)
        half[name] = ((ask - bid) / 2).mask(bad)
    return pd.DataFrame(mid), pd.DataFrame(half)


def load_intraday(path: str | Path = INTRADAY_FILE, bar: str = INTRADAY_BAR) -> pd.DataFrame:
    """Intraday FX closes resampled to `bar` (last print in each bar), one column per pair."""
    raw = pd.read_csv(path, header=None, skiprows=4, dtype=str, encoding="utf-8-sig")
    header = pd.read_csv(path, header=None, skiprows=2, nrows=1, dtype=str, encoding="utf-8-sig").iloc[0]
    cols = {}
    for j in range(0, raw.shape[1] - 1, 3):
        name = QUOTE_NAMES.get(str(header.iloc[j]).strip())
        if name is None:
            continue
        stamp = pd.to_datetime(raw[j], format="%m/%d/%Y %H:%M", errors="coerce")
        value = pd.to_numeric(raw[j + 1], errors="coerce")
        ok = stamp.notna() & value.notna()
        series = pd.Series(value[ok].to_numpy(), index=stamp[ok].to_numpy())
        cols[name] = series[~series.index.duplicated(keep="last")].sort_index()
    minute = pd.DataFrame(cols)
    out = minute.resample(bar).last().dropna(how="all")
    out.index.name = "date"
    return out


def es_roll_bars(index: pd.DatetimeIndex) -> pd.Series:
    """ES expires on the third Friday of Mar/Jun/Sep/Dec; the generic switches on the next bar.

    The roll bar is fixed by the calendar, so the flag uses no market data at all.
    """
    days = pd.Series(index, index=index)
    quarter_month = days.dt.month.isin([3, 6, 9, 12])
    third_friday = quarter_month & (days.dt.dayofweek == 4) & days.dt.day.between(15, 21)
    after = third_friday.shift(1, fill_value=False)
    # If the Friday itself is a holiday the bar after the 21st is the first of the new contract.
    return (after | (quarter_month & (days.dt.day > 21) & (days.shift(1).dt.day <= 14))).astype(bool)


if __name__ == "__main__":
    log = CleaningLog()
    mid, half = load_quotes(log=log)
    print(log.to_frame().to_string(index=False))
    print(pd.DataFrame({"first": mid.apply(pd.Series.first_valid_index), "n": mid.notna().sum(),
                        "last": mid.iloc[-1], "median_half_spread_5y": half.iloc[-1260:].median(),
                        "daily_vol": mid.diff().iloc[-1260:].std()}).to_string())
    intraday = load_intraday()
    print(intraday.shape, intraday.index[0], intraday.index[-1])
    print(intraday.notna().sum().to_string(), "\nbars/day", np.round(len(intraday) / intraday.index.normalize().nunique()))
    print("5-min vol", intraday.diff().std().to_string())
