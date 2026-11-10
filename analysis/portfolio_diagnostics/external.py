"""External calendars: Fed funds target (FRED) and Treasury coupon auction dates (TreasuryDirect).

Raw downloads are cached under data/raw/fred/ and data/raw/treasury/ (gitignored):
    fred/DFEDTAR.csv, DFEDTARU.csv, DFEDTARL.csv    fredgraph.csv?id=<series>
    treasury/auctions_Note.json, auctions_Bond.json  TA_WS/securities/search?type=<Note|Bond>&format=json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
TARGET_SWITCH = pd.Timestamp("2008-12-16")  # FRED: single target to 2008-12-15, range from 2008-12-16
AUCTION_TENORS = {"2-Year": "ust_2y", "3-Year": "ust_3y", "5-Year": "ust_5y", "7-Year": "ust_7y",
                  "10-Year": "ust_10y", "30-Year": "ust_30y"}
REGIME_BARS = 126  # trailing window for the hiking / cutting / on-hold label


def fed_target() -> pd.Series:
    """Fed funds target, %: DFEDTAR to 2008-12-15, the upper bound DFEDTARU after."""
    read = lambda s: pd.read_csv(RAW / "fred" / f"{s}.csv", parse_dates=["observation_date"], index_col=0)[s]  # noqa: E731
    old, upper = read("DFEDTAR"), read("DFEDTARU")
    out = pd.concat([old[old.index < TARGET_SWITCH], upper[upper.index >= TARGET_SWITCH]])
    return pd.to_numeric(out, errors="coerce").dropna().rename("fed_target")


def fed_regime(index: pd.DatetimeIndex, bars: int = REGIME_BARS) -> pd.Series:
    """'hiking' / 'cutting' / 'on hold' from the change in target over the trailing `bars` bars, known at t."""
    target = fed_target().reindex(index, method="ffill")
    change = target - target.shift(bars)
    return pd.Series(np.select([change > 0, change < 0], ["hiking", "cutting"], "on hold"),
                     index=index).where(change.notna())


def auctions() -> pd.DataFrame:
    """Nominal coupon auctions (TIPS and FRNs dropped), reopenings kept: date, tenor, reopening."""
    frames = []
    for kind in ("Note", "Bond"):
        d = pd.DataFrame(json.load(open(RAW / "treasury" / f"auctions_{kind}.json")))
        d = d[(d["inflationProtected"] != "Yes") & (d["floatingRate"] != "Yes")]
        d = d[d["originalSecurityTerm"].isin(AUCTION_TENORS)]
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(d["competitiveAuctionCloseDate"].str[:10]),
            "tenor": d["originalSecurityTerm"].map(AUCTION_TENORS),
            "reopening": d["reopening"].eq("Yes"),
        }))
    return pd.concat(frames).drop_duplicates().sort_values("date").reset_index(drop=True)


def auction_week(index: pd.DatetimeIndex, tenor: str = "ust_5y") -> pd.Series:
    """True on bars in the same Monday-Friday week as an auction of `tenor`. Auction dates are announced in advance."""
    a = auctions()
    weeks = set(a.loc[a["tenor"] == tenor, "date"].dt.to_period("W-FRI"))
    return pd.Series(index.to_period("W-FRI").isin(weeks), index=index)


if __name__ == "__main__":
    t = fed_target()
    print("fed target", t.index[0].date(), t.index[-1].date(), len(t))
    a = auctions()
    print(a.groupby("tenor")["date"].agg(["count", "min", "max"]))
    a.to_csv(RAW / "treasury" / "coupon_auctions.csv", index=False)
