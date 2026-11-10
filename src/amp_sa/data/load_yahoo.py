"""Yahoo Finance front-month grain futures, used as a research proxy.

Bloomberg has no CBOT wheat or corn history in the exports and the AMP
terminal only holds the live contracts, so ZW/ZC/KE come from Yahoo's
unadjusted front-month chains. Each ticker is downloaded once and cached in
data/raw/yahoo/; delete the file to refresh it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amp_sa.data.load_bbg import RAW_DIR

YAHOO_DIR = RAW_DIR / "yahoo"

# Yahoo ticker -> clean prefix. Prices are US cents per bushel.
YAHOO_TICKERS = {"ZW=F": "zw1", "ZC=F": "zc1", "KE=F": "ke1"}


def fetch(ticker: str, out_dir: Path = YAHOO_DIR) -> pd.DataFrame:
    """Daily close and volume for one ticker, from the cache if present."""
    path = out_dir / f"{ticker.replace('=', '_')}.csv"
    if path.exists():
        return pd.read_csv(path, index_col="date", parse_dates=True)
    import yfinance as yf

    raw = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    if raw.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}")
    out = raw[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
    out.index = pd.DatetimeIndex(out.index.tz_localize(None).normalize(), name="date")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return out


def load_grains(out_dir: Path = YAHOO_DIR) -> pd.DataFrame:
    """Wide frame: <prefix>_settle and <prefix>_volume per ticker. Non-positive prices are NaN."""
    cols = {}
    for ticker, prefix in YAHOO_TICKERS.items():
        frame = fetch(ticker, out_dir)
        cols[f"{prefix}_settle"] = frame["close"].where(frame["close"] > 0)
        cols[f"{prefix}_volume"] = frame["volume"]
    return pd.DataFrame(cols)


if __name__ == "__main__":
    grains = load_grains()
    print(grains.describe().T.to_string())
    print(grains.apply(pd.Series.first_valid_index).to_string())
