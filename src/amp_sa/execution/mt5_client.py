"""AMP MT5 terminal client. The only module that imports MetaTrader5.

Everything here is read-only: specs, quotes and daily bars. No order is ever
sent and Market Watch is never modified (no symbol_select), so the quote
subscription count on the account does not change.

Data pulls are deliberately light:
  * daily bars only,
  * only symbols whose history is already in the terminal's local cache
    (no new downloads from the AMP history server),
  * one request per symbol, throttled, capped per run,
  * every result is written to data/raw/amp/ and never requested again.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AMP_DIR = PROJECT_ROOT / "data" / "raw" / "amp"
TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH", "")  # full path to the AMP terminal64.exe

THROTTLE_SECONDS = 0.3
MAX_SYMBOLS_PER_RUN = 150
MAX_BARS = 5000

SPEC_FIELDS = (
    "description", "path", "trade_tick_size", "trade_tick_value", "trade_contract_size", "point",
    "margin_initial", "margin_maintenance", "bid", "ask", "last", "volume_min", "volume_step",
    "expiration_time", "select",
)


@contextmanager
def connect(path: str = TERMINAL_PATH, allow_live: bool = False):
    """Attach to the running terminal. Refuses a live account unless allow_live."""
    import MetaTrader5 as mt5

    if not (mt5.initialize(path=path) if path else mt5.initialize()):
        raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        if account is None:
            raise ConnectionError("terminal is not logged in")
        if account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and not allow_live:
            raise PermissionError(f"{account.server} is not a demo account")
        yield mt5
    finally:
        mt5.shutdown()


def account_snapshot(mt5) -> dict:
    a = mt5.account_info()
    positions = mt5.positions_get() or ()
    return {
        "server": a.server,
        "balance": a.balance,
        "equity": a.equity,
        "margin": a.margin,
        "margin_free": a.margin_free,
        "positions": {p.symbol: (p.volume if p.type == 0 else -p.volume) for p in positions},
    }


def cached_symbols(mt5) -> set[str]:
    """Symbols with history files already on disk for this server."""
    history = Path(mt5.terminal_info().data_path) / "bases" / mt5.account_info().server / "history"
    return {d.name for d in history.iterdir() if d.is_dir() and any(d.glob("*.hcc"))}


def snapshot_specs(mt5, symbols: list[str], out_dir: Path = AMP_DIR) -> pd.DataFrame:
    """Contract specs, margins and the current quote. Reads the terminal's symbol table only."""
    rows = {}
    for name in symbols:
        info = mt5.symbol_info(name)
        if info is not None:
            rows[name] = {f: getattr(info, f) for f in SPEC_FIELDS}
    specs = pd.DataFrame.from_dict(rows, orient="index")
    specs.index.name = "symbol"
    specs["snapshot_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    specs["expiration"] = pd.to_datetime(specs.pop("expiration_time"), unit="s")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs.to_parquet(out_dir / "specs.parquet")
    return specs


def pull_daily(mt5, symbols: list[str], out_dir: Path = AMP_DIR) -> pd.DataFrame:
    """Daily bars for cached, already-selected symbols. Returns a log of what happened."""
    bars_dir = out_dir / "d1"
    bars_dir.mkdir(parents=True, exist_ok=True)
    local = cached_symbols(mt5)
    log, n_requests = [], 0
    for name in symbols:
        target = bars_dir / f"{name.replace('@', '_at_')}.parquet"
        if target.exists():
            log.append((name, "skip", "already on disk"))
            continue
        if name not in local:
            log.append((name, "skip", "not in the local cache; would need a server download"))
            continue
        info = mt5.symbol_info(name)
        if info is None or not info.select:
            log.append((name, "skip", "not in Market Watch; not selecting it"))
            continue
        if n_requests >= MAX_SYMBOLS_PER_RUN:
            log.append((name, "skip", f"per-run cap of {MAX_SYMBOLS_PER_RUN} reached"))
            continue
        rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_D1, 0, MAX_BARS)
        n_requests += 1
        time.sleep(THROTTLE_SECONDS)
        if rates is None or len(rates) == 0:
            log.append((name, "empty", str(mt5.last_error())))
            continue
        bars = pd.DataFrame(rates)
        bars["date"] = pd.to_datetime(bars.pop("time"), unit="s").dt.normalize()
        bars = bars.set_index("date")[["open", "high", "low", "close", "tick_volume", "real_volume", "spread"]]
        bars.to_parquet(target)
        log.append((name, "ok", f"{len(bars)} bars {bars.index[0].date()} to {bars.index[-1].date()}"))
    return pd.DataFrame(log, columns=["symbol", "status", "detail"])


def load_daily(name: str, out_dir: Path = AMP_DIR) -> pd.DataFrame:
    """Read a pulled symbol from disk. Does not touch the terminal."""
    return pd.read_parquet(out_dir / "d1" / f"{name.replace('@', '_at_')}.parquet")


# Roots the book maps to: micros it would trade, the full-size contracts used to
# validate the research proxies, and the AMP continuous series.
PULL_ROOTS = ("ZCE", "ZWA", "CLE", "MCLE", "MZC", "MZW", "XC", "XW", "HOE", "RBE", "Z2YY", "Z5YY", "Z10Y", "Z30Y")
PULL_CONTINUOUS = ("@CLE", "@USA", "@ZSE")


def main() -> None:
    import re

    pattern = re.compile(rf"^({'|'.join(PULL_ROOTS)})[FGHJKMNQUVXZ]\d{{2}}$")
    with connect() as mt5:
        local = cached_symbols(mt5)
        symbols = sorted(s for s in local if pattern.match(s)) + [s for s in PULL_CONTINUOUS if s in local]
        specs = snapshot_specs(mt5, symbols)
        log = pull_daily(mt5, symbols)
        account = account_snapshot(mt5)
    log.to_csv(AMP_DIR / "pull_log.csv", index=False)
    print(account)
    print(f"specs: {len(specs)} symbols")
    print(log["status"].value_counts().to_string())
    print(log[log["status"] != "ok"].to_string(index=False))


if __name__ == "__main__":
    main()
