"""Bloomberg CSV loaders and the clean daily panel.

Two export layouts are supported (see README.md and the notes below):
  Layout A: one (date, value) column pair per series, each with its own dates.
  Layout B: one shared date column, then a block of field columns per ticker.

Both readers return a long frame: date, ticker, field, value. Everything that
is dropped, truncated, renamed or filled is recorded in a CleaningLog.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LAYOUT_A_FILES = ("rates_1.csv", "credit.csv", "Features_Costs_1.csv")
LAYOUT_B_FILES = ("930.csv",)
SPECS_FILE = "pnt.csv"

DATE_FORMAT = "%m/%d/%Y"
NA_PREFIX = "#N/A"

# Wrong tickers in the exports: all NaN, never load.
BAD_TICKERS = frozenset({"W11 Comdty", "W111 Comdty", "W112 Comdty"})

# Series that are monthly before this date and daily after.
TRUNCATE_BEFORE = {
    "LF98OAS Index": pd.Timestamp("2000-08-16"),
    "LUACOAS Index": pd.Timestamp("2000-08-16"),
}

# US0003M before this date, SOFRRATE on or after.
FUNDING_SWITCH_DATE = pd.Timestamp("2018-04-02")

MAX_FFILL_BARS = 3

_FUTURES_930 = ("CL1", "CL2", "CL3", "CO1", "CO2", "XB1", "HO1", "KW1")
_FUTURES_FIELDS = {"PX_SETTLE": "settle", "VOLUME": "volume", "OPEN_INT": "oi"}

# (Bloomberg ticker, field) -> clean panel column. Dict order is panel column order.
COLUMN_MAP: dict[tuple[str, str], str] = {
    # Treasury cash yields, %
    ("USGG2YR Index", "PX_LAST"): "ust_2y",
    ("USGG3YR Index", "PX_LAST"): "ust_3y",
    ("USGG5YR Index", "PX_LAST"): "ust_5y",
    ("USGG7YR Index", "PX_LAST"): "ust_7y",
    ("USGG10YR Index", "PX_LAST"): "ust_10y",
    ("USGG30YR Index", "PX_LAST"): "ust_30y",
    # Treasury futures, unadjusted generics: never compute returns on these
    ("TU1 Comdty", "PX_LAST"): "tu1_last",
    ("FV1 Comdty", "PX_LAST"): "fv1_last",
    ("TY1 Comdty", "PX_LAST"): "ty1_last",
    ("US1 Comdty", "PX_LAST"): "us1_last",
    # SOFR swaps (pre-2018 values are backfilled) and funding
    ("USOSFR2 Curncy", "PX_LAST"): "sofr_swap_2y",
    ("USOSFR5 Curncy", "PX_LAST"): "sofr_swap_5y",
    ("USOSFR10 Curncy", "PX_LAST"): "sofr_swap_10y",
    ("SOFRRATE Index", "PX_LAST"): "sofr",
    ("US0003M Index", "PX_LAST"): "libor_3m",
    # Credit. The SPRD ticker is exported under a "CDX IG" label; it is CDX HY.
    ("CDX HY CDSI GEN 5Y PRC Corp", "PX_LAST"): "cdx_hy_5y_price",
    ("CDX HY CDSI GEN 5Y SPRD Corp", "PX_LAST"): "cdx_hy_5y_spread",
    ("LF98OAS Index", "PX_LAST"): "hy_oas",
    ("LUACOAS Index", "PX_LAST"): "ig_oas",
    ("HYG US Equity", "PX_LAST"): "hyg_px",
    ("HYG US Equity", "FUND_NET_ASSET_VAL"): "hyg_nav",
    ("HYG US Equity", "EQY_SH_OUT"): "hyg_shares_out",
    ("HYG US Equity", "PX_BID"): "hyg_bid",
    ("HYG US Equity", "PX_ASK"): "hyg_ask",
    # Regime features and cost inputs
    ("VIX Index", "PX_LAST"): "vix",
    ("MOVE Index", "PX_LAST"): "move",
    ("TY1 Comdty", "PX_BID"): "ty1_bid",
    ("TY1 Comdty", "PX_ASK"): "ty1_ask",
    # Energy and wheat generics: settle for signals
    **{
        (f"{root} Comdty", bbg_field): f"{root.lower()}_{suffix}"
        for root in _FUTURES_930
        for bbg_field, suffix in _FUTURES_FIELDS.items()
    },
}

# Panel rows are the business days where at least one of these prints. These come
# from Layout A, where each series carries its own dates. ust_3y and ust_7y are
# excluded: the Layout B export is pre-filled on every weekday, holidays included.
CALENDAR_COLUMNS = ("ust_2y", "ust_5y", "ust_10y", "ust_30y")

# A Layout B futures row that repeats all of these from the prior row is a fill, not a print.
STALE_FIELDS = ("PX_SETTLE", "VOLUME", "OPEN_INT")

# Flow variables: a stale value is wrong, so these are never forward filled.
NO_FFILL_SUFFIXES = ("_volume",)

SPEC_DATE_FIELDS = ("LAST_TRADEABLE_DT", "FUT_NOTICE_FIRST")
SPEC_TEXT_FIELDS = ("FUT_CUR_GEN_TICKER", "CRNCY")

LONG_COLUMNS = ["date", "ticker", "field", "value"]


@dataclass
class CleaningLog:
    """Every drop, truncation, rename and fill applied to the raw data."""

    entries: list[dict] = field(default_factory=list)

    def add(self, source: str, action: str, target: str, detail: str = "") -> None:
        self.entries.append(
            {"source": source, "action": action, "target": target, "detail": detail}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries, columns=["source", "action", "target", "detail"])


def _read_rows(path: Path) -> list[list[str]]:
    """Read a CSV as padded rows of stripped strings. utf-8-sig drops a BOM if present."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f)]
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def _clean_ticker(raw: str) -> str:
    return " ".join(raw.split())


def _to_number(raw: pd.Series) -> tuple[pd.Series, int]:
    """Convert strings to float. Blanks and every #N/A variant become NaN.

    Returns the numeric series and the count of other strings that failed to parse.
    """
    s = raw.astype(str).str.strip()
    is_na = s.eq("") | s.str.startswith(NA_PREFIX)
    values = pd.to_numeric(s.mask(is_na).str.replace(",", "", regex=False), errors="coerce")
    n_bad = int((values.isna() & ~is_na).sum())
    return values.astype("float64"), n_bad


def _to_date(raw: pd.Series) -> pd.Series:
    return pd.to_datetime(raw, format=DATE_FORMAT, errors="coerce")


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "ticker": pd.Series(dtype="object"),
            "field": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )


def read_layout_a(path: str | Path, log: CleaningLog | None = None) -> pd.DataFrame:
    """Layout A: rows 2/3/4 hold label/ticker/field, then (date, value) pairs from row 5.

    Returns a long frame date, ticker, field, value. Human labels are kept in
    `df.attrs["labels"]` keyed by (ticker, field).
    """
    path = Path(path)
    log = log if log is not None else CleaningLog()
    src = path.name
    rows = _read_rows(path)
    labels, tickers, fields = rows[2], rows[3], rows[4]
    data = pd.DataFrame(rows[5:])

    frames, seen, label_map = [], set(), {}
    for j in range(0, len(tickers) - 1, 2):
        ticker = _clean_ticker(tickers[j])
        if not ticker:
            continue
        key = (ticker, fields[j])
        if ticker in BAD_TICKERS:
            log.add(src, "drop_column", f"{ticker} {fields[j]}", "wrong ticker")
            continue
        if key in seen:
            log.add(src, "drop_column", f"{ticker} {fields[j]}", f"duplicate series at column {j}")
            continue
        seen.add(key)
        label_map[key] = labels[j]

        has_date = data[j].ne("")
        dates = _to_date(data.loc[has_date, j])
        values, n_bad = _to_number(data.loc[has_date, j + 1])
        if n_bad:
            log.add(src, "coerce_to_nan", f"{ticker} {fields[j]}", f"{n_bad} non-numeric strings")
        if dates.isna().any():
            log.add(
                src,
                "drop_rows",
                f"{ticker} {fields[j]}",
                f"{int(dates.isna().sum())} rows with an unparseable date",
            )
        series = pd.DataFrame({"date": dates, "ticker": ticker, "field": fields[j], "value": values})
        series = series[series["date"].notna()]
        dup = series["date"].duplicated(keep="last")
        if dup.any():
            log.add(src, "drop_rows", f"{ticker} {fields[j]}", f"{int(dup.sum())} duplicate dates, kept last")
            series = series[~dup]
        frames.append(series)

    out = pd.concat(frames, ignore_index=True)[LONG_COLUMNS] if frames else _empty_long()
    out.attrs["labels"] = label_map
    return out


def read_layout_b(path: str | Path, log: CleaningLog | None = None) -> pd.DataFrame:
    """Layout B: shared date in column 0, tickers on row 3, field codes on row 5, data from row 6.

    Drops wrong tickers, repeated ticker blocks, all-NaN columns, and rows whose
    date does not parse. The export repeats the prior row on non-trading weekdays;
    futures rows where settle, volume and open interest all repeat are set to NaN.
    Returns a long frame date, ticker, field, value.
    """
    path = Path(path)
    log = log if log is not None else CleaningLog()
    src = path.name
    rows = _read_rows(path)
    tickers, codes = rows[3], rows[5]
    data = pd.DataFrame(rows[6:])

    dates = _to_date(data[0])
    bad_date = dates.isna()
    for i in data.index[bad_date]:
        stray = [v for v in data.loc[i, 1:] if v and not v.startswith(NA_PREFIX)]
        log.add(
            src,
            "drop_row",
            f"csv line {i + 7}",
            f"date {data.at[i, 0]!r} does not parse; {len(stray)} stray values discarded",
        )
    data, dates = data[~bad_date], dates[~bad_date]
    dup = dates.duplicated(keep="last")
    if dup.any():
        log.add(src, "drop_rows", "date", f"{int(dup.sum())} duplicate dates, kept last")
        data, dates = data[~dup], dates[~dup]

    starts = [j for j in range(1, len(tickers)) if tickers[j]]
    ends = starts[1:] + [len(tickers)]
    frames, first_block = [], {}
    for start, end in zip(starts, ends):
        ticker = _clean_ticker(tickers[start])
        cols = [j for j in range(start, end) if codes[j]]
        block_fields = [codes[j] for j in cols]
        if ticker in BAD_TICKERS:
            log.add(src, "drop_block", ticker, f"wrong ticker, columns {start}-{end - 1}")
            continue
        parsed = {codes[j]: _to_number(data[j]) for j in cols}
        block = pd.DataFrame({k: v[0] for k, v in parsed.items()})
        if ticker in first_block:
            same = block.equals(first_block[ticker])
            log.add(
                src,
                "drop_block",
                ticker,
                f"duplicate block, columns {start}-{end - 1}, "
                f"{'identical to' if same else 'DIFFERS from'} the first",
            )
            continue
        first_block[ticker] = block

        if set(STALE_FIELDS) <= set(block_fields):
            trio = block[list(STALE_FIELDS)]
            stale = (trio == trio.shift()).all(axis=1)
            if stale.any():
                log.add(
                    src,
                    "unfill",
                    ticker,
                    f"{int(stale.sum())} rows repeat the prior settle, volume and open interest "
                    f"(export fill on non-trading days), set to NaN",
                )
                block = block.mask(stale)

        for bbg_field in block_fields:
            values, n_bad = block[bbg_field], parsed[bbg_field][1]
            if n_bad:
                log.add(src, "coerce_to_nan", f"{ticker} {bbg_field}", f"{n_bad} non-numeric strings")
            if values.isna().all():
                log.add(src, "drop_column", f"{ticker} {bbg_field}", "all NaN")
                continue
            frames.append(
                pd.DataFrame({"date": dates, "ticker": ticker, "field": bbg_field, "value": values})
            )

    return pd.concat(frames, ignore_index=True)[LONG_COLUMNS] if frames else _empty_long()


def _load_long(raw_dir: Path, log: CleaningLog) -> pd.DataFrame:
    frames = []
    readers = [(name, read_layout_a) for name in LAYOUT_A_FILES]
    readers += [(name, read_layout_b) for name in LAYOUT_B_FILES]
    for name, reader in readers:
        path = raw_dir / name
        if not path.exists():
            log.add(name, "missing_file", name, "not in raw dir; its columns are absent from the panel")
            continue
        frame = reader(path, log)
        for (ticker, _), label in frame.attrs.get("labels", {}).items():
            if "CDX IG" in label and ticker.startswith("CDX HY"):
                log.add(
                    name,
                    "rename",
                    ticker,
                    f"export label {label!r} is wrong; this is CDX HY spread, "
                    f"named {COLUMN_MAP[(ticker, 'PX_LAST')]}",
                )
        frames.append(frame.assign(source=name))

    known = {SPECS_FILE, *LAYOUT_A_FILES, *LAYOUT_B_FILES}
    for extra in sorted(p.name for p in raw_dir.glob("*.csv") if p.name not in known):
        log.add(extra, "ignore_file", extra, "not a panel input (out of scope)")

    if not frames:
        raise FileNotFoundError(f"no Bloomberg exports found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def build_panel(
    raw_dir: str | Path = RAW_DIR,
    out_path: str | Path | None = PROCESSED_DIR / "panel_daily.parquet",
    log: CleaningLog | None = None,
) -> pd.DataFrame:
    """Wide daily panel on the business days where Treasury yields print.

    Levels are forward filled at most MAX_FFILL_BARS bars; volumes are not filled.
    Writes the panel to `out_path` and the cleaning log next to it. Pass
    `out_path=None` to skip writing.
    """
    raw_dir = Path(raw_dir)
    log = log if log is not None else CleaningLog()
    long = _load_long(raw_dir, log)

    for ticker, cutoff in TRUNCATE_BEFORE.items():
        early = long["ticker"].eq(ticker) & long["date"].lt(cutoff)
        if early.any():
            log.add(
                "panel",
                "truncate",
                ticker,
                f"{int(early.sum())} rows before {cutoff.date()} dropped (monthly history)",
            )
            long = long[~early]

    keys = pd.Series(list(zip(long["ticker"], long["field"])), index=long.index)
    long = long.assign(name=keys.map(COLUMN_MAP))
    unmapped = long[long["name"].isna()].groupby(["source", "ticker", "field"]).size()
    for (source, ticker, bbg_field), n in unmapped.items():
        log.add(source, "drop_column", f"{ticker} {bbg_field}", f"not in COLUMN_MAP, {n} rows")
    long = long[long["name"].notna()]

    # A series may only come from one file: keep the first source in load order.
    first_source = long.drop_duplicates("name").set_index("name")["source"]
    other = long["source"].ne(long["name"].map(first_source))
    for (name, source), n in long[other].groupby(["name", "source"]).size().items():
        log.add(source, "drop_column", name, f"already loaded from {first_source[name]}, {n} rows")
    long = long[~other]

    wide = long.pivot(index="date", columns="name", values="value").sort_index()
    wide = wide[[c for c in dict.fromkeys(COLUMN_MAP.values()) if c in wide.columns]]
    wide.columns.name = None

    calendar_cols = [c for c in CALENDAR_COLUMNS if c in wide.columns]
    if not calendar_cols:
        calendar_cols = [c for c in wide.columns if c.startswith("ust_")]
        log.add("panel", "fallback", "calendar", f"no Layout A yields loaded; calendar from {calendar_cols}")
    if not calendar_cols:
        raise ValueError("no Treasury yield column loaded; cannot build the bar calendar")
    prints = wide[calendar_cols].notna().any(axis=1)
    weekend = wide.index.dayofweek >= 5
    if (prints & weekend).any():
        log.add("panel", "drop_rows", "calendar", f"{int((prints & weekend).sum())} weekend yield prints")
    calendar = wide.index[prints & ~weekend]

    off = wide.loc[~wide.index.isin(calendar)]
    lost = off.notna().sum()
    log.add(
        "panel",
        "drop_rows",
        "calendar",
        f"{len(off)} dates with no Treasury yield print removed "
        f"({off.index.min().date()} to {off.index.max().date()})" if len(off) else "0 dates removed",
    )
    for name, n in lost[lost > 0].items():
        log.add("panel", "drop_rows", name, f"{int(n)} observations on non-calendar dates")

    panel = wide.reindex(calendar)
    fill_cols = [c for c in panel.columns if not c.endswith(NO_FFILL_SUFFIXES)]
    before = panel[fill_cols].notna().sum()
    panel[fill_cols] = panel[fill_cols].ffill(limit=MAX_FFILL_BARS)
    filled = panel[fill_cols].notna().sum() - before
    for name, n in filled[filled > 0].items():
        log.add("panel", "ffill", name, f"{int(n)} cells filled, limit {MAX_FFILL_BARS} bars")

    pre = panel.index < FUNDING_SWITCH_DATE
    libor = panel["libor_3m"] if "libor_3m" in panel else pd.Series(np.nan, index=panel.index)
    sofr = panel["sofr"] if "sofr" in panel else pd.Series(np.nan, index=panel.index)
    panel["funding_rate"] = libor.where(pre, sofr)
    log.add(
        "panel",
        "derive",
        "funding_rate",
        f"libor_3m before {FUNDING_SWITCH_DATE.date()} ({int(pre.sum())} bars), "
        f"sofr on or after ({int((~pre).sum())} bars); {int(panel['funding_rate'].isna().sum())} NaN",
    )

    panel.index.name = "date"
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)
        log.to_frame().to_csv(out_path.with_name("cleaning_log.csv"), index=False)
    return panel


def load_specs(
    path: str | Path = RAW_DIR / SPECS_FILE,
    out_path: str | Path | None = PROCESSED_DIR / "contract_specs.parquet",
    log: CleaningLog | None = None,
) -> pd.DataFrame:
    """Contract specs snapshot, one row per generic ticker. Row 1 holds the field codes."""
    path = Path(path)
    log = log if log is not None else CleaningLog()
    src = path.name
    rows = _read_rows(path)
    codes = rows[1][1:]
    raw = pd.DataFrame([r[1:] for r in rows[2:]], columns=codes)
    raw.insert(0, "ticker", [_clean_ticker(r[0]) for r in rows[2:]])
    raw = raw[raw["ticker"].ne("")]

    bad = raw["ticker"].isin(BAD_TICKERS)
    for ticker in raw.loc[bad, "ticker"]:
        log.add(src, "drop_row", ticker, "wrong ticker")
    raw = raw[~bad]
    dup = raw["ticker"].duplicated(keep="first")
    for ticker in raw.loc[dup, "ticker"]:
        log.add(src, "drop_row", ticker, "duplicate row, kept first")
    raw = raw[~dup]

    specs = pd.DataFrame(index=pd.Index(raw["ticker"], name="ticker"))
    for code in codes:
        col = raw[code].set_axis(specs.index)
        if code in SPEC_DATE_FIELDS:
            specs[code] = _to_date(col)
        elif code in SPEC_TEXT_FIELDS:
            specs[code] = col.mask(col.eq("") | col.str.startswith(NA_PREFIX))
        else:
            specs[code], n_bad = _to_number(col)
            if n_bad:
                log.add(src, "coerce_to_nan", code, f"{n_bad} non-numeric strings")
    specs.columns = [c.lower() for c in specs.columns]
    for name in specs.columns[specs.isna().all()]:
        log.add(src, "drop_column", name.upper(), "all NaN")
        specs = specs.drop(columns=name)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        specs.to_parquet(out_path)
    return specs


def column_ranges(panel: pd.DataFrame) -> pd.DataFrame:
    """First date, last date, observation count and NaN count per panel column."""
    return pd.DataFrame(
        {
            "first": panel.apply(pd.Series.first_valid_index),
            "last": panel.apply(pd.Series.last_valid_index),
            "n_obs": panel.notna().sum(),
            "n_nan": panel.isna().sum(),
        }
    )


def main() -> None:
    log = CleaningLog()
    panel = build_panel(log=log)
    specs = load_specs(log=log)
    log.to_frame().to_csv(PROCESSED_DIR / "cleaning_log.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 120):
        print(f"panel shape: {panel.shape}, {panel.index.min().date()} to {panel.index.max().date()}")
        print(column_ranges(panel).to_string())
        print(f"\nspecs shape: {specs.shape}")
        print(specs.to_string())
        print("\ncleaning log:")
        print(log.to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
