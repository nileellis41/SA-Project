"""Loader tests.

Synthetic fixtures exercise the parsing and cleaning rules without the raw
data. The `real_*` fixtures run the same checks on the Bloomberg exports in
data/raw and skip when those files are absent (raw data is never in git).
"""

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype

from amp_sa.data import load_bbg as lb

OAS_START = pd.Timestamp("2000-08-16")
BOM = "﻿"


def _write(path, lines, bom=True):
    path.write_text((BOM if bom else "") + "\n".join(lines) + "\n", encoding="utf-8")


def _layout_a(path, series):
    """series: list of (label, ticker, field, [(date_str, value_str), ...])."""
    n = max(len(obs) for *_, obs in series)
    lines = [
        "Start date,1/1/1990" + "," * (2 * len(series) - 2),
        "End date,9/25/2026" + "," * (2 * len(series) - 2),
        ",".join(f"{label}," for label, *_ in series),
        ",".join(f"{ticker}," for _, ticker, *_ in series),
        ",".join(f"{fld}," for _, _, fld, _ in series),
    ]
    for i in range(n):
        cells = []
        for *_, obs in series:
            cells += list(obs[i]) if i < len(obs) else ["", ""]
        lines.append(",".join(cells))
    _write(path, lines)


def _bdays(start, periods):
    return [f"{d.month}/{d.day}/{d.year}" for d in pd.bdate_range(start, periods=periods)]


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    days = _bdays("2018-03-26", 12)  # Mon 3/26/2018 .. Tue 4/10/2018, spans the funding switch

    yields = [(d, f"{2 + i / 100:.2f}") for i, d in enumerate(days)]
    sofr = [(d, f"{1.8 + i / 100:.2f}") for i, d in enumerate(days) if pd.Timestamp(d) >= lb.FUNDING_SWITCH_DATE]
    # Prints on day 0, then a gap of 6 bars, then the last day: only 3 bars may be filled.
    gappy = [(days[0], "100"), (days[7], "#N/A N/A"), (days[11], "101")]
    _layout_a(
        raw / "rates_1.csv",
        [
            ("UST 2Y yield", "USGG2YR Index", "PX_LAST", yields),
            ("UST 10Y yield", "USGG10YR Index", "PX_LAST", yields + [("3/31/2018", "9.99")]),  # Saturday
            ("SOFR", "SOFRRATE Index", "PX_LAST", sofr),
            ("10Y T-note fut", "TY1 Comdty", "PX_LAST", gappy),
        ],
    )
    monthly = [("1/31/2000", "4.82"), ("7/31/2000", "5.5")]
    _layout_a(
        raw / "credit.csv",
        [
            ("CDX HY 5Y (generic)", "CDX HY CDSI GEN 5Y PRC Corp", "PX_LAST", [(days[0], "92")]),
            ("CDX IG 5Y (generic)", "CDX HY CDSI GEN 5Y SPRD Corp", "PX_LAST", [(days[0], "548.072")]),
            ("HYG shares out", "HYG US Equity", "EQY_SH_OUT", [(days[0], "#N/A Field Not Applicable")]),
            ("HY OAS", "LF98OAS Index", "PX_LAST", monthly + [("8/16/2000", "6.1"), (days[1], "3.5")]),
            ("IG OAS", "LUACOAS Index", "PX_LAST", monthly + [("8/16/2000", "1.6"), (days[1], "1.1")]),
        ],
    )
    _layout_a(
        raw / "Features_Costs_1.csv",
        [("VIX", "VIX Index", "PX_LAST", [(days[0], "#N/A Requesting Data..."), (days[1], "17.24")])],
    )

    blocks = ["CL1 Comdty", "W11 Comdty", "W111 Comdty", "CL1 Comdty", "US0003M Index"]
    na4 = ["#N/A N/A"] * 4
    lines = [
        "Start Date,1/1/2005" + "," * 20,
        "End Date" + "," * 21,
        "," * 21,
        "," + ",".join(f"{t},,," for t in blocks),
        "," + ",".join(["Settlement Price,Last Price,Volume - Realtime,Open Interest"] * len(blocks)),
        "Dates," + ",".join(["PX_SETTLE,PX_LAST,VOLUME,OPEN_INT"] * len(blocks)),
    ]
    for i, d in enumerate(days):
        # Day 4 (Fri 3/30/2018, Good Friday) repeats day 3: an export fill, not a print.
        k = 3 if i == 4 else i
        cl = [f"{60 + k}", f"{60 + k}", f"{1000 + k}", f"{5000 + k}"]
        libor = ["#N/A N/A", f"{2.3 + i / 100:.2f}", "#N/A N/A", "#N/A N/A"]
        lines.append(",".join([d, *cl, *na4, *na4, *cl, *libor]))
    lines.append(",".join(["", *[""] * 16, "#N/A N/A", "9.99", "0", "#N/A N/A"]))  # stray last row
    _write(raw / "930.csv", lines)

    _write(
        raw / "pnt.csv",
        [
            ",Current Generic Futures Ticker,Last Tradeable Date,Tick Size,Initial Speculator Margin Limit,Currency",
            ",FUT_CUR_GEN_TICKER,LAST_TRADEABLE_DT,FUT_TICK_SIZE,FUT_INIT_SPEC_ML,CRNCY",
            "CL1 Comdty,CLX6,10/20/2026,0.01,#N/A Field Not Applicable,USD",
            "W11 Comdty,#N/A N/A,#N/A N/A,0.01,#N/A Field Not Applicable,AUD",
            "W112 Comdty,#N/A N/A,#N/A N/A,0.01,#N/A Field Not Applicable,AUD",
            "CL1 Comdty,CLX6,10/20/2026,0.01,#N/A Field Not Applicable,USD",
            "TY1 Comdty,TYZ6,12/21/2026,0.015625,#N/A Field Not Applicable,USD",
        ],
    )
    _write(raw / "intraminfx.csv", ["BarTp,Trade", "x,y"])  # out of scope: must be ignored
    return raw


@pytest.fixture(scope="module")
def synth(synth_dir):
    log = lb.CleaningLog()
    panel = lb.build_panel(synth_dir, out_path=None, log=log)
    specs = lb.load_specs(synth_dir / "pnt.csv", out_path=None, log=log)
    return panel, specs, log.to_frame()


@pytest.fixture(scope="module")
def real():
    needed = [*lb.LAYOUT_A_FILES, *lb.LAYOUT_B_FILES, lb.SPECS_FILE]
    missing = [f for f in needed if not (lb.RAW_DIR / f).exists()]
    if missing:
        pytest.skip(f"raw Bloomberg exports not present: {missing}")
    log = lb.CleaningLog()
    panel = lb.build_panel(out_path=None, log=log)
    specs = lb.load_specs(out_path=None, log=log)
    return panel, specs, log.to_frame()


@pytest.fixture(params=["synth", "real"])
def built(request):
    return request.getfixturevalue(request.param)


# --- rules that must hold on both the synthetic and the real build ---


def test_numeric_columns_hold_no_strings(built):
    panel, specs, _ = built
    assert all(is_float_dtype(t) for t in panel.dtypes)
    numeric_specs = specs.drop(columns=["fut_cur_gen_ticker", "crncy"]).select_dtypes(exclude="datetime")
    assert all(is_float_dtype(t) for t in numeric_specs.dtypes)
    assert not specs.astype(str).apply(lambda c: c.str.startswith("#N/A")).any().any()


def test_cdx_column_is_hy_spread(built):
    panel = built[0]
    assert "cdx_hy_5y_spread" in panel.columns
    assert not [c for c in panel.columns if "cdx_ig" in c]


def test_no_duplicate_or_bad_tickers(built):
    panel, specs, _ = built
    assert panel.columns.is_unique
    assert specs.index.is_unique
    for bad in ("w11", "w111", "w112"):
        assert not [c for c in panel.columns if c.split("_")[0] == bad]
    assert not set(specs.index) & lb.BAD_TICKERS


def test_oas_starts_at_daily_history(built):
    panel = built[0]
    for col in ("hy_oas", "ig_oas"):
        assert panel[col].first_valid_index() >= OAS_START


def test_funding_rate_switches_from_libor_to_sofr(built):
    panel = built[0]
    pre = panel.index < lb.FUNDING_SWITCH_DATE
    assert pre.any() and (~pre).any()
    pd.testing.assert_series_equal(panel.loc[pre, "funding_rate"], panel.loc[pre, "libor_3m"], check_names=False)
    pd.testing.assert_series_equal(panel.loc[~pre, "funding_rate"], panel.loc[~pre, "sofr"], check_names=False)
    assert panel.loc[~pre, "funding_rate"].notna().any()


def test_panel_index_unique_sorted_no_weekends(built):
    index = built[0].index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert (index.dayofweek < 5).all()


def test_calendar_is_days_where_yields_print(built):
    panel = built[0]
    cols = [c for c in lb.CALENDAR_COLUMNS if c in panel.columns]
    assert cols and panel[cols].notna().any(axis=1).all()


# --- parsing rules, synthetic only ---


def test_na_strings_become_nan(synth_dir):
    long = lb.read_layout_a(synth_dir / "Features_Costs_1.csv")
    assert list(long.columns) == ["date", "ticker", "field", "value"]
    assert is_float_dtype(long["value"])
    assert np.isnan(long["value"].iloc[0]) and long["value"].iloc[1] == 17.24

    credit = lb.read_layout_a(synth_dir / "credit.csv")
    assert credit.loc[credit["field"] == "EQY_SH_OUT", "value"].isna().all()

    b = lb.read_layout_b(synth_dir / "930.csv")
    assert is_float_dtype(b["value"])
    assert b["date"].notna().all()


def test_layout_a_reads_each_series_on_its_own_dates(synth_dir):
    long = lb.read_layout_a(synth_dir / "rates_1.csv")
    counts = long.groupby("ticker").size()
    assert counts["USGG2YR Index"] == 12
    assert counts["SOFRRATE Index"] == 7
    assert counts["TY1 Comdty"] == 3
    assert long.attrs["labels"][("USGG2YR Index", "PX_LAST")] == "UST 2Y yield"


def test_layout_b_drops_bad_duplicate_and_stray(synth_dir):
    log = lb.CleaningLog()
    long = lb.read_layout_b(synth_dir / "930.csv", log)
    assert set(long["ticker"]) == {"CL1 Comdty", "US0003M Index"}
    assert not long.duplicated(["date", "ticker", "field"]).any()
    assert set(long.loc[long["ticker"] == "US0003M Index", "field"]) == {"PX_LAST"}
    assert 9.99 not in set(long["value"])

    entries = log.to_frame()
    dropped = entries.loc[entries["action"] == "drop_block", "target"].tolist()
    assert dropped == ["W11 Comdty", "W111 Comdty", "CL1 Comdty"]
    assert (entries["action"] == "drop_row").sum() == 1


def test_layout_b_export_fill_is_undone(synth):
    panel, _, log = synth
    good_friday = pd.Timestamp("2018-03-30")
    assert np.isnan(panel.at[good_friday, "cl1_volume"])  # volume is never filled
    assert panel.at[good_friday, "cl1_settle"] == panel.at[pd.Timestamp("2018-03-29"), "cl1_settle"]
    assert (log["action"] == "unfill").sum() == 1


def test_forward_fill_stops_after_three_bars(synth):
    panel = synth[0]
    ty = panel["ty1_last"]
    assert ty.iloc[:4].eq(100).all()  # the print plus 3 filled bars
    assert ty.iloc[4:11].isna().all()
    assert ty.iloc[11] == 101


def test_weekend_and_off_calendar_rows_are_dropped(synth):
    panel, _, log = synth
    assert pd.Timestamp("2018-03-31") not in panel.index
    assert panel.index.min() == pd.Timestamp("2018-03-26")  # 2000 OAS rows are off calendar
    assert ((log["action"] == "drop_rows") & (log["target"] == "calendar")).any()


def test_cleaning_log_records_every_rule(synth):
    log = synth[2]
    actions = set(log["action"])
    assert {"rename", "truncate", "drop_block", "drop_row", "drop_column", "ffill", "ignore_file"} <= actions
    truncated = log.loc[log["action"] == "truncate", "target"].tolist()
    assert truncated == ["LF98OAS Index", "LUACOAS Index"]
    assert "intraminfx.csv" in log.loc[log["action"] == "ignore_file", "target"].tolist()
    spec_drops = log.loc[log["source"] == "pnt.csv", "target"].tolist()
    assert spec_drops == ["W11 Comdty", "W112 Comdty", "CL1 Comdty", "FUT_INIT_SPEC_ML"]


def test_panel_builds_without_optional_files(synth_dir, tmp_path):
    for name in ("rates_1.csv", "930.csv"):
        (tmp_path / name).write_bytes((synth_dir / name).read_bytes())
    log = lb.CleaningLog()
    panel = lb.build_panel(tmp_path, out_path=None, log=log)
    assert "hy_oas" not in panel.columns
    assert (log.to_frame()["action"] == "missing_file").sum() == 2


def test_outputs_round_trip_through_parquet(synth_dir, tmp_path):
    panel = lb.build_panel(synth_dir, out_path=tmp_path / "panel_daily.parquet")
    specs = lb.load_specs(synth_dir / "pnt.csv", out_path=tmp_path / "contract_specs.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "panel_daily.parquet"), panel, check_freq=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "contract_specs.parquet"), specs)
    assert (tmp_path / "cleaning_log.csv").exists()


# --- facts about the real exports ---


def test_cl1_settles_negative_on_2020_04_20(real):
    assert real[0].at[pd.Timestamp("2020-04-20"), "cl1_settle"] == -37.63


def test_real_fills_are_at_most_three_bars_old(real):
    panel = real[0]
    raw = lb.read_layout_a(lb.RAW_DIR / "rates_1.csv")
    for ticker, col in (("TY1 Comdty", "ty1_last"), ("SOFRRATE Index", "sofr")):
        prints = raw.loc[(raw["ticker"] == ticker) & raw["value"].notna(), "date"]
        printed = pd.Series(panel.index.isin(prints), index=panel.index)
        bars_since_print = printed.groupby(printed.cumsum()).cumcount()
        has_value = panel[col].notna()
        assert 1 <= bars_since_print[has_value].max() <= lb.MAX_FFILL_BARS, col
        assert not (has_value & (printed.cumsum() == 0)).any(), col


def test_real_specs_cover_the_loaded_futures(real):
    panel, specs, _ = real
    roots = {c.split("_")[0].upper() for c in panel.columns if c.endswith(("_settle", "_last"))}
    assert {f"{r} Comdty" for r in roots} <= set(specs.index)
    assert "fut_init_spec_ml" not in specs.columns
