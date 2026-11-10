# amp-mt5-sa: Systematic Cross-Asset Relative Value Book

## Context
- Personal project. Fully separate from NYU VIP work: do not reference, copy, or import VIP code or data.
- Two uses:
  1. Statistical arbitrage class deliverable. Graded on Sleeve 1 (mean reversion) and Sleeve 2 (trend), plus their combination. Sleeve 1 was presented in class 9/30.
  2. Portfolio piece for a macro quant role. Public framing: "systematic cross-asset relative value book," not "stat arb."
- Research data: Bloomberg (university terminal). Execution: AMP Futures via MT5, demo first, micros only, $2k-5k account. MT5 is used for execution only, never as a research data source.

## Architecture
```
data/raw/          Bloomberg CSV exports (gitignored, never commit)
data/processed/    clean panels as parquet (gitignored)
src/amp_sa/
  data/            loaders, cleaning, continuous series, calendar
  features/        spreads, regime features
  models/          cointegration, kalman, ou, regime, trend filters
  sleeves/         rv.py, trend.py, (carry.py later)
  portfolio/       risk budget, vol target, drawdown overlay
  backtest/        walk-forward engine, cost model, metrics
  execution/       mt5_client.py (isolated; nothing else imports MetaTrader5)
signals/signals.parquet
tests/             pytest, including the leakage suite
```

Signal output contract (`signals/signals.parquet`):
`date, instrument, sleeve (rv|trend|carry), signal, target_vol_weight`

## Non-negotiable rules
- Walk-forward only. Splits are purged and embargoed. Fit everything (cointegration screens, Kalman init, OU params, band optimization, regime models, L1 lambda, scalers) on the training window only.
- Every feature must be observable at decision time. Macro or positioning data is lagged to its release date.
- Pytest leakage suite must pass before any result is reported.
- One bar calendar, one roll rule, one cost model, shared by all sleeves.
- Use price or bp differences for energy, not log returns (CL1 settled at -37.63 on 2020-04-20).
- Raw data never goes into git.

## Strategy scope
**Sleeve 1, mean reversion (build first)**
- Universe now: Treasury cash curve in bp (2s10s, 5s30s, 2s5s10s, 5s10s30s flies), HY OAS vs IG OAS.
- Later: CL calendars, CL/Brent, 3-2-1 crack, ZW/ZC, ZW/KE, CDX IG/HY basis.
- Pipeline: Engle-Granger and Johansen screening per fold, Kalman hedge ratio, OU half-life with optimal entry/exit bands, regime filter (HMM baseline, neural regime classifier as the adaptive version), health monitor emitting `spread_disabled`.
- Compare the adaptive model against a static version (fixed hedge ratio, no regime layer).

**Sleeve 2, trend**
- Same series as Sleeve 1, traded as outrights or levels, not as the RV spreads.
- MA momentum baseline vs single-scale and multi-scale L1 trend filtering (cvxpy). Position = normalized slope x vol scaling.

**Portfolio layer**
- Risk budget across sleeves, fixed vol target, drawdown overlay. Report sleeve return correlation.

**Reporting**
- Sharpe, deflated Sharpe, max drawdown, turnover, cost drag, cost sensitivity at 1x/2x/3x.
- Regime breakdown: 2008, 2020, 2022.

## Data files (data/raw/)
All are Bloomberg Excel exports saved as CSV. Dates are m/d/yyyy. Missing values appear as `#N/A N/A`, `#N/A Field Not Applicable`, `#N/A Requesting Data...`. Treat all as NaN.

### Layout A: per-series date columns
Files: `rates_1.csv`, `credit.csv`, `Features_Costs_1.csv`
- Row 0: `Start date,...`. Row 1: `End date,...`. Row 2: human labels. Row 3: Bloomberg tickers. Row 4: field names.
- From row 5: column pairs `(date, value)`, one pair per series, each with its own dates and its own length. A label sits in the date column of each pair.
- Some files have a UTF-8 BOM.

`rates_1.csv` (1990 to 2026-09-25):
USGG2YR, USGG5YR, USGG10YR, USGG30YR (PX_LAST, yields in %); TU1, FV1, TY1, US1 Comdty (PX_LAST); USOSFR2/5/10 Curncy (SOFR swaps, from 2007; pre-2018 values are backfilled); SOFRRATE Index (from 2018-04-02).

`credit.csv` (2000 to 2026-09-25):
- Col pair 1 label "CDX HY 5Y (generic)", ticker `CDX HY CDSI GEN 5Y PRC Corp`: CDX HY price.
- Col pair 2 label "CDX IG 5Y (generic)" is MISLABELED. Ticker is `CDX HY CDSI GEN 5Y SPRD Corp`: it is CDX HY spread. Rename to `cdx_hy_5y_spread`. There is no CDX IG data yet.
- HYG US Equity: PX_LAST, FUND_NET_ASSET_VAL, EQY_SH_OUT (verify units; early values ~0.2).
- LF98OAS Index (HY OAS), LUACOAS Index (IG OAS): monthly until 2000-08-16, daily after. Truncate to >= 2000-08-16.

`Features_Costs_1.csv`:
VIX Index, MOVE Index (PX_LAST, from 1990); TY1 PX_BID/PX_ASK (from 1995, ~2,500 missing days, do not rely on it); HYG PX_BID/PX_ASK.

### Layout B: shared date column
File: `930.csv` (2005-01-03 to 2026-09-30)
- Row 0: Start Date. Row 1: End Date. Row 2: blank. Row 3: ticker in the first column of each 4-column block. Row 4: field descriptions. Row 5: `Dates` then field codes. Data from row 6. Column 0 is the shared date.
- Fields per futures ticker: PX_SETTLE, PX_LAST, VOLUME, OPEN_INT. Use PX_SETTLE for signals.
- Good: CL1, CL2, CL3, CO1, CO2, HO1, KW1 Comdty (from 2005). XB1 starts 2005-10-03 (RBOB launch). USGG3YR (from 2005), USGG7YR (from 2009-02-26), US0003M (PX_LAST only).
- Bad, drop: blocks for `W11 Comdty` and `W111 Comdty` (wrong tickers, all NaN); the second, duplicate `CL1 Comdty` and `CL2 Comdty` blocks.
- Index tickers return NaN for PX_SETTLE, VOLUME, OPEN_INT; use PX_LAST.
- Last row may hold stray values; validate the date parses.

File: `pnt.csv` (contract specs snapshot, Layout: row 0 labels, row 1 field codes, then one row per ticker)
- Fields: FUT_CUR_GEN_TICKER, LAST_TRADEABLE_DT, FUT_NOTICE_FIRST, FUT_TICK_SIZE, FUT_TICK_VAL, FUT_VAL_PT, FUT_CONT_SIZE, FUT_INIT_SPEC_ML, CRNCY.
- Drop rows `W11 Comdty`, `W112 Comdty` (wrong) and duplicate CL1/CL2 rows. FUT_INIT_SPEC_ML is NaN everywhere; margins come from AMP.

### Known data issues
- Generic futures (TU1, FV1, TY1, US1, CL1, ...) are unadjusted: price jumps at every roll (TY1 jumps 12.125 on 1999-12-22 from the coupon change). Never compute returns on raw generics. For now, trade the Treasury curve in cash-yield bp space.
- Funding rate: US0003M until 2018-04-01, SOFRRATE from 2018-04-02. LIBOR stopped being a real rate in mid-2023.
- Missing, to be pulled later: W1, W2, C1, C2 Comdty history and specs; CDX IG 5Y spread. Code must run without them.
- `pxbid.csv` (daily ES1 and FX bid/ask) and `intraminfx.csv` (one-minute FX) in the project root are now in scope (decided 2026-10-12; the original "do not load intraday, FX, ES" rule is withdrawn). `data/load_extra.py` reads them. ES and FX mids feed the trend sleeve, four cross-asset/FX spreads and the equity factor in the alpha regression; the intraday file feeds a stand-alone five-minute FX study (`sleeves/intraday_fx.py`) that is not part of the book.

## Live-trading mapping (later, Phase 4)
Tradeable as micros: micro yield futures (curve spreads and flies), MCL (calendars), micro wheat and micro corn (ZW/ZC), trend on the same. Not tradeable live: HY/IG credit, CL/Brent, crack, ZW/KE. Confirm the MT5 symbol list and margins with AMP before building this.

## Build order
1. **Loader and clean panel** (current task).
2. Walk-forward splitter with purge and embargo, plus the pytest leakage suite.
3. Sleeve 1 on the Treasury curve and HY/IG OAS.
4. Sleeve 2 trend on the same series.
5. Portfolio layer and combined report.
6. Energy and grain spreads using the same code.
7. Optional: carry sleeve, MT5 paper trading.

## Status (2026-10-12): steps 1-6 built, MT5 is dry-run only
Run order:
```
python -m amp_sa.data.load_bbg          # Bloomberg panel
python -m amp_sa.execution.mt5_client   # optional: AMP specs, quotes, cached daily bars (read-only)
python -m amp_sa.run [--force]          # all pairwise spreads, trend, two-sleeve book -> reports/appendix.md
python -m amp_sa.run_sleeve1 [--force]  # THE DELIVERABLE: reports/report.md, signals/signals.parquet
python -m amp_sa.run_lessons            # descriptive: reports/trades.md, reports/periods.md (lessons.md is hand-written)
python -m amp_sa.run_curve              # 2s10s steepener/flattener rules vs PCA: reports/curve_trades.md (findings hand-written in the module)
python -m amp_sa.run_options            # delta-hedged IV mean reversion (SPX/NDX proxies; MSFT once data/raw/msft_options.csv exists): reports/options.md
pytest                                  # includes tests/test_leakage.py
```
- Scope decision (2026-10-12): the deliverable is the mean-reversion sleeve only (the goal is to learn stat arb). Trend, the regime-switching book, carry, energy/grains/FX and intraday FX are appendix material; do not extend them.
- Sleeve 1 is built three ways and compared in `run_sleeve1.py`: pairwise static, pairwise adaptive (Kalman + regime gate), and PCA residuals on the Treasury curve (`sleeves/pca.py`, Avellaneda-Lee, k = 2 and 3). `run_sleeve1.HEADLINE` was named before the PCA model ran on real data; do not change it after the fact.
- This file's scope sections below were the starting brief. Where they conflict with the current code (universe, data in scope), the code and this status section win.
- `config.py` holds the universe (31 spreads, 20 trend series), the cost table and the walk-forward schedule. `run.BOOK` is the declared book (`adaptive_nn` + `l1_multi`); do not change it after looking at results.
- Sleeve runs are cached in `data/processed/sleeve_runs.pkl` and L1 slopes in `l1_slopes.parquet`. Delete them or pass `--force` after changing a model.
- Wheat, corn and KC wheat come from Yahoo Finance (`data/load_yahoo.py`, cached in `data/raw/yahoo/`). Grain roll bars are taken from Bloomberg KW1 open interest.
- AMP MT5 is used for specs, margins and quotes only, never research data. `mt5_client.py` is read-only: it never sends orders, never calls `symbol_select`, and only reads symbols already cached and in Market Watch. Expired contracts are not in Market Watch, so AMP has no usable history beyond the live contracts.
- `execution/limits.py` are the account guardrails (margin, daily loss, stress, size). Symbol-table margins are AMP day margins; the 4x overnight multiplier is an assumption to confirm with AMP.
- Not built: carry sleeve, order sending.

## Step 1 spec (done)
`src/amp_sa/data/load_bbg.py` provides:
- `read_layout_a(path) -> pd.DataFrame` and `read_layout_b(path) -> pd.DataFrame`, each returning a long frame: `date, ticker, field, value`.
- A config mapping (Bloomberg ticker, field) to clean names, e.g. `ust_2y`, `ust_10y`, `hy_oas`, `ig_oas`, `cdx_hy_5y_spread`, `cl1_settle`, `vix`, `move`, `sofr`, `funding_rate`.
- `build_panel() -> pd.DataFrame`: wide daily panel on the business days where Treasury yields print, no forward fill beyond 3 days, written to `data/processed/panel_daily.parquet`.
- `load_specs(path)` for `pnt.csv`, written to `data/processed/contract_specs.parquet`.
- A cleaning log that lists every dropped column, row, and truncation.

Tests (`tests/test_load_bbg.py`):
- `#N/A` strings become NaN; no string values in numeric columns.
- The CDX column is named `cdx_hy_5y_spread` and no `cdx_ig` column exists.
- No duplicate tickers; no W11/W111/W112 columns.
- HY and IG OAS have no rows before 2000-08-16.
- `funding_rate` equals US0003M before 2018-04-02 and SOFRRATE on or after.
- CL1 settle on 2020-04-20 equals -37.63.
- The panel index is unique, sorted, and has no weekends.
