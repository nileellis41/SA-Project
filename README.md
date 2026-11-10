# Sleeve 1: PCA residual mean reversion on the US Treasury curve

**Research question.** After hedging out the first two or three principal components of the Treasury curve, does each tenor's residual mean-revert strongly enough to pay its transaction costs?

The answer, from walk-forward out-of-sample tests over 1995–2026: **the signal is real before cost, and no model survives cost.**

| model | gross Sharpe | net Sharpe |
|---|---|---|
| headline | 0.55 | −0.60 |
| best variant (PCA-3 sparse) | 1.21 | −0.09 |

The best variant breaks even at 0.93x the cost table. The write-up is `submission/report.md` (and `.pdf`). Every number in it is sourced in `submission/deck_numbers.md`.

## What the code does

1. **Hedge.** On each 5-year training window, take the top k principal components of daily yield changes (2y, 3y, 5y, 7y, 10y, 30y; bp).
   - **PCA-2 full**: each tenor's residual is its change with level and slope projected out (`W = I − LLᵀ`).
   - **PCA-3 sparse**: each tenor is hedged with its three nearest tenors so that it is neutral to level, slope and curvature.
2. **Signal.** Fit an AR(1) to the cumulative residual on a trailing 60-bar window and form the s-score `z = (C − m) / σ_eq` (Avellaneda and Lee 2010). Windows whose half-life is 30 bars or more are screened out.
3. **Trade.** Enter at |z| > entry and exit at |z| ≤ exit. Bands are chosen per fold on the training window by net Sharpe. There is a stop at |z| > 4.
   - Each member is sized `signal / trailing vol` (EW span 60).
   - Members are equal-weighted into a sleeve with a 10% vol target.
4. **Cost.** 0.25 bp per Treasury leg, one way, on every position change. Credit pairs are 0.5–2 bp.
5. **Walk-forward.** 5y train, 1y test, 10-bar purge, 5-bar embargo. Every fitted quantity uses the training window only. The leakage suite (`tests/test_leakage.py`) rewrites all data after a cut-off and checks that no earlier decision changes.

The headline model, named before the PCA model was run on real data, is **PCA-2 curve + adaptive credit pairs** (`src/amp_sa/run_sleeve1.py`, `HEADLINE`).

## Data

Research data are Bloomberg terminal exports (university terminal), saved as CSV in `data/raw/`. They are **not included**: they are licensed and gitignored. The loaders expect these files:

| file | contents |
|---|---|
| `rates_1.csv` | USGG2YR/5YR/10YR/30YR yields (PX_LAST), TU1/FV1/TY1/US1, SOFR swaps, SOFRRATE |
| `930.csv` | USGG3YR, USGG7YR, US0003M, energy/grain futures (PX_SETTLE) |
| `credit.csv` | LF98OAS (HY OAS) and LUACOAS (IG OAS) indices, CDX HY 5Y price and spread, HYG |
| `Features_Costs_1.csv` | VIX, MOVE, TY1 and HYG bid/ask |
| `pnt.csv` | contract specs |

The file layouts and known data issues are documented at the top of `src/amp_sa/data/load_bbg.py`.

Small public inputs used only by the diagnostics are cached under `data/raw/`:
- FRED: `DFEDTAR`, `DFEDTARU`, `DFEDTARL` (fed funds target)
- TreasuryDirect auction calendar
- Yahoo Finance grain futures (appendix only, `src/amp_sa/data/load_yahoo.py` fetches them)

MetaTrader 5 (AMP Futures) is used for contract specs only, never for research data. It is optional: set `MT5_TERMINAL_PATH` to use `src/amp_sa/execution/mt5_client.py`.

## Setup

Python 3.12. With [uv](https://docs.astral.sh/uv/) (versions pinned in `uv.lock`):

```
uv sync
```

or with pip:

```
python -m venv .venv && .venv\Scripts\activate    # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install -e .
```

## Reproduce

Run from the repo root, in this order. Prefix each command with `uv run` if you use uv. Cached runs are written to `data/processed/`; pass `--force` to refit.

| step | command | writes |
|---|---|---|
| 1. Clean panel | `python -m amp_sa.data.load_bbg` | `data/processed/panel_daily.parquet`, `contract_specs.parquet` |
| 2. Pairwise spreads, trend, appendix | `python -m amp_sa.run [--force]` | `reports/appendix.md`, `data/processed/sleeve_runs.pkl` |
| 3. **Deliverable**: Sleeve 1 model comparison | `python -m amp_sa.run_sleeve1 [--force]` | `reports/report.md`, `reports/tables/sleeve1_*.csv`, `reports/figures/sleeve1_*.png`, `signals/signals.parquet` |
| 4. Trade- and period-level analysis | `python -m amp_sa.run_lessons` | `reports/trades.md`, `reports/periods.md` |
| 5. Diagnostics (cost audit, hedge, signal, regimes, overfitting) | `python -m analysis.portfolio_diagnostics.run_all` then `python -m analysis.portfolio_diagnostics.report` | `analysis/portfolio_diagnostics/report.md`, `tables/` |
| 6. Walk-forward experiments (time stop, resize band, entry bands, MOVE gate, LOO) | `python -m analysis.experiments.run_experiments [--force]` | `analysis/experiments/results.csv` |
| 7. Cost round 1 (freeze size, vol gate) | `python -m analysis.experiments.cost_round1 [--force]` | `results/cost_round1/` |
| 8. Submission numbers, figures, PDF | `python -m analysis.submission_numbers`, `python -m analysis.submission_figures`, `uv run --with markdown python -m analysis.submission_pdf` (needs Edge or Chrome) | `submission/tables/`, `submission/figures/`, `submission/report.pdf` |
| Tests, including the leakage suite | `python -m pytest tests analysis/experiments/tests analysis/portfolio_diagnostics/tests` | |

Steps 3–8 read cached runs and take seconds to a few minutes each. Steps 1–2 with `--force` refit every fold and take longer.

## Repository layout

```
src/amp_sa/
  data/         Bloomberg loaders, clean panel, calendar (load_bbg.py, panel.py)
  models/       cointegration, Kalman, OU fit and band optimiser (ou.py), Leung-Li (ou_optimal.py), regime, trend
  sleeves/      pca.py (PCA residuals; the deliverable), rv.py (pairwise spreads), trend.py, options.py
  portfolio/    combine.py (risk budget, vol target, drawdown overlay)
  backtest/     engine.py (P&L and cost), splits.py (walk-forward with purge and embargo), metrics.py
  execution/    mt5_client.py (read-only, optional), limits.py
  run*.py       entry points (see Reproduce)
analysis/
  portfolio_diagnostics/   steps 1-7 of the diagnostics review
  experiments/             pipeline.py (parameterised copy of pca.run_curve, bit-identical with no hook),
                           run_experiments.py, cost_round1.py
  submission_*.py          numbers, figures and PDF for submission/
tests/          leakage suite, loader tests, model tests
reports/        generated reports, tables and figures
results/        cost round 1 outputs
```

## Known limitations

- **No model is net positive at the 1x cost table.**
  - PCA-3 sparse with frozen sizing and a vol gate reaches +0.09 net Sharpe, but its deflated Sharpe is 0.009 at 35 trials, so it cannot be told apart from zero.
- **PCA-3 sparse 2010–13 near-singular hedge.** Before the 7y has a full training window, the 3y residual is hedged with a nearly singular 2y/5y/10y solve, with weights up to +68 / −121 / +69.
  - It passes the `cond < 1e8` check in `sparse_hedges`.
  - Trades entered in those four folds lost 34 risk units (13 distinct trades).
  - It is deliberately not fixed, because the fix would be chosen after seeing results.
- **PCA-3 sparse members overlap.** The 2y, 3y and 5y members are often the same portfolio, so its 6 members are 2.3 effective bets.
- **Bands often sit at the top of the grid.** The entry band is 2.5 in 45–70% of PCA-2 folds.
- **The 60-bar OU half-life is biased fast.** The median is 6 bars against 103 on the 5-year training fit, mostly from small-sample AR(1) bias, so σ_eq-based forecasts overstate the move.
- **Not modelled:** repo financing, carry, roll-down, specialness and futures basis. The curve is traded in cash-yield bp. Costs are a flat 0.25 bp per leg; a time-varying table (0.5 bp before 2008, 30y at 1.5x) makes every model worse.
- **Credit pairs are a research proxy.** HY and IG OAS cannot be traded as modelled.
- **Trial count.** 35 documented trials, or 47 including the step-8 experiments. Several parameters were proposed after full-sample results were seen; see `analysis/portfolio_diagnostics/report.md`, Step 8, provenance table.
- **History before publication.** Development predates this repository, so earlier versions are reconstructed from dated table snapshots (`reports/tables_before_breadth/`, `reports/tables_27_spreads/`).
