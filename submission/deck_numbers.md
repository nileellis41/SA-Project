# Deck numbers: Sleeve 1 (PCA curve residual mean reversion)

All results are out of sample from the walk-forward folds: 5y (1,260-bar) train, 1y (252-bar) test, 10-bar purge, 5-bar embargo, 0.25 bp per Treasury leg one way, 10% vol target. They come from the cached production runs of 2026-10-19 and the cost round of 2026-10-23. No model was refitted for this file.

**How sources are cited.**
- Code is cited as `file:line`.
- Markdown reports are cited as `file:line`.
- CSVs are cited as `file → row key` (the row is selected by its `model`, `member` or `tenor` column).
- `diag` = `analysis/portfolio_diagnostics`.
- `sub` = `submission/tables`, which is written by `analysis/submission_numbers.py`.
- **[RECOMPUTED]** marks a number that no existing table logged. It was computed for this file from the cached runs, without refitting.

---

## A. Signal mechanics (verified against code)

| item | value | source |
|---|---|---|
| Residual | Each tenor's daily yield change in bp, with the first k PCs projected out: `resid = Δy · Wᵀ`, `W = I − LLᵀ` (PCA-2 full). For PCA-3 sparse, each row is the own tenor hedged with its k nearest tenors. The signal series is the cumulative residual `C_t`. | `src/amp_sa/sleeves/pca.py:156-160`, `sparse_hedges` `pca.py:101-120` |
| AR(1) fit | `C_t = a + b·C_{t−1} + e`, fitted by rolling moments on a **60-bar** trailing window (`OU_WINDOW`) | `pca.py:36`, `rolling_ou` `pca.py:50-69` |
| Refit frequency | OU stats are **re-estimated every bar** (rolling window). PCA loadings and bands are refit **once per fold** (each 252-bar test year). | `pca.py:57-69`; `pca.py:156`, `174` |
| b | `b = cov(C_t, C_{t−1}) / var(C_{t−1})` | `pca.py:60` |
| a | `a = mean(C_t) − b·mean(C_{t−1})` | `pca.py:61` |
| m (equilibrium) | `m = a / (1 − b)` | `pca.py:66` |
| σ_eq | `σ_eq = sqrt(var(e) / (1 − b²))`, with `var(e) = var(C_t) − b²·var(C_{t−1})` (moments, floored at 0) | `pca.py:62`, `67` |
| z (s-score) | `z = (C_t − m) / σ_eq` | `pca.py:69` |
| Half-life | `HL = −ln 2 / ln b` bars; κ = ln 2 / HL per bar (logged) | `pca.py:64`, `pca.py:187` |
| Screen | Tradeable only if `0 < b < 1`, `HL < window/2 = 30 bars` and `σ_eq > 0`; otherwise z is NaN and the position is flat | `pca.py:63-68` |
| Bars passing the screen (live bars) | PCA-2 **89.8%**, PCA-3 sparse **96.0%** | `diag/tables/s5_ou_bias.csv` → `observed_screen_pass_share` |
| Bars passing, per member | PCA-2: 87.1% (7y) to 94.6% (3y). PCA-3 sparse: 95.3% to 99.2%. **[RECOMPUTED]** | `sub/c_ou_by_tenor.csv` → `screen_pass_share` |
| Entry grid | {1.0, 1.25, 1.5, 1.75, 2.0, 2.5} | `src/amp_sa/config.py:105` |
| Exit grid | {0, 0.25, 0.5, 0.75}, with exit < entry | `config.py:106`, `ou.py:82` |
| Band selection criterion | **Net** per-bar Sharpe (mean/sd of daily P&L **after 1x cost**, positions sized signal/vol) on the training window, per member per fold | `optimize_bands` `src/amp_sa/models/ou.py:73-90`; called at `pca.py:174` |
| Time stop / max holding period | **No** in production. A time stop exists only as an experiment hook (`Rule.time_stop`, tested as `time_stop_20`). | `ou.band_positions` `ou.py:44-69` has none; experiment `analysis/experiments/pipeline.py` `positions()` |
| Stop-loss | **Yes**, on the s-score: if \|z\| > `STOP_Z` = **4.0**, the position is closed and there is no re-entry until \|z\| is back inside the entry band. There is no P&L stop. | `config.py:107`, `ou.py:55-59` |
| Half-life used besides the screen? | **No** for the production PCA models (sizing and timing do not use it). It is used in the Leung-Li variant (bands from the median training HL, `pca.py:178`), which is not the headline, and in the pairwise screen and health monitor (`rv.py:52`, `rv.py:155`). | as listed |
| Member sizing | `position = signal / vol`. `vol` = EW std of daily residual change, **span 60** (min 30 bars), floored at 25% of its own expanding median. The signal is ±1 or 0. | `src/amp_sa/backtest/engine.py:19-25`, `63`; `config.py:19` |
| Sleeve sizing | Equal-weight mean over live members, then scaled to a **10%** vol target on max(EW-120 std, expanding std) of the sleeve P&L through t−1, with leverage capped at 3x. Drawdown overlay **off** for Sleeve 1. | `src/amp_sa/run_sleeve1.py:48-57`; `src/amp_sa/portfolio/combine.py:17`, `21-31`, `114`; `config.py:122-123` |
| Cost | Per bar: `|Δposition| × Σ_legs |w_leg| × 0.25 bp` one way, plus `|Δ sleeve weight| × mean |p| × unit cost`. | `engine.py:66-71`, `pca.py:171`, `combine.py` (hold_cost) |
| Timing | The signal at the close of t sets the position held over t+1. | `engine.py` docstring, lines 3-6 |
| Sign convention | `z > +entry` → state −1, i.e. short the residual **in yield units**. That profits if the residual yield falls, so in bond terms you **buy the own-tenor bond** (cheap: its yield is high relative to the factors) and take the opposite side of each hedge leg (sell legs whose W weight is negative). `z < −entry` → the mirror image. Worked case: the SVB 5y trade (section K) has z = +2.87, DV01 −$2,012/bp on the 5y (long 5y bonds) and positive $/bp on the 2y, 3y, 7y, 10y and 30y (short those). | `ou.py:65-68`; `sub/k_svb_5y_trades.csv` |

**Chosen bands per member** (median, mode and share of folds at the top of the grid) **[RECOMPUTED]**: `sub/a_bands_by_member.csv`. Each value is a per-fold band from the run diagnostics.

| model | member | folds | entry median | entry mode | share at 2.5 (grid top) | exit median | exit mode |
|---|---|---|---|---|---|---|---|
| PCA-2 / headline | 2y | 33 | 2.0 | 2.5 | 45% | 0.25 | 0 |
| PCA-2 / headline | 3y | 17 | 2.5 | 2.5 | 59% | 0.25 | 0 |
| PCA-2 / headline | 5y | 33 | 2.5 | 2.5 | 70% | 0.5 | 0.5 |
| PCA-2 / headline | 7y | 13 | 2.5 | 2.5 | 69% | 0.25 | 0 |
| PCA-2 / headline | 10y | 33 | 2.5 | 2.5 | 55% | 0.25 | 0 |
| PCA-2 / headline | 30y | 33 | 2.5 | 2.5 | 55% | 0 | 0 |
| PCA-3 sparse | 2y | 33 | 1.75 | 2.5 | 27% | 0 | 0 |
| PCA-3 sparse | 3y | 17 | 2.0 | 2.5 | 41% | 0 | 0 |
| PCA-3 sparse | 5y | 33 | 1.75 | 2.5 | 27% | 0 | 0 |
| PCA-3 sparse | 7y | 13 | 2.5 | 2.5 | 77% | 0 | 0 |
| PCA-3 sparse | 10y | 33 | 2.0 | 2.5 | 48% | 0 | 0 |
| PCA-3 sparse | 30y | 33 | 2.0 | 2.5 | 48% | 0 | 0 |
| headline credit | hy_ig_oas | 22 | 2.25 | 2.5 | 50% | 0.25 | 0.25 |
| headline credit | cdx_cash_basis | 10 | 1.375 | 1.5 | 0% | 0.5 | 0.5 |
| headline credit | hyg_cdx | 11 | 1.75 | 1.75 | 9% | 0.25 | 0 |

**Flag: the entry band sits at the top of the grid (2.5) in 45–70% of PCA-2 folds.** The optimiser would probably have chosen higher if the grid allowed it. The step-8 experiments `entry_2.5` and `entry_3.0` (`diag/report.md:627-632`) show what fixed higher bands do.

**Reversion vs round-trip cost (PCA-2, entries).** Source: `diag/tables/s5_cost_aware_entry_summary.csv`, also `diag/report.md:393`.

| measure | bp |
|---|---|
| Median OU-forecast reversion, (\|z\| − exit) × σ_eq | **1.667** |
| Median round-trip cost | **0.725** |
| Mean realised move per trade | **0.454** |
| Median realised move per trade | 0.876 |

## B. PCA, fold with test window starting 2022-12-12 (covers Dec 2022 – Nov 2023) **[RECOMPUTED]**

- **Training window:** 2018-01-22 to 2022-11-18, 1,259 daily changes, six tenors (2y, 3y, 5y, 7y, 10y, 30y).
- **Check against the log:** the recomputed eigenvectors equal the logged fold loadings (`extras["fold_loadings"]`) to a max absolute gap of 0.
- **Sign convention:** eigenvector signs are arbitrary. Here level is negative on every tenor.
- Source: `sub/b_pca_fold.csv`.

**Covariance of daily changes, bp²** (`sub/b_covariance_bp2.csv`)

| | 2y | 3y | 5y | 7y | 10y | 30y |
|---|---|---|---|---|---|---|
| 2y | 20.16 | 21.28 | 21.08 | 20.01 | 18.26 | 13.54 |
| 3y | 21.28 | 23.88 | 24.49 | 23.74 | 22.01 | 16.76 |
| 5y | 21.08 | 24.49 | 27.25 | 27.35 | 26.08 | 20.96 |
| 7y | 20.01 | 23.74 | 27.35 | 28.59 | 27.94 | 23.72 |
| 10y | 18.26 | 22.01 | 26.08 | 27.94 | 28.32 | 25.49 |
| 30y | 13.54 | 16.76 | 20.96 | 23.72 | 25.49 | 26.52 |

**Eigenvalues** (`sub/b_eigen.csv`): PC1 = 138.0 bp² (**89.2%**), PC2 = 13.88 (**9.0%**, cumulative 98.2%), PC3 = 2.14 (**1.4%**, cumulative 99.6%).

**Loadings** (`sub/b_loadings.csv`)

| tenor | level (PC1) | slope (PC2) | curvature (PC3) |
|---|---|---|---|
| 2y | −0.337 | −0.516 | −0.552 |
| 3y | −0.391 | −0.425 | −0.131 |
| 5y | −0.438 | −0.165 | +0.384 |
| 7y | −0.451 | +0.073 | +0.426 |
| 10y | −0.442 | +0.295 | +0.200 |
| 30y | −0.379 | +0.659 | −0.557 |

**Diagonal of W = I − LLᵀ** (`sub/b_W_diag.csv`)

| tenor | PCA-2 (k = 2) | PCA-3 (k = 3, full) |
|---|---|---|
| 2y | 0.620 | 0.316 |
| 3y | 0.666 | 0.649 |
| 5y | 0.781 | 0.634 |
| 7y | 0.792 | 0.610 |
| 10y | 0.718 | 0.678 |
| 30y | 0.423 | 0.113 |
| **trace** | **4.000** (= 6 − 2) | **3.000** (= 6 − 3) |

**5y package weights**, bp of yield per unit of the 5y residual (`sub/b_5y_package.csv`). The PCA-2 weights equal the 5y row of W, which confirms the logged legs.

| leg | PCA-2 full (row of W) | PCA-3 sparse (own + 3 nearest) |
|---|---|---|
| 2y | −0.233 | +0.741 |
| 3y | −0.241 | −1.347 |
| 5y | +0.781 | +1.000 |
| 7y | −0.185 | −0.356 |
| 10y | −0.145 | 0 |
| 30y | −0.057 | 0 |

## C. OU fitted values, out of sample, per tenor **[RECOMPUTED]**

These are taken over live test bars where the 60-bar screen passed, from the per-bar `[diag-logging]` OU state. HL is in bars, κ is per bar, and σ_eq is in bp of cumulative residual. The headline's curve members are the PCA-2 members, so they are identical. The credit pairs use a different (Kalman) model: their per-fold half-lives are in `diag/tables/s5_ou_by_fold_headline.csv`. Source: `sub/c_ou_by_tenor.csv`.

| model | tenor | HL median | HL IQR | κ median | σ_eq median, bp | σ_eq IQR, bp |
|---|---|---|---|---|---|---|
| PCA-2 / headline | 2y | 6.36 | 4.19–10.55 | 0.109 | 1.60 | 1.01–2.62 |
| PCA-2 / headline | 3y | 5.50 | 3.76–8.68 | 0.126 | 0.91 | 0.60–1.39 |
| PCA-2 / headline | 5y | 5.98 | 3.83–9.89 | 0.116 | 1.47 | 0.99–2.40 |
| PCA-2 / headline | 7y | 7.08 | 3.96–11.81 | 0.098 | 1.11 | 0.88–1.64 |
| PCA-2 / headline | 10y | 6.28 | 3.74–10.60 | 0.110 | 1.10 | 0.72–1.68 |
| PCA-2 / headline | 30y | 6.99 | 4.44–11.45 | 0.099 | 1.53 | 1.13–2.37 |
| PCA-3 sparse | 2y | 4.49 | 2.83–7.64 | 0.154 | 5.27 | 2.35–10.56 |
| PCA-3 sparse | 3y | 4.25 | 2.86–6.89 | 0.163 | 1.48 | 0.88–2.89 |
| PCA-3 sparse | 5y | 4.49 | 2.83–7.64 | 0.154 | 2.17 | 1.44–3.50 |
| PCA-3 sparse | 7y | 2.98 | 1.98–4.97 | 0.232 | 0.70 | 0.54–0.93 |
| PCA-3 sparse | 10y | 5.03 | 2.89–8.92 | 0.138 | 1.32 | 0.89–2.17 |
| PCA-3 sparse | 30y | 5.03 | 2.89–8.92 | 0.138 | 3.18 | 2.24–5.31 |

Logged pooled values, for context (`diag/report.md:375-385`):
- The training-window AR(1) half-life has a median of **102.8 bars (PCA-2)** and **31.7 (PCA-3 sparse)**.
- The trailing 60-bar median is **6.35** and **4.49** bars.
- At entry it is **3.54** and **3.01** bars.
- The median holding period is **10** bars.

## D. Hit ratio and conditional gain/loss **[RECOMPUTED]**

**Units: member risk units.** One unit is one trailing daily standard deviation of that member's residual: the EW span-60 vol at entry, i.e. a **per-trade daily sigma**. These are **not σ_eq and not bp**. A trade's P&L is Σ position × Δresidual over its bars, with position = 1/vol.

Definitions:
- **W** = mean net P&L of winning trades (net > 0).
- **L** = mean |net| of losing trades.
- **Expectancy** = p·W − (1−p)·L, which equals the mean net.
- **Breakeven hit rate** = L / (W + L).

Source: `sub/d_cg_cl.csv`, built from the trade tables `diag/tables/s3_trades_<model>.csv`. Flies are rebuilt with `run_members(fixed=BENCHMARKS)`.

| model | basis | trades | gross hit | net hit | mean gross | mean cost | mean net | W | L | W/L | expectancy | breakeven hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Headline | member trades | 426 | 68.1% | 54.5% | 0.714 | 1.333 | −0.619 | 1.944 | 3.684 | 0.528 | −0.619 | 65.5% |
| PCA-3 sparse | member trades (unmerged) | 691 | 86.0% | 64.7% | 1.921 | 1.907 | +0.014 | 1.528 | 2.761 | 0.554 | +0.014 | 64.4% |
| PCA-3 sparse | merged (identical copies once) | 255 | 83.1% | 57.6% | 1.744 | 2.077 | −0.333 | 1.522 | 2.857 | 0.533 | −0.333 | 65.2% |
| PCA-2 full | member trades | 344 | 67.2% | 54.4% | 0.689 | 1.394 | −0.704 | 2.037 | 3.970 | 0.513 | −0.704 | 66.1% |
| Pairwise NN | member trades | 252 | 69.0% | 54.0% | 0.479 | 1.100 | −0.621 | 1.888 | 3.564 | 0.530 | −0.621 | 65.4% |
| Fly 2s5s10s | member trades | 60 | 56.7% | 51.7% | −0.192 | 1.067 | −1.259 | 1.897 | 4.633 | 0.410 | −1.259 | 70.9% |
| Fly 3s5s7s | member trades | 43 | 74.4% | 44.2% | 1.118 | 2.265 | −1.147 | 1.737 | 3.430 | 0.506 | −1.147 | 66.4% |

**Reading the table:** every model wins 54–65% of trades net but needs about 65% to break even, because each loser costs about twice what a winner earns (W/L ≈ 0.5). The logged equivalents are in `reports/tables/lessons_trade_summary.csv`: headline net hit 54.5% and avg win/avg loss 0.528 match.

**W and L by entry |z|**, member trades (`sub/d_cg_cl_by_z.csv`; the logged counts and means match `diag/report.md:352-369`):

| model | \|z\| | trades | net hit | W | L | expectancy |
|---|---|---|---|---|---|---|
| Headline | <2 | 233 | 53.6% | 1.90 | 3.60 | −0.65 |
| Headline | 2–2.5 | 98 | 58.2% | 1.95 | 3.46 | −0.32 |
| Headline | 2.5–3 | 78 | 48.7% | 2.00 | 3.92 | −1.04 |
| Headline | 3+ | 17 | 70.6% | 2.21 | 5.31 | −0.00 |
| PCA-3 sparse | <2 | 462 | 66.5% | 1.47 | 2.48 | +0.15 |
| PCA-3 sparse | 2–2.5 | 94 | 63.8% | 1.67 | 3.23 | −0.10 |
| PCA-3 sparse | 2.5–3 | 111 | 60.4% | 1.54 | 3.82 | −0.59 |
| PCA-3 sparse | 3+ | 24 | 54.2% | 2.11 | 1.06 | +0.66 |
| PCA-2 full | <2 | 173 | 50.9% | 2.03 | 3.92 | −0.89 |
| PCA-2 full | 2–2.5 | 86 | 60.5% | 1.97 | 3.61 | −0.24 |
| PCA-2 full | 2.5–3 | 71 | 49.3% | 2.09 | 4.10 | −1.05 |
| PCA-2 full | 3+ | 14 | 85.7% | 2.21 | 9.82 | +0.49 |
| Pairwise NN | <2 | 158 | 57.6% | 1.57 | 3.12 | −0.42 |
| Pairwise NN | 2–2.5 | 43 | 51.2% | 2.67 | 5.48 | −1.31 |
| Pairwise NN | 2.5–3 | 36 | 58.3% | 2.59 | 3.01 | +0.25 |
| Pairwise NN | 3+ | 15 | 13.3% | 0.34 | 3.38 | −2.88 |

The 3+ buckets have 14–24 trades, so read them as noise.

## E. Model comparison (1x cost)

| model | gross SR | net SR | breakeven cost x | IC, 10d, entry bars (p) | trades/yr | cost % cap pa | effective bets |
|---|---|---|---|---|---|---|---|
| Headline | 0.547 | −0.599 | 0.475 | 0.166 (0.001) | 13.13 | 9.58 | 8.69 |
| PCA-3 sparse | 1.210 | −0.087 | 0.934 | 0.326 (<0.001) | 7.86 | 10.76 | 2.28 |
| PCA-2 full | 0.448 | −0.668 | 0.398 | 0.196 (<0.001) | 10.60 | 9.67 | 5.71 |
| Pairwise NN | 0.459 | −0.257 | 0.640 | 0.127 (0.044) | 7.66 | 5.55 | 14.37 |
| Fly 2s5s10s | 0.109 | −0.422 | 0.215 | — | 60 trades | 3.86 | — |
| Fly 3s5s7s | 0.382 | −0.766 | 0.315 | — | 43 trades (from 2014) | 10.12 | — |

Sources:
- **Sharpe, breakeven and cost:** `diag/tables/s2_breakeven.csv`, also in `diag/report.md:46-53`.
- **IC:** Spearman rank correlation of −z with the forward 10-bar member change, on entry bars, from `diag/tables/s5_ic_<model>.csv` → horizon 10, entry bars. The all-bars IC is 0.082, 0.211, 0.064 and 0.049.
- **Trades/yr:** `analysis/experiments/results.csv` → baseline (merged trades), and for pairwise `reports/tables/lessons_trade_summary.csv` (7.656).
- **Fly trade counts:** `diag/tables/s2_benchmark_trades.csv`.
- **Effective bets:** `diag/tables/s7_effective_bets.csv` → gross.

**Inconsistency:** `lessons_trade_summary.csv` gives headline trades/yr as **12.94**, against **13.13** in `experiments/results.csv`. They use the same 426 trades with different year denominators (calendar span vs live bars / 252). Use one and say which.

## F. Quiet vs volatile entries

- **Definition:** entry vol = the member's trailing residual vol on the entry bar, in bp/day. It is the EW span-60 std used for sizing, read off the position as `|signal| / |position|` (`diag/signals_regimes.py:52-63`).
- **Terciles** are taken over each model's trades, pooled across members and over the full sample. That makes them **ex post and descriptive**, not a tradeable rule.
- **Cutoffs, in bp/day** **[RECOMPUTED]** (`sub/f_tercile_cuts.csv`, member trades):
  - headline: 0.50 / 1.03
  - PCA-3 sparse: 0.92 / 1.54
  - PCA-2: 0.47 / 0.77
  - pairwise NN: 1.23 / 3.01

**Results**, risk units per trade (`diag/tables/s4_cost_vs_vol_<model>.csv`, `diag/report.md:302-315`)

| model | tercile | trades | median entry vol, bp | round trip / vol | mean gross | mean cost | mean net |
|---|---|---|---|---|---|---|---|
| Headline | low | 142 | 0.39 | 1.78 | 0.80 | 2.05 | −1.25 |
| Headline | mid | 142 | 0.68 | 1.01 | 0.86 | 1.13 | −0.27 |
| Headline | high | 142 | 4.40 | 0.59 | 0.49 | 0.82 | −0.33 |
| PCA-3 sparse | low | 231 | 0.66 | 2.32 | 2.07 | 2.55 | −0.48 |
| PCA-3 sparse | mid | 230 | 1.16 | 1.63 | 1.88 | 1.84 | +0.04 |
| PCA-3 sparse | high | 230 | 3.29 | 1.14 | 1.81 | 1.33 | +0.48 |
| PCA-2 full | low | 115 | 0.37 | 1.96 | 0.73 | 2.19 | −1.47 |
| PCA-2 full | mid | 114 | 0.57 | 1.16 | 1.09 | 1.28 | −0.20 |
| PCA-2 full | high | 115 | 1.08 | 0.61 | 0.26 | 0.70 | −0.45 |
| Pairwise NN | low | 84 | 0.77 | 1.30 | −0.14 | 1.63 | −1.77 |
| Pairwise NN | mid | 84 | 1.68 | 0.61 | 0.70 | 0.71 | −0.01 |
| Pairwise NN | high | 84 | 7.89 | 0.63 | 0.87 | 0.96 | −0.09 |

**Heads-up, two tercile bases.** The headline's high-tercile median (4.40 bp) is the credit pairs, which are pooled in. The cost-round-1 terciles (section H) use the baseline's **merged** trades, so its PCA-3 sparse cuts differ (0.98 / 3.27, 85 trades each).

## G. Per dollar: $1m of capital, at the 10% vol target

**Basis:** capital, not notional. Returns are additive fractions of $1m. Exposure is reported as DV01. Sources: `diag/tables/s2_breakeven.csv` (`gross_return_pa`, `cost_pa_1x`) and `diag/tables/s3_factor_exposure_summary.csv` (mean gross and net DV01).

| model | gross $/yr | cost $/yr | net $/yr | per $1 of gross: cost | per $1 of gross: take-home | mean gross DV01 $/bp | mean net DV01 $/bp |
|---|---|---|---|---|---|---|---|
| Headline | 45,500 | 95,800 | −50,300 | $2.10 | −$1.10 | 6,632 | 446 |
| PCA-3 sparse | 100,500 | 107,600 | −7,100 | $1.07 | −$0.07 | 9,374 | 175 |
| PCA-2 full | 38,500 | 96,700 | −58,200 | $2.51 | −$1.51 | 7,596 | 498 |
| Pairwise NN | 35,600 | 55,500 | −20,000 | $1.56 | −$0.56 | 3,856 | 263 |

The per-$1 columns are ratios of the logged values (cost ÷ gross, 1 − cost ÷ gross) **[RECOMPUTED]**. Net $/yr equals `ann_return` × $1m from `diag/tables/s2_kpis.csv`. The take-home figures are the reciprocal view of the breakeven multiple: 1/0.934 = $1.07 of cost per $1 of gross.

## H. Cost round 1 (`results/cost_round1/summary.md`, `results.csv`)

**1x cost**

| model | config | gross SR | net SR | trades/yr | net hit | mean net/trade | cost % pa | breakeven x |
|---|---|---|---|---|---|---|---|---|
| Headline | baseline | 0.547 | −0.599 | 13.13 | 54.5% | −0.619 | 9.58 | 0.475 |
| Headline | freeze_size | 0.557 | −0.351 | 13.13 | 56.1% | −0.454 | 8.22 | 0.612 |
| Headline | freeze + vol_gate | 0.339 | −0.224 | 8.36 | 55.9% | −0.377 | 4.53 | 0.602 |
| PCA-3 sparse | baseline | 1.210 | −0.087 | 7.86 | 57.6% | −0.333 | 10.76 | 0.934 |
| PCA-3 sparse | freeze_size | 1.168 | +0.073 | 7.86 | 60.8% | −0.118 | 9.54 | 1.065 |
| PCA-3 sparse | freeze + vol_gate | 0.559 | +0.090 | 3.38 | 61.3% | −0.098 | 3.90 | 1.189 |

**Net Sharpe at 0.5x / 2x cost** (bands and X fitted at 1x; the same positions re-priced)

| model | baseline | freeze_size | freeze + vol_gate |
|---|---|---|---|
| Headline | −0.028 / −1.637 | +0.103 / −1.196 | +0.058 / −0.763 |
| PCA-3 sparse | +0.576 / −1.284 | +0.633 / −0.950 | +0.328 / −0.375 |

**Entry-vol terciles**, 1x, mean net per trade in risk units. The cut points come from the baseline's merged trades (`results/cost_round1/terciles.csv`).

| model | tercile (bp/day) | baseline n | baseline mean net | freeze+gate n | freeze+gate mean net |
|---|---|---|---|---|---|
| Headline | low (< 0.50) | 142 | −1.25 | 35 | −1.87 |
| Headline | mid | 142 | −0.27 | 97 | −0.37 |
| Headline | high (> 1.03) | 142 | −0.33 | 131 | +0.02 |
| PCA-3 sparse | low (< 0.98) | 85 | −0.91 | 21 | −0.89 |
| PCA-3 sparse | mid | 85 | −0.24 | 45 | −0.26 |
| PCA-3 sparse | high (> 3.27) | 85 | +0.14 | 40 | +0.50 |

- **Vol gate X** was fitted per member per fold on the training window (percentiles 10–50 of trailing residual vol, maximising training net per trade). **It counts as a trial.** The X per fold is in `results/cost_round1/vol_gate_x_by_fold.csv`.
- **Is either model net positive at 1x?** The headline is not. PCA-3 sparse is marginally positive (+0.07 / +0.09), but its deflated Sharpe is 0.009, so it cannot be told apart from zero.

## I. PCA-3 sparse Sharpe basis

- **Yes.** The −0.087 net Sharpe is computed on the book where every live member is a separate position: an equal-weight mean over members at `src/amp_sa/run_sleeve1.py:48-57` (`avg(...)` = `.mean(axis=1)`), then `combine()`.
- **Overlap shifts risk share; it doesn't add leverage.** Members that are the same portfolio (e.g. 2y = 3y = 5y) take a c/n share of sleeve risk, but the 10% vol target sets total size. Before 2010 all four members are one portfolio and the book holds it once.
- **Merged basis** (each distinct portfolio given one equal share per bar; `results/cost_round1/merged_basis.csv`): net Sharpe **−0.097** (baseline) and **+0.061** (freeze + vol_gate); freeze_size alone gives +0.064. Overlap changes the result by only 0.01–0.03.
- **Do not equate trade-table sums with book P&L.** +9.37 (691 member trades) and −84.81 (255 merged trades) are unweighted risk-unit sums, and neither is the book's P&L.

## J. Project evolution

The repo has no git history, so the steps come from dated table snapshots and the documented revision lists (`reports/appendix.md:526-527`). The net/gross Sharpe for steps 1–2 is the core-universe pairwise sleeve (`scope = core`).

| # | step (date of tables) | model | gross SR | net SR | source |
|---|---|---|---|---|---|
| 1 | Pairwise sleeve, first universe (10 spreads), 2026-10-12 | static | 0.297 | −0.047 | `reports/tables_before_breadth/sleeves.csv` → core, rv |
| | | adaptive HMM | 0.466 | +0.065 | same |
| | | adaptive NN | 0.511 | −0.008 | same |
| 2 | Universe widened to 27 spreads, list fixed before running, 2026-10-12 | static | 0.136 | −0.338 | `reports/tables_27_spreads/sleeves.csv` → core, rv |
| | | adaptive HMM | 0.410 | −0.161 | same |
| | | adaptive NN | 0.459 | −0.257 | same |
| 3 | 31 spreads (current) | the core pairwise rows are identical to step 2 | | | `reports/tables/sleeves.csv` |
| 4 | Undated design revisions: adaptive model changed twice, cointegration screen tightened to both tests, vol estimate made more conservative | — | n/a | n/a (no Sharpe saved) | `reports/appendix.md:526` |
| 5 | PCA residual models; headline named in advance (`run_sleeve1.HEADLINE`) | PCA-2 full | 0.448 | −0.668 | `reports/tables/sleeve1_models.csv` |
| | | PCA-3 full | 0.923 | −0.321 | same |
| | | PCA-2 sparse | 0.936 | −0.349 | same |
| | | **PCA-3 sparse** | **1.210** | **−0.087** | same |
| | | **Headline (PCA-2 + credit)** | **0.547** | **−0.599** | same |
| | | PCA-2 / PCA-3 Leung-Li | −0.048 / 0.259 | −0.511 / −0.307 | same |
| 6 | Cost filters and 2s10s curve rules | 8 + 8 variants | | | `reports/tables/cost_filter_effect.csv`, `curve_full.csv` |
| 7 | Diagnostics and step-8 experiments, 2026-10-19 | best: PCA-3 sparse `resize_25pct` | 1.200 | −0.002 | `diag/report.md:616-638` |
| 8 | Cost round 1, 2026-10-23 | best: PCA-3 sparse freeze + vol_gate | 0.559 | +0.090 | `results/cost_round1/results.csv` |

**Trial count.**
- **35 documented** (`analysis/portfolio_diagnostics/run_all.py`, `TRIALS`): 11 model-table rows + 8 cost filters + 8 curve rules + 4 design revisions + 4 cost-round-1 configs.
- **47** if you add the 12 step-8 experiments, which were never added to `TRIALS`.

**Deflated Sharpe**

| trial set | best | net SR | DSR net | DSR gross | source |
|---|---|---|---|---|---|
| model table, 11 | PCA-3 sparse | −0.087 | 0.016 | 1.000 | `diag/tables/s7_deflated_sharpe.csv` |
| documented, 31 | PCA-3 sparse | −0.087 | 0.000 | 1.000 | same |
| documented incl. round 1, 35 | PCA-3 sparse freeze + gate | +0.090 | 0.009 | 0.609 | `results/cost_round1/deflated_sharpe.csv` |
| + step-8 experiments, 47 | same | +0.090 | 0.002 | 0.421 | same |

**PCA-3 sparse baseline deflated Sharpe at 47 trials** **[RECOMPUTED]** (trial Sharpe s.d. 0.263, as in `results/cost_round1/deflated_sharpe.csv`): net **0.0001**, gross **0.9999**. This is still 0.000 / 1.00 at the precision printed on the original slides 14, 21 and 22, so those slides only change their trial count from 31 to 47.

## K. SVB-period 5y trade, end to end **[RECOMPUTED]**

This is the only 5y-member trade open between 2023-02-15 and 2023-04-28 in the headline or PCA-3 sparse. It is the headline's `pca2_ust_5y` member (identical to PCA-2 full). Fold test start 2022-12-12, so the weights are the section B 5y row of W. Source: `sub/k_svb_5y_trades.csv`.

| item | value |
|---|---|
| Entry (signal bar; first held bar is the next day) | **2023-03-13** (the first trading day after SVB was closed on Friday 2023-03-10) |
| z at entry | **+2.87** (5y residual yield high → 5y cheap → short the residual in yield units) |
| σ_eq at entry | **2.80 bp** |
| Gap from equilibrium, z × σ_eq | **8.03 bp** |
| 60-bar half-life at entry | 4.73 bars |
| Residual vol at entry (sizing) | 1.90 bp/day → position −0.526 risk units |
| Package weights (bp of yield) | 2y −0.233, 3y −0.241, **5y +0.781**, 7y −0.185, 10y −0.145, 30y −0.057 |
| DV01 held, $/bp at $1m capital (P&L per +1 bp in each yield) | 2y +600, 3y +622, **5y −2,012 (long 5y)**, 7y +477, 10y +372, 30y +147. Net +$206/bp; gross $4,230/bp |
| Exit | **2023-04-10**, 20 bars held. This fold's band is entry 2.5 / exit 0.5, and z first reached ≤ 0.5 here (0.49). |
| Residual move, entry to exit | −6.20 bp (in the trade's favour) |
| Gross / cost / net, risk units | **+3.26 / 0.52 / +2.74** |
| Gross / cost / net, $ at $1m | **+$15,300 / $2,400 / +$12,900** |

Cost in $ includes the entry and exit charges and the daily resizes in between.

## Defect (used in report §6)

- **Hedge weights.** The PCA-3 sparse 3y member, 2010–13, carries weights of **+68.2 (2y) / −120.9 (5y) / +68.9 (10y)** per unit of 3y. Source: `diag/tables/s3_leg_weights_pca3_sparse.csv` → member pca3s_ust_3y, date 2010-05-21; text in `diag/report.md:248`.
- **Losses in the affected folds.** −34.13 over 13 merged trades, 20.06 of it from the 3y; −67.45 over 29 member trades. Source: `results/cost_round1/summary.md`, Caveats.

---

## Recomputed items

The following are marked **[RECOMPUTED]**, each computed from the cached runs without refitting:
- per-member bands (A)
- per-member screen pass share (A)
- B in full
- C per-tenor values
- D (W, L, expectancy, breakeven hit rate, z buckets, flies)
- F tercile cutoffs
- G per-$1 ratios
- K in full

Script: `analysis/submission_numbers.py`, runtime 23 s.
