# Sleeve 1 trade guide: what the trade is, why it loses, how to cut cost

A study guide built from the numbers in `analysis/portfolio_diagnostics/report.md` and its `tables/`. Nothing here is a new backtest result. Where a number is an in-sample estimate or a hypothesis, it says so.

Units used throughout:
- **bp**: yield basis points.
- **risk unit (σ)**: one trailing daily standard deviation of the traded residual. A trade that makes 1σ made one typical day's move.
- **% of capital**: at the sleeve's 10% vol target.

---

## 1. The trade in one page

The headline model is PCA-2 on the Treasury curve plus the adaptive credit pairs. Most of the cost and most of the lesson sit in the curve part, so this guide follows that.

**What you hold.** Each of the six tenors (2y, 3y, 5y, 7y, 10y, 30y) has a *residual*: its yield change with the level and slope factors (PC1, PC2) projected out. Holding one unit of the 5y residual means holding a portfolio of every tenor that is neutral to level and slope. This is the actual 5y package for the fold starting 2022-12-12 (`tables/s3_leg_weights_pca2.csv`):

| tenor | 2y | 3y | 5y | 7y | 10y | 30y |
|---|---:|---:|---:|---:|---:|---:|
| weight | −0.23 | −0.24 | **+0.78** | −0.19 | −0.14 | −0.06 |

Long the 5y yield, short a bit of everything else. It is a belly-vs-curve bet, close to a 5y-centred butterfly with weights chosen by PCA rather than by hand.

**What the signal is.** The residual is added up over time (the cumulative residual) and fitted as an OU process on the **last 60 bars**. The s-score is

```
z = (cumulative residual − OU mean) / OU stationary sigma
```

It is re-estimated every bar, so the "fair value" drifts.

**When you trade.**

| step | rule | where |
|---|---|---|
| fold screen | the 60-bar window must be mean reverting with a half-life under 30 bars (Avellaneda–Lee speed filter) | `sleeves/pca.py: rolling_ou` |
| entry | \|z\| above the entry band; bands are optimised per fold on the training window, grid 1.0–2.5 | `models/ou.py: optimize_bands` |
| exit | \|z\| back inside the exit band (0, 0.25, 0.5 or 0.75) | same |
| direction | short the residual when z > 0, long when z < 0 | |
| size | position = signal / trailing vol, so every trade carries the same risk | `backtest/engine.py` |

The curve part has **no regime gate**. The HMM and neural gates only act on the pairwise credit members of the headline.

**Typical trade (PCA-2, 344 trades):** held 10 bars (median), entered at |z| ≈ 2, residual vol 0.57 bp a day (median), round-trip cost 1.1× that day's vol. 67% of trades win before cost and 54% win after.

## 2. Two real trades, same package, opposite outcomes

Both are the PCA-2 5y residual, entered within a year of each other (`tables/s3_trades_pca2.csv`).

| | Mar 2023 (SVB week) | Mar 2024 (quiet) |
|---|---:|---:|
| entry → exit | 2023-03-13 → 04-10 (20 bars) | 2024-03-08 → 03-26 (12 bars) |
| entry z | +2.87 (short the residual) | −2.55 (long the residual) |
| residual vol at entry | **1.90 bp/day** | **0.43 bp/day** |
| one-way cost of the package | 0.41 bp | 0.42 bp |
| round-trip cost / vol | 0.43 | **1.95** |
| gross (σ) | +3.26 | +3.47 |
| cost (σ) | 0.52 | **2.18** |
| net (σ) | **+2.74** | +1.29 |
| cost as share of gross | 16% | **63%** |

Both trades were right and made about the same gross in risk units. The cost in bp was the same too (0.41 bp = 0.25 bp × Σ|w| = 0.25 × 1.64). The difference is the **vol at entry**. Position size is 1/vol, so in the quiet market the book bought 4.5× more DV01 to carry the same risk, and paid 4.5× more cost per unit of risk.

> **The core mechanic.** Cost is fixed in bp. Edge scales with vol. Every trade entered in a quiet market starts in a hole of about (2 × 0.41 bp) / vol. At 0.4 bp a day vol that hole is about 2σ, and the median trade only makes about 1.9σ gross.

## 3. Why the book loses: the arithmetic

| model | gross Sharpe | net Sharpe | gross % pa | cost % pa | breakeven cost mult |
|---|---:|---:|---:|---:|---:|
| headline | 0.55 | −0.60 | 4.6 | 9.6 | 0.48× |
| PCA-3 sparse | 1.21 | −0.09 | 10.1 | 10.8 | 0.93× |
| PCA-2 full | 0.45 | −0.67 | 3.8 | 9.7 | 0.40× |
| pairwise NN | 0.46 | −0.26 | 3.6 | 5.6 | 0.64× |

- **The signal is real.** On entry bars, −z predicts the next 10 bars of residual change with rank IC 0.17 (headline) and 0.33 (PCA-3 sparse), both at p < 0.01.
- **Cost is about twice gross for PCA-2.** Per trade: 0.69σ mean gross against 1.39σ mean cost.
- **The breakeven multiplier is the target.** The headline needs the real one-way cost to be under 0.48 × 0.25 bp ≈ 0.12 bp per leg, *or* it needs to trade less or bigger, by about the same factor.
- **The realistic cost table is worse, not better.** With 0.5 bp per leg pre-2008, 0.25 bp after and 30y at 1.5×, the headline goes to −1.12 and PCA-3 sparse to −0.63.

## 4. Where the cost comes from

Headline, % of capital a year (Step 1 of the diagnostics):

| source | % pa | share |
|---|---:|---:|
| entries | 3.98 | 42% |
| exits | 3.86 | 40% |
| daily vol resizing while held | 0.78 | 8% |
| sleeve leverage changes | 0.58 | 6% |
| CDX roll (credit legs) | 0.38 | 4% |
| re-hedging at refits (not charged) | 0.004 | ~0 |

**Cost by entry vol** (headline terciles):

| entry vol | median bp/day | round trip / vol | mean net per trade |
|---|---:|---:|---:|
| low | 0.39 | 1.78 | **−1.25σ** |
| mid | 0.68 | 1.01 | −0.27σ |
| high | 4.40 | 0.59 | −0.33σ |

So **82% of cost is opening and closing trades**, and the worst of it comes from trades opened in quiet markets. Maintenance (resizing, refits) and per-member double counting (netting) are small.

## 5. How to reduce cost, ranked

Ranked by how much of the problem each lever attacks, with the evidence behind it. "Tested" means a walk-forward experiment exists in `analysis/experiments/`.

### A. Cost-aware entry: only trade when the expected move pays for the round trip. *Biggest lever, not tested yet.*

The bands are in z units and fixed per fold, but the cost hurdle in z units changes every day with vol. Make the hurdle explicit:

```
enter only if  (|z| − z_exit) × σ_eq,bp  >  k × 2 × unit_cost_bp
```

Here σ_eq,bp is the OU stationary sigma in bp, the expected reversion is (|z| − exit) σ_eq, and k ≥ 1 is a safety margin fitted on the training window like the bands. The `[diag-logging]` change to `pca.py` now saves σ_eq per bar, so the inputs already exist.

- **Why it should work:** it targets the low-vol tercile, which loses 1.25σ a trade for the headline and 0.48σ for PCA-3 sparse.
- **Catch, measured from the `[diag-logging]` OU state:** at k = 1 the rule barely binds. 92% of PCA-2 entries already pass: the median expected reversion is 1.67 bp against a 0.73 bp round trip. The 60-bar σ_eq and half-life overstate the move that actually comes; the mean realised move is 0.45 bp a trade. So k has to be fitted and will come out well above 1, or the rule has to use realised vol directly (the vol-floor variant below). The 92% figure and the per-entry check are in report.md, Step 5 (`tables/s5_cost_aware_entry_<model>.csv`). Half the PCA-2 entries would be dropped only at k ≈ 2.4.
- **In-sample upper bound (look-ahead, do not report as a result):** dropping the low-vol tercile changes PCA-3 sparse's mean net per trade from about 0.0σ to **+0.26σ**. The headline improves from −0.62σ to −0.30σ but stays negative. The tercile cut-offs are full-sample, so a causal rule will do worse than this.
- **Simpler cousin:** a vol floor, i.e. no new entries when trailing residual vol is below X bp a day, with X fitted per fold.

### B. Stop resizing held positions. *Tested in part: +0.06 to +0.09 Sharpe.*

Resizing and leverage changes are 14–17% of cost. `resize_25pct` (rebalance only if the target moves more than 25%) helped every model, and PCA-3 sparse reached a net Sharpe of −0.00. The natural next step is to **freeze size at entry**: hold fixed DV01 until exit, and let the drawdown overlay handle risk. That removes nearly all of that 14–17%.

### C. Slow the signal down. *Hypothesis, not tested.*

The 60-bar OU fit gives a median half-life of 6.5 bars. The 5-year training-window fit of the same residual gives 103 bars (PCA-2) and 32 bars (PCA-3 sparse). Most of that gap is small-sample bias, not the speed filter: an AR(1) fitted on 60 bars reads a slow process as fast. Simulating the training-window process and fitting it on 60 bars gives 8.3 bars (PCA-2) before any screen and 7.6 after it; the screen passes about 9 windows in 10 (report.md, Step 5). So the 60-bar half-life and σ_eq are noisy, optimistic estimates, and they make trades short and frequent. A longer OU window (120 or 250 bars) or a longer minimum holding period means fewer round trips per unit of edge. The risk is that gross falls with it, as in point F.

### D. Fix the PCA-3 sparse construction defect. *Construction fix.*

In 2010–2013, before the 7y has a full training window, the 3y residual is hedged with a nearly singular 2y/5y/10y solve (weights +68, −121, +69). Cost is proportional to Σ|w|, so those folds pay huge cost: trades entered there lost 67 risk units net, against +9 for the whole model. Fix it by capping Σ|w| (or the condition number much lower than 1e8), or by not trading the member until the 7y has a full window. This is a correctness fix more than a tuning choice, but it was found after seeing results, so count it as a trial.

### E. Cheaper execution. *Changes the cost table, not the strategy. Verify before using.*

The 0.25 bp per leg assumes cash Treasuries. Treasury futures quote in ticks. A half tick divided by contract DV01 comes to roughly 0.1 bp for TU/FV/TY/US in liquid hours (order of magnitude only, before commissions and roll). That is about 0.4× the table, below the headline's 0.48× breakeven and well below PCA-3 sparse's 0.93×. Two warnings:
- The Bloomberg TY1 bid/ask in `Features_Costs_1.csv` has about 2,500 missing days and should not be used to prove this.
- Micro yield futures have different ticks and commissions. Confirm the actual cost with AMP before modelling it.

Whatever the number, use it as a **separate cost scenario**, not as a new baseline.

### F. Not worth pursuing (already measured)

| idea | result |
|---|---|
| leg netting across members | saves 0.5–5.5% of cost; members rarely hold offsetting legs at once |
| charging refit re-hedges | at most 0.035% of capital a year |
| entry band 2.5 / 3.0 alone | cuts cost but leaves 0.6–6 trades a year; too few to trust |
| Leung–Li optimal levels (the PDF in the repo root) | turnover falls 5× (0.88 vs 4.37), but PCA-2 gross Sharpe falls to −0.05; the edge disappears with the trades |
| 20-bar time stop | hurts PCA-3 sparse by −0.26 Sharpe |
| MOVE top-tercile gate | small help, +0.05 to +0.14 |

The Leung–Li row teaches the most. Trading less is not enough: the levers that work (A, B, D) keep the good trades and drop the expensive ones.

## 6. Do it without fooling yourself

- **31 trials are already documented.** PCA-3 sparse's deflated net Sharpe is 0.000 at N = 31. Every lever above adds at least one trial.
- **Write the rule before you run it.** That means the formula, the fitted parameters and the training-window objective. The repo already does this for `run_sleeve1.HEADLINE` and `run.BOOK`; treat a cost fix the same way.
- **Suggested order:** first B (freeze size; mechanical, almost no free parameters), then A (cost-aware entry, one parameter k fitted per fold), then D (construction fix). Run each through `analysis/experiments/pipeline.py`, which reproduces production bit for bit when no hook is set, and report net Sharpe at both the flat and the time-varying cost with the updated trial count.
- **Remember what is missing.** Financing, carry and roll-down are not in P&L. A cost fix that tilts the book toward long-carry tenors will look better in attribution than it would trade.

## 7. Where to look next

| to see | open |
|---|---|
| every trade with entry z, vol, cost, gross, net | `portfolio_diagnostics/tables/s3_trades_<model>.csv` |
| leg weights by fold | `tables/s3_leg_weights_<model>.csv` |
| cost vs entry vol by trade | `tables/s4_cost_vs_vol_<model>.csv` |
| the ten worst trade paths | `tables/s5_worst_paths_<model>.csv` |
| win rate by entry \|z\| | `tables/s5_z_buckets_<model>.csv` |
| everything as interactive charts | `portfolio_diagnostics/dashboard.html` |
| the experiment harness | `experiments/pipeline.py`, `experiments/run_experiments.py` |
