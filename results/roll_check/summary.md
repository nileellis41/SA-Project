# Roll contamination check: result (2026-11-09)

Design and kill criterion: `analysis/experiments/roll_check.md` (pre-registered; one recorded deviation).
Code: `analysis/experiments/roll_check.py`. Tables: `event_study.csv`, `event_counts.csv`, `results.csv`, `verdict.csv`.
A data check, not a strategy trial: the trial count stays at 35.

## 1. The generic switches the bar after the auction
Idiosyncratic move e = dy minus the maturity-interpolated dy of the neighbouring tenors, on the bar after each
new-issue auction:

| tenor | events | mean abs e vs ordinary | mean e (bp) | t |
|---|---|---|---|---|
| 2y | 427 | 1.82x | +1.02 | +5.6 |
| 3y | 224 | 1.72x | +0.71 | +1.2 |
| 5y | 375 | 1.85x | +0.79 | +6.3 |
| 7y | 212 | 3.05x | +0.99 | +11.4 |
| 10y | 135 | 2.48x | -0.16 | -0.6 |
| 30y | 99 | 1.63x | -1.17 | -3.9 |

There is no excess on the issue (settlement) date, and none on 10y/30y reopening auctions, which are the control.
The jump has a consistent sign: the new, longer issue yields more than the aged one on the upward-sloping front end, and less at
the 30y. It is about 1 bp, more than twice the 0.45 bp realised reversion per trade.

## 2. Re-run with the switch bar cleaned (walk-forward, production rules)

| model | gross Sharpe baseline | placebo range (3 draws) | switch cleaned | net 1x | net time-varying |
|---|---|---|---|---|---|
| Headline | 0.55 | 0.25 to 0.46 | **-0.04** | -1.05 | -1.50 |
| PCA-2 full | 0.45 | 0.18 to 0.41 | **-0.18** | -1.11 | -1.57 |
| PCA-3 sparse | 1.21 | 0.60 to 1.01 | **0.31** | -0.92 | -1.12 |

The all-bars IC of PCA-2 at 10 bars falls from 0.064 to -0.005. For PCA-3 sparse it falls from 0.211 to 0.063.

**Kill criterion met for all three models** (treated < 2/3 of its placebo). Each treated result is also below
every placebo draw. The gross edge reported in §1 of the report is mostly a data artifact. A position held over the
switch bar books the change of security as P&L. The jump has a predictable sign, so the bands can learn to
collect it. A real bond position never sees this jump.

The pre-registered auction-day and issue-day treatments are in `verdict.csv`. Auction-day cleaning also trips
the criterion for PCA-3 sparse, which is consistent with the switch leaking into its neighbours' interpolation.

## Caveats
- Three placebo draws give a range, not a distribution. A wider placebo set (more shifts) would put a p-value on the drop.
- Cleaning also removes any genuine relative-value move a tenor makes on its switch bar (about 12 bars a year per tenor).
  The placebo covers the average cost of that, not a move specific to the switch day.
- The cleaner fix is a series without the switch: Fed GSW fitted yields or FRED constant-maturity (DGS*) yields.
