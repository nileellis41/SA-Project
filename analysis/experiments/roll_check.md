# Backlog item 1: on-the-run roll contamination (data check)

Pre-registered 2026-11-09, before any of the numbers below were computed.
This is a data check, not a strategy trial: no rule, band or parameter of the
models changes, so it does not add to the trial count (35 documented).
Code: `analysis/experiments/roll_check.py`. Output: `results/roll_check/`.

## Question
USGG generic yields switch to the newly auctioned issue. The switch can put a
jump in a tenor's daily change that is not a tradeable move. Does the gross
edge of the PCA residual models come partly from such jumps "reverting"?

## Event calendar
From the cached TreasuryDirect auction files (`data/raw/treasury/`), nominal
coupons only (no TIPS, no FRNs), original term 2, 3, 5, 7, 10, 30 years.
Only **new issues** (`reopening == No`) can switch the generic. Reopenings are
kept as a control: they have the same auction-day market event but no switch.
Two candidate switch dates per new issue, each mapped to the first panel bar
on or after the calendar date:
- **auction**: `competitiveAuctionCloseDate`
- **issue**: `issueDate` (settlement)

Both calendars are published before the event, so using them is not look-ahead.

## Measurement (descriptive, no model)
Idiosyncratic move of tenor T on bar t:
`e_T,t = dy_T,t - interp(dy of the nearest available shorter and longer tenor)`,
interpolated linearly in maturity. Edge tenors (or a missing side) use the
nearest available tenor's change (flat). Units: bp.

Per tenor and date type, event study over offsets -3..+5 bars:
- mean |e| on event bars divided by mean |e| on all bars that are not within
  ±5 bars of any event of that tenor (ratio > 1 = excess movement)
- mean signed e and its t-stat (a consistent sign points to an artifact)
- 10y and 30y: the same at offset 0 for reopening auctions (control)

## Treatment (walk-forward re-run)
For each date type, on each tenor's event bars, replace dy_T by the
interpolated neighbour value (so e = 0 on that bar for that tenor only) and
rebuild the level series as the cumulative sum of the cleaned changes. All
other bars and tenors are unchanged. The cleaned levels go through
`analysis/experiments/pipeline.run_members` with the default `Rule()` (the
production model), for PCA-2 full and PCA-3 sparse. The headline is the
cleaned PCA-2 members plus the unchanged adaptive credit pairs, as in
`run_experiments.py`.

**Placebo**, per date type: the same replacement on dates shifted +6 bars,
dropping any placebo bar within 2 bars of a real auction or issue bar of the
same tenor (new issue or reopening). This measures the mechanical effect of
removing a tenor's idiosyncratic move on that many ordinary days.

Configs: baseline (cached production runs), auction, auction placebo, issue,
issue placebo. Models: Headline, PCA-2 full, PCA-3 sparse.

## Reported
Gross Sharpe, net Sharpe at the flat 1x table and at the time-varying table
(`portfolio_diagnostics.book.time_varying_cost`), trades per year, and the
all-bars and entry-bar IC (Spearman of -z with the forward 10-bar member
change, same definition as diagnostics step 5) on the cleaned changes.

## Deviation, recorded 2026-11-09 after the event study and before any re-run
The event study (`results/roll_check/event_study.csv`) puts the jump on the
bar **after** the auction for every tenor: mean |e| is 1.6-3.0x ordinary at
auction +1, with a consistent sign (t = +5.6 2y, +6.3 5y, +11.4 7y, -3.9 30y).
There is no excess on the issue date, and none on reopening auctions (the control).
So Bloomberg switches the generic one bar after the auction, which is
neither pre-registered date. A third date type, **switch** = auction bar + 1,
is added with its own placebo, under the same kill criterion. The two
pre-registered types are still run and reported. This choice uses only the
calendar and the raw yields, not any backtest result. The large negative t on the 7y auction
day is the 5y switch (the 5y auctions the day before) leaking into the 7y's
neighbour interpolation, not a 7y event.

## Kill criterion (fixed now)
For each model and date type, compare treated with its placebo:
**if gross Sharpe(treated) < 2/3 x gross Sharpe(placebo), the gross edge is
partly a data artifact** and §1 of the report must be rewritten.
The comparison is against the placebo, not the raw baseline, because removing
a tenor's move on ~12 days a year costs some gross Sharpe on any days; the
placebo nets that out. The drop against the raw baseline (the backlog's
literal test) is reported alongside and flagged if it also exceeds 1/3.
