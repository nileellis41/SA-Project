# What Sleeve 1 teaches: trades, periods and patterns

Written 2026-10-14 from the tables made by `python -m amp_sa.run_lessons` (sample to 2026-09-25). The detail is in [trades.md](trades.md) and [periods.md](periods.md). Rerunning the script refreshes those two files but not this one.

**Units.** Trade P&L is in daily sigmas: 1.0 means a trade made one typical day's move of that residual or spread. Period P&L is in % of capital at the 10% volatility target.

**One warning before the lessons.** Everything below is found by looking at out-of-sample results. Any rule built from it ("skip low-volatility entries", "drop the 7-year") is a new trial, fitted on data the walk-forward has already used. It has to be declared before it is run, tested walk-forward, and counted in the deflated Sharpe. Treat these as hypotheses, not fixes.

---

## 1. The average trade pays more in cost than it earns

| model | gross per trade | cost per trade | net per trade | hit rate, gross | hit rate, net |
|:--|--:|--:|--:|--:|--:|
| Headline (PCA-2 + credit) | 0.71 | 1.33 | −0.62 | 68% | 54% |
| PCA-3 sparse | 1.74 | 2.08 | −0.33 | 83% | 58% |
| Pairwise adaptive, neural | 0.48 | 1.10 | −0.62 | 69% | 54% |

The signal is real: before costs, 7 or 8 trades in 10 make money. A round trip costs 1.1 to 2.1 daily sigmas, which is about the size of the edge. Costs turn many small winners into small losers.

**Arithmetic to remember.** The average winner is about 0.53 times the average loser in every model. That is the usual mean-reversion shape: many small wins and a few large losses. At that payoff ratio a strategy needs a hit rate of 1 / (1 + 0.53) ≈ 65% to break even. The gross hit rates clear 65%; the net hit rates (54–58%) do not.

**What to take from it.** Before building a model, put the expected gross per trade next to the round-trip cost, both in daily sigmas. The `cost_over_daily_vol` diagnostic (one-way cost of 0.4 to 1.4 sigma) told this story before any backtest did.

## 2. Long-held trades lose the money, and a time stop would not have saved them

Trades still open after 20 bars are 16–24% of trades and lose more than everything else combined:

| model | trades held past 20 bars | their net | net of all other trades |
|:--|--:|--:|--:|
| Headline | 67 | −167 | −96 |
| PCA-3 sparse | 61 | −172 | **+87** |
| Pairwise adaptive, neural | 60 | −150 | −7 |

Sparse PCA-3 makes money, in total, in every holding bucket up to 20 bars. The OU half-lives on the training windows are 3 to 7 bars, so a trade that is 20 bars old is 3 to 6 half-lives past what the model expects.

The obvious fix is a time stop. The data says it would not help. These trades lost their money **in the first 20 bars** (headline −65, PCA-3 sparse −75) and *gained* some back afterwards (+31 and +47). Closing them at bar 20 would have locked in the loss and given up the recovery. They also cost more to carry: 2.0 sigmas per trade against 1.2 on the other headline trades, and 2.4 against 2.0 for sparse PCA-3.

**What to take from it.** A trade that runs long is a trade that went wrong early: the residual moved away instead of back. One possible mechanism, not tested: the s-score's mean and sigma are refitted every bar on a 60-bar trailing window, so when a residual trends, the mean follows it. The s-score never reaches the 4-sigma stop, and the trade closes "at the mean" at a loss. Before proposing an exit rule, look at the s-score path of a few of these trades (worst list in [trades.md](trades.md)).

## 3. Cost is fixed in basis points; volatility is not

The cost table charges 0.25 bp per leg whatever the market does. The residual's volatility changes a lot:

| training window ending | 2-year daily vol | PCA-3 sparse cost / daily vol |
|:--|--:|--:|
| 2012 fold | 6.7 bp | 0.36 |
| 2016 fold | 2.3 bp | 0.84 |
| 2021 fold | 3.2 bp | 1.40 |
| 2025 fold | 6.7 bp | 1.12 |

Between 2014 and 2022 the front end was pinned near zero and moved 2–3 bp a day instead of 6–7. The same quarter of a basis point became two to four times as expensive relative to what a trade could earn. That is why sparse PCA-3's best era for the signal was its worst for cost: in 2013–2019 it had a gross Sharpe of **1.96** and still lost money net (gross 18.2% a year, cost 19.3%). See the figure in [periods.md](periods.md).

The same pattern shows up by rates volatility. Trades entered when MOVE was in its top third cost 1.0 sigma on average for the headline, against 1.7 in the bottom third. PCA-3 sparse has a positive net Sharpe (+0.24) only in high-MOVE days.

**What to take from it.** A fixed entry band in sigma units is the wrong rule when cost is fixed in bp. Make the entry threshold depend on cost over volatility, or trade smaller when volatility is low. This is the most promising cost idea in this review, and it is a candidate for the "cost-aware entry" step in the deck.

## 4. Six instruments were really two or three bets

In the sparse PCA-3 model each tenor is hedged with its three nearest neighbours. The 2y, 3y and 5y residuals end up with the same four legs (2y, 3y, 5y, 7y), and the 10y and 30y residuals share 5y, 7y, 10y, 30y. Same legs means the same portfolio, so the "six-instrument" sleeve held one trade three times over. Before 2010, with only 2y, 5y, 10y and 30y in the panel, all four residuals were one portfolio.

| model | members | identical pairs | effective independent bets |
|:--|--:|--:|--:|
| PCA-3 sparse | 6 | 4 | 2.3 |
| PCA-2 full | 6 | 0 | 5.7 |
| Headline | 9 | 0 | 8.7 |

Its high gross Sharpe is therefore less diversified than it looks, and the equal weighting gives the 2y-3y-5y-7y portfolio triple weight. One upside: that early four-tenor portfolio (2y, 5y, 10y, 30y) was the best single construction in the study. It made 84 trades with a 77% net hit rate and +0.52 sigma per trade net.

**What to take from it.** Count effective bets, not instruments. For a factor model this is a quick check: if two rows of the hedge matrix use the same legs, they are the same trade.

## 5. The worst days are macro shocks hitting one leg

| date | sleeve loss | main loser | market that day |
|:--|--:|:--|:--|
| 2021-02-25 | −6.9% (headline), −7.0% (PCA-3 sparse) | 5y and 2-3-5 residuals | 10y +14 bp, weak 7-year auction |
| 2008-09-15 | −6.0% (headline) | HY vs IG OAS | Lehman; 10y −33 bp |
| 2024-08-02 | −4.4% (headline) | 30y residual | weak payrolls; 10y −19 bp |
| 2020-03-10 | −5.0% (PCA-3 sparse) | 2y residual | 10y +26 bp, MOVE 164 |

The single worst trade in the headline was a short 5y residual opened 2021-02-16. It lost 12.5 daily sigmas in seven bars, most of it on the 25th. The 4-sigma stop could not help: on daily bars, a stop only fires at the close, after the move.

The HY-vs-IG trade entered at the close on Friday 2008-09-12 lost 7.9 sigmas on Monday. Mean reversion sells insurance: it bets that a dislocation will close, and it loses most on the days the dislocation is news.

**What to take from it.** Size for the gap, not the stop. Known event days (auctions, payrolls, FOMC) are where the belly residuals get hit. Two ways to test this: flatten or shrink positions into scheduled events, or cap any one residual's share of sleeve risk.

## 6. Where the losing stretches came from

The headline's worst six months was **Dec 2018 to Jun 2019**: −21%, half of it from costs (10%) and the rest from the 7y and 2y residuals as the 10y rallied 75 bp. Its second worst was **Sep 2008 to Mar 2009** (−18%), driven by the HY/IG credit pair (−9.6%), not the curve. The third is the most recent: **Mar to Sep 2026**, −17.5%, of which **15.1% was cost** and only 2.4% gross loss.

The best stretches share no single market pattern. The top three were Oct 2022 to Apr 2023 (+12.7%, led by the CDX cash basis), Nov 2011 to Apr 2012 (+10.6%, 2y and 10y residuals) and Apr to Sep 1996 (+9.3%).

The headline's cost has risen lately: 20% of capital in 2025 and 25% in 2026, against 4–12% in most years (1996, 2014, 2017 and 2018 were also high, at 17–21%). Each curve trade cost 2.9–3.1 sigmas in 2025–2026, against 1.2–2.5 in 2014–2024. The training-window cost-to-volatility ratio (about 0.6) does not explain it on its own. This is worth a closer look before the final presentation.

**What to take from it.** Split every bad period into "the signal was wrong" and "we paid too much to trade it". Recent losses are mostly the second kind.

## 7. Smaller observations

- **The 7-year residual is the worst member.** In the headline it averaged −2.5 sigmas per trade over 14 trades. Two of the three worst headline trades are 7y residuals held 44–49 bars. It has the highest cost-to-volatility of any tenor and only exists from 2014, so the sample is small.
- **Credit pairs mostly lost.** HY vs IG averaged −0.65 per trade. The CDX cash basis is the only headline member with a positive net per trade (+0.05), and it drove the best six-month stretch.
- **Very short trades are pure cost for the headline.** Trades closed within 2 bars earned 0.23 sigma gross and paid 0.81. They are whipsaws through the band. Sparse PCA-3's short trades did make money, so this is about the full PCA-2 residual, not about short holds in general.
- **The headline relies on a few trades.** Its best 10% of trades made 74% of gross P&L. For sparse PCA-3 the figure is 28%: its edge is spread across many trades, which is what a real but small signal looks like.
- **Long and short PCA residuals behave about the same.** The pairwise model is different: its long-spread trades made almost nothing gross (0.01 sigma per trade, against 0.80 for short-spread trades). That points at specific spreads, which [trades.md](trades.md) can't separate yet; it is not a reason to trade one side only.
- **Re-sizing open positions is not the main cost.** Daily volatility re-scaling of open trades is 9–21% of cost, and sleeve-level re-scaling is under 2% of capital in most years. The cost is in entries and exits.

## What to test next, in order

These come from the lessons above. Each is a new trial, to be declared before it is run and tested walk-forward.

1. **Entry threshold scaled by cost over volatility** (lessons 1 and 3). Enter only when the expected move back to the mean, in bp, clears a multiple of the round-trip cost in bp.
2. **De-duplicate the sparse hedges** (lesson 4). Trade each distinct leg set once, so sparse PCA-3 becomes two or three portfolios instead of six with copies.
3. **Look at the s-score path of the long losers** before designing any exit (lesson 2). The rolling-mean hypothesis is easy to check by plotting the residual, its 60-bar mean and the s-score through one of the 2019 or 2025 7y trades.
4. **Event-day exposure** (lesson 5). Measure the P&L on auction, payroll and FOMC days before deciding whether to flatten into them.
5. **Explain the 2025–2026 cost jump** (lesson 6) before reporting recent performance.
