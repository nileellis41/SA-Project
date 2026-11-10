# Sleeve 1 portfolio diagnostics

Out-of-sample walk-forward results only, from the cached production runs. P&L is in fractions of capital at the sleeve's 10% volatility target (additive, no compounding) unless marked as risk units (1 = one trailing daily standard deviation of the member). Dollar figures are for a notional $1m book. Models: `headline` = Headline (PCA-2 + credit), `pca3_sparse` = PCA-3 sparse, `pca2` = PCA-2 full, `pairwise_nn` = Pairwise adaptive NN.

## Findings

1. **No model makes money after cost, and the cost table is the reason.** Net Sharpe is −0.60 for the headline, −0.09 for PCA-3 sparse, −0.67 for PCA-2 full and −0.26 for pairwise adaptive. Gross Sharpe is 0.55, 1.21, 0.45 and 0.46. The PCA models pay 9.6–10.8% of capital a year in cost, against gross P&L of 3.8–10.1%.
2. **The signal is real.** On entry bars, the rank correlation between −z and the next 10 bars of residual change is 0.33 for PCA-3 sparse, 0.17 for the headline and 0.13 for pairwise (all p < 0.05). The fixed 2s5s10s and 3s5s7s flies, traded with the same rules, have gross Sharpe of only 0.11 and 0.38.
3. **Breakeven cost is below the 1x table for every model.** The cost multiple at which mean net P&L is zero is 0.93x for PCA-3 sparse, 0.48x for the headline, 0.40x for PCA-2 full and 0.64x for pairwise. A more realistic table (0.5 bp before 2008, 0.25 bp after, 30y at 1.5x) moves net Sharpe to −1.12 for the headline and −0.63 for PCA-3 sparse.
4. **Cost comes from entries and exits, not from maintenance.** Entries and exits are 75–87% of cost. Daily vol resizing is 8–10%, sleeve leverage changes 4–8%, and re-weighting legs at refits is at most 0.035% of capital a year. Netting legs across members saves only 0.5–5.5%, because members rarely hold offsetting legs at the same time.
5. **Low-volatility entries are where the money is lost.** For the headline, the lowest-vol third of trades was entered when the residual moved 0.39 bp a day. A round trip cost 1.8x that daily vol, and those trades averaged −1.25 sigma net. In the highest-vol third, round trip was 0.6x vol. PCA-3 sparse makes +0.49 sigma per trade in its high-vol third and loses −0.48 in its low-vol third. A cost-aware entry rule at k = 1 would not remove them: 92% of PCA-2 entries already pass it (median forecast reversion 1.67 bp against a 0.73 bp round trip), because the 60-bar OU forecasts far more than arrives (mean realised move 0.45 bp a trade). k has to be fitted, and halves the entries only at about 2.4.
6. **The hedge does its job; the trade it isolates is small.** PCA-2 residuals move 0.5–1.1 bp a day against 5–6 bp for raw yields, so 96–99% of the variance is removed. Against causal trailing-1y loadings, leakage to the hedged factors is 2–7% of gross DV01. PCA-2 deliberately keeps curvature: 27% of its gross DV01, and 73 of its 237 risk units of gross P&L.
7. **PCA-3 sparse has a construction defect in 2010–2013, and it does fewer bets than it appears.** Before the 7y has a full training window, the 2y/3y/5y residual is hedged with 2y, 5y and 10y, which nearly span the same factors. Leg weights reach −121 on the 5y per unit of 3y. Trades entered in those four folds lost 67 risk units net, against +9 for the whole model. The 2y, 3y and 5y members are the same portfolio, so its 6 members are 2.3 effective bets.
8. **The OU fit the signal uses is much faster than the training-window fit, and that is mostly small-sample bias.** On each 5-year training window, the cumulative residual's AR(1) half-life has a median of 103 bars for PCA-2 and 32 for PCA-3 sparse. The trailing 60-bar fit behind the s-score has a median of 4.5–6.4 bars, and trades are held 10 bars. An AR(1) fitted on 60 bars is biased towards fast reversion: simulating the training-window process and fitting it on 60 bars gives 8.3 bars (PCA-2) and 7.1 (PCA-3 sparse) before any screen. The speed screen passes 88–92% of simulated windows and moves this only to 7.6 and 6.7. Selection at entry does the rest (3.0–3.5 bars).
9. **None of the six experiments turns net P&L positive.** Holding positions unless the target moves more than 25% helps every model (+0.06 to +0.09 Sharpe; PCA-3 sparse reaches −0.00). Entry at 2.5 or 3.0 cuts cost but leaves 0.6–6 trades a year. The MOVE gate helps a little, the 20-bar time stop hurts PCA-3 sparse (−0.26), and the leave-one-out residual is worse everywhere. Every parameter except leave-one-out was chosen after full-sample results were seen.
10. **Overfitting check: PCA-3 sparse's deflated Sharpe is 0.016 net with 11 trials and 0.000 with all 31 documented trials.** The gross P&L clears the bar (deflated Sharpe 1.00); the net does not. Results are not driven by auction weeks (p > 0.09) and correlate weakly with rates, MOVE and HY OAS (|ρ| < 0.1). The worst windows are the GFC for the headline (−17%) and February 2021 for PCA-3 sparse (−10%).

## What the P&L does not include

- **Financing (repo), carry, roll-down and specialness are not modeled.** The curve is traded in cash-yield bp; the funding rate is loaded into the panel but never enters P&L. A residual position that is long high-carry tenors earns carry that is not counted, and vice versa.
- **The alpha attribution will show carry as alpha.** `amp_sa/backtest/attribution.py` regresses P&L on price-move factors only, so any return that comes from collecting carry appears in the intercept.
- **Futures roll** is modeled only for the credit pairs (CDX series roll, 2x cost). The Treasury curve has no roll because it is traded as cash yields.
- **Costs are per package, not netted** (experiment 6 measures the gap), and the same 0.25 bp per Treasury leg is used for every tenor and every date (step 2 shows a time-varying alternative).

## Key numbers (net of 1x cost)

> **What this tells you:** every model loses money after cost, and every model makes money before it. The four models' net Sharpe of −0.09 to −0.67 against 0.45–1.21 gross means the signal exists but is smaller than what it costs to trade. Drawdowns above 100% of capital appear because P&L is added up, not compounded, over 31 years of losing; a real book would have stopped long before.

Returns, vol, drawdowns and VaR/ES are fractions of capital (0.05 = 5%). `longest_dd_bars` is the longest spell below a previous peak.

| model       | start      |   ann_return |   ann_vol |   sharpe |   gross_sharpe |   sortino |   calmar |   max_dd |   longest_dd_bars |   worst_day |   worst_week |   var_99 |   es_99 |
|:------------|:-----------|-------------:|----------:|---------:|---------------:|----------:|---------:|---------:|------------------:|------------:|-------------:|---------:|--------:|
| headline    | 1995-05-19 |       -0.050 |     0.084 |   -0.599 |          0.547 |    -0.795 |   -0.028 |   -1.788 |              7635 |      -0.069 |       -0.092 |    0.018 |   0.026 |
| pca3_sparse | 1995-05-19 |       -0.007 |     0.082 |   -0.087 |          1.210 |    -0.123 |   -0.013 |   -0.540 |              5307 |      -0.073 |       -0.111 |    0.016 |   0.025 |
| pca2        | 1995-05-19 |       -0.058 |     0.087 |   -0.668 |          0.448 |    -0.876 |   -0.028 |   -2.044 |              7635 |      -0.082 |       -0.100 |    0.019 |   0.028 |
| pairwise_nn | 1995-05-19 |       -0.020 |     0.078 |   -0.257 |          0.459 |    -0.359 |   -0.018 |   -1.128 |              5731 |      -0.129 |       -0.130 |    0.014 |   0.023 |
| fly_2s5s10s | 1995-05-19 |       -0.030 |     0.072 |   -0.422 |          0.109 |    -0.545 |   -0.029 |   -1.055 |              3672 |      -0.147 |       -0.107 |    0.014 |   0.027 |
| fly_3s5s7s  | 2014-09-17 |       -0.069 |     0.090 |   -0.766 |          0.382 |    -0.891 |   -0.078 |   -0.884 |              2950 |      -0.153 |       -0.189 |    0.016 |   0.031 |

### Breakeven cost

> **What this tells you:** how cheap trading would have to be. PCA-3 sparse breaks even at 0.93x the cost table, so it is close; the headline needs costs cut by more than half (0.48x). Gross Sharpe is the ceiling: no cost cut can do better.

Net Sharpe at multiples of the 1x cost table with positions and sleeve weights held at the 1x run. `breakeven_mult` is where mean net P&L is zero.

| model       |   gross_sharpe |   net_sharpe_0x |   net_sharpe_0.25x |   net_sharpe_0.5x |   net_sharpe_0.75x |   net_sharpe_1x |   breakeven_mult |   gross_return_pa |   cost_pa_1x |
|:------------|---------------:|----------------:|-------------------:|------------------:|-------------------:|----------------:|-----------------:|------------------:|-------------:|
| headline    |          0.547 |           0.547 |              0.260 |            -0.028 |             -0.316 |          -0.599 |            0.475 |             0.046 |        0.096 |
| pca3_sparse |          1.210 |           1.210 |              0.900 |             0.576 |              0.245 |          -0.087 |            0.934 |             0.101 |        0.108 |
| pca2        |          0.448 |           0.448 |              0.167 |            -0.115 |             -0.394 |          -0.668 |            0.398 |             0.038 |        0.097 |
| pairwise_nn |          0.459 |           0.459 |              0.281 |             0.101 |             -0.079 |          -0.257 |            0.640 |             0.036 |        0.056 |
| fly_2s5s10s |          0.109 |           0.109 |             -0.018 |            -0.154 |             -0.291 |          -0.422 |            0.215 |             0.008 |        0.039 |
| fly_3s5s7s  |          0.382 |           0.382 |              0.079 |            -0.223 |             -0.509 |          -0.766 |            0.315 |             0.032 |        0.101 |

### Flat vs time-varying cost

> **What this tells you:** the flat 0.25 bp table is generous. Charging the wider pre-2008 spreads and a dearer 30y makes every model worse, mostly before 2008.

Time-varying: Treasury legs 0.5 bp one way before 2008 and 0.25 bp after; the 30y at 1.5x the other tenors. The sleeve is rebuilt with re-priced cost; positions and bands are unchanged (bands were optimised at the flat table).

| model       |   gross_sharpe |   net_sharpe_flat |   net_sharpe_time_varying |   cost_pa_flat |   cost_pa_time_varying |   net_sharpe_time_varying_pre2008 |   net_sharpe_time_varying_2008on |
|:------------|---------------:|------------------:|--------------------------:|---------------:|-----------------------:|----------------------------------:|---------------------------------:|
| headline    |          0.547 |            -0.599 |                    -1.120 |          0.096 |                  0.138 |                            -1.590 |                           -0.783 |
| pca3_sparse |          1.210 |            -0.087 |                    -0.633 |          0.108 |                  0.148 |                            -1.186 |                           -0.170 |
| pca2        |          0.448 |            -0.668 |                    -1.186 |          0.097 |                  0.140 |                            -1.597 |                           -0.901 |
| pairwise_nn |          0.459 |            -0.257 |                    -0.545 |          0.056 |                  0.076 |                            -0.611 |                           -0.498 |

### P&L by year (net, % of capital)

> **What this tells you:** losses are spread over many years, not one bad episode. No model has a run of positive years long enough to suggest a regime where it works after cost.

|   year |   fly_2s5s10s |   fly_3s5s7s |   headline |   pairwise_nn |   pca2 |   pca3_sparse |
|-------:|--------------:|-------------:|-----------:|--------------:|-------:|--------------:|
| 1995.0 |          -5.0 |        nan   |        1.7 |          -1.1 |    1.7 |           0.6 |
| 1996.0 |          -0.1 |        nan   |        7.8 |           0.0 |    7.8 |           6.9 |
| 1997.0 |         -12.4 |        nan   |       -4.9 |           0.0 |   -4.9 |          -7.6 |
| 1998.0 |           3.2 |        nan   |      -11.6 |          -2.7 |  -11.6 |          -4.8 |
| 1999.0 |          -9.0 |        nan   |       -5.3 |          16.6 |   -5.3 |          -3.9 |
| 2000.0 |           0.0 |        nan   |       -7.1 |          13.1 |   -7.1 |           2.6 |
| 2001.0 |          -7.2 |        nan   |      -11.1 |          13.3 |  -11.1 |           9.2 |
| 2002.0 |           5.3 |        nan   |      -15.5 |         -12.5 |  -15.5 |           7.5 |
| 2003.0 |          18.6 |        nan   |        3.9 |          10.8 |    3.9 |          -2.6 |
| 2004.0 |          -7.2 |        nan   |       -2.3 |          -1.3 |   -2.3 |          -0.9 |
| 2005.0 |           0.0 |        nan   |       -5.1 |         -13.2 |   -6.4 |           6.0 |
| 2006.0 |           0.0 |        nan   |       10.6 |          -0.2 |    9.8 |           5.7 |
| 2007.0 |           0.0 |        nan   |      -10.8 |         -18.6 |   -8.9 |         -19.3 |
| 2008.0 |           3.2 |        nan   |       -6.6 |         -14.0 |   -3.3 |           0.3 |
| 2009.0 |           0.0 |        nan   |       -7.7 |           0.4 |   -5.4 |           9.7 |
| 2010.0 |           0.0 |        nan   |      -11.9 |         -10.4 |   -8.5 |          -3.6 |
| 2011.0 |           4.7 |        nan   |        0.5 |           7.4 |    1.4 |          -4.7 |
| 2012.0 |           1.0 |        nan   |        4.9 |          -1.4 |    4.4 |           0.7 |
| 2013.0 |           0.0 |        nan   |       -3.5 |           0.6 |   -4.2 |          -9.8 |
| 2014.0 |          -5.2 |          0.0 |        9.4 |         -11.8 |    5.8 |          14.8 |
| 2015.0 |           8.3 |        -22.3 |      -10.6 |         -12.0 |   -9.7 |          -1.4 |
| 2016.0 |           0.0 |        -11.6 |        4.4 |           4.1 |    6.0 |           5.0 |
| 2017.0 |          -5.0 |         -1.4 |       -8.2 |          -2.1 |  -11.4 |           0.9 |
| 2018.0 |         -12.2 |        -15.8 |      -18.3 |          -1.1 |  -24.1 |         -12.5 |
| 2019.0 |          -2.1 |         -4.0 |      -20.6 |          -5.0 |  -23.1 |          -4.6 |
| 2020.0 |         -10.6 |          0.6 |        4.0 |         -15.0 |    3.3 |         -11.5 |
| 2021.0 |           3.0 |        -12.3 |      -18.6 |          -7.3 |  -22.7 |         -11.8 |
| 2022.0 |          -9.4 |        -17.1 |        3.3 |           5.9 |   -2.9 |           0.4 |
| 2023.0 |           0.0 |          2.4 |        5.9 |           1.3 |    4.5 |           2.8 |
| 2024.0 |         -11.1 |          0.0 |       -5.9 |           0.0 |   -6.9 |           8.4 |
| 2025.0 |         -32.0 |         -4.6 |      -16.6 |          -5.7 |  -20.7 |          -5.2 |
| 2026.0 |         -17.0 |          0.0 |      -17.3 |          -3.4 |  -21.6 |          -0.4 |

Benchmarks: the fixed-weight flies (belly +1, wings −0.5 each, in bp) traded with the PCA rules (60-bar OU s-score, training-window bands, same folds and costs). Trades: fly_2s5s10s 60, fly_3s5s7s 43.

## Step 1. Cost model audit

> **What this tells you:** what you pay and when. 75–87% of cost is opening and closing trades. Keeping a position the right size (resizing, leverage changes) is the rest, and re-hedging at refits is free in practice. So cutting cost means fewer, better round trips, not tidier maintenance.

- **How cost is charged** (`amp_sa/backtest/engine.py`): per member and bar, `|p_t − p_{t−1}| × Σ_legs |w_leg| × c_leg`, with `p = signal / trailing vol` in risk units and `c` = 0.25 bp per Treasury tenor one way (2 bp HY OAS, 0.5 bp IG OAS, 1 bp CDX HY). It is per package: each member pays for all its legs on its own position change. Members are not netted.
- **Entry / exit / resize**: an entry is opening from flat or the new side of a flip, an exit is closing to flat or the old side of a flip, and a resize is the same side held through a change in `signal / vol`. Resizes are charged every bar at the same rate. The sleeve also pays `|Δ sleeve weight| × mean |p| × unit cost` when leverage changes (`amp_sa/portfolio/combine.py`).
- **Does cost scale with DV01?** Not directly: the charge is bp per unit of residual. Because the position is 1 / vol units, a quiet market means more DV01 per risk unit and the same bp cost is a larger share of the edge.
- **Not modeled**: financing, carry, roll-down, specialness; futures roll only on the credit pairs (see above).
- **Unpriced**: when leg weights change under a held position (PCA refit, Kalman β), the engine does not charge the re-hedge. Priced leg by leg it is 0.002–0.04% of capital a year: negligible.

| model       | type                           |   pct_cap_per_year |   usd_per_year_at_1m |   share_of_cost |
|:------------|:-------------------------------|-------------------:|---------------------:|----------------:|
| headline    | entry                          |              3.983 |            39825.772 |           0.416 |
| headline    | exit                           |              3.864 |            38644.858 |           0.403 |
| headline    | resize                         |              0.776 |             7762.783 |           0.081 |
| headline    | roll                           |              0.377 |             3769.305 |           0.039 |
| headline    | sleeve_rescale                 |              0.580 |             5795.479 |           0.060 |
| headline    | unpriced_rehedge (not charged) |              0.004 |               38.912 |           0.000 |
| pca3_sparse | entry                          |              4.718 |            47177.326 |           0.438 |
| pca3_sparse | exit                           |              4.694 |            46940.832 |           0.436 |
| pca3_sparse | resize                         |              0.935 |             9348.451 |           0.087 |
| pca3_sparse | roll                           |              0.000 |                0.000 |           0.000 |
| pca3_sparse | sleeve_rescale                 |              0.415 |             4151.867 |           0.039 |
| pca3_sparse | unpriced_rehedge (not charged) |              0.037 |              366.321 |           0.003 |
| pca2        | entry                          |              4.165 |            41650.782 |           0.431 |
| pca2        | exit                           |              3.892 |            38921.712 |           0.403 |
| pca2        | resize                         |              0.795 |             7948.267 |           0.082 |
| pca2        | roll                           |              0.000 |                0.000 |           0.000 |
| pca2        | sleeve_rescale                 |              0.816 |             8157.428 |           0.084 |
| pca2        | unpriced_rehedge (not charged) |              0.003 |               31.360 |           0.000 |
| pairwise_nn | entry                          |              2.156 |            21557.030 |           0.388 |
| pairwise_nn | exit                           |              2.021 |            20212.647 |           0.364 |
| pairwise_nn | resize                         |              0.543 |             5434.754 |           0.098 |
| pairwise_nn | roll                           |              0.452 |             4523.689 |           0.081 |
| pairwise_nn | sleeve_rescale                 |              0.382 |             3817.338 |           0.069 |
| pairwise_nn | unpriced_rehedge (not charged) |              0.002 |               23.493 |           0.000 |

Per member (risk units and % of capital, with the median one-way unit cost in bp): `tables/s1_cost_split_by_member.csv`.

Headline members: one-way and round-trip cost of one unit of the package (bp), cost by type summed over the sample (risk units), and the charged total in % of capital per year of the member's own live period.

| member         |   unit_cost_bp |   round_trip_bp |   entry_risk |   exit_risk |   resize_risk |   roll_risk |   charged_pct_cap_pa |
|:---------------|---------------:|----------------:|-------------:|------------:|--------------:|------------:|---------------------:|
| pca2_ust_2y    |           0.37 |            0.73 |        35.07 |       36.87 |          7.01 |        0.00 |                 1.34 |
| pca2_ust_3y    |           0.40 |            0.80 |        55.37 |       57.41 |         10.58 |        0.00 |                 3.32 |
| pca2_ust_5y    |           0.38 |            0.75 |        28.01 |       28.67 |          5.62 |        0.00 |                 1.05 |
| pca2_ust_7y    |           0.42 |            0.84 |        11.25 |       10.60 |          3.79 |        0.00 |                 0.94 |
| pca2_ust_10y   |           0.36 |            0.73 |        53.23 |       54.60 |         11.63 |        0.00 |                 1.92 |
| pca2_ust_30y   |           0.32 |            0.64 |        32.19 |       33.29 |          5.57 |        0.00 |                 1.19 |
| hy_ig_oas      |           3.56 |            7.12 |        10.48 |       11.01 |          2.89 |        0.00 |                 0.63 |
| cdx_cash_basis |           2.44 |            4.87 |        15.09 |       15.32 |          1.49 |       21.23 |                 2.48 |
| hyg_cdx        |           0.03 |            0.06 |         1.55 |        1.52 |          0.96 |        6.98 |                 0.45 |

> **What this tells you:** the two spread packages (hy_ig_oas, cdx_cash_basis) cost 6-10x a curve residual per unit (2.4-3.6 bp against about 0.4 bp one way), and cdx_cash_basis also pays the CDX roll. Credit trades rarely, so the curve members still pay most of the bill.

## Step 2. Portfolio KPIs

> **What this tells you:** the same models gross and net side by side. Gross drawdowns of the four models are 20–37%, and three of them recover; no net drawdown recovers (`recovery_bars` blank), because cost is a steady drag that an edge this small cannot outrun.

Gross and net, all models and the two benchmarks: `tables/s2_kpis.csv`; by year: `tables/s2_yearly.csv`. Headline numbers above.

| model       | basis   |   ann_return |   ann_vol |   sharpe |   sortino |   calmar |   max_dd |   recovery_bars |   worst_week |   es_99 |
|:------------|:--------|-------------:|----------:|---------:|----------:|---------:|---------:|----------------:|-------------:|--------:|
| headline    | gross   |        0.046 |     0.083 |    0.547 |     0.805 |    0.199 |   -0.228 |         894.000 |       -0.065 |   0.024 |
| headline    | net     |       -0.050 |     0.084 |   -0.599 |    -0.795 |   -0.028 |   -1.788 |         nan     |       -0.092 |   0.026 |
| pca3_sparse | gross   |        0.101 |     0.083 |    1.210 |     2.000 |    0.504 |   -0.199 |         448.000 |       -0.090 |   0.023 |
| pca3_sparse | net     |       -0.007 |     0.082 |   -0.087 |    -0.123 |   -0.013 |   -0.540 |         nan     |       -0.111 |   0.025 |
| pca2        | gross   |        0.038 |     0.086 |    0.448 |     0.656 |    0.148 |   -0.260 |         nan     |       -0.072 |   0.025 |
| pca2        | net     |       -0.058 |     0.087 |   -0.668 |    -0.876 |   -0.028 |   -2.044 |         nan     |       -0.100 |   0.028 |
| pairwise_nn | gross   |        0.036 |     0.077 |    0.459 |     0.693 |    0.095 |   -0.373 |        3598.000 |       -0.119 |   0.021 |
| pairwise_nn | net     |       -0.020 |     0.078 |   -0.257 |    -0.359 |   -0.018 |   -1.128 |         nan     |       -0.130 |   0.023 |
| fly_2s5s10s | gross   |        0.008 |     0.076 |    0.109 |     0.184 |    0.016 |   -0.533 |         nan     |       -0.082 |   0.023 |
| fly_2s5s10s | net     |       -0.030 |     0.072 |   -0.422 |    -0.545 |   -0.029 |   -1.055 |         nan     |       -0.107 |   0.027 |
| fly_3s5s7s  | gross   |        0.032 |     0.084 |    0.382 |     0.513 |    0.165 |   -0.193 |         nan     |       -0.124 |   0.025 |
| fly_3s5s7s  | net     |       -0.069 |     0.090 |   -0.766 |    -0.891 |   -0.078 |   -0.884 |         nan     |       -0.189 |   0.031 |

## Step 3. Hedge performance

> **What this tells you:** whether the book really trades residuals. It does: with drifting loadings, level and slope leak about 7% of gross DV01 for PCA-2 and the three hedged factors about 12% for PCA-3 sparse, and the residuals move a fraction of a raw yield. The exceptions are by design (PCA-2 keeps curvature) or a defect (PCA-3 sparse 2010–2013, below).

Factor exposure of the netted book, measured two ways. Against the fold's own loadings (the specified test), the factors a PCA model hedges are zero by construction, so that number cannot reveal a failure. Against PCA-3 loadings re-estimated every 21 bars from the prior 252 bars (causal), the exposure shows how much the factors drift away from the hedge.

| model       | loadings                                                          |   mean_abs_pc1_per_gross_dv01 |   mean_abs_pc2_per_gross_dv01 |   mean_abs_pc3_per_gross_dv01 |   mean_gross_dv01_usd_per_bp |   mean_abs_net_dv01_usd_per_bp |
|:------------|:------------------------------------------------------------------|------------------------------:|------------------------------:|------------------------------:|-----------------------------:|-------------------------------:|
| headline    | fold loadings (spec; zero by construction for hedged PCA factors) |                         0.000 |                         0.000 |                         0.299 |                     6631.882 |                        445.621 |
| headline    | trailing 1y loadings (causal drift check)                         |                         0.016 |                         0.051 |                         0.275 |                     6631.882 |                        445.621 |
| pca3_sparse | fold loadings (spec; zero by construction for hedged PCA factors) |                         0.000 |                         0.000 |                         0.000 |                     9373.582 |                        174.621 |
| pca3_sparse | trailing 1y loadings (causal drift check)                         |                         0.006 |                         0.020 |                         0.099 |                     9373.582 |                        174.621 |
| pca2        | fold loadings (spec; zero by construction for hedged PCA factors) |                         0.000 |                         0.000 |                         0.299 |                     7596.442 |                        498.020 |
| pca2        | trailing 1y loadings (causal drift check)                         |                         0.016 |                         0.051 |                         0.275 |                     7596.442 |                        498.020 |
| pairwise_nn | fold loadings (spec; zero by construction for hedged PCA factors) |                         0.041 |                         0.106 |                         0.188 |                     3856.121 |                        263.435 |
| pairwise_nn | trailing 1y loadings (causal drift check)                         |                         0.059 |                         0.124 |                         0.196 |                     3856.121 |                        263.435 |

Per-trade attribution, summed over trades, risk units. Treasury-leg changes split into leaked factor P&L (PC1–3) and the residual beyond three factors; non-Treasury legs (credit pairs, swaps) are `other_legs`.

Fold loadings:

| model       |   trades |   pc1 |   pc2 |   pc3 |   residual |   other_legs |   gross |   cost |    net |
|:------------|---------:|------:|------:|------:|-----------:|-------------:|--------:|-------:|-------:|
| headline    |      426 |  -0.0 |  -0.0 |  91.2 |      145.9 |         67.3 |   304.4 |  567.9 | -263.5 |
| pca3_sparse |      691 |   0.0 |  -0.0 |   0.0 |     1327.4 |          0.0 |  1327.4 | 1318.0 |    9.4 |
| pca2        |      344 |  -0.0 |  -0.0 |  91.2 |      145.9 |          0.0 |   237.1 |  479.4 | -242.3 |
| pairwise_nn |      252 |  47.1 | -55.6 |  31.6 |       88.9 |          8.7 |   120.7 |  277.3 | -156.6 |

Trailing loadings:

| model       |   trades |   pc1 |   pc2 |   pc3 |   residual |   other_legs |   gross |   cost |    net |
|:------------|---------:|------:|------:|------:|-----------:|-------------:|--------:|-------:|-------:|
| headline    |      426 |  37.6 |   9.3 |  73.5 |      116.7 |         67.3 |   304.4 |  567.9 | -263.5 |
| pca3_sparse |      691 |  12.1 |   7.1 |  42.7 |     1265.5 |          0.0 |  1327.4 | 1318.0 |    9.4 |
| pca2        |      344 |  37.6 |   9.3 |  73.5 |      116.7 |          0.0 |   237.1 |  479.4 | -242.3 |
| pairwise_nn |      252 |  35.8 | -63.4 |   8.3 |      131.3 |          8.7 |   120.7 |  277.3 | -156.6 |

Refit turnover. `unpriced_reweight_pa` is the leg trade forced by new weights under a held position, which the engine does not charge. `charged_on_boundary_bars_pa` is the cost the engine charged on the last bar of each fold and the first bar of the next (band state restarts each fold, so positions can jump), and `boundary_bars_vs_mean` compares those two bars with an average pair.

| model       |   unpriced_reweight_pa |   charged_on_boundary_bars_pa |   boundary_bars_vs_mean |
|:------------|-----------------------:|------------------------------:|------------------------:|
| headline    |                 0.0000 |                        0.0026 |                  3.3383 |
| pca3_sparse |                 0.0003 |                        0.0017 |                  1.9922 |
| pca2        |                 0.0000 |                        0.0028 |                  3.5588 |
| pairwise_nn |                 0.0000 |                        0.0012 |                  2.7480 |

Residual package vs raw yields (bp per day, live bars):

| model       | member        |   resid_vol_bp |   own_leg_vol_bp |   ust_5y_vol_bp |   variance_removed_vs_own |
|:------------|:--------------|---------------:|-----------------:|----------------:|--------------------------:|
| pca2        | pca2_ust_2y   |           1.14 |             5.49 |            6.09 |                      0.96 |
| pca2        | pca2_ust_3y   |           0.55 |             4.82 |            5.31 |                      0.99 |
| pca2        | pca2_ust_5y   |           0.90 |             6.09 |            6.09 |                      0.98 |
| pca2        | pca2_ust_7y   |           0.61 |             5.42 |            5.38 |                      0.99 |
| pca2        | pca2_ust_10y  |           0.74 |             5.82 |            6.09 |                      0.98 |
| pca2        | pca2_ust_30y  |           1.07 |             5.27 |            6.09 |                      0.96 |
| pca3_sparse | pca3s_ust_2y  |          12.40 |             5.49 |            6.09 |                     -4.10 |
| pca3_sparse | pca3s_ust_3y  |          53.19 |             4.82 |            5.31 |                   -120.63 |
| pca3_sparse | pca3s_ust_5y  |           1.49 |             6.09 |            6.09 |                      0.94 |
| pca3_sparse | pca3s_ust_7y  |           0.50 |             5.42 |            5.38 |                      0.99 |
| pca3_sparse | pca3s_ust_10y |           1.03 |             5.82 |            6.09 |                      0.97 |
| pca3_sparse | pca3s_ust_30y |           2.89 |             5.27 |            6.09 |                      0.70 |

In PCA-3 sparse the 2y and 3y members come out with negative variance removed. The package is mostly hedge legs: in 2010–2013 the 3y unit carried weights of +68 (2y), −121 (5y) and +69 (10y), from a nearly singular 3x3 solve that passes the `cond < 1e8` check in `sparse_hedges`. Leg weights by fold: `tables/s3_leg_weights_<model>.csv`. Daily gross and net DV01 by leg: `tables/s3_dv01_daily_<model>.csv`.

DV01 by leg, averaged over live days: bp of capital per 1 bp move, and dollars per bp on a $1m book. Gross sums members' |exposure|; net is the book after members offset. Daily PC1–3 exposure in both units: `tables/s3_factor_exposure_daily_<model>.csv`.

| model       | leg           |   mean_gross_bp_of_cap_per_bp |   mean_abs_net_bp_of_cap_per_bp |   mean_gross_usd_per_bp |   mean_abs_net_usd_per_bp |   net_over_gross |
|:------------|:--------------|------------------------------:|--------------------------------:|------------------------:|--------------------------:|-----------------:|
| headline    | ust_2y        |                         12.46 |                           11.39 |                 1245.79 |                   1138.64 |             0.91 |
| headline    | ust_3y        |                          9.69 |                            8.65 |                  968.59 |                    864.69 |             0.89 |
| headline    | ust_5y        |                         15.33 |                           13.25 |                 1532.54 |                   1324.54 |             0.86 |
| headline    | ust_7y        |                          6.65 |                            5.33 |                  665.48 |                    532.89 |             0.80 |
| headline    | ust_10y       |                         15.39 |                           14.21 |                 1539.21 |                   1420.70 |             0.92 |
| headline    | ust_30y       |                         14.15 |                           13.50 |                 1415.46 |                   1350.43 |             0.95 |
| headline    | hy_oas        |                          0.58 |                            0.58 |                   58.14 |                     58.14 |             1.00 |
| headline    | ig_oas        |                          1.27 |                            1.27 |                  127.02 |                    127.02 |             1.00 |
| headline    | cdx_hy_spread |                          0.24 |                            0.24 |                   24.20 |                     24.20 |             1.00 |
| headline    | hyg           |                          7.77 |                            7.77 |                  776.86 |                    776.86 |             1.00 |
| headline    | cdx_hy_price  |                          5.36 |                            5.36 |                  535.66 |                    535.66 |             1.00 |
| pca3_sparse | ust_2y        |                         11.16 |                           11.16 |                 1116.11 |                   1115.94 |             1.00 |
| pca3_sparse | ust_3y        |                         14.07 |                           13.92 |                 1407.37 |                   1391.87 |             0.99 |
| pca3_sparse | ust_5y        |                         27.14 |                           26.68 |                 2713.54 |                   2667.51 |             0.98 |
| pca3_sparse | ust_7y        |                         10.56 |                           10.18 |                 1056.37 |                   1018.46 |             0.96 |
| pca3_sparse | ust_10y       |                         22.62 |                           22.60 |                 2261.99 |                   2260.36 |             1.00 |
| pca3_sparse | ust_30y       |                          9.19 |                            9.19 |                  919.45 |                    919.45 |             1.00 |

> **What this tells you:** a 1 bp move in one Treasury leg moves the headline by roughly 0.07–0.15 bp of capital ($700–1,500 on $1m). Net is 80–95% of gross for the headline and 96–100% for PCA-3 sparse, so members seldom offset each other, which is why netting saves little (experiment 6).

## Step 4. Cost attribution

> **What this tells you:** why low-volatility trades lose. Cost is fixed in bp, but the position is sized to 1 / vol, so in a quiet market the same risk needs more DV01 and the round trip eats more of the edge. In the low-vol third a round trip costs 1.3–2.3x a day's move, and those trades lose for every model.

Cost by leg (% of capital a year): the engine charge allocated to legs by |w| × c, and the leg-level cost of the netted book.

| model       | leg           |   engine_allocated_pa |   netted_leg_pa |
|:------------|:--------------|----------------------:|----------------:|
| headline    | cdx_hy_price  |                0.0010 |          0.0004 |
| headline    | cdx_hy_spread |                0.0029 |          0.0018 |
| headline    | hy_oas        |                0.0068 |          0.0052 |
| headline    | hyg           |                0.0005 |          0.0002 |
| headline    | ig_oas        |                0.0018 |          0.0019 |
| headline    | ust_10y       |                0.0160 |          0.0159 |
| headline    | ust_2y        |                0.0132 |          0.0134 |
| headline    | ust_30y       |                0.0152 |          0.0155 |
| headline    | ust_3y        |                0.0099 |          0.0098 |
| headline    | ust_5y        |                0.0167 |          0.0165 |
| headline    | ust_7y        |                0.0059 |          0.0055 |
| pca3_sparse | ust_10y       |                0.0217 |          0.0230 |
| pca3_sparse | ust_2y        |                0.0130 |          0.0135 |
| pca3_sparse | ust_30y       |                0.0088 |          0.0094 |
| pca3_sparse | ust_3y        |                0.0181 |          0.0182 |
| pca3_sparse | ust_5y        |                0.0298 |          0.0308 |
| pca3_sparse | ust_7y        |                0.0120 |          0.0121 |

Cost per trade against the residual's volatility at entry (risk units; vol in bp a day, read off the position as `|signal| / |p|`). Terciles are over each model's trades.

| model       | vol_tercile   |   trades |   median_entry_vol_bp |   mean_round_trip_over_vol |   mean_cost |   mean_gross |   mean_net |
|:------------|:--------------|---------:|----------------------:|---------------------------:|------------:|-------------:|-----------:|
| headline    | low vol       |      142 |                  0.39 |                       1.78 |        2.05 |         0.80 |      -1.25 |
| headline    | mid vol       |      142 |                  0.68 |                       1.01 |        1.13 |         0.86 |      -0.27 |
| headline    | high vol      |      142 |                  4.40 |                       0.59 |        0.82 |         0.49 |      -0.33 |
| pca3_sparse | low vol       |      231 |                  0.66 |                       2.32 |        2.55 |         2.07 |      -0.48 |
| pca3_sparse | mid vol       |      230 |                  1.16 |                       1.63 |        1.84 |         1.88 |       0.04 |
| pca3_sparse | high vol      |      230 |                  3.29 |                       1.14 |        1.33 |         1.81 |       0.48 |
| pca2        | low vol       |      115 |                  0.37 |                       1.96 |        2.19 |         0.73 |      -1.47 |
| pca2        | mid vol       |      114 |                  0.57 |                       1.16 |        1.28 |         1.09 |      -0.20 |
| pca2        | high vol      |      115 |                  1.08 |                       0.61 |        0.70 |         0.26 |      -0.45 |
| pairwise_nn | low vol       |       84 |                  0.77 |                       1.30 |        1.63 |        -0.14 |      -1.77 |
| pairwise_nn | mid vol       |       84 |                  1.68 |                       0.61 |        0.71 |         0.70 |      -0.01 |
| pairwise_nn | high vol      |       84 |                  7.89 |                       0.63 |        0.96 |         0.87 |      -0.09 |

## Step 5. Signal diagnostics

> **What this tells you:** the signal points the right way (positive IC at every horizon), but the OU model behind it promises bigger, faster reversions than arrive. That is why a cost filter built on the OU forecast barely filters.

Information coefficient: Spearman correlation of −z with the forward h-bar change of the member (bp). Positive means the signal points the right way.

| model       |   horizon | scope      |     n |    ic |   p_value |
|:------------|----------:|:-----------|------:|------:|----------:|
| headline    |         5 | all bars   | 47101 | 0.072 |     0.000 |
| headline    |         5 | entry bars |   425 | 0.168 |     0.000 |
| headline    |        10 | all bars   | 47056 | 0.082 |     0.000 |
| headline    |        10 | entry bars |   425 | 0.166 |     0.001 |
| headline    |        20 | all bars   | 46966 | 0.072 |     0.000 |
| headline    |        20 | entry bars |   423 | 0.091 |     0.060 |
| pca3_sparse |         5 | all bars   | 39061 | 0.163 |     0.000 |
| pca3_sparse |         5 | entry bars |   691 | 0.223 |     0.000 |
| pca3_sparse |        10 | all bars   | 39031 | 0.211 |     0.000 |
| pca3_sparse |        10 | entry bars |   688 | 0.326 |     0.000 |
| pca3_sparse |        20 | all bars   | 38971 | 0.146 |     0.000 |
| pca3_sparse |        20 | entry bars |   688 | 0.260 |     0.000 |
| pca2        |         5 | all bars   | 36525 | 0.058 |     0.000 |
| pca2        |         5 | entry bars |   343 | 0.150 |     0.005 |
| pca2        |        10 | all bars   | 36495 | 0.064 |     0.000 |
| pca2        |        10 | entry bars |   343 | 0.196 |     0.000 |
| pca2        |        20 | all bars   | 36435 | 0.041 |     0.000 |
| pca2        |        20 | entry bars |   341 | 0.078 |     0.153 |
| pairwise_nn |         5 | all bars   | 98368 | 0.052 |     0.000 |
| pairwise_nn |         5 | entry bars |   252 | 0.168 |     0.007 |
| pairwise_nn |        10 | all bars   | 98278 | 0.049 |     0.000 |
| pairwise_nn |        10 | entry bars |   252 | 0.127 |     0.044 |
| pairwise_nn |        20 | all bars   | 98098 | 0.050 |     0.000 |
| pairwise_nn |        20 | entry bars |   252 | 0.126 |     0.045 |

Win rate and mean P&L by |z| at entry (risk units). Entry bands are optimised per fold from 1.0 to 2.5, so many trades open below 2.

| model       | entry_abs_z   |   trades |   win_rate_net |   mean_gross |   mean_cost |   mean_net |
|:------------|:--------------|---------:|---------------:|-------------:|------------:|-----------:|
| headline    | <2            |      233 |           0.54 |         0.64 |        1.29 |      -0.65 |
| headline    | 2-2.5         |       98 |           0.58 |         1.03 |        1.35 |      -0.32 |
| headline    | 2.5-3         |       78 |           0.49 |         0.46 |        1.49 |      -1.04 |
| headline    | 3+            |       17 |           0.71 |         1.11 |        1.11 |      -0.00 |
| pca3_sparse | <2            |      462 |           0.66 |         1.92 |        1.77 |       0.15 |
| pca3_sparse | 2-2.5         |       94 |           0.64 |         1.86 |        1.97 |      -0.10 |
| pca3_sparse | 2.5-3         |      111 |           0.60 |         1.77 |        2.36 |      -0.59 |
| pca3_sparse | 3+            |       24 |           0.54 |         2.82 |        2.16 |       0.66 |
| pca2        | <2            |      173 |           0.51 |         0.45 |        1.35 |      -0.89 |
| pca2        | 2-2.5         |       86 |           0.60 |         1.14 |        1.38 |      -0.24 |
| pca2        | 2.5-3         |       71 |           0.49 |         0.50 |        1.55 |      -1.05 |
| pca2        | 3+            |       14 |           0.86 |         1.78 |        1.29 |       0.49 |
| pairwise_nn | <2            |      158 |           0.58 |         0.74 |        1.16 |      -0.42 |
| pairwise_nn | 2-2.5         |       43 |           0.51 |        -0.23 |        1.08 |      -1.31 |
| pairwise_nn | 2.5-3         |       36 |           0.58 |         1.31 |        1.06 |       0.25 |
| pairwise_nn | 3+            |       15 |           0.13 |        -2.20 |        0.68 |      -2.88 |

OU half-life: AR(1) on each training window's cumulative residual (that fold's weights), the median trailing 60-bar half-life the signal used in the test block, and the actual median holding. Per fold: `tables/s5_ou_by_fold_<model>.csv`. For pairwise, the half-life is the adaptive spread's from the backtest's fold diagnostics.

| model       |   members |   median_train_ar1_half_life |   share_folds_not_mean_reverting |   median_rolling60_half_life |   median_holding_bars |
|:------------|----------:|-----------------------------:|---------------------------------:|-----------------------------:|----------------------:|
| headline    |         6 |                        102.8 |                              0.1 |                          6.5 |                   9.0 |
| pca3_sparse |         6 |                         31.7 |                              0.0 |                          4.7 |                  10.0 |
| pca2        |         6 |                        102.8 |                              0.1 |                          6.5 |                  10.0 |
| pairwise_nn |        18 |                         26.0 |                            nan   |                        nan   |                   7.0 |

Why the 60-bar half-life is so short. Each fold's training-window AR(1) is taken as the true process, simulated 500 times and fitted on 60 bars exactly as `pca.rolling_ou` does, before and after its screen (0 < b < 1, half-life under 30 bars). The training b is itself biased slightly fast, so this understates the bias. `kendall_implied` is the textbook first-order correction b − (1 + 3b) / 60.

| model       |   fold_member_fits |   median_train_half_life |   kendall_implied_60bar_half_life |   sim_60bar_half_life_before_screen |   sim_screen_pass_share |   sim_60bar_half_life_after_screen |   observed_screen_pass_share |   observed_60bar_half_life |   observed_half_life_at_entry |   median_holding_bars |
|:------------|-------------------:|-------------------------:|----------------------------------:|------------------------------------:|------------------------:|-----------------------------------:|-----------------------------:|---------------------------:|------------------------------:|----------------------:|
| pca2        |                152 |                   102.81 |                              9.14 |                                8.31 |                    0.88 |                               7.59 |                         0.90 |                       6.35 |                          3.54 |                 10.00 |
| pca3_sparse |                162 |                    31.72 |                              7.60 |                                7.13 |                    0.92 |                               6.74 |                         0.96 |                       4.49 |                          3.01 |                 10.00 |

> **What this tells you:** most of the gap from 103 to about 6 bars is the estimator, not the market and not the screen. A 60-bar AR(1) fit on a slowly reverting series reads it as fast: 8.3 bars before the screen, 7.6 after it. The screen passes about 9 in 10 windows, so it selects little. Entries happen when the fit looks fastest (3.0–3.5 bars), yet trades last 10. Treat the 60-bar half-life and σ_eq as noisy, optimistic estimates.

Fix A (cost-aware entry) checked on the entries that happened: enter only if (|z| − exit band) × σ_eq > k × round trip. All inputs are known on the entry bar. `realised_move_bp` is what the residual then did, signed so that reversion is positive. `k_that_halves_entries` is the median of forecast / round trip. Per entry: `tables/s5_cost_aware_entry_<model>.csv`.

| model       |   entries |   pass_share_k1 |   median_expected_reversion_bp |   median_round_trip_bp |   mean_realised_move_bp |   median_realised_move_bp |   k_that_halves_entries |   mean_net_passing |   mean_net_failing |
|:------------|----------:|----------------:|-------------------------------:|-----------------------:|------------------------:|--------------------------:|------------------------:|-------------------:|-------------------:|
| pca2        |       344 |           0.919 |                          1.667 |                  0.725 |                   0.454 |                     0.876 |                   2.441 |             -0.562 |             -2.310 |
| pca3_sparse |       691 |           0.863 |                          3.002 |                  1.698 |                   3.985 |                     2.486 |                   1.600 |              0.167 |             -0.948 |

> **What this tells you:** at k = 1 the rule keeps 92% of PCA-2 entries and 86% of PCA-3 sparse's, so it would change little. For PCA-2 the OU forecast (1.67 bp median) is nearly 4x the mean realised move (0.45 bp). The entries it would drop do lose more, so the idea is sound, but k must be fitted per fold and will be well above 1.

Residual paths of the ten worst trades per model: `tables/s5_worst_paths_<model>.csv` and the dashboard.

## Step 6. Regimes and robustness

> **What this tells you:** the losses are not one regime's fault. Every Fed regime loses net, auction weeks are no different, and the P&L barely correlates with rates, MOVE or credit. Stress windows matter at the margin: the GFC hurt the headline and February 2021 hurt PCA-3 sparse. The curve sat out March–December 2022.

Stress windows (net, gross and cost in fractions of capital):

| model       | window           |    net |   gross |   cost |   max_dd |   days |
|:------------|:-----------------|-------:|--------:|-------:|---------:|-------:|
| headline    | GFC Sep08-Mar09  | -0.173 |  -0.141 |  0.032 |   -0.181 |    152 |
| headline    | Taper May-Sep13  |  0.032 |   0.038 |  0.006 |   -0.019 |    109 |
| headline    | Covid Feb-Apr20  |  0.017 |   0.066 |  0.050 |   -0.048 |     54 |
| headline    | Feb 2021 selloff | -0.060 |  -0.044 |  0.016 |   -0.073 |     31 |
| headline    | 2022 hikes       |  0.033 |   0.095 |  0.062 |   -0.062 |    219 |
| pca3_sparse | GFC Sep08-Mar09  |  0.034 |   0.037 |  0.003 |   -0.014 |    152 |
| pca3_sparse | Taper May-Sep13  | -0.023 |  -0.007 |  0.017 |   -0.108 |    109 |
| pca3_sparse | Covid Feb-Apr20  | -0.075 |  -0.026 |  0.049 |   -0.075 |     54 |
| pca3_sparse | Feb 2021 selloff | -0.104 |  -0.081 |  0.024 |   -0.110 |     31 |
| pca3_sparse | 2022 hikes       |  0.004 |   0.027 |  0.023 |   -0.025 |    219 |
| pca2        | GFC Sep08-Mar09  | -0.136 |  -0.113 |  0.023 |   -0.144 |    152 |
| pca2        | Taper May-Sep13  |  0.036 |   0.043 |  0.007 |   -0.022 |    109 |
| pca2        | Covid Feb-Apr20  |  0.017 |   0.068 |  0.051 |   -0.069 |     54 |
| pca2        | Feb 2021 selloff | -0.070 |  -0.049 |  0.022 |   -0.088 |     31 |
| pca2        | 2022 hikes       |  0.000 |   0.000 |  0.000 |    0.000 |    219 |
| pairwise_nn | GFC Sep08-Mar09  | -0.149 |  -0.109 |  0.039 |   -0.182 |    152 |
| pairwise_nn | Taper May-Sep13  |  0.000 |   0.000 |  0.000 |    0.000 |    109 |
| pairwise_nn | Covid Feb-Apr20  | -0.148 |  -0.111 |  0.038 |   -0.165 |     54 |
| pairwise_nn | Feb 2021 selloff | -0.015 |   0.006 |  0.021 |   -0.033 |     31 |
| pairwise_nn | 2022 hikes       |  0.036 |   0.102 |  0.066 |   -0.066 |    219 |
| fly_2s5s10s | GFC Sep08-Mar09  |  0.032 |   0.041 |  0.009 |   -0.049 |    152 |
| fly_2s5s10s | Taper May-Sep13  |  0.000 |   0.000 |  0.000 |    0.000 |    109 |
| fly_2s5s10s | Covid Feb-Apr20  |  0.000 |   0.000 |  0.000 |    0.000 |     54 |
| fly_2s5s10s | Feb 2021 selloff |  0.013 |   0.017 |  0.004 |   -0.032 |     31 |
| fly_2s5s10s | 2022 hikes       | -0.094 |  -0.076 |  0.018 |   -0.094 |    219 |
| fly_3s5s7s  | GFC Sep08-Mar09  |  0.000 |   0.000 |  0.000 |  nan     |      0 |
| fly_3s5s7s  | Taper May-Sep13  |  0.000 |   0.000 |  0.000 |  nan     |      0 |
| fly_3s5s7s  | Covid Feb-Apr20  | -0.023 |  -0.011 |  0.012 |   -0.023 |     54 |
| fly_3s5s7s  | Feb 2021 selloff | -0.153 |  -0.128 |  0.025 |   -0.163 |     31 |
| fly_3s5s7s  | 2022 hikes       | -0.130 |  -0.099 |  0.032 |   -0.130 |    219 |

Fed regime: hiking / cutting / on hold from the change in the fed funds target (DFEDTAR to 2008-12-15, then the upper bound DFEDTARU) over the trailing 126 bars.

| model       | fed_regime   |   days |   net_pa |   gross_pa |   sharpe |
|:------------|:-------------|-------:|---------:|-----------:|---------:|
| headline    | cutting      |   2081 |   -0.065 |      0.020 |   -0.792 |
| headline    | hiking       |   2389 |   -0.037 |      0.069 |   -0.481 |
| headline    | on hold      |   3579 |   -0.056 |      0.039 |   -0.629 |
| pca3_sparse | cutting      |   2081 |   -0.004 |      0.071 |   -0.060 |
| pca3_sparse | hiking       |   2389 |   -0.003 |      0.124 |   -0.041 |
| pca3_sparse | on hold      |   3579 |   -0.012 |      0.097 |   -0.147 |
| pca2        | cutting      |   2081 |   -0.068 |      0.014 |   -0.827 |
| pca2        | hiking       |   2389 |   -0.053 |      0.053 |   -0.633 |
| pca2        | on hold      |   3579 |   -0.062 |      0.037 |   -0.670 |
| pairwise_nn | cutting      |   2081 |   -0.026 |      0.045 |   -0.286 |
| pairwise_nn | hiking       |   2389 |   -0.004 |      0.042 |   -0.062 |
| pairwise_nn | on hold      |   3579 |   -0.027 |      0.025 |   -0.346 |

Calendar eras, as the brief asked (1995–2004 vs 2009+). These are data eras: the 3y data starts in 2005 and the 7y in 2009, but neither trades until it has a full 5-year training window (next tables).

| model       | era       |   days |   net_pa |   gross_pa |   sharpe |
|:------------|:----------|-------:|---------:|-----------:|---------:|
| headline    | 1995-2004 |   2509 |   -0.045 |      0.037 |   -0.509 |
| headline    | 2005-2008 |   1042 |   -0.029 |      0.060 |   -0.339 |
| headline    | 2009+     |   4624 |   -0.058 |      0.047 |   -0.713 |
| pca3_sparse | 1995-2004 |   2509 |    0.007 |      0.094 |    0.080 |
| pca3_sparse | 2005-2008 |   1042 |   -0.018 |      0.076 |   -0.192 |
| pca3_sparse | 2009+     |   4624 |   -0.012 |      0.110 |   -0.163 |
| pca2        | 1995-2004 |   2509 |   -0.045 |      0.037 |   -0.509 |
| pca2        | 2005-2008 |   1042 |   -0.021 |      0.071 |   -0.240 |
| pca2        | 2009+     |   4624 |   -0.074 |      0.032 |   -0.853 |
| pairwise_nn | 1995-2004 |   2509 |    0.036 |      0.094 |    0.422 |
| pairwise_nn | 2005-2008 |   1042 |   -0.111 |     -0.065 |   -1.197 |
| pairwise_nn | 2009+     |   4624 |   -0.030 |      0.026 |   -0.441 |

When each tenor's data starts and when its member first trades:

| model   | member       | data_start   | first_traded   |
|:--------|:-------------|:-------------|:---------------|
| pca2    | pca2_ust_2y  | 1990-01-02   | 1994-12-02     |
| pca2    | pca2_ust_3y  | 2005-01-03   | 2010-05-21     |
| pca2    | pca2_ust_5y  | 1990-01-02   | 1994-12-02     |
| pca2    | pca2_ust_7y  | 2009-02-26   | 2014-04-02     |
| pca2    | pca2_ust_10y | 1990-01-02   | 1994-12-02     |
| pca2    | pca2_ust_30y | 1990-01-02   | 1994-12-02     |

The same split by the tenors the PCA actually used:

| model       | era                       |   days |   net_pa |   gross_pa |   sharpe |
|:------------|:--------------------------|-------:|---------:|-----------:|---------:|
| headline    | 2010-05 to 2014-04: + 3y  |   1008 |   -0.015 |      0.023 |   -0.186 |
| headline    | 2014-04 on: all six       |   3255 |   -0.071 |      0.061 |   -0.879 |
| headline    | to 2010-05: 2y 5y 10y 30y |   3912 |   -0.042 |      0.038 |   -0.483 |
| pca3_sparse | 2010-05 to 2014-04: + 3y  |   1008 |   -0.049 |     -0.025 |   -0.746 |
| pca3_sparse | 2014-04 on: all six       |   3255 |   -0.010 |      0.151 |   -0.125 |
| pca3_sparse | to 2010-05: 2y 5y 10y 30y |   3912 |    0.006 |      0.091 |    0.067 |
| pca2        | 2010-05 to 2014-04: + 3y  |   1008 |   -0.017 |      0.026 |   -0.190 |
| pca2        | 2014-04 on: all six       |   3255 |   -0.098 |      0.038 |   -1.097 |
| pca2        | to 2010-05: 2y 5y 10y 30y |   3912 |   -0.036 |      0.043 |   -0.423 |
| pairwise_nn | 2010-05 to 2014-04: + 3y  |   1008 |   -0.036 |     -0.011 |   -0.497 |
| pairwise_nn | 2014-04 on: all six       |   3255 |   -0.036 |      0.029 |   -0.579 |
| pairwise_nn | to 2010-05: 2y 5y 10y 30y |   3912 |   -0.003 |      0.053 |   -0.028 |

> **What this tells you:** the 3y trades from 2010 and the 7y from 2014, so the six-tenor curve covers only the last 12 years. That era is the worst for the headline and PCA-2 (net Sharpe −0.88 and −1.10), so adding tenors did not help them. PCA-3 sparse's weak 2010–2014 era is the construction defect in Step 3.

5y auction weeks (TreasuryDirect auction dates, reopenings included) vs other weeks:

| model       |   auction_week_days |   other_days |   auction_week_bp_per_day |   other_bp_per_day |   auction_week_sharpe |   other_sharpe |   welch_t |   p_value |
|:------------|--------------------:|-------------:|--------------------------:|-------------------:|----------------------:|---------------:|----------:|----------:|
| headline    |                1662 |         6513 |                    -4.050 |             -1.470 |                -1.119 |         -0.452 |    -1.668 |     0.096 |
| pca3_sparse |                1662 |         6513 |                    -0.757 |             -0.159 |                -0.178 |         -0.055 |    -0.341 |     0.733 |
| pca2        |                1662 |         6513 |                    -3.584 |             -1.984 |                -0.940 |         -0.590 |    -0.984 |     0.325 |
| pairwise_nn |                1662 |         6513 |                    -0.860 |             -0.776 |                -0.242 |         -0.263 |    -0.056 |     0.956 |

Correlation of daily net P&L with market moves:

| model       |   hy_oas_change |   move_change |   move_level |   rates_level_change |
|:------------|----------------:|--------------:|-------------:|---------------------:|
| headline    |          -0.068 |        -0.070 |        0.003 |                0.051 |
| pairwise_nn |          -0.090 |        -0.051 |       -0.003 |                0.040 |
| pca2        |          -0.026 |        -0.056 |        0.011 |                0.022 |
| pca3_sparse |          -0.018 |        -0.036 |        0.008 |               -0.012 |

Headline: curve members vs credit members, daily net contribution. Stress = MOVE at or above its trailing 5-year 80th percentile.

| days   |    n |   corr_curve_credit |
|:-------|-----:|--------------------:|
| all    | 5533 |               0.003 |
| normal | 4526 |              -0.009 |
| stress | 1007 |               0.034 |

Headline: curve vs credit performance. Daily capital contribution at the sleeve weight, fractions of capital a year; the sleeve-rescale cost is not split and is left out. Credit-live days start 2005-07-07 (hy_ig_oas); the CDX pairs start in 2016-17. `var_share_credit_live` is each part's share of book variance on those days.

| part   | scope            |   days |   gross_pa |   cost_pa |   net_pa |   net_vol_pa |   gross_sharpe |   net_sharpe |   cost_over_gross |   var_share_credit_live |
|:-------|:-----------------|-------:|-----------:|----------:|---------:|-------------:|---------------:|-------------:|------------------:|------------------------:|
| curve  | full sample      |   8175 |      0.037 |     0.077 |   -0.040 |        0.078 |          0.474 |       -0.511 |             2.087 |                 nan     |
| curve  | credit-live days |   5533 |      0.037 |     0.080 |   -0.043 |        0.074 |          0.499 |       -0.580 |             2.179 |                   0.807 |
| credit | full sample      |   8175 |      0.009 |     0.013 |   -0.004 |        0.030 |          0.293 |       -0.146 |             1.506 |                 nan     |
| credit | credit-live days |   5533 |      0.013 |     0.019 |   -0.006 |        0.036 |          0.356 |       -0.178 |             1.506 |                   0.193 |

By member, each over its own live days. `in_market_share` is the share of live days with a position.

| part   | member         | first_live   |   in_market_share |   gross_pa |   cost_pa |   net_pa |   net_vol_pa |   gross_sharpe |   net_sharpe |   cost_over_gross |
|:-------|:---------------|:-------------|------------------:|-----------:|----------:|---------:|-------------:|---------------:|-------------:|------------------:|
| curve  | pca2_ust_2y    | 1995-05-19   |             0.129 |      0.008 |     0.014 |   -0.006 |        0.032 |          0.240 |       -0.176 |             1.724 |
| curve  | pca2_ust_3y    | 2010-05-21   |             0.155 |      0.022 |     0.033 |   -0.011 |        0.028 |          0.776 |       -0.406 |             1.525 |
| curve  | pca2_ust_5y    | 1995-05-19   |             0.076 |      0.002 |     0.011 |   -0.009 |        0.028 |          0.058 |       -0.320 |             6.545 |
| curve  | pca2_ust_7y    | 2014-04-02   |             0.090 |     -0.003 |     0.009 |   -0.013 |        0.024 |         -0.138 |       -0.523 |           nan     |
| curve  | pca2_ust_10y   | 1995-05-19   |             0.117 |      0.013 |     0.019 |   -0.007 |        0.032 |          0.393 |       -0.212 |             1.532 |
| curve  | pca2_ust_30y   | 1995-05-19   |             0.099 |      0.005 |     0.012 |   -0.008 |        0.030 |          0.153 |       -0.251 |             2.650 |
| credit | hy_ig_oas      | 2005-07-07   |             0.076 |     -0.000 |     0.006 |   -0.006 |        0.028 |         -0.000 |       -0.228 |           nan     |
| credit | cdx_cash_basis | 2017-06-28   |             0.130 |      0.027 |     0.025 |    0.002 |        0.027 |          1.010 |        0.077 |             0.923 |
| credit | hyg_cdx        | 2016-07-29   |             0.080 |      0.002 |     0.004 |   -0.002 |        0.023 |          0.100 |       -0.099 |             1.990 |

Net by stress window (fractions of capital):

| window           |   curve_net |   credit_net |
|:-----------------|------------:|-------------:|
| GFC Sep08-Mar09  |      -0.091 |       -0.080 |
| Taper May-Sep13  |       0.032 |        0.000 |
| Covid Feb-Apr20  |       0.031 |       -0.013 |
| Feb 2021 selloff |      -0.060 |        0.000 |
| 2022 hikes       |       0.000 |        0.033 |

PCA-2 curve members in 2022 (all one fold, starting 2021-12-23): the entry band, the largest |z|, the share of live bars where the 60-bar screen blanked the signal, and days holding a position.

| member       | period       | fold_start   |   entry_band |   max_abs_z |   screen_blank_share |   days_in_market |   live_days |
|:-------------|:-------------|:-------------|-------------:|------------:|---------------------:|-----------------:|------------:|
| pca2_ust_2y  | Jan-Feb 2022 | 2021-12-23   |        2.500 |       0.887 |                0.707 |                0 |          41 |
| pca2_ust_2y  | Mar-Dec 2022 | 2021-12-23   |        2.500 |       1.344 |                0.493 |                0 |         219 |
| pca2_ust_3y  | Jan-Feb 2022 | 2021-12-23   |        2.500 |       1.396 |                0.000 |                0 |          41 |
| pca2_ust_3y  | Mar-Dec 2022 | 2021-12-23   |        2.500 |       2.112 |                0.169 |                0 |         219 |
| pca2_ust_5y  | Jan-Feb 2022 | 2021-12-23   |        2.000 |       0.815 |                0.634 |               13 |          41 |
| pca2_ust_5y  | Mar-Dec 2022 | 2021-12-23   |        2.000 |       1.992 |                0.425 |                0 |         219 |
| pca2_ust_7y  | Jan-Feb 2022 | 2021-12-23   |        2.500 |       0.462 |                0.659 |                0 |          41 |
| pca2_ust_7y  | Mar-Dec 2022 | 2021-12-23   |        2.500 |       1.714 |                0.338 |                0 |         219 |
| pca2_ust_10y | Jan-Feb 2022 | 2021-12-23   |        2.500 |       1.009 |                0.220 |                0 |          41 |
| pca2_ust_10y | Mar-Dec 2022 | 2021-12-23   |        2.500 |       2.049 |                0.183 |                0 |         219 |
| pca2_ust_30y | Jan-Feb 2022 | 2021-12-23   |        2.500 |       0.932 |                0.854 |                0 |          41 |
| pca2_ust_30y | Mar-Dec 2022 | 2021-12-23   |        2.500 |       1.733 |                0.434 |                0 |         219 |

Credit is a small diversifier: its gross Sharpe is lower than the curve's, each pair holds a position on 8-13% of its live days, and cdx_cash_basis is the only member of the headline with positive net P&L (48 trades since 2017, too few to separate from luck). `cost_over_gross` is blank where gross P&L is not positive. Credit and curve lost together in the GFC but not in 2020 or 2022. The curve is flat from March to December 2022: the fold starting 2021-12-23 set entry at 2.5 for every tenor but the 5y (2.0), no residual's |z| passed 2.11 from March (the 5y peaked at 1.99), and the 60-bar speed screen blanked 17–49% of bars. Only the 5y traded in 2022, for 13 days in January–February.

## Step 7. Overfitting

> **What this tells you:** how much of the best result is luck from trying many things. PCA-3 sparse is the best of 31 documented trials. Its gross P&L survives that adjustment; its net P&L does not. Its six members behave like 2.3 independent bets, so its sample is thinner than the member count suggests.

Variants tried before and around PCA-3 sparse. The repo has no git history, so the count comes from the tables and the appendix's own list of design revisions. Per-fold band and Kalman choices are fitted in-sample within each fold and are not counted as trials.

| source                                                                                               |   trials |
|:-----------------------------------------------------------------------------------------------------|---------:|
| model table rows (reports/tables/sleeve1_models.csv)                                                 |       11 |
| cost-filter variants not already in the model table (cost_filter_effect.csv, 12 rows - 4 duplicates) |        8 |
| 2s10s curve rules other than PCA (curve_full.csv, 10 rows - 2 PCA rows)                              |        8 |
| design revisions listed in appendix.md (adaptive x2, cointegration screen, vol estimate)             |        4 |

| trial_set                    |   n_trials |   trial_sharpe_sd |   pca3_sparse_sharpe_net |   dsr_net |   pca3_sparse_sharpe_gross |   dsr_gross |
|:-----------------------------|-----------:|------------------:|-------------------------:|----------:|---------------------------:|------------:|
| narrow: model table          |         11 |             0.179 |                   -0.087 |     0.016 |                      1.210 |       1.000 |
| broad: all documented trials |         31 |             0.244 |                   -0.087 |     0.000 |                      1.210 |       1.000 |

The deflated Sharpe is the probability that the true Sharpe exceeds zero after selecting the best of N trials (Bailey and López de Prado), using the dispersion of the trials' Sharpe ratios. The 4 revisions have no saved Sharpe, so the broad set uses the dispersion of the 27 that do.

Effective number of independent bets, from the correlation of members' daily P&L (reproduces trades.md):

| model       |   members_gross |   effective_bets_gross |   effective_bets_net |   trades_md_effective_bets |
|:------------|----------------:|-----------------------:|---------------------:|---------------------------:|
| headline    |               9 |                   8.69 |                 8.67 |                       8.69 |
| pca3_sparse |               6 |                   2.28 |                 2.28 |                       2.28 |
| pca2        |               6 |                   5.71 |                 5.68 |                       5.71 |
| pairwise_nn |              17 |                  14.37 |                14.38 |                      14.37 |

## Step 8. Experiments

> **What this tells you:** what simple rule changes do out of sample. Trading less (higher entry band, rebalancing only on big changes, a MOVE gate) cuts cost faster than it cuts gross, so net improves, but nothing turns positive. Most parameters were picked after seeing results, so treat the improvements as upper bounds.

Walk-forward, each against the unchanged baseline, built with `analysis/experiments/pipeline.py`. With no hook set it reproduces the production runs bit for bit (tested). Bands are re-optimised on each training window under the experiment's rule. Headline = the experiment's PCA-2 curve plus the unchanged credit pairs. Returns in fractions of capital a year; `trades` merges identical trades across members, as trades.md does.

| experiment       | model        |   trades |   trades_per_year |   gross_pa |   cost_pa |   net_pa |   sharpe_net |   sharpe_gross |   max_dd |   d_sharpe_net |   d_net_pa |
|:-----------------|:-------------|---------:|------------------:|-----------:|----------:|---------:|-------------:|---------------:|---------:|---------------:|-----------:|
| baseline         | PCA-2 full   |      344 |            10.604 |      0.038 |     0.097 |   -0.058 |       -0.668 |          0.448 |   -2.044 |          0.000 |      0.000 |
| baseline         | Headline     |      426 |            13.132 |      0.046 |     0.096 |   -0.050 |       -0.599 |          0.547 |   -1.788 |          0.000 |      0.000 |
| baseline         | PCA-3 sparse |      255 |             7.861 |      0.101 |     0.108 |   -0.007 |       -0.087 |          1.210 |   -0.540 |          0.000 |      0.000 |
| time_stop_20     | PCA-2 full   |      340 |            10.481 |      0.042 |     0.098 |   -0.056 |       -0.648 |          0.487 |   -2.003 |          0.019 |      0.002 |
| time_stop_20     | Headline     |      422 |            13.008 |      0.048 |     0.097 |   -0.050 |       -0.590 |          0.576 |   -1.787 |          0.009 |      0.001 |
| time_stop_20     | PCA-3 sparse |      235 |             7.244 |      0.083 |     0.112 |   -0.028 |       -0.350 |          1.021 |   -1.001 |         -0.264 |     -0.021 |
| resize_25pct     | PCA-2 full   |      344 |            10.604 |      0.037 |     0.090 |   -0.053 |       -0.601 |          0.433 |   -1.884 |          0.066 |      0.006 |
| resize_25pct     | Headline     |      426 |            13.132 |      0.045 |     0.090 |   -0.045 |       -0.536 |          0.537 |   -1.647 |          0.063 |      0.005 |
| resize_25pct     | PCA-3 sparse |      255 |             7.861 |      0.099 |     0.099 |   -0.000 |       -0.002 |          1.200 |   -0.445 |          0.085 |      0.007 |
| entry_2.5        | PCA-2 full   |      113 |             3.483 |      0.035 |     0.066 |   -0.030 |       -0.335 |          0.400 |   -1.206 |          0.333 |      0.028 |
| entry_2.5        | Headline     |      195 |             6.011 |      0.039 |     0.075 |   -0.036 |       -0.397 |          0.442 |   -1.375 |          0.202 |      0.014 |
| entry_2.5        | PCA-3 sparse |       86 |             2.709 |      0.075 |     0.077 |   -0.003 |       -0.034 |          0.940 |   -0.549 |          0.053 |      0.004 |
| entry_3.0        | PCA-2 full   |       19 |             0.604 |      0.018 |     0.032 |   -0.013 |       -0.193 |          0.276 |   -0.482 |          0.475 |      0.045 |
| entry_3.0        | Headline     |      101 |             3.212 |      0.027 |     0.050 |   -0.023 |       -0.270 |          0.342 |   -0.866 |          0.328 |      0.028 |
| entry_3.0        | PCA-3 sparse |       22 |             0.700 |      0.032 |     0.038 |   -0.006 |       -0.081 |          0.475 |   -0.339 |          0.006 |      0.001 |
| move_top_tercile | PCA-2 full   |      226 |             6.973 |      0.018 |     0.055 |   -0.037 |       -0.525 |          0.262 |   -1.339 |          0.143 |      0.021 |
| move_top_tercile | Headline     |      308 |             9.504 |      0.027 |     0.064 |   -0.037 |       -0.509 |          0.380 |   -1.371 |          0.090 |      0.013 |
| move_top_tercile | PCA-3 sparse |      131 |             4.038 |      0.070 |     0.073 |   -0.003 |       -0.036 |          0.934 |   -0.472 |          0.051 |      0.004 |
| loo_residual     | PCA-2 full   |      384 |            11.837 |      0.038 |     0.102 |   -0.064 |       -0.725 |          0.428 |   -2.335 |         -0.058 |     -0.006 |
| loo_residual     | Headline     |      466 |            14.365 |      0.044 |     0.102 |   -0.058 |       -0.677 |          0.521 |   -2.124 |         -0.078 |     -0.007 |
| loo_residual     | PCA-3 sparse |      442 |            13.625 |      0.067 |     0.132 |   -0.066 |       -0.771 |          0.787 |   -2.185 |         -0.684 |     -0.059 |

Where each parameter came from. Anything marked *after results* was proposed after full-sample results were seen, so even a walk-forward improvement is partly an in-sample choice:

| experiment       | provenance                                                                                                |
|:-----------------|:----------------------------------------------------------------------------------------------------------|
| baseline         | unchanged production model                                                                                |
| time_stop_20     | after results: lessons.md looked at trades held past 20 bars on the full sample                           |
| resize_25pct     | after results: 25% chosen in this review's brief; not tuned here                                          |
| entry_2.5        | after results: lessons.md/trades.md bucketed trades by |z| on the full sample                             |
| entry_3.0        | after results: as entry_2.5                                                                               |
| move_top_tercile | after results: lessons.md found high-MOVE trades did better on the full sample; cutoff itself is trailing |
| loo_residual     | no tuned parameter; construction change only                                                              |

LOO residual: each tenor is regressed (training window) on the k factors of the other tenors; it hedges with every other tenor, so PCA-3 sparse's comparison is against a full, not sparse, construction.

### Experiment 6: leg netting

> **What this tells you:** charging cost on the whole book's net trade instead of member by member saves 0.5–5.5%. Members rarely trade opposite sides of the same tenor on the same day.

Positions unchanged. Each tenor's exposure is netted across members daily and cost is charged only on the netted leg change; CDX roll charges are kept. `cost_leg_unnetted_pa` is the same leg-level pricing without netting (the engine-equivalent); the engine itself also charges sleeve rescaling slightly differently.

| model       |   cost_engine_pa |   cost_leg_unnetted_pa |   cost_leg_netted_pa |   saved_vs_engine_pa |   saved_vs_unnetted_share |   sharpe_net_engine |   sharpe_net_netted |   net_pa_netted |
|:------------|-----------------:|-----------------------:|---------------------:|---------------------:|--------------------------:|--------------------:|--------------------:|----------------:|
| headline    |            0.096 |                  0.094 |                0.090 |                0.006 |                     0.043 |              -0.599 |              -0.528 |          -0.044 |
| pca3_sparse |            0.108 |                  0.108 |                0.107 |                0.001 |                     0.005 |              -0.087 |              -0.079 |          -0.006 |
| pca2        |            0.097 |                  0.094 |                0.089 |                0.008 |                     0.055 |              -0.668 |              -0.581 |          -0.050 |
| pairwise_nn |            0.056 |                  0.055 |                0.054 |                0.002 |                     0.018 |              -0.257 |              -0.234 |          -0.018 |

## Changes made to existing code

- `src/amp_sa/sleeves/pca.py`: added `# [diag-logging]` lines that save per-bar OU state (z, half-life, kappa, σ_eq), fold loadings and test-window factor scores to `extras`. After rerunning with `--force`, all 360 P&L, cost, position, signal, turnover and unit-cost series of the 24 PCA runs were bit-identical to the previous cache (`analysis/portfolio_diagnostics/check_logging_identical.py`), as were `reports/` and `signals/signals.parquet`.
- External data cached (gitignored): `data/raw/fred/DFEDTAR.csv`, `DFEDTARU.csv`, `DFEDTARL.csv`; `data/raw/treasury/auctions_Note.json`, `auctions_Bond.json`, `coupon_auctions.csv`.
- Note: the repo's `.gitignore` ignores `*.csv`, so `tables/` will not be committed unless that rule is changed.
