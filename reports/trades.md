# Sleeve 1, trade by trade

Out-of-sample trades from the walk-forward runs, sample ending 2026-09-25. A trade is a run of bars with the same non-zero position in one instrument (a PCA residual or a pairwise spread). P&L is in risk units: one unit is one trailing daily standard deviation of that instrument, so `mean_gross` of 0.5 means the average trade earned half a day's typical move before costs. `cost` is entry, exit and any re-sizing in between.

The four models: the headline (PCA-2 curve plus adaptive credit pairs, named in advance), the best variant found afterwards (sparse PCA-3), the full PCA-2 projection, and the best pairwise model.

## The average trade

| model                     |   trades |   trades_per_year |   median_bars |   hit_rate_gross |   hit_rate_net |   mean_gross |   mean_cost |   mean_net |   cost_share_of_gross |   resize_share_of_cost |   avg_win_over_avg_loss |   top_decile_share_of_gross |
|:--------------------------|---------:|------------------:|--------------:|-----------------:|---------------:|-------------:|------------:|-----------:|----------------------:|-----------------------:|------------------------:|----------------------------:|
| Headline (PCA-2 + credit) |      426 |             12.94 |          9.00 |             0.68 |           0.54 |         0.71 |        1.33 |      -0.62 |                  1.87 |                   0.13 |                    0.53 |                        0.74 |
| PCA-3 sparse              |      255 |              7.75 |         10.00 |             0.83 |           0.58 |         1.74 |        2.08 |      -0.33 |                  1.19 |                   0.10 |                    0.53 |                        0.28 |
| PCA-2 full                |      344 |             10.45 |         10.00 |             0.67 |           0.54 |         0.69 |        1.39 |      -0.70 |                  2.02 |                   0.09 |                    0.51 |                        0.78 |
| Pairwise adaptive, neural |      252 |              7.66 |          7.00 |             0.69 |           0.54 |         0.48 |        1.10 |      -0.62 |                  2.30 |                   0.21 |                    0.53 |                        1.14 |

`cost_share_of_gross` is total cost over total gross: above 1 means costs took more than the signal earned. `resize_share_of_cost` is the part of cost paid while a trade is open, because the position is signal divided by trailing volatility and is re-sized every bar. `top_decile_share_of_gross` is the share of gross P&L made by the best tenth of trades.

## By holding period

| model                     | holding   |   trades |   mean_gross |   mean_cost |   mean_net |   hit_rate_net |   total_net |
|:--------------------------|:----------|---------:|-------------:|------------:|-----------:|---------------:|------------:|
| Headline (PCA-2 + credit) | 1-2 bars  |       56 |         0.23 |        0.81 |      -0.58 |           0.46 |      -32.45 |
| Headline (PCA-2 + credit) | 3-5       |       85 |         0.97 |        1.15 |      -0.18 |           0.60 |      -15.65 |
| Headline (PCA-2 + credit) | 6-10      |       97 |         0.86 |        1.34 |      -0.48 |           0.61 |      -46.43 |
| Headline (PCA-2 + credit) | 11-20     |      121 |         1.32 |        1.33 |      -0.01 |           0.64 |       -1.79 |
| Headline (PCA-2 + credit) | 21+       |       67 |        -0.51 |        1.99 |      -2.50 |           0.28 |     -167.21 |
| PCA-3 sparse              | 1-2 bars  |       17 |         2.76 |        2.01 |       0.75 |           0.76 |       12.70 |
| PCA-3 sparse              | 3-5       |       38 |         2.49 |        2.14 |       0.35 |           0.76 |       13.30 |
| PCA-3 sparse              | 6-10      |       73 |         2.75 |        1.99 |       0.76 |           0.75 |       55.60 |
| PCA-3 sparse              | 11-20     |       66 |         1.97 |        1.89 |       0.08 |           0.59 |        5.58 |
| PCA-3 sparse              | 21+       |       61 |        -0.45 |        2.37 |      -2.82 |           0.18 |     -171.99 |
| PCA-2 full                | 1-2 bars  |       23 |         0.24 |        1.03 |      -0.78 |           0.43 |      -18.02 |
| PCA-2 full                | 3-5       |       63 |         0.88 |        1.25 |      -0.37 |           0.56 |      -23.40 |
| PCA-2 full                | 6-10      |       90 |         0.85 |        1.35 |      -0.49 |           0.62 |      -44.44 |
| PCA-2 full                | 11-20     |      110 |         1.16 |        1.35 |      -0.19 |           0.63 |      -21.13 |
| PCA-2 full                | 21+       |       58 |        -0.49 |        1.84 |      -2.33 |           0.29 |     -135.31 |
| Pairwise adaptive, neural | 1-2 bars  |       63 |         0.07 |        0.78 |      -0.71 |           0.44 |      -44.42 |
| Pairwise adaptive, neural | 3-5       |       49 |         1.12 |        0.86 |       0.26 |           0.67 |       12.59 |
| Pairwise adaptive, neural | 6-10      |       39 |         0.93 |        1.13 |      -0.20 |           0.51 |       -7.89 |
| Pairwise adaptive, neural | 11-20     |       41 |         1.84 |        1.04 |       0.80 |           0.71 |       32.83 |
| Pairwise adaptive, neural | 21+       |       60 |        -0.83 |        1.66 |      -2.49 |           0.43 |     -149.68 |

### Trades held longer than 20 bars

Gross P&L of those trades up to bar 20 and after it. The `after` column is roughly what a time stop at 20 bars would have avoided (before its own exit cost); it is a pointer for a walk-forward test, not a result.

| model                     |   trades_held_past_stop |   share_of_trades |   gross_first_bars |   gross_after |   cost_total |   net_total |   net_all_other_trades |
|:--------------------------|------------------------:|------------------:|-------------------:|--------------:|-------------:|------------:|-----------------------:|
| Headline (PCA-2 + credit) |                      67 |              0.16 |             -65.31 |         31.21 |       133.11 |     -167.21 |                 -96.32 |
| PCA-3 sparse              |                      61 |              0.24 |             -74.79 |         47.20 |       144.40 |     -171.99 |                  87.18 |
| PCA-2 full                |                      58 |              0.17 |             -54.28 |         25.75 |       106.78 |     -135.31 |                -106.99 |
| Pairwise adaptive, neural |                      60 |              0.24 |             -60.85 |         10.80 |        99.63 |     -149.68 |                  -6.90 |

## By rates volatility at entry

MOVE index on the entry date, split into terciles over each model's own trades.

| model                     | move_at_entry   |   trades |   mean_gross |   mean_cost |   mean_net |   hit_rate_net |   total_net |
|:--------------------------|:----------------|---------:|-------------:|------------:|-----------:|---------------:|------------:|
| Headline (PCA-2 + credit) | low MOVE        |      142 |         0.95 |        1.71 |      -0.76 |           0.51 |     -108.50 |
| Headline (PCA-2 + credit) | mid MOVE        |      142 |         0.42 |        1.28 |      -0.86 |           0.53 |     -122.47 |
| Headline (PCA-2 + credit) | high MOVE       |      142 |         0.77 |        1.00 |      -0.23 |           0.60 |      -32.56 |
| PCA-3 sparse              | low MOVE        |       85 |         2.10 |        2.45 |      -0.35 |           0.51 |      -30.17 |
| PCA-3 sparse              | mid MOVE        |       85 |         1.62 |        2.22 |      -0.61 |           0.55 |      -51.57 |
| PCA-3 sparse              | high MOVE       |       85 |         1.52 |        1.56 |      -0.04 |           0.67 |       -3.07 |
| PCA-2 full                | low MOVE        |      115 |         0.92 |        1.75 |      -0.83 |           0.54 |      -95.19 |
| PCA-2 full                | mid MOVE        |      114 |         0.17 |        1.36 |      -1.19 |           0.48 |     -135.18 |
| PCA-2 full                | high MOVE       |      115 |         0.97 |        1.08 |      -0.10 |           0.61 |      -11.93 |
| Pairwise adaptive, neural | low MOVE        |       84 |         0.09 |        1.53 |      -1.44 |           0.37 |     -120.90 |
| Pairwise adaptive, neural | mid MOVE        |       84 |         0.70 |        0.99 |      -0.28 |           0.57 |      -23.78 |
| Pairwise adaptive, neural | high MOVE       |       84 |         0.65 |        0.79 |      -0.14 |           0.68 |      -11.90 |

## Long and short residuals

Long means the position profits if the residual (or spread) rises back toward its mean.

| model                     | side           |   trades |   mean_gross |   mean_cost |   mean_net |   hit_rate_net |   total_net |
|:--------------------------|:---------------|---------:|-------------:|------------:|-----------:|---------------:|------------:|
| Headline (PCA-2 + credit) | long residual  |      194 |         0.72 |        1.30 |      -0.58 |           0.57 |     -111.87 |
| Headline (PCA-2 + credit) | short residual |      232 |         0.71 |        1.36 |      -0.65 |           0.53 |     -151.66 |
| PCA-3 sparse              | long residual  |      121 |         1.84 |        2.04 |      -0.21 |           0.60 |      -25.13 |
| PCA-3 sparse              | short residual |      134 |         1.66 |        2.11 |      -0.45 |           0.55 |      -59.68 |
| PCA-2 full                | long residual  |      166 |         0.72 |        1.35 |      -0.63 |           0.55 |     -104.78 |
| PCA-2 full                | short residual |      178 |         0.66 |        1.44 |      -0.77 |           0.53 |     -137.53 |
| Pairwise adaptive, neural | long residual  |      102 |         0.01 |        0.97 |      -0.97 |           0.55 |      -98.60 |
| Pairwise adaptive, neural | short residual |      150 |         0.80 |        1.19 |      -0.39 |           0.53 |      -57.98 |

## By instrument

| model                     | instrument                                                  |   trades |   mean_gross |   mean_cost |   mean_net |   hit_rate_net |   total_net |
|:--------------------------|:------------------------------------------------------------|---------:|-------------:|------------:|-----------:|---------------:|------------:|
| Headline (PCA-2 + credit) | cdx_cash_basis                                              |       48 |         1.15 |        1.11 |       0.05 |           0.62 |        2.23 |
| Headline (PCA-2 + credit) | hy_ig_oas                                                   |       26 |         0.29 |        0.94 |      -0.65 |           0.42 |      -16.77 |
| Headline (PCA-2 + credit) | hyg_cdx                                                     |        8 |         0.54 |        1.38 |      -0.84 |           0.50 |       -6.68 |
| Headline (PCA-2 + credit) | pca2_ust_10y                                                |       76 |         0.95 |        1.57 |      -0.62 |           0.55 |      -47.50 |
| Headline (PCA-2 + credit) | pca2_ust_2y                                                 |       81 |         0.57 |        0.97 |      -0.40 |           0.62 |      -32.47 |
| Headline (PCA-2 + credit) | pca2_ust_30y                                                |       69 |         0.51 |        1.03 |      -0.52 |           0.52 |      -36.11 |
| Headline (PCA-2 + credit) | pca2_ust_3y                                                 |       55 |         1.55 |        2.23 |      -0.68 |           0.47 |      -37.38 |
| Headline (PCA-2 + credit) | pca2_ust_5y                                                 |       49 |         0.17 |        1.27 |      -1.10 |           0.53 |      -53.89 |
| Headline (PCA-2 + credit) | pca2_ust_7y                                                 |       14 |        -0.71 |        1.79 |      -2.50 |           0.50 |      -34.96 |
| PCA-3 sparse              | pca3s_ust_10y                                               |        8 |        -0.50 |        2.37 |      -2.87 |           0.12 |      -22.96 |
| PCA-3 sparse              | pca3s_ust_10y = pca3s_ust_30y                               |       17 |         1.90 |        2.93 |      -1.03 |           0.41 |      -17.49 |
| PCA-3 sparse              | pca3s_ust_2y                                                |        8 |        -0.75 |        1.95 |      -2.70 |           0.25 |      -21.59 |
| PCA-3 sparse              | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |       83 |         2.10 |        2.22 |      -0.12 |           0.59 |       -9.60 |
| PCA-3 sparse              | pca3s_ust_2y = pca3s_ust_5y = pca3s_ust_10y = pca3s_ust_30y |       84 |         1.99 |        1.46 |       0.52 |           0.77 |       44.02 |
| PCA-3 sparse              | pca3s_ust_30y                                               |        8 |        -0.50 |        2.35 |      -2.85 |           0.12 |      -22.80 |
| PCA-3 sparse              | pca3s_ust_3y                                                |        2 |        -1.01 |        1.18 |      -2.18 |           0.50 |       -4.37 |
| PCA-3 sparse              | pca3s_ust_3y = pca3s_ust_5y                                 |        1 |         0.67 |        1.12 |      -0.45 |           0.00 |       -0.45 |
| PCA-3 sparse              | pca3s_ust_5y                                                |        8 |        -0.92 |        1.75 |      -2.66 |           0.25 |      -21.31 |
| PCA-3 sparse              | pca3s_ust_7y                                                |       36 |         2.60 |        2.83 |      -0.23 |           0.53 |       -8.27 |

## The best and worst trades

| model                     | kind   | instrument                                                  |   copies |   side | entry      | exit       |   bars |   gross |   cost |    net |
|:--------------------------|:-------|:------------------------------------------------------------|---------:|-------:|:-----------|:-----------|-------:|--------:|-------:|-------:|
| Headline (PCA-2 + credit) | worst  | pca2_ust_5y                                                 |        1 |  -1.00 | 2021-02-16 | 2021-02-25 |   7.00 |  -12.49 |   1.87 | -14.37 |
| Headline (PCA-2 + credit) | worst  | pca2_ust_7y                                                 |        1 |  -1.00 | 2019-03-22 | 2019-05-30 |  49.00 |  -10.96 |   2.83 | -13.80 |
| Headline (PCA-2 + credit) | worst  | pca2_ust_7y                                                 |        1 |   1.00 | 2025-03-21 | 2025-05-22 |  44.00 |  -10.61 |   2.46 | -13.07 |
| Headline (PCA-2 + credit) | worst  | pca2_ust_10y                                                |        1 |   1.00 | 2026-07-01 | 2026-07-31 |  22.00 |   -6.94 |   4.00 | -10.94 |
| Headline (PCA-2 + credit) | worst  | pca2_ust_10y                                                |        1 |   1.00 | 2011-07-28 | 2011-08-04 |   5.00 |   -8.46 |   0.71 |  -9.16 |
| Headline (PCA-2 + credit) | worst  | pca2_ust_3y                                                 |        1 |  -1.00 | 2018-02-01 | 2018-02-09 |   6.00 |   -6.90 |   2.16 |  -9.06 |
| Headline (PCA-2 + credit) | worst  | hy_ig_oas                                                   |        1 |   1.00 | 2008-09-12 | 2008-09-15 |   1.00 |   -7.93 |   0.83 |  -8.77 |
| Headline (PCA-2 + credit) | worst  | pca2_ust_30y                                                |        1 |   1.00 | 2024-07-12 | 2024-08-02 |  15.00 |   -6.92 |   1.74 |  -8.66 |
| Headline (PCA-2 + credit) | best   | pca2_ust_30y                                                |        1 |   1.00 | 2000-07-31 | 2000-08-10 |   8.00 |    7.81 |   0.80 |   7.01 |
| Headline (PCA-2 + credit) | best   | pca2_ust_30y                                                |        1 |  -1.00 | 2008-03-07 | 2008-03-14 |   5.00 |    7.27 |   0.45 |   6.82 |
| Headline (PCA-2 + credit) | best   | pca2_ust_2y                                                 |        1 |  -1.00 | 2000-12-14 | 2001-01-03 |  14.00 |    7.00 |   0.46 |   6.54 |
| Headline (PCA-2 + credit) | best   | hy_ig_oas                                                   |        1 |  -1.00 | 2014-08-05 | 2014-09-02 |  20.00 |    8.09 |   1.55 |   6.54 |
| Headline (PCA-2 + credit) | best   | pca2_ust_3y                                                 |        1 |   1.00 | 2014-04-25 | 2014-05-07 |   8.00 |    6.81 |   0.58 |   6.22 |
| Headline (PCA-2 + credit) | best   | hy_ig_oas                                                   |        1 |  -1.00 | 2014-10-16 | 2014-10-28 |   8.00 |    6.44 |   1.00 |   5.45 |
| Headline (PCA-2 + credit) | best   | pca2_ust_30y                                                |        1 |   1.00 | 2003-04-30 | 2003-05-07 |   5.00 |    6.25 |   0.93 |   5.32 |
| Headline (PCA-2 + credit) | best   | pca2_ust_2y                                                 |        1 |   1.00 | 2012-08-15 | 2012-08-31 |  12.00 |    5.63 |   0.51 |   5.12 |
| PCA-3 sparse              | worst  | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |  -1.00 | 2021-01-11 | 2021-02-25 |  33.00 |   -7.86 |   3.75 | -11.61 |
| PCA-3 sparse              | worst  | pca3s_ust_10y = pca3s_ust_30y                               |        2 |   1.00 | 2011-06-06 | 2011-08-10 |  47.00 |   -8.78 |   1.32 | -10.10 |
| PCA-3 sparse              | worst  | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |   1.00 | 2013-04-15 | 2013-06-03 |  35.00 |   -8.02 |   1.79 |  -9.82 |
| PCA-3 sparse              | worst  | pca3s_ust_2y = pca3s_ust_5y = pca3s_ust_10y = pca3s_ust_30y |        4 |  -1.00 | 2007-11-19 | 2007-11-21 |   2.00 |   -7.25 |   1.70 |  -8.95 |
| PCA-3 sparse              | worst  | pca3s_ust_2y                                                |        1 |  -1.00 | 2007-06-22 | 2007-07-26 |  24.00 |   -5.18 |   3.47 |  -8.64 |
| PCA-3 sparse              | worst  | pca3s_ust_7y                                                |        1 |  -1.00 | 2020-02-28 | 2020-03-16 |  11.00 |   -5.92 |   2.48 |  -8.40 |
| PCA-3 sparse              | worst  | pca3s_ust_5y                                                |        1 |   1.00 | 2007-06-22 | 2007-07-26 |  24.00 |   -5.49 |   2.81 |  -8.30 |
| PCA-3 sparse              | worst  | pca3s_ust_30y                                               |        1 |   1.00 | 2007-06-22 | 2007-07-26 |  24.00 |   -5.49 |   2.81 |  -8.30 |
| PCA-3 sparse              | best   | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |   1.00 | 2013-08-22 | 2013-09-18 |  19.00 |    5.78 |   0.77 |   5.01 |
| PCA-3 sparse              | best   | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |   1.00 | 2024-06-03 | 2024-06-11 |   6.00 |    5.75 |   1.19 |   4.56 |
| PCA-3 sparse              | best   | pca3s_ust_2y = pca3s_ust_5y = pca3s_ust_10y = pca3s_ust_30y |        4 |  -1.00 | 2000-02-03 | 2000-02-04 |   1.00 |    5.10 |   0.69 |   4.41 |
| PCA-3 sparse              | best   | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |  -1.00 | 2014-06-26 | 2014-07-09 |   9.00 |    4.79 |   0.47 |   4.32 |
| PCA-3 sparse              | best   | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |  -1.00 | 2015-06-24 | 2015-07-08 |  10.00 |    5.94 |   1.84 |   4.10 |
| PCA-3 sparse              | best   | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |   1.00 | 2014-12-12 | 2014-12-23 |   7.00 |    4.64 |   0.57 |   4.07 |
| PCA-3 sparse              | best   | pca3s_ust_2y = pca3s_ust_3y = pca3s_ust_5y                  |        3 |  -1.00 | 2014-05-28 | 2014-06-11 |  10.00 |    4.98 |   1.04 |   3.94 |
| PCA-3 sparse              | best   | pca3s_ust_10y = pca3s_ust_30y                               |        2 |   1.00 | 2014-02-05 | 2014-02-13 |   6.00 |    5.49 |   1.66 |   3.83 |

## How many independent bets

Sparse hedges can give two tenors the same legs, so the same trade shows up under several names. Trades above are merged when entry, exit and P&L match (`copies`). `effective_bets` uses the correlation of the members' daily gross P&L: 1 if they all move as one, the member count if they are uncorrelated.

| model                     |   members |   identical_pairs |   effective_bets |   mean_pairwise_corr |
|:--------------------------|----------:|------------------:|-----------------:|---------------------:|
| Headline (PCA-2 + credit) |         9 |                 0 |             8.69 |                 0.03 |
| PCA-3 sparse              |         6 |                 4 |             2.28 |                 0.47 |
| PCA-2 full                |         6 |                 0 |             5.71 |                 0.08 |
| Pairwise adaptive, neural |        17 |                 0 |            14.37 |                 0.02 |

Full trade list: `tables/lessons_all_trades.csv`.
