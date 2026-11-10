"""Universe, costs and parameters shared by every sleeve.

One bar calendar (the Treasury calendar of the Bloomberg panel), one roll rule
(data/panel.py), one cost model (COST_PER_UNIT), one walk-forward schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

ANN = 252

# Walk-forward schedule, in bars.
TRAIN_BARS = 5 * ANN
TEST_BARS = ANN
PURGE_BARS = 10  # longest label horizon (the neural regime label looks 10 bars ahead)
EMBARGO_BARS = 5

VOL_SPAN = 60  # EW span for the risk estimate of every position


@dataclass(frozen=True)
class Spread:
    name: str
    group: str
    y: str
    x: tuple[str, ...]


SPREADS: tuple[Spread, ...] = (
    Spread("ust_2s10s", "rates", "ust_10y", ("ust_2y",)),
    Spread("ust_5s30s", "rates", "ust_30y", ("ust_5y",)),
    Spread("ust_fly_2s5s10s", "rates", "ust_5y", ("ust_2y", "ust_10y")),
    Spread("ust_fly_5s10s30s", "rates", "ust_10y", ("ust_5y", "ust_30y")),
    Spread("hy_ig_oas", "credit", "hy_oas", ("ig_oas",)),
    # ---- breadth expansion, listed before any of it was run
    Spread("ust_2s5s", "rates", "ust_5y", ("ust_2y",)),
    Spread("ust_5s10s", "rates", "ust_10y", ("ust_5y",)),
    Spread("ust_10s30s", "rates", "ust_30y", ("ust_10y",)),
    Spread("ust_fly_2s3s5s", "rates", "ust_3y", ("ust_2y", "ust_5y")),
    Spread("ust_fly_3s5s7s", "rates", "ust_5y", ("ust_3y", "ust_7y")),
    Spread("ust_fly_5s7s10s", "rates", "ust_7y", ("ust_5y", "ust_10y")),
    Spread("ust_fly_7s10s30s", "rates", "ust_10y", ("ust_7y", "ust_30y")),
    Spread("swap_spread_2y", "rates", "sofr_swap_2y", ("ust_2y",)),
    Spread("swap_spread_5y", "rates", "sofr_swap_5y", ("ust_5y",)),
    Spread("swap_spread_10y", "rates", "sofr_swap_10y", ("ust_10y",)),
    Spread("swap_fly_2s5s10s", "rates", "sofr_swap_5y", ("sofr_swap_2y", "sofr_swap_10y")),
    Spread("cdx_cash_basis", "credit", "cdx_hy_spread", ("hy_oas",)),
    Spread("hyg_cdx", "credit", "hyg", ("cdx_hy_price",)),
    Spread("brent_calendar", "energy", "co1", ("co2",)),
    Spread("cl_calendar_2_3", "energy", "cl2", ("cl3",)),
    Spread("gasoline_heating_oil", "energy", "xb1", ("ho1",)),
    Spread("zc_ke", "grains", "zc1", ("ke1",)),
    # ---- cross-asset and FX, from the ES / FX quote file; listed before running
    Spread("usdcad_crude", "cross", "usdcad", ("cl1",)),
    Spread("usdjpy_rates", "cross", "usdjpy", ("ust_10y",)),
    Spread("es_credit", "cross", "es1", ("hy_oas",)),
    Spread("fx_eur_jpy_cad", "fx", "eurusd", ("usdjpy", "usdcad")),
    Spread("cl_calendar", "energy", "cl1", ("cl2",)),
    Spread("wti_brent", "energy", "cl1", ("co1",)),
    Spread("crack_321", "energy", "products_321", ("cl1",)),
    Spread("zw_zc", "grains", "zw1", ("zc1",)),
    Spread("zw_ke", "grains", "zw1", ("ke1",)),
)

# Sleeve 2 trades the same series as outrights or levels.
TREND_SERIES: dict[str, str] = {
    "ust_2y": "rates", "ust_5y": "rates", "ust_10y": "rates", "ust_30y": "rates",
    "hy_oas": "credit", "ig_oas": "credit",
    "ust_3y": "rates", "ust_7y": "rates", "cdx_hy_spread": "credit",
    "cl1": "energy", "co1": "energy", "xb1": "energy", "ho1": "energy",
    "zw1": "grains", "zc1": "grains", "ke1": "grains",
    "es1": "equity", "eurusd": "fx", "usdjpy": "fx", "usdcad": "fx",
}

# The class deliverable is graded on the rates and credit book; energy and
# grains are the step-6 extension run through the same code.
CORE_GROUPS = ("rates", "credit")

# One-way cost per unit traded, in the units of the series: half the bid/ask
# plus commission. Rates and OAS are in bp, crude in $/bbl, grains in cents/bu.
# Futures half-spreads are one tick, cross-checked against the AMP quote
# snapshot (data/raw/amp/specs.parquet). This is the 1x case; reports also run 2x and 3x.
COST_PER_UNIT: dict[str, float] = {
    "ust_2y": 0.25, "ust_5y": 0.25, "ust_10y": 0.25, "ust_30y": 0.25,
    "hy_oas": 2.0, "ig_oas": 0.5,
    "ust_3y": 0.25, "ust_7y": 0.25,
    "sofr_swap_2y": 0.25, "sofr_swap_5y": 0.25, "sofr_swap_10y": 0.25,
    "cdx_hy_spread": 1.0, "cdx_hy_price": 0.03,
    "hyg": 0.01,  # median quoted half-spread in the Bloomberg bid/ask history is $0.005
    "cl1": 0.012, "cl2": 0.012, "cl3": 0.012, "co1": 0.012, "co2": 0.012,
    "xb1": 0.03, "ho1": 0.03,  # cents per gallon
    "products_321": 0.035,  # 14 x (2 XB + HO) at one 0.0001 $/gal tick per leg, plus commission
    "zw1": 0.35, "zc1": 0.35, "ke1": 0.35,
    # Median quoted half-spread over the last five years of the daily bid/ask export
    # (indicative composite quotes, so conservative for FX). ES adds commission to one tick.
    "es1": 0.375, "eurusd": 0.0002, "usdjpy": 0.035, "usdcad": 0.00025,
}
# Extra cost per unit held through a roll day (close one contract, open the next).
ROLL_COST_MULT = 2.0

# Sleeve 1
COINT_PVALUE = 0.10
KALMAN_DELTAS = (1e-5, 1e-4, 1e-3)
ENTRY_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
EXIT_GRID = (0.0, 0.25, 0.5, 0.75)
STOP_Z = 4.0
HEALTH_WINDOW = ANN
MAX_HALF_LIFE = 60.0
MIN_HALF_LIFE = 1.0
NN_LABEL_HORIZON = 10
NN_GATE = 0.5

# Sleeve 2
MA_FAST, MA_SLOW = 20, 100
L1_WINDOW = 250
L1_STEP = 5
L1_LAMBDA_FRACS = (0.02, 0.1, 0.5)

# Portfolio
SLEEVE_RISK_BUDGET = {"rv": 0.5, "trend": 0.5}
TARGET_VOL = 0.10
MAX_LEVERAGE_SCALE = 3.0
DD_STEPS = ((0.10, 0.5), (0.15, 0.25))  # drawdown threshold -> exposure multiplier

REGIME_YEARS = (2008, 2020, 2022)
