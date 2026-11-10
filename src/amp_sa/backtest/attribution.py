"""Alpha and beta: how much of a strategy's return is explained by plain market exposure.

Each factor is the daily P&L of a simple long position, scaled to unit
volatility over the regression sample so betas are comparable:

  duration   long 10-year Treasury      (minus the change in the 10y yield)
  steepener  long the 2s10s curve       (change in 10y minus 2y)
  credit     long high yield            (minus the change in HY OAS)
  crude      long front-month WTI       (roll-free change in CL1)
  equity     long S&P 500 futures       (roll-free change in ES1)

These are price moves only. Carry and roll-down are not in the factors, so a
strategy that merely collects carry would show up here as alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from amp_sa.config import ANN
from amp_sa.data.panel import changes

HAC_LAGS = 10
SPECS = {
    "rates": ("duration", "steepener"),
    "macro": ("duration", "steepener", "credit", "crude", "equity"),
}


def factor_returns(levels: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
    d = changes(levels, rolls)
    out = pd.DataFrame(index=levels.index)
    out["duration"] = -d["ust_10y"]
    out["steepener"] = d["ust_10y"] - d["ust_2y"]
    if "hy_oas" in d:
        out["credit"] = -d["hy_oas"]
    if "cl1" in d:
        out["crude"] = d["cl1"]
    if "es1" in d:
        out["equity"] = d["es1"]
    return out


def regress(pnl: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of daily P&L on unit-volatility factors, Newey-West standard errors."""
    data = pd.concat([pnl.rename("y"), factors], axis=1).dropna()
    if len(data) < 250:
        return {}
    x = data[factors.columns] / data[factors.columns].std()
    fit = sm.OLS(data["y"], sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    resid_vol = fit.resid.std() * np.sqrt(ANN)
    alpha = fit.params["const"] * ANN
    out = {
        "start": data.index[0].date(), "bars": len(data),
        "ann_return": data["y"].mean() * ANN,
        "alpha": alpha, "alpha_t": fit.tvalues["const"],
        "sharpe": data["y"].mean() / data["y"].std() * np.sqrt(ANN),
        "appraisal_ratio": alpha / resid_vol if resid_vol > 0 else np.nan,
        "r_squared": fit.rsquared,
    }
    for name in factors.columns:
        # Beta per one-sigma factor day, annualised: the return a year of that exposure is worth
        # per unit of factor Sharpe. beta_t is the test that the exposure is non-zero.
        out[f"beta_{name}"] = fit.params[name] * np.sqrt(ANN)
        out[f"t_{name}"] = fit.tvalues[name]
    return out


def attribution_table(series: dict[str, pd.Series], levels: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
    factors = factor_returns(levels, rolls)
    rows = []
    for spec, names in SPECS.items():
        names = [n for n in names if n in factors]
        for label, pnl in series.items():
            result = regress(pnl, factors[names])
            if result:
                rows.append({"factors": spec, "series": label, **result})
    return pd.DataFrame(rows)
