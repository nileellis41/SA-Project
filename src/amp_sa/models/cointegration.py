"""Cointegration screens, run on a training window only."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from amp_sa import config


@dataclass(frozen=True)
class Screen:
    eg_pvalue: float
    johansen_trace: float
    johansen_crit90: float
    alpha: float
    beta: np.ndarray  # OLS hedge ratios of y on x

    @property
    def johansen_pass(self) -> bool:
        return self.johansen_trace > self.johansen_crit90

    @property
    def passed(self) -> bool:
        # Both tests must agree. Johansen alone rejects too easily with three series.
        return self.eg_pvalue < config.COINT_PVALUE and self.johansen_pass


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), coef[1:]


def screen(y: pd.Series, x: pd.DataFrame) -> Screen:
    """Engle-Granger and Johansen (trace, r=0) on one training window."""
    yv, xv = y.to_numpy(float), x.to_numpy(float).reshape(len(y), -1)
    ok = np.isfinite(yv) & np.isfinite(xv).all(axis=1)
    yv, xv = yv[ok], xv[ok]
    alpha, beta = ols(yv, xv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eg_p = float(coint(yv, xv, trend="c", autolag="AIC")[1])
    except Exception:
        eg_p = 1.0
    try:
        joh = coint_johansen(np.column_stack([yv, xv]), det_order=0, k_ar_diff=1)
        trace, crit = float(joh.lr1[0]), float(joh.cvt[0, 0])
    except Exception:
        trace, crit = 0.0, np.inf
    return Screen(eg_p, trace, crit, alpha, beta)
