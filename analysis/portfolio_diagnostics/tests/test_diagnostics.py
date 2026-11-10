"""Computations in the diagnostics that could be silently wrong: factor exposure, OU fit, sign conventions, costs."""

import numpy as np
import pandas as pd

from analysis.portfolio_diagnostics import book as bk
from analysis.portfolio_diagnostics.hedge import PCS, orient, trade_ids
from analysis.portfolio_diagnostics.signals_regimes import ar1

TENORS = list(bk.TREASURY)


def _loadings(seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(6, 3)))
    return pd.DataFrame(q, index=TENORS, columns=list(PCS))


def test_full_projection_has_zero_exposure_to_hedged_factors():
    L = _loadings()
    L2 = L[["pc1", "pc2"]].to_numpy()
    project = np.eye(6) - L2 @ L2.T
    expo = project @ L.to_numpy()  # each residual's exposure to pc1..pc3
    assert np.allclose(expo[:, :2], 0.0, atol=1e-12)
    assert not np.allclose(expo[:, 2], 0.0)  # PCA-2 leaves curvature


def test_factor_leak_plus_residual_equals_member_change():
    L = _loadings(1).to_numpy()
    rng = np.random.default_rng(2)
    w, dy = rng.normal(size=6), rng.normal(size=(50, 6))
    f = dy @ L
    leak = (w @ L) * f
    resid = dy @ w - leak.sum(axis=1)
    assert np.allclose(leak.sum(axis=1) + resid, dy @ w)
    assert np.allclose(resid, (dy - f @ L.T) @ w)  # residual is w . (dy - L f)


def test_orient_fixes_signs():
    L = orient(-_loadings(3))
    assert L["pc1"].sum() > 0
    assert L["pc2"].iloc[-1] > L["pc2"].iloc[0]
    assert L["pc3"].iloc[0] + L["pc3"].iloc[-1] - 2 * L.loc["ust_5y", "pc3"] > 0


def test_ar1_recovers_known_parameters():
    rng = np.random.default_rng(4)
    b, sigma, n = 0.95, 1.0, 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 + b * x[t - 1] + rng.normal(0, sigma)
    fit = ar1(x)
    assert abs(fit["b"] - b) < 0.01
    assert abs(fit["half_life"] - np.log(2) / -np.log(b)) < 2.0
    assert abs(fit["sigma_eq"] - sigma / np.sqrt(1 - b**2)) < 0.3


def test_ar1_flags_random_walk_and_explosive():
    rng = np.random.default_rng(5)
    walk = np.cumsum(rng.normal(size=5000)) + np.arange(5000) * 0.01
    assert np.isnan(ar1(np.exp(np.arange(100) / 20.0))["half_life"])  # b > 1
    assert ar1(walk)["b"] > 0.99


def test_time_varying_cost_table():
    idx = pd.to_datetime(["2007-12-31", "2008-01-02"])
    c = bk.time_varying_cost(["ust_2y", "ust_30y", "hy_oas"], idx)
    assert list(c["ust_2y"]) == [0.5, 0.25]
    assert list(c["ust_30y"]) == [0.75, 0.375]
    assert list(c["hy_oas"]) == [2.0, 2.0]


def test_trade_ids_match_run_lessons_convention():
    idx = pd.bdate_range("2020-01-01", periods=8)
    pos = pd.Series([0, 1, 1.2, -1, -1, 0, 0, 0.5], index=idx)
    tid, held = trade_ids(pos)
    assert list(tid.fillna(0)) == [0, 1, 1, 2, 2, 0, 0, 3]  # a flip starts a new trade
    assert list(held.fillna(0)) == [0, 0, 1, 1, 2, 2, 0, 0]


def test_window_fit_matches_rolling_ou_and_its_screen():
    from amp_sa.sleeves import pca
    from analysis.portfolio_diagnostics.signals_regimes import simulate_window_fits
    rng = np.random.default_rng(6)
    bh, hl, passes = simulate_window_fits(0.9, 60, 50, rng)
    rng = np.random.default_rng(6)  # same draws, run through the production fit
    x0 = rng.normal(0.0, 1.0 / np.sqrt(1 - 0.81), 50)
    e = rng.normal(size=(50, 60))
    for i in range(50):
        path = [x0[i]]
        for t in range(60):
            path.append(0.9 * path[-1] + e[i, t])
        _, h, _ = pca.rolling_ou(pd.DataFrame({"c": path}), 60)
        h = h["c"].iloc[-1]
        assert (np.isfinite(h) == passes[i]) and (not passes[i] or np.isclose(h, hl[i]))


def test_short_window_ar1_is_biased_fast_as_kendall_says():
    from analysis.portfolio_diagnostics.signals_regimes import half_life, kendall_b, simulate_window_fits
    bh, _, _ = simulate_window_fits(0.9, 60, 20000, np.random.default_rng(7))
    assert abs(bh.mean() - kendall_b(0.9, 60)) < 0.015
    assert half_life(bh.mean()) < 0.6 * half_life(0.9)
    assert np.isnan(half_life(1.0)) and np.isnan(half_life(-0.2))


def test_realised_moves_signed_by_side_and_split_at_flips():
    from analysis.portfolio_diagnostics.signals_regimes import realised_moves
    idx = pd.bdate_range("2020-01-01", periods=7)
    pos = pd.Series([0, 1, 1, -2, -2, 0, 0], index=idx)
    change = pd.Series([9, 9, -1, -2, 3, -4, 9], index=idx, dtype=float)
    r = realised_moves(pos, change)
    # long from bar 1: held on bars 2-3 -> -1 - 2; short from bar 3: held on bars 4-5 -> -(3 - 4)
    assert list(r.index) == [idx[1], idx[3]]
    assert list(r) == [-3.0, 1.0]


def test_era_traded_labels():
    from analysis.portfolio_diagnostics.signals_regimes import era_traded
    idx = pd.to_datetime(["2005-01-03", "2010-05-21", "2014-04-01", "2014-04-02"])
    lab = era_traded(idx, pd.Timestamp("2010-05-21"), pd.Timestamp("2014-04-02"))
    assert list(lab) == ["to 2010-05: 2y 5y 10y 30y", "2010-05 to 2014-04: + 3y", "2010-05 to 2014-04: + 3y", "2014-04 on: all six"]
