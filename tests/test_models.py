"""Model correctness on data with a known answer, and the execution guardrails."""

import numpy as np
import pandas as pd
import pytest

from amp_sa.backtest import metrics
from amp_sa.execution import limits as lim
from amp_sa.execution import plan
from amp_sa.models import cointegration, kalman, ou
from amp_sa.models.trend import L1Filter


def test_l1_interior_point_matches_cvxpy():
    rng = np.random.default_rng(0)
    filt = L1Filter(120)
    for _ in range(3):
        y = np.cumsum(rng.normal(size=120))
        y = (y - y.mean()) / y.std()
        for frac in (0.02, 0.1, 0.5):
            lam = frac * filt.lambda_max(y)
            np.testing.assert_allclose(filt.fit(y, lam), filt.fit_cvxpy(y, lam), atol=2e-3)


def test_l1_recovers_a_kinked_line_and_lambda_max_gives_a_straight_line():
    t = np.arange(120.0)
    truth = np.where(t < 60, 0.1 * t, 6 - 0.2 * (t - 60))
    y = truth + np.random.default_rng(1).normal(0, 0.05, 120)
    filt = L1Filter(120)
    x = filt.fit(y, 2.0)
    assert x[-1] - x[-2] == pytest.approx(-0.2, abs=0.02)
    line = filt.fit(y, 1.01 * filt.lambda_max(y))
    assert np.abs(np.diff(line, 2)).max() < 1e-3


def test_kalman_tracks_a_drifting_hedge_ratio():
    rng = np.random.default_rng(2)
    n = 3000
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    beta = np.linspace(0.9, 1.1, n)
    y = 10 + beta * x + rng.normal(0, 0.5, n)
    fast, slow = kalman.run(y, x, 1e-2), kalman.run(y, x, 1e-7)
    error = lambda p: np.abs(p.prior[1000:, 1] - beta[1000:]).mean()  # noqa: E731
    assert error(fast) < 0.03
    assert error(fast) < error(slow)  # a near-static filter cannot follow the drift


def test_ou_fit_recovers_half_life_and_mean():
    rng = np.random.default_rng(3)
    b, s = 0.5 ** (1 / 10), np.zeros(20000)
    for t in range(1, len(s)):
        s[t] = 3 * (1 - b) + b * s[t - 1] + rng.normal()
    fit = ou.fit_ou(s)
    assert fit.half_life == pytest.approx(10, rel=0.15)
    assert fit.mu == pytest.approx(3, abs=0.5)
    assert not ou.fit_ou(np.cumsum(rng.normal(size=500)) * 0 + np.arange(500.0)).mean_reverting


def test_band_rule_enters_exits_and_stops():
    z = np.array([0, 2.1, 1.0, 0.4, 0, -2.1, -0.4, 5.0, 3.0, 2.2, 1.0, 2.1])
    pos = ou.band_positions(z, entry=2.0, exit_=0.5, stop=4.0)
    assert pos.tolist() == [0, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1]


def test_cointegration_screen_separates_cointegrated_from_independent():
    rng = np.random.default_rng(4)
    x = pd.DataFrame({"x": np.cumsum(rng.normal(size=1500))})
    noise = np.zeros(1500)
    for t in range(1, 1500):
        noise[t] = 0.8 * noise[t - 1] + rng.normal()
    good = cointegration.screen(pd.Series(2 * x["x"].to_numpy() + noise), x)
    bad = cointegration.screen(pd.Series(np.cumsum(rng.normal(size=1500))), x)
    assert good.passed and good.beta[0] == pytest.approx(2, abs=0.1)
    assert not bad.passed


def test_deflated_sharpe_falls_with_more_and_wider_trials():
    pnl = pd.Series(np.random.default_rng(5).normal(0.05, 1, 2500))
    one = metrics.deflated_sharpe(pnl, [metrics.sharpe(pnl)])
    many = metrics.deflated_sharpe(pnl, [0.8, 0.2, -0.3, 0.5, 0.0, 1.1, -0.6, 0.4])
    assert 0 <= many < one <= 1


def test_max_drawdown_is_peak_to_trough():
    assert metrics.max_drawdown(pd.Series([0.1, -0.05, -0.1, 0.2, -0.01])) == pytest.approx(-0.15)


# ---------------------------------------------------------------- AMP guardrails

SPECS = pd.DataFrame({"margin_initial": [90.0, 240.0]}, index=["Z10YX26", "MCLEZ26"])
VOL = pd.Series({"Z10YX26": 60.0, "MCLEZ26": 150.0})
NONE = pd.Series(dtype=float)


def test_small_book_passes_limits():
    assert lim.check(pd.Series({"Z10YX26": 1.0}), NONE, SPECS, VOL, equity=5000) == []


def test_margin_stress_size_and_loss_limits_trip():
    msgs = lim.check(pd.Series({"MCLEZ26": 4.0}), NONE, SPECS, VOL, equity=3000)
    joined = " ".join(msgs)
    assert "margin" in joined and "stress" in joined and "more than" in joined
    stopped = lim.check(pd.Series({"Z10YX26": 1.0}), NONE, SPECS, VOL, equity=5000, day_pnl=-600)
    assert any("no opening orders" in m for m in stopped)
    # Closing out is always allowed after a loss.
    assert lim.check(NONE, pd.Series({"Z10YX26": 1.0}), SPECS, VOL, equity=5000, day_pnl=-600) == []


def test_daily_loss_limit_sits_far_inside_amp_liquidation():
    assert lim.Limits().max_daily_loss_frac <= lim.AMP_DAILY_LOSS_LIQUIDATION / 4


def test_unknown_symbol_is_refused():
    assert lim.check(pd.Series({"NOPE": 1.0}), NONE, SPECS, VOL, equity=5000)


def test_contract_mapping_rounds_and_skips_untradeable_legs():
    specs = pd.DataFrame(
        {"expiration": pd.to_datetime(["2026-10-13", "2026-11-25", "2026-11-20", "2026-12-20"])},
        index=["Z10YV26", "Z10YX26", "MCLEX26", "MCLEZ26"],
    )
    legs = pd.Series({"ust_10y": 0.004, "cl1": -0.0002, "cl2": 0.0002, "hy_oas": 0.01})
    table = plan.target_contracts(legs, specs, capital=5000, asof=pd.Timestamp("2026-10-12")).set_index("series")
    assert table.loc["ust_10y", "symbol"] == "Z10YX26"  # the contract expiring tomorrow is skipped
    assert table.loc["ust_10y", "contracts"] == 2
    assert table.loc["cl1", "symbol"] == "MCLEX26" and table.loc["cl2", "symbol"] == "MCLEZ26"
    assert table.loc["cl1", "contracts"] == 0
    assert pd.isna(table.loc["hy_oas", "symbol"]) and table.loc["hy_oas", "contracts"] == 0
