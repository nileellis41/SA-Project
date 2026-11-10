"""Leakage suite. Must pass before any result is reported.

The core check is a perturbation test: rewrite every observation from bar CUT
onward and re-run. Anything decided before CUT (signals, positions, hedge
ratios, regime probabilities, portfolio weights) must come out identical. A
model fitted on the full sample, a smoother, a centred window or a scaler
fitted across the split would all fail it.
"""

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from amp_sa import config
from amp_sa.backtest.engine import account, trailing_vol
from amp_sa.backtest.splits import folds_for, walk_forward
from amp_sa.config import Spread
from amp_sa.data.panel import detect_rolls
from amp_sa.features import regime as rf
from amp_sa.models import kalman
from amp_sa.models.regime import HMMRegime
from amp_sa.models.trend import l1_slopes
from amp_sa.portfolio.combine import combine, drawdown_multiplier
from amp_sa.sleeves import rv, trend
from conftest import CUT, perturb

SPREAD = Spread("test_2s10s", "rates", "ust_10y", ("ust_2y",))


def assert_same_before_cut(a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame, cut: int = CUT):
    left, right = np.asarray(a.iloc[:cut], dtype=float), np.asarray(b.iloc[:cut], dtype=float)
    np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True)


def assert_differs_after_cut(a, b, cut: int = CUT):
    left, right = np.asarray(a.iloc[cut:], dtype=float), np.asarray(b.iloc[cut:], dtype=float)
    assert not np.allclose(np.nan_to_num(left), np.nan_to_num(right)), "perturbation had no effect; test is vacuous"


# ---------------------------------------------------------------- splits

def test_folds_train_strictly_before_test_with_gap():
    folds = walk_forward(3000)
    assert folds
    gap = config.PURGE_BARS + config.EMBARGO_BARS
    for f in folds:
        assert f.train_end - f.train_start == config.TRAIN_BARS
        assert f.test_start - f.train_end == gap
        assert f.train_start >= 0 and f.train_end <= f.test_start < f.test_end


def test_test_blocks_tile_without_overlap():
    folds = walk_forward(3000)
    covered = np.concatenate([np.arange(f.test_start, f.test_end) for f in folds])
    assert len(covered) == len(set(covered))
    assert covered[0] == config.TRAIN_BARS + config.PURGE_BARS + config.EMBARGO_BARS and covered[-1] == 2999
    assert (np.diff(covered) == 1).all()


def test_purge_covers_the_longest_label_horizon():
    assert config.PURGE_BARS >= config.NN_LABEL_HORIZON


def test_folds_start_at_first_valid_bar():
    s = pd.Series(np.r_[np.full(50, np.nan), np.arange(2000.0)])
    assert folds_for(s)[0].train_start == 50


# ---------------------------------------------------------------- building blocks

def test_trailing_vol_is_causal(panel):
    levels, _ = panel
    a, b = levels["ust_10y"].diff(), perturb(levels)["ust_10y"].diff()
    assert_same_before_cut(trailing_vol(a), trailing_vol(b))


def test_roll_flag_uses_no_future_open_interest():
    oi = pd.Series([100.0, 90, 80, 10, 400, 390, 380, 5, 300])
    flags = detect_rolls(oi)
    assert list(flags[flags].index) == [4, 8]
    changed = oi.copy()
    changed.iloc[6:] = [1, 1000, 1]
    assert detect_rolls(changed).iloc[:6].equals(flags.iloc[:6])


def test_kalman_prior_is_causal(panel):
    levels, _ = panel
    other = perturb(levels)
    a = kalman.run(levels["ust_10y"].to_numpy(), levels[["ust_2y"]].to_numpy(), 1e-4)
    b = kalman.run(other["ust_10y"].to_numpy(), other[["ust_2y"]].to_numpy(), 1e-4)
    np.testing.assert_allclose(a.prior[: CUT + 1], b.prior[: CUT + 1], equal_nan=True)  # prior[CUT] is known at CUT-1
    np.testing.assert_allclose(a.innovation[:CUT], b.innovation[:CUT], equal_nan=True)
    assert not np.allclose(a.prior[CUT + 5 :], b.prior[CUT + 5 :])


def test_l1_slopes_are_causal(panel):
    levels, _ = panel
    a = l1_slopes(levels["ust_10y"], window=60, step=5)
    b = l1_slopes(perturb(levels)["ust_10y"], window=60, step=5)
    assert a.iloc[:CUT].notna().any().all()
    assert_same_before_cut(a, b)
    assert_differs_after_cut(a, b)


def test_hmm_filter_is_causal_and_the_smoother_is_not(panel):
    """The reason the HMM is run as a forward filter instead of hmmlearn's predict_proba."""
    rng = np.random.default_rng(7)
    state = (np.sin(np.arange(760) / 25) > 0).astype(float)  # persistent regimes that overlap heavily
    x = np.column_stack([state + rng.normal(0, 1, 760), -state + rng.normal(0, 1, 760)])
    model = HMMRegime.fit(x[:300])
    full, prefix = model.filter(x), model.filter(x[:400])
    np.testing.assert_allclose(full[:400], prefix)

    raw = GaussianHMM(2, covariance_type="diag", n_iter=50, random_state=0).fit(model.scaler.transform(x[:300]))
    z = model.scaler.transform(x)
    smoothed_full, smoothed_prefix = raw.predict_proba(z), raw.predict_proba(z[:400])
    assert np.abs(smoothed_full[:400] - smoothed_prefix).max() > 1e-6, "smoother should depend on later bars"


def test_reversion_label_is_blank_where_its_horizon_is_unknown():
    s = pd.Series(np.sin(np.arange(100) / 5.0))
    label = rf.reversion_label(s, s * 3, horizon=10)
    assert label.iloc[-10:].isna().all()
    # A label inside the window never reads past the window it was given.
    longer = pd.Series(np.r_[s.to_numpy(), np.full(50, 1e6)])
    relabel = rf.reversion_label(longer, longer * 3, horizon=10)
    pd.testing.assert_series_equal(label.iloc[:90], relabel.iloc[:90])


def test_market_features_are_causal(panel):
    levels, _ = panel
    other = perturb(levels)
    a = rf.market_features(levels, levels["ust_10y"].diff())
    b = rf.market_features(other, other["ust_10y"].diff())
    assert_same_before_cut(a, b)


# ---------------------------------------------------------------- sleeves end to end

@pytest.mark.parametrize("variant", rv.VARIANTS)
def test_rv_sleeve_ignores_the_future(panel, small_schedule, variant):
    levels, rolls = panel
    a = rv.run_spread(SPREAD, levels, rolls)
    b = rv.run_spread(SPREAD, perturb(levels), rolls)
    assert a.live.iloc[:CUT].any(), "no out-of-sample bars before the cut"
    for field in ("signal", "position", "gross", "cost"):
        assert_same_before_cut(getattr(a.results[variant], field), getattr(b.results[variant], field))
    assert_same_before_cut(a.extras["legs_adaptive"], b.extras["legs_adaptive"])
    assert_same_before_cut(a.extras["spread_disabled"], b.extras["spread_disabled"])
    assert_differs_after_cut(a.results[variant].gross + a.results["static"].position,
                             b.results[variant].gross + b.results["static"].position)


def test_rv_fold_parameters_come_from_the_training_window(panel, small_schedule):
    """Folds whose training window ends before the cut report identical fitted parameters."""
    levels, rolls = panel
    a = rv.run_spread(SPREAD, levels, rolls).diagnostics
    b = rv.run_spread(SPREAD, perturb(levels), rolls).diagnostics
    cut_date = levels.index[CUT].date()
    early = a["train_end"] < cut_date
    assert early.any() and (~early).any()
    pd.testing.assert_frame_equal(a[early].drop(columns="disabled_share"), b[early].drop(columns="disabled_share"))


@pytest.mark.parametrize("variant", trend.VARIANTS)
def test_trend_sleeve_ignores_the_future(panel, small_schedule, variant):
    levels, rolls = panel
    other = perturb(levels)
    runs = []
    for lv in (levels, other):
        slopes = trend.compute_slopes(lv, rolls, series=("ust_10y",), n_jobs=1, cache=None)
        runs.append(trend.run_series("ust_10y", lv, rolls, slopes["ust_10y"]))
    a, b = runs
    assert a.live.iloc[:CUT].any()
    for field in ("signal", "position", "gross", "cost"):
        assert_same_before_cut(getattr(a.results[variant], field), getattr(b.results[variant], field))
    assert_differs_after_cut(a.results[variant].signal, b.results[variant].signal)


# ---------------------------------------------------------------- accounting and portfolio

def test_position_earns_the_next_bar_not_the_current_one():
    idx = pd.RangeIndex(6)
    change = pd.Series([0.0, 0.0, 0.0, 5.0, 0.0, 0.0], index=idx)
    one = pd.Series(1.0, index=idx)
    same_bar = account(pd.Series([0, 0, 0, 1, 0, 0.0], index=idx), change, 0.0, vol=one)
    prior_bar = account(pd.Series([0, 0, 1, 0, 0, 0.0], index=idx), change, 0.0, vol=one)
    assert same_bar.gross.sum() == 0.0  # a signal formed on the move's own bar cannot capture it
    assert prior_bar.gross.sum() == 5.0


def test_roll_bars_cost_extra_and_earn_nothing_on_their_own():
    idx = pd.RangeIndex(4)
    result = account(pd.Series(1.0, index=idx), pd.Series(0.0, index=idx), 0.1,
                     is_roll=pd.Series([False, False, True, False], index=idx), vol=pd.Series(1.0, index=idx))
    assert result.cost.tolist() == pytest.approx([0.1, 0.0, config.ROLL_COST_MULT * 0.1, 0.0])


def test_portfolio_weights_are_lagged():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=600)
    make = lambda: pd.Series(rng.normal(0.02, 1, 600), index=idx)  # noqa: E731
    zero = pd.Series(0.0, index=idx)
    g1, g2 = make(), make()
    a = combine({"rv": (g1, zero), "trend": (g2, zero)})
    h1, h2 = g1.copy(), g2.copy()
    h1.iloc[400:] = rng.normal(0, 25, 200)
    h2.iloc[400:] = rng.normal(0, 25, 200)
    b = combine({"rv": (h1, zero), "trend": (h2, zero)})
    # Weights for bar 400 are set from P&L through bar 399.
    np.testing.assert_allclose(a.weights.iloc[:401], b.weights.iloc[:401], equal_nan=True)
    assert not np.allclose(a.weights.iloc[420:].fillna(0), b.weights.iloc[420:].fillna(0))


def test_drawdown_overlay_reacts_one_bar_late():
    pnl = pd.Series([0.0, -0.12, 0.0, 0.0, 0.2, 0.0])
    mult = drawdown_multiplier(pnl, steps=((0.10, 0.5),))
    assert mult.tolist() == [1.0, 1.0, 0.5, 0.5, 0.5, 1.0]


# ---------------------------------------------------------------- regime-switching allocation

def _regime_world(seed: int = 3, n: int = 1500):
    """Two persistent regimes; sleeve `a` earns in the calm one, sleeve `b` in the stressed one."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    stressed = (np.sin(np.arange(n) / 60) > 0.3).astype(float)
    features = pd.DataFrame({"log_vix": 2.7 + 0.8 * stressed + rng.normal(0, 0.1, n),
                             "log_move": 4.3 + 0.6 * stressed + rng.normal(0, 0.1, n)}, index=idx)
    pnl = pd.DataFrame({"a": rng.normal(0.15 * (1 - stressed) - 0.05 * stressed, 1.0),
                        "b": rng.normal(0.15 * stressed - 0.05 * (1 - stressed), 1.0)}, index=idx)
    return pnl, features, pd.Series(stressed, index=idx)


def test_regime_budget_ignores_the_future():
    from amp_sa.portfolio.regime_switch import regime_budget

    pnl, features, _ = _regime_world()
    cut = 1200
    rng = np.random.default_rng(11)
    pnl2, features2 = pnl.copy(), features.copy()
    pnl2.iloc[cut:] = rng.normal(0, 30, (len(pnl) - cut, 2))
    features2.iloc[cut:] = features2.iloc[cut:].to_numpy()[::-1] + rng.normal(0, 1, (len(pnl) - cut, 2))
    a, _ = regime_budget(pnl, features, min_train=400, test_bars=200)
    b, _ = regime_budget(pnl2, features2, min_train=400, test_bars=200)
    assert a.iloc[:cut].notna().any().all()
    # Row `cut` sizes bar `cut` from the regime probability at cut-1, so it must also be unchanged.
    np.testing.assert_allclose(a.iloc[: cut + 1], b.iloc[: cut + 1], equal_nan=True)
    assert not np.allclose(a.iloc[cut + 1 :].fillna(0), b.iloc[cut + 1 :].fillna(0))


def test_regime_budget_backs_the_sleeve_that_earns_in_each_regime():
    from amp_sa.portfolio.regime_switch import regime_budget

    pnl, features, stressed = _regime_world()
    budget, diag = regime_budget(pnl, features, min_train=400, test_bars=200)
    live = budget["a"].notna()
    assert len(diag) >= 3
    np.testing.assert_allclose(budget[live].sum(axis=1), 1.0)
    was_stressed = stressed.shift(1)[live] == 1  # the budget at t reflects the regime seen at t-1
    assert budget.loc[live, "b"][was_stressed].mean() > 0.8
    assert budget.loc[live, "a"][~was_stressed].mean() > 0.8


def test_weight_changes_are_charged_when_a_hold_cost_is_given():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(5)
    sleeves = {k: (pd.Series(rng.normal(0, 1, 400), index=idx), pd.Series(0.0, index=idx)) for k in ("rv", "trend")}
    flip = pd.DataFrame({"rv": np.where(np.arange(400) // 20 % 2 == 0, 1.0, 0.0)}, index=idx)
    flip["trend"] = 1.0 - flip["rv"]
    hold = {k: pd.Series(0.5, index=idx) for k in sleeves}
    free = combine(sleeves, budget=flip, use_overlay=False)
    charged = combine(sleeves, budget=flip, use_overlay=False, hold_cost=hold)
    assert free.cost.fillna(0).sum() == 0.0
    assert charged.cost.fillna(0).sum() > 0.0
    pd.testing.assert_series_equal(free.gross, charged.gross)


# ---------------------------------------------------------------- ES / FX extras

def test_es_roll_flag_is_the_bar_after_the_third_friday():
    from amp_sa.data.load_extra import es_roll_bars

    idx = pd.bdate_range("2026-03-02", "2026-06-30")
    flags = es_roll_bars(idx)
    assert list(flags[flags].index.strftime("%Y-%m-%d")) == ["2026-03-23", "2026-06-22"]


def test_intraday_regime_features_are_yesterdays_close(monkeypatch):
    from amp_sa.sleeves import intraday_fx

    days = pd.bdate_range("2026-03-16", periods=5)
    daily = pd.DataFrame({"vix": [10.0, 20, 30, 40, 50], "move": [1.0, 2, 3, 4, 5]}, index=days)
    stamps = pd.DatetimeIndex(["2026-03-18 09:00", "2026-03-18 23:55", "2026-03-19 00:05", "2026-03-21 12:00"])
    bars = pd.DataFrame({"eurusd": 1.0, "usdjpy": 150.0, "usdcad": 1.4}, index=stamps)
    monkeypatch.setattr(intraday_fx, "load_intraday", lambda: bars.copy())
    panel, _ = intraday_fx.intraday_panel(daily)
    # Wed 18th sees Tuesday's close; Thu 19th sees Wednesday's; Sat 21st sees Thursday's (Friday's is shifted).
    assert panel["vix"].tolist() == [20.0, 20.0, 30.0, 40.0]


def test_intraday_schedule_restores_the_daily_config():
    from amp_sa.sleeves import intraday_fx

    before = (config.TRAIN_BARS, config.TEST_BARS, dict(config.COST_PER_UNIT))
    with intraday_fx.intraday_schedule():
        assert config.TRAIN_BARS == intraday_fx.TRAIN_DAYS * intraday_fx.BARS_PER_DAY
        assert config.COST_PER_UNIT["eurusd"] == intraday_fx.INTRADAY_COST["eurusd"]
    assert (config.TRAIN_BARS, config.TEST_BARS, dict(config.COST_PER_UNIT)) == before


# ---------------------------------------------------------------- PCA residual sleeve

def _curve_panel(seed: int = 4, n: int = 760):
    """Five tenors driven by level and slope, each with its own mean-reverting residual."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n, name="date")
    level, slope = np.cumsum(rng.normal(0, 5, n)), np.cumsum(rng.normal(0, 2, n))
    tenors = {"ust_2y": -1.0, "ust_3y": -0.6, "ust_5y": 0.0, "ust_10y": 0.6, "ust_30y": 1.0}
    cols = {}
    for name, tilt in tenors.items():
        gap = np.zeros(n)
        for t in range(1, n):
            gap[t] = 0.85 * gap[t - 1] + rng.normal(0, 1.5)
        cols[name] = 400 + level + tilt * slope + gap
    return pd.DataFrame(cols, index=idx)


@pytest.mark.parametrize("k", [2, 3])
def test_pca_sleeve_ignores_the_future(small_schedule, k):
    from amp_sa.sleeves import pca

    levels = _curve_panel()
    a, b = pca.run_curve(levels, k), pca.run_curve(perturb(levels), k)
    assert len(a) == 5 and all(r.live.iloc[:CUT].any() for r in a)
    for ra, rb in zip(a, b):
        for field in ("signal", "position", "gross", "cost"):
            assert_same_before_cut(getattr(ra.results["pca"], field), getattr(rb.results["pca"], field))
        assert_same_before_cut(ra.extras["legs_adaptive"], rb.extras["legs_adaptive"])
    assert_differs_after_cut(a[0].results["pca"].gross, b[0].results["pca"].gross)


def test_pca_residual_portfolios_are_factor_neutral():
    from amp_sa.sleeves import pca

    levels = _curve_panel()
    changes = levels.diff().dropna()
    loadings, explained = pca.factor_loadings(changes, 2)
    project = np.eye(5) - loadings @ loadings.T
    assert explained > 0.8
    np.testing.assert_allclose(project @ loadings, 0.0, atol=1e-10)  # no exposure to level or slope


def test_s_score_window_is_trailing():
    from amp_sa.sleeves import pca

    rng = np.random.default_rng(8)
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = 0.8 * x[t - 1] + rng.normal()
    c = pd.DataFrame({"a": x})
    full = pca.s_scores(c, 60)
    changed = c.copy()
    changed.iloc[300:] = rng.normal(0, 50, (100, 1))
    np.testing.assert_allclose(full.iloc[:300], pca.s_scores(changed, 60).iloc[:300], equal_nan=True)
    assert full.iloc[100:].notna().mean().iloc[0] > 0.5


def test_sparse_hedges_are_factor_neutral_with_k_plus_one_legs():
    from amp_sa.sleeves import pca

    changes = _curve_panel().diff().dropna()
    for k in (2, 3):
        loadings, _ = pca.factor_loadings(changes, k)
        weights = pca.sparse_hedges(loadings)
        np.testing.assert_allclose(weights @ loadings, 0.0, atol=1e-9)
        assert ((np.abs(weights) > 1e-12).sum(axis=1) == k + 1).all()
        np.testing.assert_allclose(np.diag(weights), 1.0)


def test_sparse_pca_sleeve_ignores_the_future(small_schedule):
    from amp_sa.sleeves import pca

    levels = _curve_panel()
    a, b = pca.run_curve(levels, 2, sparse=True), pca.run_curve(perturb(levels), 2, sparse=True)
    assert len(a) == 5
    for ra, rb in zip(a, b):
        for variant in ("pca", "pca_ll"):
            assert_same_before_cut(ra.results[variant].position, rb.results[variant].position)
