"""The experiment pipeline must equal production when no hook is set; each hook must do only what it says."""

import pickle

import numpy as np
import pandas as pd
import pytest

from amp_sa.backtest.engine import account
from amp_sa.data.load_bbg import PROCESSED_DIR
from amp_sa.models.ou import band_positions
from analysis.experiments.pipeline import (Rule, account_sticky, fit_vol_gate, freeze_result, loo_rows, positions,
                                          run_members)


def test_positions_match_production_without_hooks():
    rng = np.random.default_rng(1)
    z = np.cumsum(rng.normal(0, 0.4, 3000))
    z[::97] = np.nan
    for entry, exit_ in ((1.0, 0.0), (1.5, 0.5), (2.0, 0.25)):
        assert np.array_equal(positions(z, entry, exit_), band_positions(z, entry, exit_))


def test_time_stop_closes_after_n_bars_and_waits():
    z = np.r_[0.0, np.full(30, 2.5), 0.5, 2.5, 2.5]
    pos = positions(z, 2.0, 0.0, time_stop=5)
    assert list(pos[1:7]) == [-1, -1, -1, -1, -1, 0]  # set on bar 1, held over bars 2..6 (5 bars), closed at bar 6
    assert (pos[7:31] == 0).all()  # |z| never came back inside the entry band
    assert pos[31] == 0 and pos[32] == -1  # inside the band at 31, a new trade at 32


def test_gate_blocks_entries_only():
    z = np.array([0.0, 2.5, 2.5, 1.0, -0.1, 2.5, 2.5])
    allow = np.array([True, False, True, False, False, False, True])
    pos = positions(z, 2.0, 0.0, allow=allow)
    assert list(pos) == [0, 0, -1, -1, 0, 0, -1]  # held through a closed gate, exits by the band


def test_sticky_account_with_zero_tolerance_equals_engine():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(2)
    signal = pd.Series(np.sign(rng.normal(size=200)) * (rng.random(200) > 0.5), index=idx)
    change = pd.Series(rng.normal(size=200), index=idx)
    vol = pd.Series(rng.uniform(0.5, 2, 200), index=idx)
    unit = pd.Series(0.3, index=idx)
    a, b = account(signal, change, unit, vol=vol), account_sticky(signal, change, unit, vol, 0.0)
    assert np.allclose(a.position, b.position) and np.allclose(a.cost, b.cost) and np.allclose(a.gross, b.gross)


def test_sticky_account_skips_small_resizes():
    idx = pd.bdate_range("2020-01-01", periods=4)
    signal = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    vol = pd.Series([1.0, 1.1, 1.5, 1.5], index=idx)  # target 1, 0.91, 0.67, 0.67
    res = account_sticky(signal, pd.Series(0.0, index=idx), pd.Series(1.0, index=idx), vol, 0.25)
    assert np.allclose(res.position, [1.0, 1.0, 1 / 1.5, 1 / 1.5])


def test_freeze_holds_entry_size_until_exit():
    idx = pd.bdate_range("2020-01-01", periods=7)
    signal = pd.Series([0, 1, 1, 1, 0, -1, -1], index=idx, dtype=float)
    vol = pd.Series([1.0, 2.0, 4.0, 1.0, 1.0, 0.5, 1.0], index=idx)
    change = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], index=idx)
    base = account(signal, change, pd.Series(0.1, index=idx), vol=vol)
    res = freeze_result(base)
    assert np.allclose(res.position, [0, 0.5, 0.5, 0.5, 0, -2, -2])
    assert np.allclose(res.gross, [0, 0, 0.5, 0.5, 0.5, 0, -2])
    assert np.allclose(res.cost, 0.1 * np.array([0, 0.5, 0, 0, 0.5, 2, 0]))


def test_freeze_is_identity_when_size_never_changes():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(4)
    signal = pd.Series(np.sign(rng.normal(size=300)) * (rng.random(300) > 0.5), index=idx)
    base = account(signal, pd.Series(rng.normal(size=300), index=idx), pd.Series(0.3, index=idx),
                   vol=pd.Series(1.5, index=idx))
    res = freeze_result(base)
    assert np.allclose(res.gross, base.gross) and np.allclose(res.cost, base.cost)


def test_vol_gate_picks_floor_from_training_only():
    rng = np.random.default_rng(5)
    n = 2000
    z = np.cumsum(rng.normal(0, 0.4, n)) % 6 - 3
    vol = np.where(np.arange(n) % 400 < 200, 0.1, 1.0)  # quiet half: cost dominates
    change = rng.normal(0, 1, n) * vol
    q, x, _ = fit_vol_gate(z, change, vol, np.full(n, 0.5), 1.5, 0.0, Rule(freeze_size=True))
    assert q in (10, 20, 30, 40, 50) and x == np.nanpercentile(vol, q)


def test_loo_residual_is_orthogonal_to_the_other_tenors_factors():
    rng = np.random.default_rng(3)
    f = rng.normal(size=(800, 2))
    load = rng.normal(size=(5, 2))
    dw = pd.DataFrame(f @ load.T + 0.1 * rng.normal(size=(800, 5)))
    rows = loo_rows(dw, 2)
    for i in range(5):
        others = [j for j in range(5) if j != i]
        resid = dw.to_numpy() @ rows[i]
        assert rows[i, i] == 1.0
        from amp_sa.sleeves.pca import factor_loadings
        fo = dw.iloc[:, others].to_numpy() @ factor_loadings(dw.iloc[:, others], 2)[0]
        assert np.allclose(fo.T @ resid, 0.0, atol=1e-8)


@pytest.mark.skipif(not (PROCESSED_DIR / "pca_runs.pkl").exists(), reason="needs the cached PCA runs")
@pytest.mark.parametrize("key,k,sparse", [(2, 2, False), ("3s", 3, True)])
def test_baseline_reproduces_cached_runs(key, k, sparse):
    from amp_sa.data.panel import load_research_panel
    levels, _ = load_research_panel()
    cached = pickle.load(open(PROCESSED_DIR / "pca_runs.pkl", "rb"))[key]
    mine = run_members(levels, k=k, sparse=sparse, rule=Rule())
    assert [r.name for r in mine] == [r.name for r in cached]
    for a, b in zip(mine, cached):
        for f in ("signal", "position", "gross", "cost"):
            x, y = getattr(a.results["pca"], f), getattr(b.results["pca"], f)
            assert np.array_equal(x.to_numpy(), y.to_numpy(), equal_nan=True), (a.name, f)
        assert a.extras["legs_adaptive"].equals(b.extras["legs_adaptive"])
