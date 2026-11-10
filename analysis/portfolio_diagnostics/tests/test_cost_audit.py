"""The cost split must add back to what the engine charged, and the leg-level cost must match it when legs are fixed."""

import numpy as np
import pandas as pd

from analysis.portfolio_diagnostics.cost_audit import leg_level_cost, split_cost
from amp_sa.backtest.engine import account

IDX = pd.bdate_range("2020-01-01", periods=10)


def test_split_classifies_and_adds_up():
    signal = pd.Series([0, 1, 1, 1, -1, -1, 0, 0, 1, 0], index=IDX, dtype=float)
    vol = pd.Series([1, 1, 2, 2, 1, 1, 1, 1, 0.5, 1], index=IDX, dtype=float)
    res = account(signal, pd.Series(0.0, index=IDX), 0.5, vol=vol)
    s = split_cost(res.position, res.cost, res.unit_cost)
    assert np.allclose(s.sum(axis=1), res.cost)
    # bar 1: entry 1 unit; bar 2: resize 1 -> 0.5; bar 4: flip 0.5 -> -1 (exit 0.5, entry 1);
    # bar 6: exit 1; bar 8: entry 2; bar 9: exit 2
    assert np.allclose(s["entry"].to_numpy(), 0.5 * np.array([0, 1, 0, 0, 1, 0, 0, 0, 2, 0]))
    assert np.allclose(s["exit"].to_numpy(), 0.5 * np.array([0, 0, 0, 0, 0.5, 0, 1, 0, 0, 2]))
    assert np.allclose(s["resize"].to_numpy(), 0.5 * np.array([0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0]))
    assert np.allclose(s["roll"], 0.0)


def test_roll_charge_lands_in_roll():
    signal = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=IDX, dtype=float)
    is_roll = pd.Series(False, index=IDX)
    is_roll.iloc[2] = True
    res = account(signal, pd.Series(0.0, index=IDX), 0.5, is_roll=is_roll, vol=pd.Series(1.0, index=IDX))
    s = split_cost(res.position, res.cost, res.unit_cost)
    assert np.isclose(s["roll"].iloc[2], 2.0 * 1.0 * 0.5)
    assert np.allclose(s.sum(axis=1), res.cost)


def test_leg_level_equals_charged_with_fixed_legs():
    signal = pd.Series([0, 1, 1, 1, -1, -1, 0, 0, 1, 0], index=IDX, dtype=float)
    vol = pd.Series([1, 1, 2, 2, 1, 1, 1, 1, 0.5, 1], index=IDX, dtype=float)
    legs = pd.DataFrame({"a": 1.0, "b": -0.6, "c": 0.3}, index=IDX)
    c = pd.Series({"a": 0.25, "b": 0.25, "c": 0.5})
    unit = float(legs.iloc[0].abs() @ c)
    res = account(signal, pd.Series(0.0, index=IDX), unit, vol=vol)
    true, _ = leg_level_cost(res.position, legs, c)
    assert np.allclose(true, res.cost)


def test_leg_level_prices_a_refit_with_unchanged_position():
    # Held at 1 unit throughout; legs change on bar 5. Engine charges nothing after entry, a leg trader pays the reweight.
    pos = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], index=IDX, dtype=float)
    legs = pd.DataFrame({"a": 1.0, "b": -0.5}, index=IDX)
    legs.loc[IDX[5]:, "b"] = -0.8
    c = pd.Series({"a": 0.25, "b": 0.25})
    true, _ = leg_level_cost(pos, legs, c)
    # bar 1 entry: 1*(1 + 0.5)*0.25; the trade at the close of bar 4 moves b from -0.5 to -0.8: 0.3 * 0.25
    expected = np.zeros(10)
    expected[1], expected[4] = 1.5 * 0.25, 0.3 * 0.25
    assert np.allclose(true.to_numpy(), expected)
