"""Shared synthetic panel: small enough that the whole walk-forward runs in seconds."""

import numpy as np
import pandas as pd
import pytest

from amp_sa import config

N_BARS = 760
CUT = 600  # bars at or after this position are rewritten in the perturbed panel


def make_panel(seed: int = 0, n: int = N_BARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-05", periods=n, name="date")
    common = 400 + np.cumsum(rng.normal(0, 5, n))
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.9 * gap[t - 1] + rng.normal(0, 2)
    levels = pd.DataFrame(
        {
            "ust_2y": common,
            "ust_10y": 50 + 0.8 * common + gap,
            "vix": np.exp(3 + 0.2 * np.sin(np.arange(n) / 40) + rng.normal(0, 0.05, n)),
            "move": np.exp(4.5 + 0.2 * np.cos(np.arange(n) / 55) + rng.normal(0, 0.05, n)),
        },
        index=index,
    )
    rolls = pd.DataFrame(False, index=index, columns=levels.columns)
    return levels, rolls


def perturb(levels: pd.DataFrame, cut: int = CUT, seed: int = 99) -> pd.DataFrame:
    """Same data before `cut`; a different, much wilder path from `cut` on."""
    rng = np.random.default_rng(seed)
    out = levels.copy()
    n = len(levels) - cut
    for col in out.columns:
        shock = np.cumsum(rng.normal(0, 5 * levels[col].diff().std(), n))
        out.iloc[cut:, out.columns.get_loc(col)] = np.abs(levels[col].iloc[cut - 1] + shock) + 1e-3
    return out


@pytest.fixture
def small_schedule(monkeypatch):
    """A walk-forward schedule that fits the synthetic panel."""
    monkeypatch.setattr(config, "TRAIN_BARS", 300)
    monkeypatch.setattr(config, "TEST_BARS", 100)
    monkeypatch.setattr(config, "HEALTH_WINDOW", 100)
    monkeypatch.setattr(config, "L1_WINDOW", 60)


@pytest.fixture(scope="session")
def panel():
    return make_panel()
