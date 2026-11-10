"""Shared loading for the portfolio diagnostics: the four models, their members and the built sleeves.

Reads the cached walk-forward runs only (data/processed/sleeve_runs.pkl, pca_runs.pkl).
Nothing here re-fits or changes a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amp_sa import config, run
from amp_sa import run_sleeve1 as s1

OUT_DIR = Path(__file__).resolve().parent
TABLE_DIR = OUT_DIR / "tables"

MODELS = {  # short label -> model name in run_sleeve1.models (same four as run_lessons)
    "headline": s1.HEADLINE,
    "pca3_sparse": "PCA-3 sparse (4 legs per residual)",
    "pca2": "PCA-2 curve (level, slope hedged)",
    "pairwise_nn": "pairwise adaptive, neural gate",
}
CAPITAL = 1_000_000  # dollars, only to express fractions of capital as a DV01 / dollar view


@dataclass
class Model:
    label: str
    name: str
    items: list  # (StrategyRun, variant)
    port: object  # amp_sa.portfolio.combine.Portfolio


def legs_for(run_, variant: str) -> pd.DataFrame:
    """Leg weights per bar (columns = series) of one unit of the member, as the backtest saved them."""
    if "legs_adaptive" in run_.extras and variant != "static":
        return run_.extras["legs_adaptive"]
    return run_.extras["legs_static"]


def load_models() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Model]]:
    levels, rolls, rv_runs, _ = run.load_runs()
    pca_runs = s1.load_pca_runs(levels)
    all_items = s1.models(rv_runs, pca_runs)
    out = {}
    for label, name in MODELS.items():
        items = all_items[name]
        port, _, _ = s1.sleeve(items)
        out[label] = Model(label, name, items, port)
    return levels, rolls, out


def leg_cost_vector(columns) -> pd.Series:
    return pd.Series({c: config.COST_PER_UNIT[c] for c in columns})
