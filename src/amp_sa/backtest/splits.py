"""Walk-forward splits with purge and embargo.

Each fold trains on a rolling window that ends `purge + embargo` bars before
the test block starts. The purge removes training labels whose horizon would
reach into the test block; the embargo is an extra buffer for serial correlation.
Test blocks tile the sample without overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from amp_sa import config


@dataclass(frozen=True)
class Fold:
    k: int
    train_start: int  # positions into the index, end-exclusive
    train_end: int
    test_start: int
    test_end: int

    def train(self, obj):
        return obj.iloc[self.train_start : self.train_end]

    def test(self, obj):
        return obj.iloc[self.test_start : self.test_end]

    def warm(self, obj):
        """Train start through test end: for causal filters that need a warm-up."""
        return obj.iloc[self.train_start : self.test_end]


def walk_forward(
    n_bars: int,
    train_bars: int | None = None,
    test_bars: int | None = None,
    purge: int | None = None,
    embargo: int | None = None,
    first_bar: int = 0,
) -> list[Fold]:
    """Folds over positions [first_bar, n_bars). The last test block may be short.

    Arguments left as None take the shared schedule in config.
    """
    train_bars = config.TRAIN_BARS if train_bars is None else train_bars
    test_bars = config.TEST_BARS if test_bars is None else test_bars
    purge = config.PURGE_BARS if purge is None else purge
    embargo = config.EMBARGO_BARS if embargo is None else embargo
    gap = purge + embargo
    folds, test_start = [], first_bar + train_bars + gap
    while test_start < n_bars:
        train_end = test_start - gap
        folds.append(
            Fold(len(folds), train_end - train_bars, train_end, test_start, min(test_start + test_bars, n_bars))
        )
        test_start += test_bars
    return folds


def folds_for(series: pd.DataFrame | pd.Series, **kwargs) -> list[Fold]:
    """Folds starting at the first bar where every column of `series` is observed."""
    valid = series.notna().all(axis=1) if isinstance(series, pd.DataFrame) else series.notna()
    if not valid.any():
        return []
    first = int(valid.to_numpy().argmax())
    return walk_forward(len(series), first_bar=first, **kwargs)
