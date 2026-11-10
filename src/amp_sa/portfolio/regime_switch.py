"""Regime-switching allocation between the sleeves.

The book has two sleeves because they earn in different conditions. This
module decides, bar by bar, how much of the risk budget each one gets:

  1. A Gaussian HMM on market features (log VIX, log MOVE) defines the regimes.
  2. On each training window, every sleeve's Sharpe is measured within each
     regime, and the regime is assigned to the sleeve that did best in it.
  3. Out of sample the HMM runs as a forward filter and the budget is the
     probability-weighted mix of those assignments.

Walk-forward with an expanding window: the HMM, the per-regime scores and the
assignments are refitted every `test_bars` on data that ends a purge and
embargo before the block they are applied to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.config import ANN
from amp_sa.models.regime import HMMRegime

MIN_TRAIN_BARS = 3 * ANN
MIN_STATE_BARS = 60
N_STATES = 2


def regime_features(levels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"log_vix": np.log(levels["vix"]), "log_move": np.log(levels["move"])})


def _sharpe(x: np.ndarray) -> float:
    sd = x.std()
    return float(x.mean() / sd) if len(x) >= MIN_STATE_BARS and sd > 0 else np.nan


def regime_budget(
    sleeve_net: pd.DataFrame,
    features: pd.DataFrame,
    min_train: int = MIN_TRAIN_BARS,
    test_bars: int | None = None,
    n_states: int = N_STATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (budget, diagnostics).

    sleeve_net: each sleeve's net P&L in risk units (NaN before it is live).
    budget:     risk share per sleeve, already lagged: row t sizes bar t and
                uses the filtered regime probability at t-1.
    """
    test_bars = config.TEST_BARS if test_bars is None else test_bars
    gap = config.PURGE_BARS + config.EMBARGO_BARS
    names = list(sleeve_net.columns)
    live = sleeve_net.notna().all(axis=1).to_numpy()
    budget_now = pd.DataFrame(np.nan, index=sleeve_net.index, columns=names)  # known at the close of t
    diag = []
    if not live.any():
        return budget_now, pd.DataFrame()
    first = int(live.argmax())
    pnl = sleeve_net.to_numpy(float)
    x = features.reindex(sleeve_net.index).to_numpy(float)

    for test_start in range(first + min_train + gap, len(sleeve_net), test_bars):
        train_end = test_start - gap
        test_end = min(test_start + test_bars, len(sleeve_net))
        model = HMMRegime.fit(x[first:train_end], n_states=n_states)
        if model is None:
            budget_now.iloc[test_start:test_end] = 1.0 / len(names)
            continue
        probs = model.filter(x[first:test_end])  # causal, warmed up from the first live bar
        state = probs.argmax(axis=1)

        # The regime seen at t-1 is the one a position held over bar t was sized in.
        prev_state = state[: train_end - first - 1]
        train_pnl = pnl[first + 1 : train_end]
        assign = np.full((n_states, len(names)), 1.0 / len(names))
        row = {"test_start": sleeve_net.index[test_start].date(), "train_bars": train_end - first}
        for k in range(n_states):
            in_state = prev_state == k
            scores = np.array([_sharpe(train_pnl[in_state, j]) for j in range(len(names))])
            if np.isfinite(scores).all():
                assign[k] = 0.0
                assign[k, int(scores.argmax())] = 1.0
            row[f"state{k}_share"] = float(in_state.mean())
            row[f"state{k}_vix"] = float(np.exp(model.scaler.mean[0] + model.scaler.std[0] * model.means[k, 0]))
            for j, name in enumerate(names):
                row[f"state{k}_{name}_sharpe"] = scores[j] * np.sqrt(ANN)
            row[f"state{k}_assigned"] = names[int(assign[k].argmax())] if assign[k].max() == 1.0 else "split"
        diag.append(row)
        budget_now.iloc[test_start:test_end] = (probs @ assign)[test_start - first : test_end - first]

    return budget_now.shift(1), pd.DataFrame(diag)
