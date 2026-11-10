"""Regime models. Both are fitted on a training window and applied causally.

HMM baseline: a Gaussian HMM on standardized regime features. Out of sample it
is run as a forward filter, P(state_t | features through t). hmmlearn's own
predict/predict_proba are smoothers that condition on the whole sequence and
would leak the future, so they are not used.

Neural classifier: a small MLP that predicts whether a stretched spread
reverts over the next few bars.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from sklearn.neural_network import MLPClassifier


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        std = np.nanstd(x, axis=0)
        return cls(np.nanmean(x, axis=0), np.where(std > 0, std, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class HMMRegime:
    scaler: Scaler
    start: np.ndarray
    trans: np.ndarray
    means: np.ndarray  # (states, features)
    stds: np.ndarray

    @classmethod
    def fit(cls, x_train: np.ndarray, n_states: int = 2, seed: int = 0) -> "HMMRegime | None":
        x_train = x_train[np.isfinite(x_train).all(axis=1)]
        if len(x_train) < 100:
            return None
        scaler = Scaler.fit(x_train)
        model = GaussianHMM(n_states, covariance_type="diag", n_iter=50, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(scaler.transform(x_train))
            except Exception:
                return None
        variances = np.array([np.diag(c) for c in model.covars_])  # covars_ is expanded to full matrices
        return cls(scaler, model.startprob_, model.transmat_, model.means_, np.sqrt(variances))

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Forward filter: row t is P(state_t | x_1..x_t). Rows with NaN repeat the prediction."""
        z = self.scaler.transform(x)
        n_states = len(self.start)
        out = np.full((len(z), n_states), np.nan)
        # Emission log-density of each row under each state: a function of that row alone.
        emit = norm.logpdf(z[:, None, :], self.means[None], self.stds[None]).sum(axis=2)
        p = self.start.copy()
        for t in range(len(z)):
            if t > 0:
                p = p @ self.trans
            if np.isfinite(emit[t]).all():
                p = p * np.exp(emit[t] - emit[t].max())
            p = p / p.sum() if p.sum() > 0 else np.full(n_states, 1.0 / n_states)
            out[t] = p
        return out


@dataclass
class NeuralRegime:
    scaler: Scaler
    model: MLPClassifier

    @classmethod
    def fit(cls, x_train: np.ndarray, label: np.ndarray, seed: int = 0) -> "NeuralRegime | None":
        ok = np.isfinite(x_train).all(axis=1) & np.isfinite(label)
        if ok.sum() < 200 or len(np.unique(label[ok])) < 2:
            return None
        scaler = Scaler.fit(x_train[ok])
        model = MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-2, solver="lbfgs", max_iter=200, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(scaler.transform(x_train[ok]), label[ok].astype(int))
        return cls(scaler, model)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """P(reverts) per row, NaN where a feature is missing. Each row uses only itself."""
        out = np.full(len(x), np.nan)
        ok = np.isfinite(x).all(axis=1)
        if ok.any():
            out[ok] = self.model.predict_proba(self.scaler.transform(x[ok]))[:, 1]
        return out
