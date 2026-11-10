"""Bit-identical check after the [diag-logging] change to amp_sa/sleeves/pca.py.

    python -m analysis.portfolio_diagnostics.check_logging_identical

Compares every Result series (signal, position, gross, cost, turnover, unit_cost),
`live`, the saved leg weights and the fold diagnostics of every PCA run, both
variants, between data/processed/pca_runs_before_logging.pkl and pca_runs.pkl.
Exact equality (NaN == NaN), no tolerance.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

from amp_sa.data.load_bbg import PROCESSED_DIR

FIELDS = ("signal", "position", "gross", "cost", "turnover", "unit_cost")


def identical(a, b) -> bool:
    if isinstance(a, pd.DataFrame):
        return a.shape == b.shape and a.index.equals(b.index) and list(a.columns) == list(b.columns) and a.equals(b)
    return a.index.equals(b.index) and np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def main() -> int:
    old = pickle.load(open(PROCESSED_DIR / "pca_runs_before_logging.pkl", "rb"))
    new = pickle.load(open(PROCESSED_DIR / "pca_runs.pkl", "rb"))
    rows = []
    assert list(old) == list(new), (list(old), list(new))
    for k in old:
        assert [r.name for r in old[k]] == [r.name for r in new[k]], k
        for ro, rn in zip(old[k], new[k]):
            for v in ro.results:
                for f in FIELDS:
                    rows.append((rn.name, v, f, identical(getattr(ro.results[v], f), getattr(rn.results[v], f))))
            rows.append((rn.name, "-", "live", identical(ro.live, rn.live)))
            rows.append((rn.name, "-", "legs_adaptive", identical(ro.extras["legs_adaptive"], rn.extras["legs_adaptive"])))
            rows.append((rn.name, "-", "diagnostics", ro.diagnostics.reset_index(drop=True).equals(rn.diagnostics.reset_index(drop=True))))
    t = pd.DataFrame(rows, columns=["run", "variant", "series", "identical"])
    print(f"{len(t)} series compared, {int(t['identical'].sum())} identical, {int((~t['identical']).sum())} differ")
    if not t["identical"].all():
        print(t[~t["identical"]].to_string(index=False))
        return 1
    print(t.groupby(["variant", "series"])["identical"].agg(["count", "all"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
