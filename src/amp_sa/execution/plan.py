"""Map research positions to AMP micro contracts. Dry run only: nothing here sends an order.

Research series -> AMP root and the $ value of a one-unit move of the series:
  Treasury yields (bp)  -> micro yield futures, $10 per bp
  CL generics ($/bbl)   -> micro WTI, $100 per $1
  wheat, corn (c/bu)    -> micro wheat / corn, $5 per cent
Credit OAS, Brent, the crack and KC wheat have no micro on AMP and are skipped.
"""

from __future__ import annotations

import pandas as pd

# research column -> (AMP root, dollars per unit of the series, 1 = front contract / 2 = second)
CONTRACT_MAP: dict[str, tuple[str, float, int]] = {
    "ust_2y": ("Z2YY", 10.0, 1), "ust_5y": ("Z5YY", 10.0, 1), "ust_10y": ("Z10Y", 10.0, 1),
    "ust_30y": ("Z30Y", 10.0, 1),
    "cl1": ("MCLE", 100.0, 1), "cl2": ("MCLE", 100.0, 2),
    "zw1": ("MZW", 5.0, 1), "zc1": ("MZC", 5.0, 1),
}


def front_symbols(specs: pd.DataFrame, asof: pd.Timestamp, min_days: int = 5) -> dict[tuple[str, int], str]:
    """(root, n) -> the n-th nearest listed contract that expires at least `min_days` out."""
    out = {}
    live = specs[specs["expiration"] > asof + pd.Timedelta(days=min_days)].sort_values("expiration")
    for root in {v[0] for v in CONTRACT_MAP.values()}:
        names = [s for s in live.index if s[: len(root)] == root and len(s) == len(root) + 3]
        for n, name in enumerate(names, start=1):
            out[(root, n)] = name
    return out


def target_contracts(leg_units: pd.Series, specs: pd.DataFrame, capital: float, asof: pd.Timestamp) -> pd.DataFrame:
    """leg_units: research column -> position in units of the series per $1 of capital.

    Returns one row per mapped leg with the exact and the rounded contract count.
    """
    symbols = front_symbols(specs, asof)
    rows = []
    for col, units in leg_units.items():
        if col not in CONTRACT_MAP:
            rows.append({"series": col, "symbol": None, "exact": float("nan"), "contracts": 0, "note": "no AMP micro"})
            continue
        root, dollars_per_unit, nth = CONTRACT_MAP[col]
        symbol = symbols.get((root, nth))
        exact = capital * units / dollars_per_unit
        rows.append({
            "series": col, "symbol": symbol, "exact": exact,
            "contracts": int(round(exact)) if symbol else 0,
            "note": "" if symbol else f"no listed {root} contract in the spec snapshot",
        })
    return pd.DataFrame(rows)
