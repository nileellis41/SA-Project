"""Leung-Li cost filter for the mean-reversion sleeve.

For each spread and fold, on the training window only:
  * fit the OU process of the static (OLS-hedged) spread,
  * express the round-trip cost in units of the spread's stationary sigma,
  * run the Leung-Li no-trade test: with entry cost c_hat, exit cost c and the
    sleeve's stop at STOP_Z sigma, is one round trip worth more than staying out?

The decision for a fold uses nothing after its training window, so it is a
legitimate ex-ante filter. `apply` re-scores the cached sleeve runs with the
folds that fail switched off.

    python -m amp_sa.sleeves.cost_filter
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from amp_sa import config
from amp_sa.backtest.splits import folds_for
from amp_sa.features.spreads import hedged, unit_cost
from amp_sa.models import cointegration
from amp_sa.models.ou_optimal import OUOptimal, fit_ou

DT = 1.0 / config.ANN
RATE = 0.05
BAND_POINTS = 200
RULE_OF_THUMB = 0.7  # round-trip cost / stationary sigma above which the simulation found no edge


def fold_decisions(spread: config.Spread, levels: pd.DataFrame) -> pd.DataFrame:
    """One row per fold: OU fit, cost in sigma units and the no-trade verdict, from the training window."""
    cols = [spread.y, *spread.x]
    y, x = levels[spread.y], levels[list(spread.x)]
    rows = []
    for fold in folds_for(levels[cols]):
        ytr, xtr = fold.train(y), fold.train(x)
        scr = cointegration.screen(ytr, xtr)
        s = hedged(ytr, xtr, scr.beta).dropna().to_numpy()
        cost = float(unit_cost(spread, scr.beta, ytr.index).iloc[0])  # one way, in spread units
        row = {"spread": spread.name, "group": spread.group, "fold": fold.k,
               "test_start": levels.index[fold.test_start], "test_end": levels.index[fold.test_end - 1],
               "cost_one_way": cost, "sigma_eq": np.nan, "round_trip_over_sigma": np.nan,
               "half_life": np.nan, "leung_li_trade": False, "take_profit_z": np.nan}
        try:
            theta, mu, sigma, _ = fit_ou(s, DT)
            sigma_eq = sigma / np.sqrt(2 * mu)
            row.update(sigma_eq=sigma_eq, round_trip_over_sigma=2 * cost / sigma_eq,
                       half_life=np.log(2) / mu / DT)
            model = OUOptimal(theta, mu, sigma, r=RATE, r_hat=RATE, c=cost, c_hat=cost)
            band = model.entry_band(theta - config.STOP_Z * sigma_eq, n=BAND_POINTS)
            if band is not None and band.get("d") is not None:
                row.update(leung_li_trade=True, take_profit_z=(band["b"] - theta) / sigma_eq)
        except (ValueError, AssertionError, ZeroDivisionError, FloatingPointError):
            pass  # no mean reversion on this window, or the solver found no root: treat as no-trade
        rows.append(row)
    return pd.DataFrame(rows)


def keep_mask(decisions: pd.DataFrame, index: pd.DatetimeIndex, rule: str) -> pd.Series:
    """True on bars whose fold passes `rule` ('leung_li' or 'ratio')."""
    keep = pd.Series(False, index=index)
    for row in decisions.itertuples():
        ok = row.leung_li_trade if rule == "leung_li" else row.round_trip_over_sigma < RULE_OF_THUMB
        if ok:
            keep.loc[row.test_start : row.test_end] = True
    return keep


def main() -> None:
    from amp_sa import run
    from amp_sa.backtest import metrics
    from amp_sa.portfolio.combine import combine

    levels, _, rv_runs, _ = run.load_runs()
    spreads = {s.name: s for s in config.SPREADS}
    decisions = {r.name: fold_decisions(spreads[r.name], levels) for r in rv_runs}
    table = pd.concat(decisions.values(), ignore_index=True)
    table.to_csv(run.REPORT_DIR / "tables" / "cost_filter_folds.csv", index=False)

    by_spread = table.groupby("spread", sort=False).agg(
        group=("group", "first"), folds=("fold", "count"),
        round_trip_over_sigma=("round_trip_over_sigma", "median"), half_life=("half_life", "median"),
        leung_li_pass=("leung_li_trade", "mean"),
        ratio_pass=("round_trip_over_sigma", lambda r: float((r < RULE_OF_THUMB).mean())),
    )
    for r in rv_runs:
        for v in ("static", "adaptive_nn"):
            by_spread.loc[r.name, f"oos_net_{v}"] = metrics.sharpe(r.net(v))
            by_spread.loc[r.name, f"oos_gross_{v}"] = metrics.sharpe(r.results[v].gross.where(r.live))
    by_spread.reset_index().to_csv(run.REPORT_DIR / "tables" / "cost_filter_by_spread.csv", index=False)

    def sleeve_sharpe(variant: str, groups, rule: str | None) -> dict:
        gross, cost = {}, {}
        for r in rv_runs:
            if groups is not None and r.group not in groups:
                continue
            keep = keep_mask(decisions[r.name], levels.index, rule) if rule else pd.Series(True, index=levels.index)
            res = r.results[variant]
            gross[r.name] = res.gross.where(keep, 0.0).where(r.live)
            cost[r.name] = res.cost.where(keep, 0.0).where(r.live)
        g, c = pd.DataFrame(gross).mean(axis=1), pd.DataFrame(cost).mean(axis=1)
        port = combine({"s": (g, c)}, budget={"s": 1.0}, use_overlay=False)
        return {"sharpe_net": metrics.sharpe(port.net), "sharpe_gross": metrics.sharpe(port.gross),
                "max_dd": metrics.max_drawdown(port.net), "sharpe_2x_cost": metrics.sharpe(port.net_at(2.0))}

    rows = []
    for scope, groups in run.SCOPES.items():
        for variant in ("static", "adaptive_nn"):
            for label, rule in (("no filter", None), ("Leung-Li no-trade test", "leung_li"),
                                (f"round trip < {RULE_OF_THUMB} sigma", "ratio")):
                rows.append({"scope": scope, "variant": variant, "filter": label,
                             **sleeve_sharpe(variant, groups, rule)})
    effect = pd.DataFrame(rows)
    effect.to_csv(run.REPORT_DIR / "tables" / "cost_filter_effect.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(by_spread.sort_values("round_trip_over_sigma").to_string())
        print(effect.to_string(index=False))


if __name__ == "__main__":
    main()
