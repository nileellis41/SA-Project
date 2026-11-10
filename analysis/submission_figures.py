"""Slide figures for submission/figures/ (PNG, 300 dpi). Reads logged tables and cached runs only.

    uv run python -m analysis.submission_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.portfolio_diagnostics.common import load_models  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "figures"
SUB = ROOT / "submission" / "tables"
CR1 = ROOT / "results" / "cost_round1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = {"headline": "Headline (PCA-2 + credit)", "pca3_sparse": "PCA-3 sparse", "pca2": "PCA-2 full",
          "pairwise_nn": "Pairwise NN"}
TENORS = ["2y", "3y", "5y", "7y", "10y", "30y"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def bars(ax, labels, groups: dict[str, list[float]], fmt="{:.2f}"):
    """Grouped bars, 2px surface gap between neighbours, value labels on the bars."""
    n = len(groups)
    width = 0.8 / n
    x = np.arange(len(labels))
    for j, (name, vals) in enumerate(groups.items()):
        pos = x - 0.4 + width * (j + 0.5)
        b = ax.bar(pos, vals, width * 0.92, color=SERIES[j], label=name, edgecolor=SURFACE, linewidth=1)
        ax.bar_label(b, [fmt.format(v) for v in vals], fontsize=7, color=INK2, padding=2)
    ax.set_xticks(x, labels)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.grid(axis="x", visible=False)


def loadings() -> None:
    L = pd.read_csv(SUB / "b_loadings.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for j, c in enumerate(("level", "slope", "curvature")):
        ax.plot(TENORS, L[c], color=SERIES[j], marker="o", markersize=5, label=c)
        ax.annotate(c, (5, L[c].iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=8)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_title("PCA loadings by tenor, fold trained 2018-01 to 2022-11")
    ax.set_ylabel("loading (unit eigenvector)")
    ax.legend(loc="upper left", ncol=3)
    ax.text(0, -0.18, "PC1 89.2%, PC2 9.0%, PC3 1.4% of variance. Signs are arbitrary (level is negative on every tenor).",
            transform=ax.transAxes, fontsize=7, color=INK2)
    save(fig, "03_pca_loadings")


def w_diagonal() -> None:
    W = pd.read_csv(SUB / "b_W_diag.csv").query("tenor != 'trace'")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars(ax, TENORS, {"PCA-2 (trace 4)": W["W_diag_pca2"].tolist(), "PCA-3 full (trace 3)": W["W_diag_pca3"].tolist()})
    ax.set_title("Diagonal of W = I − LLᵀ: share of each tenor left in its own residual")
    ax.set_ylabel("W_ii")
    ax.set_ylim(0, 0.98)
    ax.legend(loc="upper right", ncol=2)
    save(fig, "03_pca_W_diagonal")


def cumulative(models) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
    for ax, (k, label) in zip(axes.flat, MODELS.items()):
        p = models[k].port
        live = p.net.notna()
        for j, (name, s) in enumerate((("gross", p.gross), ("net of 1x cost", p.net))):
            cum = s[live].cumsum() * 100
            ax.plot(cum.index, cum, color=SERIES[j], linewidth=1.5, label=name)
            ax.annotate(f"{cum.iloc[-1]:+.0f}%", (cum.index[-1], cum.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK2)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel("cumulative, % of capital")
    axes.flat[0].legend(loc="upper left")
    fig.suptitle("Cumulative P&L at the 10% vol target, out of sample 1995–2026 (additive)", fontweight="bold", fontsize=11)
    save(fig, "05_cumulative_gross_vs_net")


def hit_cgcl() -> None:
    d = pd.read_csv(SUB / "d_cg_cl.csv")
    d = d[~d["basis"].str.startswith("merged")]
    names = ["Headline", "PCA-3 sparse", "PCA-2 full", "Pairwise NN", "fly_2s5s10s", "fly_3s5s7s"]
    d = d.set_index("model").loc[names]
    labels = ["Headline", "PCA-3\nsparse", "PCA-2\nfull", "Pairwise\nNN", "Fly\n2s5s10s", "Fly\n3s5s7s"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    bars(a, labels, {"gross hit": (d["hit_gross"] * 100).tolist(), "net hit": (d["hit_net"] * 100).tolist(),
                     "breakeven hit L/(W+L)": (d["breakeven_hit_rate_L_over_W_plus_L"] * 100).tolist()}, "{:.0f}")
    a.set_title("Hit rate vs the hit rate needed to break even")
    a.set_ylabel("% of trades")
    a.legend(loc="upper left", ncol=3, fontsize=7)
    a.set_ylim(0, 108)
    bars(b, labels, {"W: mean net gain of winners": d["W_mean_net_gain_of_winners"].tolist(),
                     "L: mean net loss of losers": d["L_mean_net_loss_of_losers"].tolist()})
    b.set_title("Conditional gain and loss per trade")
    b.set_ylabel("risk units (daily σ at entry)")
    b.legend(loc="upper left", fontsize=7)
    save(fig, "07_hit_ratio_cg_cl")


def terciles() -> None:
    t = pd.read_csv(CR1 / "terciles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, model in zip(axes, ("Headline", "PCA-3 sparse")):
        x = t[t["model"] == model]
        groups = {lab: x[x["config"] == c]["mean_net"].tolist()
                  for lab, c in (("baseline", "baseline"), ("freeze + vol gate", "freeze_size+vol_gate"))}
        bars(ax, ["low vol", "mid vol", "high vol"], groups)
        cuts = x["entry_vol_bp_max"].iloc[:2].tolist()
        ax.set_title(f"{model}: entry-vol terciles\n(cuts {cuts[0]:.2f} / {cuts[1]:.2f} bp/day)")
        ax.set_ylabel("mean net P&L per trade, risk units")
    axes[0].legend(loc="lower right", fontsize=7)
    save(fig, "08_entry_vol_terciles")


def cost_sensitivity() -> None:
    r = pd.read_csv(CR1 / "results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, model in zip(axes, ("Headline", "PCA-3 sparse")):
        for j, (cfg, lab) in enumerate((("baseline", "baseline"), ("freeze_size", "freeze size"),
                                        ("freeze_size+vol_gate", "freeze + vol gate"))):
            x = r[(r["model"] == model) & (r["config"] == cfg)]
            one = x.loc[x["cost_mult"] == 1.0, "sharpe_net"].iloc[0]
            ax.plot(x["cost_mult"], x["sharpe_net"], color=SERIES[j], marker="o", markersize=5,
                    label=f"{lab} (1x: {one:+.2f})")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks([0.5, 1.0, 2.0], ["0.5x", "1x", "2x"])
        ax.set_title(f"{model}: net Sharpe vs cost multiple")
        ax.set_xlabel("multiple of the cost table (1x = 0.25 bp per Treasury leg)")
        ax.legend(loc="lower left", fontsize=7)
    axes[0].set_ylabel("net Sharpe")
    save(fig, "09_cost_sensitivity")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, models = load_models()
    loadings()
    w_diagonal()
    cumulative(models)
    hit_cgcl()
    terciles()
    cost_sensitivity()
    for f in sorted(OUT.glob("*.png")):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
