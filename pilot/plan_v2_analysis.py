"""
Phase 1 analysis deliverables for experimental_plan_v2.md.

All analyses operate on the completed primary grid in outputs/primary_results.csv
and emit the new CSV/figure artifacts promised in the v2 plan:

    - t3_threshold.csv        + figures/figT3.pdf
    - bias_variance.csv       + figures/figBV.pdf
    - regime_diagram.csv      + figures/figRegime.pdf
    - crossover.csv

Run:
    python plan_v2_analysis.py
    python plan_v2_analysis.py --plans 2 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG_DIR = OUT / "figures"
RESULTS = OUT / "primary_results.csv"

T3_CSV = OUT / "t3_threshold.csv"
BV_CSV = OUT / "bias_variance.csv"
REGIME_CSV = OUT / "regime_diagram.csv"
CROSS_CSV = OUT / "crossover.csv"

REP_ORDER = ["raw", "concept", "learned"]
REP_LABELS = {
    "raw": "Raw $S$",
    "concept": "Concept $C$",
    "learned": "Learned $Z$",
}
REP_COLORS = {
    "raw": "#d62728",
    "concept": "#2ca02c",
    "learned": "#1f77b4",
}

CANONICAL_TAU = "homogeneous"
CANONICAL_ESTIMATOR = "dr"
CANONICAL_OUTCOME = "lgbm"
T3_EPSILONS = [0.05, 0.10]
CROSSOVER_PAIRS = [
    ("raw", "concept"),
    ("raw", "learned"),
    ("learned", "concept"),
]


def load_primary() -> pd.DataFrame:
    if not RESULTS.exists():
        raise SystemExit(f"No primary results at {RESULTS}. Run primary.py first.")
    return pd.read_csv(RESULTS)


def canonical(df: pd.DataFrame, tau_regime: str = CANONICAL_TAU) -> pd.DataFrame:
    return df[
        (df["tau_regime"] == tau_regime)
        & (df["estimator"] == CANONICAL_ESTIMATOR)
        & (df["outcome_model"] == CANONICAL_OUTCOME)
    ].copy()


def _lighten(color: str, amount: float = 0.4) -> tuple[float, float, float]:
    base = np.array(to_rgb(color))
    return tuple(base + (1.0 - base) * amount)


def _fmt_threshold(value: float | None, status: str, upper: float) -> str:
    if status == "gt_max":
        return f">{int(upper/1000)}k"
    if value is None:
        return ""
    return f"{value:.0f}"


def _fmt_cross(value: float | None, status: str) -> str:
    if status == "gt_max":
        return ">20k"
    if status == "lt_min":
        return "<1k"
    if value is None:
        return ""
    return f"{value:.0f}"


def _threshold_crossing(Ns: list[int], biases: list[float], epsilon: float) -> tuple[float | None, str]:
    if not Ns:
        return None, "missing"
    if biases[0] <= epsilon:
        return float(Ns[0]), "within_grid"
    for i in range(len(Ns) - 1):
        n0, n1 = float(Ns[i]), float(Ns[i + 1])
        b0, b1 = float(biases[i]), float(biases[i + 1])
        if b0 > epsilon and b1 <= epsilon:
            if abs(b1 - b0) < 1e-12:
                return n1, "within_grid"
            t = (epsilon - b0) / (b1 - b0)
            return n0 + t * (n1 - n0), "within_grid"
        if b1 <= epsilon:
            return n1, "within_grid"
    return None, "gt_max"


def build_t3_threshold(df: pd.DataFrame) -> pd.DataFrame:
    sub = canonical(df, tau_regime=CANONICAL_TAU)
    rows = []
    max_n = float(sub["N"].max())
    for gamma in sorted(sub["gamma"].unique()):
        for rep in REP_ORDER:
            cell = (sub[sub["gamma"] == gamma]
                      .query("rep == @rep")
                      .groupby("N")["ate_bias"]
                      .median()
                      .sort_index())
            Ns = cell.index.astype(int).tolist()
            biases = cell.to_numpy(dtype=float).tolist()
            for epsilon in T3_EPSILONS:
                n_t3, status = _threshold_crossing(Ns, biases, epsilon)
                rows.append(dict(
                    rep=rep,
                    gamma=float(gamma),
                    epsilon=float(epsilon),
                    N_T3=_fmt_threshold(n_t3, status, max_n),
                    N_T3_value=float(n_t3) if n_t3 is not None else np.nan,
                    status=status,
                ))
    out = pd.DataFrame(rows)
    out.to_csv(T3_CSV, index=False)
    return out


def plot_t3_threshold(df_t3: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gammas = sorted(df_t3["gamma"].unique())
    epsilons = sorted(df_t3["epsilon"].unique())

    fig, axes = plt.subplots(1, len(epsilons), figsize=(5.2 * len(epsilons), 3.8), sharey=True)
    if len(epsilons) == 1:
        axes = [axes]

    width = 0.22
    x = np.arange(len(gammas))

    for ax, epsilon in zip(axes, epsilons):
        panel = df_t3[df_t3["epsilon"] == epsilon]
        for j, rep in enumerate(REP_ORDER):
            vals = []
            hatches = []
            for gamma in gammas:
                row = panel[(panel["gamma"] == gamma) & (panel["rep"] == rep)].iloc[0]
                v = float(row["N_T3_value"]) if not pd.isna(row["N_T3_value"]) else 20000.0
                if row["status"] == "gt_max":
                    v = 20000.0
                    hatches.append("///")
                else:
                    hatches.append("")
                vals.append(v)
            bars = ax.bar(
                x + j * width - width,
                vals,
                width,
                color=REP_COLORS[rep],
                alpha=0.85,
                label=REP_LABELS[rep],
            )
            for bar, hatch in zip(bars, hatches):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_edgecolor("black")
                    bar.set_linewidth(0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{g:g}" for g in gammas])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("$\\gamma$")
        ax.set_title(f"$\\epsilon = {epsilon:.2f}$", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("$N_{T3}$")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig T3 — Sample-Efficiency Threshold by Representation", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figT3.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


def build_bias_variance(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["N", "gamma", "tau_regime", "rep", "estimator", "outcome_model"]
    rows = []
    for keys, cell in df.groupby(group_cols, sort=True):
        ate_hats = cell["ate_hat"].to_numpy(dtype=float)
        ate_true_values = cell["ate_true"].to_numpy(dtype=float)
        ate_true = float(ate_true_values[0])
        if not np.allclose(ate_true_values, ate_true, atol=1e-12):
            raise RuntimeError(f"ate_true is not constant within cell {keys}")

        mean_hat = float(np.mean(ate_hats))
        bias_signed = mean_hat - ate_true
        bias_sq = bias_signed ** 2
        # Use empirical population variance so mse = bias^2 + variance holds exactly.
        variance = float(np.mean((ate_hats - mean_hat) ** 2))
        mse = bias_sq + variance
        sanity = float(np.mean((ate_hats - ate_true) ** 2))
        if not np.isclose(mse, sanity, atol=1e-12):
            raise RuntimeError(f"MSE sanity check failed for cell {keys}: {mse} vs {sanity}")

        rows.append(dict(
            N=int(keys[0]),
            gamma=float(keys[1]),
            tau_regime=keys[2],
            rep=keys[3],
            estimator=keys[4],
            outcome_model=keys[5],
            bias_signed=bias_signed,
            bias_sq=bias_sq,
            variance=variance,
            mse=mse,
        ))
    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    out.to_csv(BV_CSV, index=False)
    return out


def plot_bias_variance(df_bv: pd.DataFrame) -> None:
    sub = df_bv[
        (df_bv["tau_regime"] == CANONICAL_TAU)
        & (df_bv["estimator"] == CANONICAL_ESTIMATOR)
        & (df_bv["outcome_model"] == CANONICAL_OUTCOME)
        & (df_bv["gamma"].isin([0.3, 0.6]))
    ]
    gammas = sorted(sub["gamma"].unique())
    Ns = sorted(sub["N"].unique())

    fig, axes = plt.subplots(1, len(gammas), figsize=(6.0 * len(gammas), 4.2), sharey=True)
    if len(gammas) == 1:
        axes = [axes]

    offsets = {"raw": 0.82, "concept": 1.00, "learned": 1.22}
    widths = {N: N * 0.12 for N in Ns}

    for ax, gamma in zip(axes, gammas):
        panel = sub[sub["gamma"] == gamma]
        for rep in REP_ORDER:
            rep_df = panel[panel["rep"] == rep].set_index("N").sort_index()
            positions = np.array([N * offsets[rep] for N in Ns], dtype=float)
            bias_vals = rep_df.loc[Ns, "bias_sq"].to_numpy(dtype=float)
            var_vals = rep_df.loc[Ns, "variance"].to_numpy(dtype=float)
            width_vals = np.array([widths[N] for N in Ns], dtype=float)

            ax.bar(
                positions,
                bias_vals,
                width=width_vals,
                color=REP_COLORS[rep],
                alpha=0.95,
                linewidth=0,
            )
            ax.bar(
                positions,
                var_vals,
                bottom=bias_vals,
                width=width_vals,
                color=_lighten(REP_COLORS[rep], amount=0.45),
                alpha=0.95,
                linewidth=0,
            )

        ax.set_xscale("log")
        ax.set_xticks(Ns)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("Sample size $N$")
        ax.set_title(f"$\\gamma = {gamma}$", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    rep_handles = [mpatches.Patch(color=REP_COLORS[r], label=REP_LABELS[r]) for r in REP_ORDER]
    seg_handles = [
        mpatches.Patch(color="#444444", label="Bias$^2$"),
        mpatches.Patch(color="#cccccc", label="Variance"),
    ]
    axes[0].legend(handles=rep_handles + seg_handles, fontsize=8, loc="upper right")
    axes[0].set_ylabel("ATE MSE Decomposition")
    fig.suptitle("Fig BV — Bias-Variance Decomposition of $\\hat{\\tau}$", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figBV.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


def build_regime_diagram(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["estimator"] == CANONICAL_ESTIMATOR)
        & (df["outcome_model"] == CANONICAL_OUTCOME)
    ]
    rows = []
    for (tau_regime, gamma, N), cell in sub.groupby(["tau_regime", "gamma", "N"], sort=True):
        med = cell.groupby("rep")["ate_bias"].median().to_dict()
        ordered = sorted((float(v), k) for k, v in med.items())
        winner = ordered[0][1]
        margin = ordered[1][0] - ordered[0][0] if len(ordered) > 1 else 0.0
        rows.append(dict(
            gamma=float(gamma),
            N=int(N),
            tau_regime=tau_regime,
            winner=winner,
            margin=float(margin),
            concept_bias=float(med.get("concept", np.nan)),
            raw_bias=float(med.get("raw", np.nan)),
            learned_bias=float(med.get("learned", np.nan)),
        ))
    out = pd.DataFrame(rows).sort_values(["tau_regime", "gamma", "N"]).reset_index(drop=True)
    out.to_csv(REGIME_CSV, index=False)
    return out


def plot_regime_diagram(df_regime: pd.DataFrame) -> None:
    tau_regimes = [t for t in ["homogeneous", "heterogeneous"] if t in df_regime["tau_regime"].unique()]
    gammas = sorted(df_regime["gamma"].unique(), reverse=True)
    Ns = sorted(df_regime["N"].unique())

    fig, axes = plt.subplots(1, len(tau_regimes), figsize=(5.6 * len(tau_regimes), 3.8), sharey=True)
    if len(tau_regimes) == 1:
        axes = [axes]

    max_margin = max(float(df_regime["margin"].max()), 1e-9)
    for ax, tau_regime in zip(axes, tau_regimes):
        panel = df_regime[df_regime["tau_regime"] == tau_regime]
        for row_i, gamma in enumerate(gammas):
            for col_i, N in enumerate(Ns):
                cell = panel[(panel["gamma"] == gamma) & (panel["N"] == N)]
                if cell.empty:
                    face = (0.95, 0.95, 0.95, 1.0)
                    margin = np.nan
                else:
                    row = cell.iloc[0]
                    alpha = 0.35 + 0.65 * float(row["margin"]) / max_margin
                    face = (*to_rgb(REP_COLORS[row["winner"]]), alpha)
                    margin = float(row["margin"])
                rect = mpatches.Rectangle((col_i, row_i), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5)
                ax.add_patch(rect)
                if not np.isnan(margin):
                    ax.text(col_i + 0.5, row_i + 0.52, f"{margin:.3f}", ha="center", va="center", fontsize=9)

        ax.set_xlim(0, len(Ns))
        ax.set_ylim(0, len(gammas))
        ax.set_xticks(np.arange(len(Ns)) + 0.5)
        ax.set_xticklabels([f"{N:,}" for N in Ns])
        ax.set_yticks(np.arange(len(gammas)) + 0.5)
        ax.set_yticklabels([f"{g:g}" for g in gammas])
        ax.invert_yaxis()
        ax.set_xlabel("Sample size $N$")
        ax.set_title(f"{tau_regime.capitalize()} $\\tau^*$", fontsize=11)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("$\\gamma$")
    legend = [mpatches.Patch(color=REP_COLORS[r], label=REP_LABELS[r]) for r in REP_ORDER]
    axes[-1].legend(handles=legend, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Fig Regime — Winner per $(\\gamma, N)$ Cell", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figRegime.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _cross_in_log_n(Ns: list[int], a: list[float], b: list[float]) -> tuple[float | None, float | None, str]:
    if not Ns:
        return None, None, "missing"

    diffs = [float(x - y) for x, y in zip(a, b)]
    if all(d > 0 for d in diffs):
        return None, None, "gt_max"
    if all(d <= 0 for d in diffs):
        return None, None, "lt_min"

    for i in range(len(Ns) - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        if d0 == 0.0:
            return float(Ns[i]), float(a[i]), "within_grid"
        if d0 > 0 and d1 <= 0:
            x0, x1 = math.log(float(Ns[i])), math.log(float(Ns[i + 1]))
            t = d0 / (d0 - d1)
            x = x0 + t * (x1 - x0)
            bias_at_cross = float(a[i] + t * (a[i + 1] - a[i]))
            return float(math.exp(x)), bias_at_cross, "within_grid"
    return None, None, "missing"


def build_crossover(df: pd.DataFrame) -> pd.DataFrame:
    sub = canonical(df)
    rows = []
    for gamma in sorted(sub["gamma"].unique()):
        panel = sub[sub["gamma"] == gamma]
        curves = {
            rep: (panel[panel["rep"] == rep]
                      .groupby("N")["ate_bias"]
                      .median()
                      .sort_index())
            for rep in REP_ORDER
        }
        Ns = sorted(set().union(*[set(c.index.tolist()) for c in curves.values() if not c.empty]))
        for left, right in CROSSOVER_PAIRS:
            left_curve = curves[left].reindex(Ns)
            right_curve = curves[right].reindex(Ns)
            n_cross, bias_cross, status = _cross_in_log_n(
                Ns,
                left_curve.to_numpy(dtype=float).tolist(),
                right_curve.to_numpy(dtype=float).tolist(),
            )
            rows.append(dict(
                gamma=float(gamma),
                tau_regime=CANONICAL_TAU,
                pair=f"{left}_vs_{right}",
                N_cross=_fmt_cross(n_cross, status),
                bias_at_cross=float(bias_cross) if bias_cross is not None else np.nan,
                N_cross_value=float(n_cross) if n_cross is not None else np.nan,
                status=status,
            ))
    out = pd.DataFrame(rows).sort_values(["tau_regime", "gamma", "pair"]).reset_index(drop=True)
    out.to_csv(CROSS_CSV, index=False)
    return out


def plot_crossover(df_cross: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    pair_labels = {
        "raw_vs_concept": "Raw vs Concept",
        "raw_vs_learned": "Raw vs Learned",
        "learned_vs_concept": "Learned vs Concept",
    }
    pair_colors = {
        "raw_vs_concept": "#222222",
        "raw_vs_learned": "#d62728",
        "learned_vs_concept": "#1f77b4",
    }
    status_markers = {"within_grid": "o", "lt_min": "<", "gt_max": ">"}

    gammas = sorted(df_cross["gamma"].unique())
    for pair in ["raw_vs_concept", "raw_vs_learned", "learned_vs_concept"]:
        sub = df_cross[df_cross["pair"] == pair].sort_values("gamma")
        xs = []
        ys = []
        for _, row in sub.iterrows():
            xs.append(float(row["gamma"]))
            status = row["status"]
            if status == "within_grid":
                ys.append(float(row["N_cross_value"]))
            elif status == "lt_min":
                ys.append(1000.0)
            else:
                ys.append(20000.0)

        ax.plot(xs, ys, color=pair_colors[pair], linewidth=1.5, alpha=0.8, label=pair_labels[pair])

        for _, row in sub.iterrows():
            x = float(row["gamma"])
            status = row["status"]
            if status == "within_grid":
                y = float(row["N_cross_value"])
            elif status == "lt_min":
                y = 1000.0
            else:
                y = 20000.0
            ax.scatter(x, y, s=54, color=pair_colors[pair], marker=status_markers[status], zorder=3)
            if status != "within_grid":
                ax.text(x, y * (0.93 if status == "gt_max" else 1.08), row["N_cross"],
                        color=pair_colors[pair], fontsize=8, ha="center",
                        va="top" if status == "gt_max" else "bottom")

    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("$N^{\\mathrm{cross}}$")
    ax.set_yscale("log")
    ax.set_xticks(gammas)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig Cross — Crossover Point vs Confounding Strength", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figCross.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", type=int, default=[1, 2, 3, 5],
                    help="subset of plan numbers to execute: 1 2 3 5")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_primary()
    print(f"[plan_v2] loaded {len(df)} rows from {RESULTS}")

    if 1 in args.plans:
        df_t3 = build_t3_threshold(df)
        plot_t3_threshold(df_t3)
        print(f"[plan_v2] wrote {T3_CSV} and figures/figT3.pdf")

    if 2 in args.plans:
        df_bv = build_bias_variance(df)
        plot_bias_variance(df_bv)
        print(f"[plan_v2] wrote {BV_CSV} and figures/figBV.pdf")

    if 3 in args.plans:
        df_regime = build_regime_diagram(df)
        plot_regime_diagram(df_regime)
        print(f"[plan_v2] wrote {REGIME_CSV} and figures/figRegime.pdf")

    if 5 in args.plans:
        df_cross = build_crossover(df)
        plot_crossover(df_cross)
        print(f"[plan_v2] wrote {CROSS_CSV}")
        print(df_cross.to_string(index=False))


if __name__ == "__main__":
    main()
