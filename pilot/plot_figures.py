"""
Figure generation for the Beyond Pattern Matching chess causality study.

Reads primary_results.csv and produces Figs 2–5 as described in
docs/experimental_notes.md §5:

    Fig 2: ATE bias vs N  (H1 money figure — sample efficiency)
    Fig 3: Heatmap of ATE-RMSE over (γ, N) at fixed estimator, per rep
    Fig 4: Nuisance-swap robustness bar chart (H2 figure)
    Fig 5: Propensity overlap histograms per rep at γ=0.6

Run:
    python plot_figures.py                     # all figures
    python plot_figures.py --figs 2 4          # subset
    python plot_figures.py --tau-regime heterogeneous  # non-default
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "outputs" / "primary_results.csv"
FIG_DIR = HERE / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["raw", "concept", "learned"]
REP_LABELS = {"raw": "Raw $S$ (1152-d)", "concept": "Concept $C$ (12-d)", "learned": "Learned $Z$ (32-d)"}
REP_COLORS = {"raw": "#d62728", "concept": "#2ca02c", "learned": "#1f77b4"}
REP_MARKERS = {"raw": "s", "concept": "o", "learned": "D"}

CANONICAL_GAMMA = 0.3
CANONICAL_TAU = "homogeneous"
CANONICAL_ESTIMATOR = "dr"
CANONICAL_OUTCOME = "lgbm"


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    df["ate_rmse_sq"] = df["ate_bias"] ** 2  # for RMSE aggregation
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: ATE bias vs N  (one panel per γ, three lines)
# ─────────────────────────────────────────────────────────────────────────────

def fig2(df: pd.DataFrame, tau_regime: str = CANONICAL_TAU) -> None:
    sub = df[
        (df["tau_regime"] == tau_regime)
        & (df["estimator"] == CANONICAL_ESTIMATOR)
        & (df["outcome_model"] == CANONICAL_OUTCOME)
    ]
    gammas = sorted(sub["gamma"].unique())
    n_panels = len(gammas)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, g in zip(axes, gammas):
        panel = sub[sub["gamma"] == g]
        for rep in REP_ORDER:
            rp = panel[panel["rep"] == rep]
            if rp.empty:
                continue
            agg = rp.groupby("N")["ate_bias"].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
            agg.columns = ["median", "q25", "q75"]
            agg = agg.sort_index()
            Ns = agg.index.to_numpy()
            ax.plot(Ns, agg["median"], marker=REP_MARKERS[rep], color=REP_COLORS[rep],
                    label=REP_LABELS[rep], linewidth=1.5, markersize=5)
            ax.fill_between(Ns, agg["q25"], agg["q75"], color=REP_COLORS[rep], alpha=0.15)
        ax.set_title(f"$\\gamma = {g}$", fontsize=11)
        ax.set_xlabel("Sample size $N$")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("ATE Bias  $|\\hat{\\tau} - \\tau^*|$")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Fig 2 — ATE Bias vs Sample Size ({tau_regime} $\\tau^*$)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / f"fig2_ate_bias_vs_N_{tau_regime}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Heatmap ATE-RMSE over (γ, N), one subplot per rep
# ─────────────────────────────────────────────────────────────────────────────

def fig3(df: pd.DataFrame, tau_regime: str = CANONICAL_TAU) -> None:
    sub = df[
        (df["tau_regime"] == tau_regime)
        & (df["estimator"] == CANONICAL_ESTIMATOR)
        & (df["outcome_model"] == CANONICAL_OUTCOME)
    ]
    reps = [r for r in REP_ORDER if r in sub["rep"].unique()]
    fig, axes = plt.subplots(1, len(reps), figsize=(4.5 * len(reps), 3.5), sharey=True)
    if len(reps) == 1:
        axes = [axes]

    vmin, vmax = None, None
    # First pass to get global color range
    for rep in reps:
        rp = sub[sub["rep"] == rep]
        piv = rp.groupby(["gamma", "N"])["ate_rmse_sq"].mean().reset_index()
        piv["rmse"] = np.sqrt(piv["ate_rmse_sq"])
        lo, hi = piv["rmse"].min(), piv["rmse"].max()
        vmin = lo if vmin is None else min(vmin, lo)
        vmax = hi if vmax is None else max(vmax, hi)

    for ax, rep in zip(axes, reps):
        rp = sub[sub["rep"] == rep]
        piv = rp.groupby(["gamma", "N"])["ate_rmse_sq"].mean().reset_index()
        piv["rmse"] = np.sqrt(piv["ate_rmse_sq"])
        heat = piv.pivot(index="gamma", columns="N", values="rmse")
        heat = heat.sort_index(ascending=False)
        sns.heatmap(heat, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=vmin, vmax=vmax, cbar=(rep == reps[-1]),
                    cbar_kws={"label": "ATE RMSE"} if rep == reps[-1] else {})
        ax.set_title(REP_LABELS[rep], fontsize=10)
        ax.set_xlabel("$N$")
        if rep == reps[0]:
            ax.set_ylabel("$\\gamma$")
        else:
            ax.set_ylabel("")

    fig.suptitle(f"Fig 3 — ATE RMSE Heatmap ($\\gamma$ × $N$, {tau_regime} $\\tau^*$)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / f"fig3_heatmap_{tau_regime}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Nuisance-swap robustness  (H2 figure)
#   ATE bias for DR-LightGBM vs DR-Linear outcome model, per rep
# ─────────────────────────────────────────────────────────────────────────────

def fig4(df: pd.DataFrame, tau_regime: str = CANONICAL_TAU, gamma: float = CANONICAL_GAMMA) -> None:
    sub = df[
        (df["tau_regime"] == tau_regime)
        & (df["gamma"] == gamma)
        & (df["estimator"] == "dr")
    ]
    reps = [r for r in REP_ORDER if r in sub["rep"].unique()]
    outcomes = sorted(sub["outcome_model"].unique())  # ['lgbm', 'linear']
    outcome_labels = {"lgbm": "LightGBM", "linear": "Linear (Ridge)"}

    # Aggregate: median bias per (rep, outcome_model) across seeds
    # Show at N=2000 where the robustness gap is most pronounced
    Ns = sorted(sub["N"].unique())
    canonical_N = 2000 if 2000 in Ns else Ns[-1]
    sub = sub[sub["N"] == canonical_N]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    width = 0.35
    x = np.arange(len(reps))

    for i, om in enumerate(outcomes):
        vals = []
        errs_lo, errs_hi = [], []
        for rep in reps:
            cell = sub[(sub["rep"] == rep) & (sub["outcome_model"] == om)]["ate_bias"]
            med = cell.median() if len(cell) else 0
            q25 = cell.quantile(0.25) if len(cell) else 0
            q75 = cell.quantile(0.75) if len(cell) else 0
            vals.append(med)
            errs_lo.append(med - q25)
            errs_hi.append(q75 - med)
        ax.bar(x + i * width - width / 2, vals, width,
               label=outcome_labels.get(om, om),
               yerr=[errs_lo, errs_hi], capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABELS[r] for r in reps], fontsize=9)
    ax.set_ylabel("ATE Bias  $|\\hat{\\tau} - \\tau^*|$")
    ax.set_title(f"Fig 4 — Nuisance-Swap Robustness ($N$={canonical_N:,}, $\\gamma$={gamma})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"fig4_robustness_{tau_regime}_g{gamma}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Propensity overlap histograms per rep
#   Requires re-fitting propensity models to get scores — or reading from
#   the primary results diagnostics. Since primary_results.csv only stores
#   ate_bias, we re-fit propensity on a canonical cell and plot p(T=1|X).
# ─────────────────────────────────────────────────────────────────────────────

def fig5(df_pool: pd.DataFrame | None = None,
         gamma: float = 0.6, N: int = 5000, seed: int = 0) -> None:
    """Propensity overlap histograms. Needs access to the decision pool."""
    try:
        from pilot import (
            DECISIONS_PATH, AGGRESSION_THRESHOLD,
            CONCEPT_FEATURES, CONTEXT_FEATURES, S_FEATURES,
            _make_propensity_model,
        )
        from primary import generate_outcomes_regime, fit_learned_Z, ALPHA_U
    except ImportError as e:
        print(f"  [fig5] skipped — import error: {e}")
        return

    if df_pool is None:
        df_pool = pd.read_parquet(DECISIONS_PATH)
        df_pool["T"] = (df_pool["agg"] >= AGGRESSION_THRESHOLD).astype(int)

    # Generate outcomes for this gamma
    df_y = generate_outcomes_regime(df_pool, gamma=gamma, tau_regime="homogeneous",
                                    alpha_U=ALPHA_U, seed=42)
    rng = np.random.default_rng(1000 + seed)
    N_actual = min(N, len(df_y))
    idx = rng.choice(len(df_y), size=N_actual, replace=False)
    sub = df_y.iloc[idx].reset_index(drop=True)
    T_obs = sub["T_obs"].to_numpy()
    X_ctx = sub[CONTEXT_FEATURES].to_numpy(dtype=np.float32)

    rep_data = {
        "raw": np.concatenate([sub[S_FEATURES].to_numpy(dtype=np.float32), X_ctx], axis=1),
        "concept": np.concatenate([sub[CONCEPT_FEATURES].to_numpy(dtype=np.float32), X_ctx], axis=1),
    }
    rep_data["learned"] = fit_learned_Z(rep_data["raw"], T_obs, dim=32, seed=seed)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, rep in zip(axes, REP_ORDER):
        X = rep_data[rep]
        clf = _make_propensity_model(X.shape[1], seed)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf.fit(Xs, T_obs)
        p = clf.predict_proba(Xs)[:, 1]

        ax.hist(p[T_obs == 0], bins=40, alpha=0.6, label="$T=0$", color="#1f77b4", density=True)
        ax.hist(p[T_obs == 1], bins=40, alpha=0.6, label="$T=1$", color="#d62728", density=True)
        ax.axvline(0.02, color="gray", linestyle="--", linewidth=0.8, label="clip = 0.02")
        ax.set_title(REP_LABELS[rep], fontsize=10)
        ax.set_xlabel("$\\hat{e}(X_{adj})$")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Fig 5 — Propensity Overlap ($\\gamma$={gamma}, $N$={N_actual:,})", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / f"fig5_propensity_overlap_g{gamma}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def fig6() -> None:
    """Fig 6 (appendix): Game-outcome ablation — SE comparison across reps."""
    ablation_csv = HERE / "outputs" / "ablation_game_outcome.csv"
    if not ablation_csv.exists():
        print("  [fig6] skipped — no ablation results. Run ablation_game_outcome.py first.")
        return
    ab = pd.read_csv(ablation_csv)
    dr = ab[(ab["estimator"] == "dr") & (ab["outcome_model"] == "lgbm")]
    if dr.empty:
        print("  [fig6] skipped — no DR-LightGBM rows in ablation results.")
        return

    Ns = sorted(dr["N"].unique())
    reps = [r for r in REP_ORDER if r in dr["rep"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: ATE point estimates with error bars (median ± IQR over seeds)
    ax = axes[0]
    width = 0.25
    x = np.arange(len(Ns))
    for j, rep in enumerate(reps):
        meds, lo, hi = [], [], []
        for N in Ns:
            cell = dr[(dr["N"] == N) & (dr["rep"] == rep)]["ate_hat"]
            m = cell.median()
            meds.append(m)
            lo.append(m - cell.quantile(0.25))
            hi.append(cell.quantile(0.75) - m)
        ax.bar(x + j * width - width, meds, width, label=REP_LABELS[rep],
               color=REP_COLORS[rep], yerr=[lo, hi], capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{N:,}" for N in Ns], fontsize=9)
    ax.set_xlabel("$N$")
    ax.set_ylabel("ATE Estimate")
    ax.set_title("(a) ATE point estimate (game-outcome $Y$)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: Median SE per rep vs N
    ax = axes[1]
    for rep in reps:
        se_vals = []
        for N in Ns:
            cell = dr[(dr["N"] == N) & (dr["rep"] == rep)]["ate_se"]
            se_vals.append(cell.median())
        ax.plot(Ns, se_vals, marker=REP_MARKERS[rep], color=REP_COLORS[rep],
                label=REP_LABELS[rep], linewidth=1.5, markersize=5)
    ax.set_xlabel("$N$")
    ax.set_ylabel("Median SE of ATE")
    ax.set_title("(b) Standard error (tighter = better adjustment)", fontsize=10)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 6 — Game-Outcome Sanity Ablation (Oracle-Correlation Defense)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig6_ablation_game_outcome.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--tau-regime", default=CANONICAL_TAU)
    args = ap.parse_args()

    if not RESULTS.exists():
        raise SystemExit(f"No results at {RESULTS}. Run primary.py first.")

    df = load()
    print(f"Loaded {len(df)} result rows from {RESULTS}")
    print(f"  reps: {sorted(df['rep'].unique())}")
    print(f"  N: {sorted(df['N'].unique())}")
    print(f"  gamma: {sorted(df['gamma'].unique())}")
    print(f"  tau_regime: {sorted(df['tau_regime'].unique())}")
    print()

    if 2 in args.figs:
        print("Fig 2 — ATE bias vs N:")
        fig2(df, tau_regime=args.tau_regime)

    if 3 in args.figs:
        print("Fig 3 — ATE-RMSE heatmap:")
        fig3(df, tau_regime=args.tau_regime)

    if 4 in args.figs:
        print("Fig 4 — Nuisance-swap robustness:")
        fig4(df, tau_regime=args.tau_regime)

    if 5 in args.figs:
        print("Fig 5 — Propensity overlap:")
        fig5()

    if 6 in args.figs:
        print("Fig 6 — Game-outcome ablation:")
        fig6()

    print("\nDone.")


if __name__ == "__main__":
    main()
