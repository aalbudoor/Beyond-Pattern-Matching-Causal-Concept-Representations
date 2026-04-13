"""
Concept-level causal queries (§"Concept-Level Causal Queries" in neurips.tex).

Three stratified CATE queries under the canonical setting
(heterogeneous tau*, gamma=0.3, alpha=0.3, N=5000, 10 seeds):

    Q1 — CATE by material balance bin {behind, equal, ahead}
         Planted ground truth: tau*_behind = tau*_equal = 0.1, tau*_ahead = 0.4
    Q2 — Conditional ATE at king_shield >= 75th percentile
         (explicitly *conditional*, not a do() intervention — see paper text)
    Q3 — CATE by space-control tercile {low, mid, high}
         Planted ground truth: tau*_* = E[tau*|bin], which is constant if space
         is orthogonal to material (false-positive check).

Implementation: for each stratum we fit a per-rep LinearDRLearner on the subset
with the standard adjustment set W = [representation, context] and read off the
ATE. This is equivalent to stratifying, not a genuine interventional do().

Run:
    python concept_queries.py
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pilot import (
    DECISIONS_PATH, AGGRESSION_THRESHOLD,
    CONCEPT_FEATURES, CONTEXT_FEATURES, S_FEATURES,
    MIN_PROPENSITY,
    _make_propensity_model,
)
from primary import (
    generate_outcomes_regime, fit_learned_Z, _make_outcome_model,
    ALPHA_U, TAU_HET_BASE, TAU_HET_HEIGHT,
)

OUT = Path(__file__).resolve().parent / "outputs"
QUERY_RESULTS = OUT / "concept_queries.csv"

# Canonical setting (heterogeneous tau* so material has planted heterogeneity)
CANONICAL_N = 5000
CANONICAL_GAMMA = 0.3
CANONICAL_TAU_REGIME = "heterogeneous"
CANONICAL_SEEDS = list(range(10))
REPS = ["raw", "concept", "learned"]

# =============================================================================
# Stratum definitions
# =============================================================================

def material_bin(df: pd.DataFrame) -> pd.Series:
    """Q1 stratifier: sign of material balance."""
    m = df["c_material"].to_numpy()
    out = np.full(len(m), "equal", dtype=object)
    out[m < -0.5] = "behind"
    out[m > 0.5] = "ahead"
    return pd.Series(out, index=df.index)


def king_shield_mask(df: pd.DataFrame, q: float = 0.75) -> np.ndarray:
    """Q2 conditioning: boolean mask for king_shield >= q-percentile."""
    threshold = df["c_king_shield"].quantile(q)
    return (df["c_king_shield"] >= threshold).to_numpy()


def space_tercile(df: pd.DataFrame) -> pd.Series:
    """Q3 stratifier: tercile of space-control feature."""
    s = df["c_space"]
    t33, t67 = s.quantile(0.33), s.quantile(0.67)
    out = np.full(len(s), "mid", dtype=object)
    out[s <= t33] = "low"
    out[s > t67] = "high"
    return pd.Series(out, index=df.index)


# =============================================================================
# Per-stratum ATE fit (reuses primary.py's DR-LGBM pipeline)
# =============================================================================

def fit_ate_on_subset(X_adj: np.ndarray, T: np.ndarray, Y: np.ndarray,
                      seed: int) -> tuple[float, float]:
    """Standard DR-LightGBM ATE + inferential SE on a sub-population."""
    from econml.dr import LinearDRLearner
    if len(Y) < 50 or T.sum() < 5 or (1 - T).sum() < 5:
        return float("nan"), float("nan")
    est = LinearDRLearner(
        model_propensity=_make_propensity_model(X_adj.shape[1], seed),
        model_regression=_make_outcome_model("lgbm", seed),
        min_propensity=MIN_PROPENSITY,
        cv=5,
        random_state=seed,
    )
    try:
        est.fit(Y=Y, T=T, X=None, W=X_adj)
        ate = float(est.ate(X=None, T0=0, T1=1))
        try:
            inf = est.ate_inference(X=None, T0=0, T1=1)
            se = float(inf.stderr_mean)
        except Exception:
            se = float("nan")
        return ate, se
    except Exception:
        return float("nan"), float("nan")


# =============================================================================
# Ground-truth CATE per stratum under heterogeneous tau*(X)
# =============================================================================

def planted_cate_in_subset(df_sub: pd.DataFrame) -> float:
    """True CATE on a sub-population for the heterogeneous regime.

    tau*(X) = TAU_HET_BASE + TAU_HET_HEIGHT * 1{c_material > 0}
    """
    frac_ahead = float((df_sub["c_material"] > 0).mean())
    return TAU_HET_BASE + TAU_HET_HEIGHT * frac_ahead


# =============================================================================
# Result row
# =============================================================================

@dataclass
class QueryRun:
    query: str          # Q1, Q2, Q3
    stratum: str        # behind/equal/ahead, high_king, low/mid/high
    rep: str
    seed: int
    n_sub: int
    ate_hat: float
    ate_se: float
    ate_true: float
    ate_bias: float
    wall_s: float


CELL_KEYS = ("query", "stratum", "rep", "seed")


def load_existing() -> set:
    if not QUERY_RESULTS.exists():
        return set()
    df = pd.read_csv(QUERY_RESULTS)
    return set(map(tuple, df[list(CELL_KEYS)].itertuples(index=False, name=None))) if not df.empty else set()


def append_row(row: QueryRun) -> None:
    new_file = not QUERY_RESULTS.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(QUERY_RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([fld.name for fld in fields(QueryRun)])
        w.writerow([getattr(row, fld.name) for fld in fields(QueryRun)])


# =============================================================================
# Main loop
# =============================================================================

def run_queries(n: int = CANONICAL_N,
                seeds: list[int] | None = None,
                king_q: float = 0.75) -> None:
    if not DECISIONS_PATH.exists():
        raise SystemExit(f"[queries] no decisions at {DECISIONS_PATH}. Run pilot.py first.")
    if seeds is None:
        seeds = CANONICAL_SEEDS

    df_pool = pd.read_parquet(DECISIONS_PATH)
    df_pool["T"] = (df_pool["agg"] >= AGGRESSION_THRESHOLD).astype(int)
    print(f"[queries] pool loaded: {len(df_pool)} decisions")

    # Generate outcomes once (heterogeneous regime, canonical gamma)
    df_y = generate_outcomes_regime(
        df_pool, gamma=CANONICAL_GAMMA, tau_regime=CANONICAL_TAU_REGIME,
        alpha_U=ALPHA_U, seed=42,
    )

    done = load_existing()

    # Build all (query, stratum, rep, seed) cells
    queries_cells = []
    for seed in seeds:
        for rep in REPS:
            # Q1: material strata
            for stratum in ("behind", "equal", "ahead"):
                queries_cells.append(("Q1", stratum, rep, seed))
            # Q2: king_shield conditional
            queries_cells.append(("Q2", "high_king_shield", rep, seed))
            # Q3: space terciles
            for stratum in ("low", "mid", "high"):
                queries_cells.append(("Q3", stratum, rep, seed))

    remaining = [c for c in queries_cells if c not in done]
    print(f"[queries] total cells = {len(queries_cells)}, done = {len(queries_cells) - len(remaining)}, remaining = {len(remaining)}")

    pbar = tqdm(total=len(queries_cells), initial=len(queries_cells) - len(remaining), desc="queries")

    # Cache the per-seed sub-sample once per seed (reused across queries and reps)
    seed_cache: dict = {}

    for (query, stratum, rep, seed) in remaining:
        if seed not in seed_cache:
            rng = np.random.default_rng(1000 + seed)
            idx = rng.choice(len(df_y), size=n, replace=False)
            sub = df_y.iloc[idx].reset_index(drop=True)

            X_ctx = sub[CONTEXT_FEATURES].to_numpy(dtype=np.float32)
            X_raw = np.concatenate(
                [sub[S_FEATURES].to_numpy(dtype=np.float32), X_ctx], axis=1,
            )
            X_con = np.concatenate(
                [sub[CONCEPT_FEATURES].to_numpy(dtype=np.float32), X_ctx], axis=1,
            )
            T_obs = sub["T_obs"].to_numpy()
            X_lrn = fit_learned_Z(X_raw, T_obs, dim=32, seed=seed)

            seed_cache[seed] = dict(
                sub=sub,
                T=T_obs,
                Y=sub["Y_obs"].to_numpy(),
                X={"raw": X_raw, "concept": X_con, "learned": X_lrn},
                mat_bin=material_bin(sub),
                king_mask=king_shield_mask(sub, q=king_q),
                space_bin=space_tercile(sub),
            )
            # cap cache
            if len(seed_cache) > 4:
                oldest = next(iter(seed_cache))
                if oldest != seed:
                    del seed_cache[oldest]

        s = seed_cache[seed]

        # Select sub-population mask per query
        if query == "Q1":
            mask = (s["mat_bin"] == stratum).to_numpy()
        elif query == "Q2":
            mask = s["king_mask"]
        elif query == "Q3":
            mask = (s["space_bin"] == stratum).to_numpy()
        else:
            raise ValueError(query)

        X_adj_full = s["X"][rep]
        X_sub = X_adj_full[mask]
        T_sub = s["T"][mask]
        Y_sub = s["Y"][mask]
        df_sub = s["sub"].iloc[mask]

        t0 = time.time()
        ate, se = fit_ate_on_subset(X_sub, T_sub, Y_sub, seed=seed)
        dt = time.time() - t0

        ate_true = planted_cate_in_subset(df_sub)
        bias = abs(ate - ate_true) if not np.isnan(ate) else float("nan")

        append_row(QueryRun(
            query=query, stratum=stratum, rep=rep, seed=seed,
            n_sub=int(mask.sum()),
            ate_hat=ate, ate_se=se, ate_true=ate_true, ate_bias=bias, wall_s=dt,
        ))
        pbar.update(1)

    pbar.close()
    print(f"[queries] done. results -> {QUERY_RESULTS}")


# =============================================================================
# Report + figure
# =============================================================================

def report() -> None:
    if not QUERY_RESULTS.exists():
        print("[queries] no results to report.")
        return
    df = pd.read_csv(QUERY_RESULTS)
    print("\n" + "=" * 64)
    print("CONCEPT-LEVEL CAUSAL QUERIES — SUMMARY")
    print("=" * 64)

    for query in ("Q1", "Q2", "Q3"):
        qdf = df[df["query"] == query]
        if qdf.empty:
            continue
        label = {"Q1": "Q1: CATE by material balance",
                 "Q2": "Q2: Conditional ATE at high king_shield (75th pctile+)",
                 "Q3": "Q3: CATE by space tercile (false-positive check)"}[query]
        print(f"\n{label}")
        for stratum in sorted(qdf["stratum"].unique()):
            print(f"  stratum = {stratum}:")
            for rep in REPS:
                cell = qdf[(qdf["stratum"] == stratum) & (qdf["rep"] == rep)]
                if cell.empty:
                    continue
                ate_med = cell["ate_hat"].median()
                se_med = cell["ate_se"].median()
                bias_med = cell["ate_bias"].median()
                tau_true = cell["ate_true"].median()
                n_med = int(cell["n_sub"].median())
                print(f"    {rep:>8s}: ATE={ate_med:+.3f}  SE={se_med:.3f}  "
                      f"true={tau_true:.3f}  |bias|={bias_med:.3f}  n_sub={n_med}")


def plot_figure() -> None:
    """Figure Q-CATE: 3-panel figure with one subplot per query."""
    if not QUERY_RESULTS.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(QUERY_RESULTS)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rep_colors = {"raw": "#d62728", "concept": "#2ca02c", "learned": "#1f77b4"}
    rep_labels = {"raw": "Raw $S$", "concept": "Concept $C$", "learned": "Learned $Z$"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    def _panel(ax, query: str, stratum_order: list[str], title: str, show_truth: bool):
        qdf = df[df["query"] == query]
        x = np.arange(len(stratum_order))
        width = 0.25
        for j, rep in enumerate(REPS):
            meds, errs_lo, errs_hi = [], [], []
            for stratum in stratum_order:
                cell = qdf[(qdf["stratum"] == stratum) & (qdf["rep"] == rep)]["ate_hat"].dropna()
                if cell.empty:
                    meds.append(np.nan); errs_lo.append(0); errs_hi.append(0); continue
                m = cell.median()
                meds.append(m)
                errs_lo.append(m - cell.quantile(0.25))
                errs_hi.append(cell.quantile(0.75) - m)
            ax.bar(x + j * width - width, meds, width, label=rep_labels[rep],
                   color=rep_colors[rep], yerr=[errs_lo, errs_hi], capsize=3, alpha=0.85)
        # Planted-truth reference line(s)
        if show_truth:
            truths = []
            for stratum in stratum_order:
                cell = qdf[(qdf["stratum"] == stratum)]["ate_true"]
                if not cell.empty:
                    truths.append(cell.median())
                else:
                    truths.append(np.nan)
            for xi, t in zip(x, truths):
                if not np.isnan(t):
                    ax.hlines(t, xi - 1.5 * width, xi + 1.5 * width,
                              colors="black", linestyles="--", linewidth=1.2)
            ax.plot([], [], "k--", label="planted $\\tau^*$")
        ax.set_xticks(x)
        ax.set_xticklabels(stratum_order, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("CATE estimate")
        ax.grid(True, axis="y", alpha=0.3)

    _panel(axes[0], "Q1", ["behind", "equal", "ahead"],
           "(a) Q1: CATE by material balance", show_truth=True)
    axes[0].legend(fontsize=7, loc="upper left")

    _panel(axes[1], "Q2", ["high_king_shield"],
           "(b) Q2: Conditional ATE at high king-shield", show_truth=True)

    _panel(axes[2], "Q3", ["low", "mid", "high"],
           "(c) Q3: CATE by space tercile (false-positive check)", show_truth=True)

    fig.suptitle("Fig Q-CATE — Concept-Level Causal Queries (heterogeneous $\\tau^*$, $\\gamma$=0.3, $N$=5,000)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    out = fig_dir / "figQ_concept_queries.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[queries] figure -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=CANONICAL_N)
    ap.add_argument("--seeds", nargs="+", type=int, default=CANONICAL_SEEDS)
    ap.add_argument("--king-shield-q", type=float, default=0.75)
    ap.add_argument("--force", action="store_true", help="delete existing CSV before re-running")
    args = ap.parse_args()

    if args.force and QUERY_RESULTS.exists():
        QUERY_RESULTS.unlink()

    run_queries(n=args.n, seeds=args.seeds, king_q=args.king_shield_q)
    report()
    plot_figure()
