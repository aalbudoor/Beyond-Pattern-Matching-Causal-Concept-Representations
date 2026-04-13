"""
Plan 7b: concept-plus-residual with supervised residual compression.

Constructs a low-dimensional representation:

    X_conplus = [C, T-PLS_16(S - E[S|C]), context]

and runs the canonical restricted grid from experimental_plan_v2.md:
    - N in {1000, 2000, 5000, 10000, 20000}
    - gamma in {0.3, 0.6}
    - tau_regime in {homogeneous}
    - estimator = DR-Learner
    - outcome model = LightGBM
    - 10 seeds

Outputs:
    outputs/conplus_supervised_results.csv
    outputs/figures/figConplus.pdf

Run:
    python conplus_supervised.py
    python conplus_supervised.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA  # noqa: F401
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pilot import OUT, CONCEPT_FEATURES, CONTEXT_FEATURES, S_FEATURES
from primary import dr_ate, generate_outcomes_regime, load_pool

CONPLUS_RESULTS = OUT / "conplus_supervised_results.csv"
PRIMARY_RESULTS = OUT / "primary_results.csv"
FIG_DIR = OUT / "figures"

DEFAULT_NS = [1000, 2000, 5000, 10000, 20000]
DEFAULT_GAMMAS = [0.3, 0.6]
DEFAULT_TAU_REGIMES = ["homogeneous"]
DEFAULT_SEEDS = list(range(10))
RESIDUAL_DIM = 16

REP_LABELS = {
    "raw": "Raw $S$",
    "concept": "Concept $C$",
    "learned": "Learned $Z$",
    "conplus_tpls": "Concept + T-PLS residual",
}
REP_COLORS = {
    "raw": "#d62728",
    "concept": "#2ca02c",
    "learned": "#1f77b4",
    "conplus_tpls": "#9467bd",
}
REP_MARKERS = {"raw": "s", "concept": "o", "learned": "D", "conplus_tpls": "v"}


@dataclass
class ConplusRun:
    N: int
    gamma: float
    tau_regime: str
    rep: str
    estimator: str
    outcome_model: str
    seed: int
    ate_hat: float
    ate_true: float
    ate_bias: float
    wall_s: float


CELL_KEYS = ("N", "gamma", "tau_regime", "rep", "estimator", "outcome_model", "seed")


def load_existing() -> set[tuple]:
    if not CONPLUS_RESULTS.exists():
        return set()
    df = pd.read_csv(CONPLUS_RESULTS)
    if df.empty:
        return set()
    return set(map(tuple, df[list(CELL_KEYS)].itertuples(index=False, name=None)))


def append_row(row: ConplusRun) -> None:
    new_file = not CONPLUS_RESULTS.exists()
    CONPLUS_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(CONPLUS_RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([fld.name for fld in fields(ConplusRun)])
        w.writerow([getattr(row, fld.name) for fld in fields(ConplusRun)])


def fit_conplus_tpls(S: np.ndarray,
                     C: np.ndarray,
                     X_context: np.ndarray,
                     T: np.ndarray,
                     seed: int,
                     n_components: int = RESIDUAL_DIM) -> np.ndarray:
    """T-supervised residual compression."""
    _ = seed

    reg = LinearRegression().fit(C, S)
    residual = S - reg.predict(C)

    scaler = StandardScaler(with_mean=True, with_std=True)
    residual_std = scaler.fit_transform(residual)

    k = min(n_components, residual_std.shape[1], max(residual_std.shape[0] - 1, 1))
    if k <= 0:
        return np.concatenate([C.astype(np.float32), X_context.astype(np.float32)], axis=1)

    pls = PLSRegression(n_components=k, scale=False, max_iter=500)
    pls.fit(residual_std, T.astype(float).reshape(-1, 1))
    R = pls.transform(residual_std).astype(np.float32)

    return np.concatenate([C.astype(np.float32), R, X_context.astype(np.float32)], axis=1)


def run_grid(df_pool: pd.DataFrame,
             Ns: list[int],
             gammas: list[float],
             tau_regimes: list[str],
             seeds: list[int],
             dry_run: bool) -> None:
    cells = []
    done = load_existing()
    for tau_regime in tau_regimes:
        for gamma in gammas:
            for N in Ns:
                if N > len(df_pool):
                    continue
                for seed in seeds:
                    cells.append(dict(
                        N=N,
                        gamma=gamma,
                        tau_regime=tau_regime,
                        rep="conplus_tpls",
                        estimator="dr",
                        outcome_model="lgbm",
                        seed=seed,
                    ))

    remaining = [c for c in cells if tuple(c[k] for k in CELL_KEYS) not in done]
    print(f"[conplus_sup] grid total = {len(cells)}; done = {len(cells) - len(remaining)}; remaining = {len(remaining)}")
    if dry_run:
        return

    outcome_cache: dict[tuple[float, str], pd.DataFrame] = {}
    sub_cache: dict[tuple, dict] = {}
    pbar = tqdm(total=len(cells), initial=len(cells) - len(remaining), desc="conplus_tpls fits")

    for cell in remaining:
        y_key = (cell["gamma"], cell["tau_regime"])
        if y_key not in outcome_cache:
            outcome_cache[y_key] = generate_outcomes_regime(
                df_pool,
                gamma=cell["gamma"],
                tau_regime=cell["tau_regime"],
                alpha_U=0.3,
                seed=42,
            )
        df_y = outcome_cache[y_key]

        sub_key = (cell["gamma"], cell["tau_regime"], cell["N"], cell["seed"])
        if sub_key not in sub_cache:
            rng = np.random.default_rng(1000 + cell["seed"])
            idx = rng.choice(len(df_y), size=cell["N"], replace=False)
            sub = df_y.iloc[idx].reset_index(drop=True)
            X_context = sub[CONTEXT_FEATURES].to_numpy(dtype=np.float32)
            S = sub[S_FEATURES].to_numpy(dtype=np.float32)
            C = sub[CONCEPT_FEATURES].to_numpy(dtype=np.float32)
            X_conplus = fit_conplus_tpls(
                S,
                C,
                X_context,
                T=sub["T_obs"].to_numpy(dtype=np.int64),
                seed=cell["seed"],
            )
            sub_cache[sub_key] = dict(
                Y=sub["Y_obs"].to_numpy(dtype=np.float32),
                T=sub["T_obs"].to_numpy(dtype=np.int64),
                tau_true=float(sub["_tau_star_mean"].iloc[0]),
                X=X_conplus,
            )
            if len(sub_cache) > 6:
                oldest = next(iter(sub_cache))
                if oldest != sub_key:
                    del sub_cache[oldest]

        s = sub_cache[sub_key]
        t0 = time.time()
        try:
            ate = dr_ate(s["X"], s["T"], s["Y"], seed=cell["seed"], outcome="lgbm")
        except Exception as e:
            tqdm.write(f"FAIL {cell}: {e}")
            pbar.update(1)
            continue
        dt = time.time() - t0
        append_row(ConplusRun(
            N=cell["N"],
            gamma=cell["gamma"],
            tau_regime=cell["tau_regime"],
            rep="conplus_tpls",
            estimator="dr",
            outcome_model="lgbm",
            seed=cell["seed"],
            ate_hat=float(ate),
            ate_true=s["tau_true"],
            ate_bias=abs(float(ate) - s["tau_true"]),
            wall_s=dt,
        ))
        pbar.update(1)

    pbar.close()
    print(f"[conplus_sup] results -> {CONPLUS_RESULTS}")


def plot_figure() -> None:
    conplus_pca_path = OUT / "conplus_results.csv"
    conplus_tpls_path = CONPLUS_RESULTS
    if not all(p.exists() for p in [PRIMARY_RESULTS, conplus_tpls_path]):
        print("[conplus_sup] skipped plot — missing primary or conplus_tpls results.")
        return

    base = pd.read_csv(PRIMARY_RESULTS)
    base = base[
        (base["tau_regime"] == "homogeneous")
        & (base["estimator"] == "dr")
        & (base["outcome_model"] == "lgbm")
        & (base["gamma"].isin(DEFAULT_GAMMAS))
    ].copy()
    cp_tpls = pd.read_csv(conplus_tpls_path)
    cp_tpls = cp_tpls[
        (cp_tpls["tau_regime"] == "homogeneous")
        & (cp_tpls["gamma"].isin(DEFAULT_GAMMAS))
    ].copy()
    if cp_tpls.empty:
        print("[conplus_sup] skipped plot — no conplus_tpls rows found.")
        return

    dfs = [base, cp_tpls]
    if conplus_pca_path.exists():
        cp_pca = pd.read_csv(conplus_pca_path)
        cp_pca = cp_pca[cp_pca["rep"] == "conplus"].copy()
        cp_pca["rep"] = "conplus_pca"
        dfs.append(cp_pca)

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    rep_order = ["raw", "concept", "learned", "conplus_pca", "conplus_tpls"]
    rep_labels = {
        **REP_LABELS,
        "conplus_pca": "Concept + PCA residual",
        "conplus_tpls": "Concept + T-PLS residual",
    }
    rep_colors = {
        **REP_COLORS,
        "conplus_pca": "#ff7f0e",
        "conplus_tpls": "#9467bd",
    }
    rep_markers = {
        **REP_MARKERS,
        "conplus_pca": "^",
        "conplus_tpls": "v",
    }
    gammas = sorted(merged["gamma"].unique())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.6 * len(gammas), 3.7), sharey=True)
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        panel = merged[merged["gamma"] == gamma]
        for rep in rep_order:
            rp = panel[panel["rep"] == rep]
            if rp.empty:
                continue
            agg = rp.groupby("N")["ate_bias"].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            ).sort_index()
            Ns = agg.index.to_numpy(dtype=float)
            ax.plot(
                Ns,
                agg["median"],
                marker=rep_markers[rep],
                color=rep_colors[rep],
                linewidth=1.6,
                markersize=5,
                label=rep_labels[rep],
            )
            ax.fill_between(Ns, agg["q25"], agg["q75"], color=rep_colors[rep], alpha=0.14)
        ax.set_title(f"$\\gamma = {gamma}$", fontsize=11)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("Sample size $N$")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("ATE Bias  $|\\hat{\\tau} - \\tau^*|$")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Fig Conplus — Concept Plus Residual on the Canonical Grid", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "figConplus.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[conplus_sup] figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    ap.add_argument("--ns", nargs="+", type=int, default=DEFAULT_NS)
    ap.add_argument("--tau-regimes", nargs="+", default=DEFAULT_TAU_REGIMES,
                    choices=["homogeneous", "heterogeneous"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df_pool = load_pool()
    run_grid(
        df_pool=df_pool,
        Ns=args.ns,
        gammas=args.gammas,
        tau_regimes=args.tau_regimes,
        seeds=args.seeds,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        plot_figure()


if __name__ == "__main__":
    main()
