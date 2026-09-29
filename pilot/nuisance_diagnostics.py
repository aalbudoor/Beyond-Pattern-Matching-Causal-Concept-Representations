"""
Plan 8: nuisance-fit diagnostics.

Standalone canonical-grid re-run that records:
    - propensity AUC (5-fold out-of-fold)
    - outcome R^2 (5-fold out-of-fold, averaged across treatment arms)

Outputs:
    outputs/nuisance_diagnostics.csv
    outputs/figures/figNuisance.pdf

Run:
    python nuisance_diagnostics.py
    python nuisance_diagnostics.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from pilot import CONCEPT_FEATURES, CONTEXT_FEATURES, S_FEATURES, _make_propensity_model
from primary import (
    OUT,
    _make_outcome_model,
    fit_learned_Z,
    generate_outcomes_regime,
    load_pool,
)

RESULTS_PATH = OUT / "nuisance_diagnostics.csv"
FIG_DIR = OUT / "figures"

DEFAULT_NS = [1000, 2000, 5000, 10000, 20000]
DEFAULT_GAMMAS = [0.3]
DEFAULT_TAU_REGIMES = ["homogeneous"]
DEFAULT_REPS = ["raw", "concept", "learned"]
DEFAULT_SEEDS = list(range(10))

REP_LABELS = {"raw": "Raw $S$", "concept": "Concept $C$", "learned": "Learned $Z$"}
REP_COLORS = {"raw": "#d62728", "concept": "#2ca02c", "learned": "#1f77b4"}
REP_MARKERS = {"raw": "s", "concept": "o", "learned": "D"}


@dataclass
class DiagRun:
    N: int
    gamma: float
    tau_regime: str
    rep: str
    seed: int
    prop_auc_cv: float
    outcome_r2_cv: float
    outcome_r2_t0: float
    outcome_r2_t1: float


CELL_KEYS = ("N", "gamma", "tau_regime", "rep", "seed")


def load_existing() -> set[tuple]:
    if not RESULTS_PATH.exists():
        return set()
    df = pd.read_csv(RESULTS_PATH)
    if df.empty:
        return set()
    return set(map(tuple, df[list(CELL_KEYS)].itertuples(index=False, name=None)))


def append_row(row: DiagRun) -> None:
    new_file = not RESULTS_PATH.exists()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([fld.name for fld in fields(DiagRun)])
        w.writerow([getattr(row, fld.name) for fld in fields(DiagRun)])


def propensity_auc_cv(X: np.ndarray, T: np.ndarray, seed: int) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = np.zeros(len(T), dtype=float)
    for fold, (tr, te) in enumerate(cv.split(X, T)):
        model = _make_propensity_model(X.shape[1], seed + fold)
        model.fit(X[tr], T[tr])
        probs[te] = model.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(T, probs))


def _arm_r2_cv(X: np.ndarray, Y: np.ndarray, seed: int) -> float:
    if len(Y) < 10:
        return float("nan")
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.full(len(Y), np.nan, dtype=float)
    for fold, (tr, te) in enumerate(cv.split(X)):
        model = _make_outcome_model("lgbm", seed + fold)
        model.fit(X[tr], Y[tr])
        preds[te] = model.predict(X[te])
    mask = ~np.isnan(preds)
    if mask.sum() < 10:
        return float("nan")
    return float(r2_score(Y[mask], preds[mask]))


def outcome_r2_cv(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int) -> tuple[float, float, float]:
    t0 = T == 0
    t1 = T == 1
    r2_t0 = _arm_r2_cv(X[t0], Y[t0], seed=seed) if t0.sum() >= 10 else float("nan")
    r2_t1 = _arm_r2_cv(X[t1], Y[t1], seed=seed + 17) if t1.sum() >= 10 else float("nan")

    vals, weights = [], []
    if not np.isnan(r2_t0):
        vals.append(r2_t0)
        weights.append(int(t0.sum()))
    if not np.isnan(r2_t1):
        vals.append(r2_t1)
        weights.append(int(t1.sum()))
    overall = float(np.average(vals, weights=weights)) if vals else float("nan")
    return overall, r2_t0, r2_t1


def run_grid(df_pool: pd.DataFrame,
             Ns: list[int],
             gammas: list[float],
             tau_regimes: list[str],
             reps: list[str],
             seeds: list[int],
             dry_run: bool) -> None:
    done = load_existing()
    cells = []
    for tau_regime in tau_regimes:
        for gamma in gammas:
            for N in Ns:
                if N > len(df_pool):
                    continue
                for seed in seeds:
                    for rep in reps:
                        cells.append(dict(
                            N=N,
                            gamma=gamma,
                            tau_regime=tau_regime,
                            rep=rep,
                            seed=seed,
                        ))

    remaining = [c for c in cells if tuple(c[k] for k in CELL_KEYS) not in done]
    print(f"[nuisance] grid total = {len(cells)}; done = {len(cells) - len(remaining)}; remaining = {len(remaining)}")
    if dry_run:
        return

    outcome_cache: dict[tuple[float, str], pd.DataFrame] = {}
    sub_cache: dict[tuple, dict] = {}
    pbar = tqdm(total=len(cells), initial=len(cells) - len(remaining), desc="nuisance")

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
            X_raw = np.concatenate([sub[S_FEATURES].to_numpy(dtype=np.float32), X_context], axis=1)
            X_con = np.concatenate([sub[CONCEPT_FEATURES].to_numpy(dtype=np.float32), X_context], axis=1)
            T_obs = sub["T_obs"].to_numpy(dtype=np.int64)
            X_lrn = fit_learned_Z(X_raw, T_obs, dim=32, seed=cell["seed"])

            sub_cache[sub_key] = dict(
                Y=sub["Y_obs"].to_numpy(dtype=np.float32),
                T=T_obs,
                X={"raw": X_raw, "concept": X_con, "learned": X_lrn},
            )
            if len(sub_cache) > 6:
                oldest = next(iter(sub_cache))
                if oldest != sub_key:
                    del sub_cache[oldest]

        s = sub_cache[sub_key]
        X_adj = s["X"][cell["rep"]]
        prop_auc = propensity_auc_cv(X_adj, s["T"], seed=cell["seed"])
        out_r2, out_r2_t0, out_r2_t1 = outcome_r2_cv(X_adj, s["T"], s["Y"], seed=cell["seed"])
        append_row(DiagRun(
            N=cell["N"],
            gamma=cell["gamma"],
            tau_regime=cell["tau_regime"],
            rep=cell["rep"],
            seed=cell["seed"],
            prop_auc_cv=prop_auc,
            outcome_r2_cv=out_r2,
            outcome_r2_t0=out_r2_t0,
            outcome_r2_t1=out_r2_t1,
        ))
        pbar.update(1)

    pbar.close()
    print(f"[nuisance] results -> {RESULTS_PATH}")


def plot_figure(gamma: float = 0.3, tau_regime: str = "homogeneous") -> None:
    if not RESULTS_PATH.exists():
        return
    df = pd.read_csv(RESULTS_PATH)
    sub = df[(df["gamma"] == gamma) & (df["tau_regime"] == tau_regime)]
    if sub.empty:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)

    for rep in DEFAULT_REPS:
        rp = sub[sub["rep"] == rep]
        if rp.empty:
            continue
        agg = rp.groupby("N")[["prop_auc_cv", "outcome_r2_cv"]].median().sort_index()
        Ns = agg.index.to_numpy(dtype=float)
        axes[0].plot(Ns, agg["prop_auc_cv"], marker=REP_MARKERS[rep], color=REP_COLORS[rep],
                     linewidth=1.6, markersize=5, label=REP_LABELS[rep])
        axes[1].plot(Ns, agg["outcome_r2_cv"], marker=REP_MARKERS[rep], color=REP_COLORS[rep],
                     linewidth=1.6, markersize=5, label=REP_LABELS[rep])

    axes[0].set_title("(a) Propensity AUC", fontsize=10)
    axes[0].set_ylabel("5-fold AUC")
    axes[1].set_title("(b) Outcome $R^2$", fontsize=10)
    axes[1].set_ylabel("5-fold $R^2$")
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("Sample size $N$")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    fig.suptitle("Fig Nuisance — Mechanistic Diagnostics on the Canonical Grid", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "figNuisance.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[nuisance] figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    ap.add_argument("--ns", nargs="+", type=int, default=DEFAULT_NS)
    ap.add_argument("--tau-regimes", nargs="+", default=DEFAULT_TAU_REGIMES,
                    choices=["homogeneous", "heterogeneous"])
    ap.add_argument("--reps", nargs="+", default=DEFAULT_REPS, choices=DEFAULT_REPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df_pool = load_pool()
    run_grid(
        df_pool=df_pool,
        Ns=args.ns,
        gammas=args.gammas,
        tau_regimes=args.tau_regimes,
        reps=args.reps,
        seeds=args.seeds,
        dry_run=args.dry_run,
    )
    if not args.dry_run and len(args.gammas) == 1 and len(args.tau_regimes) == 1:
        plot_figure(gamma=args.gammas[0], tau_regime=args.tau_regimes[0])


if __name__ == "__main__":
    main()
