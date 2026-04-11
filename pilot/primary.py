"""
Primary grid runner for the Beyond Pattern Matching chess causality study.

Runs the full locked grid from experimental_notes.md:
    - N           in {1000, 2000, 5000, 10000, 20000}
    - gamma       in {0.0, 0.3, 0.6}                       (outcome-side latent confounding)
    - alpha_U     = 0.3                                    (treatment-side latent confounding, fixed)
    - tau_regime  in {homogeneous, heterogeneous}
    - reps        in {raw, concept, learned}               (learned = linear deconfounding Z)
    - estimators  in {dr-lgbm (primary), tlearner-lgbm (baseline), dr-linear (robustness swap)}
    - seeds       0..9 (10 seeds)

Total cells: 5 N x 3 gamma x 2 tau_regime x 3 rep x 3 estimator x 10 seeds = 2700 fits.

Resumable: writes outputs/primary_results.csv incrementally and skips cells already present.
Pre-DGP T is recomputed from the cached `agg` score at the new AGGRESSION_THRESHOLD,
so primary.py doesn't depend on the threshold pilot.py was last extracted with.

Run:
    python primary.py                  # full grid, resumes
    python primary.py --reps raw concept   # subset of representations
    python primary.py --gammas 0.3 0.6     # subset of confounding levels
    python primary.py --dry-run            # print grid + counts, fit nothing
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse all feature / DGP infrastructure from the pilot module
from pilot import (
    CACHE, OUT, DECISIONS_PATH,
    GROUP_A_FEATURES, GROUP_B_FEATURES,
    CONCEPT_FEATURES, CONTEXT_FEATURES, S_FEATURES,
    AGGRESSION_THRESHOLD,
    MIN_PROPENSITY,
    _make_propensity_model,
    _standardize,
)

PRIMARY_RESULTS = OUT / "primary_results.csv"
TARGET_POOL = 25_000      # need at least this many decisions for N=20k + buffer
ALPHA_U = 0.3
TAU_HOMO = 0.3
TAU_HET_BASE = 0.1        # tau*(X) = 0.1 + 0.3 * 1{material > 0}
TAU_HET_HEIGHT = 0.3


# =============================================================================
# Decision pool: load cached parquet, recompute T at the current threshold
# =============================================================================

def load_pool() -> pd.DataFrame:
    if not DECISIONS_PATH.exists():
        raise SystemExit(
            f"[primary] no decision cache at {DECISIONS_PATH}. "
            "Run pilot.py first to extract decisions."
        )
    df = pd.read_parquet(DECISIONS_PATH)
    if len(df) < TARGET_POOL:
        raise SystemExit(
            f"[primary] cache has only {len(df)} decisions; need >= {TARGET_POOL} "
            f"for N=20k runs. Re-extract with:\n"
            f"  python pilot.py --rebuild --max-decisions 30000\n"
            f"You may also want to bump POSITION_SAMPLE_PER_GAME in pilot.py."
        )
    if "agg" not in df.columns:
        raise SystemExit("[primary] cached decisions missing the `agg` column; rebuild needed.")
    df = df.copy()
    df["T"] = (df["agg"] >= AGGRESSION_THRESHOLD).astype(int)
    p = float(df["T"].mean())
    print(f"[primary] pool loaded: {len(df)} decisions, P(T_rule=1)={p:.3f} at threshold={AGGRESSION_THRESHOLD}")
    return df


# =============================================================================
# DGP: outcome generation supporting both tau* regimes
# =============================================================================

def generate_outcomes_regime(df: pd.DataFrame, gamma: float, tau_regime: str,
                             alpha_U: float, seed: int) -> pd.DataFrame:
    """
    Y(0) = phi(X) + psi(C) + gamma * U + eps0
    Y(1) = Y(0) + tau*(X)
        homogeneous:   tau*(X) = TAU_HOMO
        heterogeneous: tau*(X) = TAU_HET_BASE + TAU_HET_HEIGHT * 1{c_material > 0}
    T_obs = soft re-assignment from rule-based T via logit shift alpha_U * U.
    Stockfish eval is never used in Y; psi only draws from Group A concepts.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    d = _standardize(df, ["delta_elo", "tc_seconds", "ply"] + GROUP_A_FEATURES)

    phi = (
        0.4 * d["delta_elo"].to_numpy()
        + 0.2 * d["tc_seconds"].to_numpy()
        + 0.1 * d["ply"].to_numpy()
    )
    psi = (
        0.3 * d["c_material"].to_numpy()
        + 0.2 * d["c_king_shield"].to_numpy()
    )
    U = rng.standard_normal(n)
    eps0 = rng.standard_normal(n) * 0.5

    Y0 = phi + psi + gamma * U + eps0

    if tau_regime == "homogeneous":
        tau = np.full(n, TAU_HOMO, dtype=np.float64)
    elif tau_regime == "heterogeneous":
        tau = TAU_HET_BASE + TAU_HET_HEIGHT * (df["c_material"].to_numpy() > 0).astype(np.float64)
    else:
        raise ValueError(f"unknown tau_regime: {tau_regime}")

    Y1 = Y0 + tau

    base_logit = np.where(df["T"].to_numpy() == 1, 1.2, -1.2)
    logit = base_logit + alpha_U * U
    p = 1.0 / (1.0 + np.exp(-logit))
    T_obs = (rng.random(n) < p).astype(int)

    Y_obs = T_obs * Y1 + (1 - T_obs) * Y0

    out = df.copy()
    out["Y_obs"] = Y_obs
    out["T_obs"] = T_obs
    out["_tau_per_unit"] = tau
    out["_tau_star_mean"] = float(tau.mean())
    return out


# =============================================================================
# Learned representation Z: orthogonal-deconfounding + PCA (linear, fast)
# =============================================================================

def fit_learned_Z(X: np.ndarray, T: np.ndarray, dim: int = 32, seed: int = 0) -> np.ndarray:
    """Linear deconfounding-flavoured embedding (Kuang 2020 simplified).

    Strategy: standardize -> fit a regularized propensity LogReg -> remove the
    1-d treatment-discriminative direction from the feature space ->
    PCA-compress the residual to `dim` dimensions. Trains in seconds, has
    a clean interpretation, and is uncoupled from outcomes (so no Y leakage).

    Compromise vs the paper version: Z is fit once on the N-sample and then
    fed to LinearDRLearner as W. EconML's CV happens on top, but the Z map
    itself is shared across folds. Documented in the threats-to-validity
    subsection.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=5000, C=0.1, solver="lbfgs", random_state=seed)
    clf.fit(Xs, T)
    beta = clf.coef_[0]
    norm = float(np.linalg.norm(beta))
    if norm < 1e-9:
        X_perp = Xs
    else:
        bn = beta / norm
        X_perp = Xs - np.outer(Xs @ bn, bn)

    k = min(dim, X_perp.shape[1], X_perp.shape[0] - 1)
    pca = PCA(n_components=k, random_state=seed)
    Z = pca.fit_transform(X_perp)
    return Z.astype(np.float32)


# =============================================================================
# Estimators
# =============================================================================

def _make_outcome_model(name: str, seed: int):
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            verbose=-1, random_state=seed,
        )
    if name == "linear":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=seed)
    raise ValueError(name)


def dr_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int, outcome: str) -> float:
    from econml.dr import LinearDRLearner
    est = LinearDRLearner(
        model_propensity=_make_propensity_model(X.shape[1], seed),
        model_regression=_make_outcome_model(outcome, seed),
        min_propensity=MIN_PROPENSITY,
        cv=5,
        random_state=seed,
    )
    est.fit(Y=Y, T=T, X=None, W=X)
    return float(est.ate(X=None, T0=0, T1=1))


def t_learner_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int, outcome: str) -> float:
    from econml.metalearners import TLearner
    models = [_make_outcome_model(outcome, seed), _make_outcome_model(outcome, seed + 1)]
    est = TLearner(models=models)
    est.fit(Y=Y, T=T, X=X)
    cates = est.effect(X)
    return float(np.mean(cates))


# =============================================================================
# Result row + incremental persistence
# =============================================================================

@dataclass
class PrimaryRun:
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


def _cell_id(d: dict) -> tuple:
    return tuple(d[k] for k in CELL_KEYS)


def load_existing() -> set[tuple]:
    if not PRIMARY_RESULTS.exists():
        return set()
    df = pd.read_csv(PRIMARY_RESULTS)
    if df.empty:
        return set()
    return set(map(tuple, df[list(CELL_KEYS)].itertuples(index=False, name=None)))


def append_row(row: PrimaryRun) -> None:
    new_file = not PRIMARY_RESULTS.exists()
    PRIMARY_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PRIMARY_RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([f.name for f in fields(PrimaryRun)])
        w.writerow([getattr(row, f.name) for f in fields(PrimaryRun)])


# =============================================================================
# Grid driver
# =============================================================================

DEFAULT_NS = [1000, 2000, 5000, 10000, 20000]
DEFAULT_GAMMAS = [0.0, 0.3, 0.6]
DEFAULT_TAU_REGIMES = ["homogeneous", "heterogeneous"]
DEFAULT_REPS = ["raw", "concept", "learned"]
DEFAULT_ESTIMATORS = [
    ("dr", "lgbm"),         # primary
    ("dr", "linear"),       # robustness swap
    ("tlearner", "lgbm"),   # baseline
]
DEFAULT_SEEDS = list(range(10))


def run_grid(df_pool: pd.DataFrame,
             Ns: list[int],
             gammas: list[float],
             tau_regimes: list[str],
             reps: list[str],
             estimators: list[tuple[str, str]],
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
                        for est_name, out_model in estimators:
                            cell = dict(
                                N=N, gamma=gamma, tau_regime=tau_regime,
                                rep=rep, estimator=est_name, outcome_model=out_model,
                                seed=seed,
                            )
                            cells.append(cell)
    total = len(cells)
    remaining = [c for c in cells if _cell_id(c) not in done]
    print(f"[primary] grid total = {total}; already done = {total - len(remaining)}; remaining = {len(remaining)}")

    if dry_run:
        print("[primary] dry-run; nothing fitted.")
        return

    pbar = tqdm(total=total, initial=total - len(remaining), desc="fits")

    # Cache outcome generation per (gamma, tau_regime) so we only build Y once.
    outcome_cache: dict[tuple[float, str], pd.DataFrame] = {}

    # Cache per-(gamma, tau_regime, N, seed) the sub-sample and its X representations
    # so multiple estimator+outcome combos reuse the same data.
    sub_cache: dict[tuple, dict] = {}

    for cell in remaining:
        key_y = (cell["gamma"], cell["tau_regime"])
        if key_y not in outcome_cache:
            outcome_cache[key_y] = generate_outcomes_regime(
                df_pool, gamma=cell["gamma"], tau_regime=cell["tau_regime"],
                alpha_U=ALPHA_U, seed=42,
            )
        df_y = outcome_cache[key_y]

        sub_key = (cell["gamma"], cell["tau_regime"], cell["N"], cell["seed"])
        if sub_key not in sub_cache:
            rng = np.random.default_rng(1000 + cell["seed"])
            idx = rng.choice(len(df_y), size=cell["N"], replace=False)
            sub = df_y.iloc[idx].reset_index(drop=True)
            X_context = sub[CONTEXT_FEATURES].to_numpy(dtype=np.float32)
            X_raw = np.concatenate(
                [sub[S_FEATURES].to_numpy(dtype=np.float32), X_context], axis=1,
            )
            X_con = np.concatenate(
                [sub[CONCEPT_FEATURES].to_numpy(dtype=np.float32), X_context], axis=1,
            )
            T_obs = sub["T_obs"].to_numpy()
            X_lrn = fit_learned_Z(X_raw, T_obs, dim=32, seed=cell["seed"])
            sub_cache[sub_key] = dict(
                Y=sub["Y_obs"].to_numpy(),
                T=T_obs,
                tau_true=float(sub["_tau_star_mean"].iloc[0]),
                X={"raw": X_raw, "concept": X_con, "learned": X_lrn},
            )
            # Bound the cache: keep at most 6 sub-samples in memory at once.
            if len(sub_cache) > 6:
                oldest = next(iter(sub_cache))
                if oldest != sub_key:
                    del sub_cache[oldest]

        s = sub_cache[sub_key]
        X_adj = s["X"][cell["rep"]]
        Y, T_obs, tau_true = s["Y"], s["T"], s["tau_true"]

        t0 = time.time()
        try:
            if cell["estimator"] == "dr":
                ate = dr_ate(X_adj, T_obs, Y, seed=cell["seed"], outcome=cell["outcome_model"])
            elif cell["estimator"] == "tlearner":
                ate = t_learner_ate(X_adj, T_obs, Y, seed=cell["seed"], outcome=cell["outcome_model"])
            else:
                raise ValueError(cell["estimator"])
        except Exception as e:
            tqdm.write(f"FAIL {cell}: {e}")
            pbar.update(1)
            continue
        dt = time.time() - t0
        bias = abs(ate - tau_true)

        row = PrimaryRun(
            N=cell["N"], gamma=cell["gamma"], tau_regime=cell["tau_regime"],
            rep=cell["rep"], estimator=cell["estimator"],
            outcome_model=cell["outcome_model"], seed=cell["seed"],
            ate_hat=ate, ate_true=tau_true, ate_bias=bias, wall_s=dt,
        )
        append_row(row)
        pbar.update(1)
    pbar.close()
    print(f"[primary] done. results -> {PRIMARY_RESULTS}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=DEFAULT_REPS,
                    choices=["raw", "concept", "learned"])
    ap.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    ap.add_argument("--ns", nargs="+", type=int, default=DEFAULT_NS)
    ap.add_argument("--tau-regimes", nargs="+", default=DEFAULT_TAU_REGIMES,
                    choices=["homogeneous", "heterogeneous"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--estimators", nargs="+", default=None,
                    help="subset of: dr-lgbm dr-linear tlearner-lgbm")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.estimators:
        m = {"dr-lgbm": ("dr", "lgbm"), "dr-linear": ("dr", "linear"),
             "tlearner-lgbm": ("tlearner", "lgbm")}
        estimators = [m[e] for e in args.estimators]
    else:
        estimators = DEFAULT_ESTIMATORS

    df_pool = load_pool()
    run_grid(
        df_pool=df_pool,
        Ns=args.ns,
        gammas=args.gammas,
        tau_regimes=args.tau_regimes,
        reps=args.reps,
        estimators=estimators,
        seeds=args.seeds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
