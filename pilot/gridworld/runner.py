"""
Gridworld secondary environment for docs/experimental_plan_v2.md.

Run:
    python gridworld/runner.py
    python gridworld/runner.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE.parent
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

from gridworld.dgp import (  # noqa: E402
    ALPHA_U,
    CONCEPT_FEATURES,
    CONTEXT_FEATURES,
    S_FEATURES,
    generate_outcomes_regime,
)
from gridworld.env import DEFAULT_POOL_SIZE, sample_decision_points  # noqa: E402
from gridworld.features import row_from_point  # noqa: E402
from gridworld.treatment import assign_treatment_rule  # noqa: E402

CACHE = HERE / "cache"
OUT = PILOT_DIR / "outputs"
FIG_DIR = OUT / "figures"
POOL_PATH = CACHE / "gridworld_pool.parquet"
RESULTS_PATH = OUT / "gridworld_results.csv"

DEFAULT_NS = [500, 1000, 2000, 5000, 10000]
DEFAULT_GAMMAS = [0.0, 0.3, 0.6]
DEFAULT_TAU_REGIMES = ["homogeneous"]
DEFAULT_REPS = ["raw", "concept", "learned"]
DEFAULT_ESTIMATORS = [("dr", "lgbm"), ("dr", "linear")]
DEFAULT_SEEDS = list(range(10))
MIN_PROPENSITY = 0.02

REP_LABELS = {"raw": "Raw $S$", "concept": "Concept $C$", "learned": "Learned $Z$"}
REP_COLORS = {"raw": "#d62728", "concept": "#2ca02c", "learned": "#1f77b4"}
REP_MARKERS = {"raw": "s", "concept": "o", "learned": "D"}


@dataclass
class GridworldRun:
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


def _make_propensity_model(n_features: int, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    C = 0.1 if n_features > 200 else 1.0
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", LogisticRegression(max_iter=5000, C=C, solver="lbfgs", random_state=seed)),
    ])


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


def fit_learned_Z(X: np.ndarray, T: np.ndarray, dim: int = 32, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
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
    return pca.fit_transform(X_perp).astype(np.float32)


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
    return float(np.mean(est.effect(X)))


def build_pool(force: bool = False, n_points: int = DEFAULT_POOL_SIZE) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    if POOL_PATH.exists() and not force:
        df = pd.read_parquet(POOL_PATH)
        print(f"[gridworld] cached pool found ({len(df)} rows); skipping generation.")
        return df

    print(f"[gridworld] generating {n_points:,} decision points...")
    points = sample_decision_points(n=n_points, seed=0)
    rows = [row_from_point(p) for p in points]
    df = pd.DataFrame(rows)
    df, intercept = assign_treatment_rule(df, seed=0, target_rate=0.4)
    df.to_parquet(POOL_PATH)
    print(f"[gridworld] saved pool to {POOL_PATH}")
    print(f"[gridworld] P(T_rule=1)={df['T'].mean():.3f} at intercept={intercept:.3f}")
    return df


def load_existing() -> set[tuple]:
    if not RESULTS_PATH.exists():
        return set()
    df = pd.read_csv(RESULTS_PATH)
    if df.empty:
        return set()
    return set(map(tuple, df[list(CELL_KEYS)].itertuples(index=False, name=None)))


def append_row(row: GridworldRun) -> None:
    new_file = not RESULTS_PATH.exists()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([fld.name for fld in fields(GridworldRun)])
        w.writerow([getattr(row, fld.name) for fld in fields(GridworldRun)])


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
                            cells.append(dict(
                                N=N,
                                gamma=gamma,
                                tau_regime=tau_regime,
                                rep=rep,
                                estimator=est_name,
                                outcome_model=out_model,
                                seed=seed,
                            ))

    remaining = [c for c in cells if tuple(c[k] for k in CELL_KEYS) not in done]
    print(f"[gridworld] grid total = {len(cells)}; done = {len(cells) - len(remaining)}; remaining = {len(remaining)}")
    if dry_run:
        return

    pbar = tqdm(total=len(cells), initial=len(cells) - len(remaining), desc="gridworld")
    outcome_cache: dict[tuple[float, str], pd.DataFrame] = {}
    sub_cache: dict[tuple, dict] = {}

    for cell in remaining:
        y_key = (cell["gamma"], cell["tau_regime"])
        if y_key not in outcome_cache:
            outcome_cache[y_key] = generate_outcomes_regime(
                df_pool,
                gamma=cell["gamma"],
                tau_regime=cell["tau_regime"],
                alpha_U=ALPHA_U,
                seed=42,
            )
        df_y = outcome_cache[y_key]

        sub_key = (cell["gamma"], cell["tau_regime"], cell["N"], cell["seed"])
        if sub_key not in sub_cache:
            rng = np.random.default_rng(2000 + cell["seed"])
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
                tau_true=float(sub["_tau_star_mean"].iloc[0]),
                X={"raw": X_raw, "concept": X_con, "learned": X_lrn},
            )
            if len(sub_cache) > 6:
                oldest = next(iter(sub_cache))
                if oldest != sub_key:
                    del sub_cache[oldest]

        s = sub_cache[sub_key]
        X_adj = s["X"][cell["rep"]]
        t0 = time.time()
        try:
            if cell["estimator"] == "dr":
                ate = dr_ate(X_adj, s["T"], s["Y"], seed=cell["seed"], outcome=cell["outcome_model"])
            elif cell["estimator"] == "tlearner":
                ate = t_learner_ate(X_adj, s["T"], s["Y"], seed=cell["seed"], outcome=cell["outcome_model"])
            else:
                raise ValueError(cell["estimator"])
        except Exception as e:
            tqdm.write(f"FAIL {cell}: {e}")
            pbar.update(1)
            continue
        dt = time.time() - t0
        append_row(GridworldRun(
            N=cell["N"],
            gamma=cell["gamma"],
            tau_regime=cell["tau_regime"],
            rep=cell["rep"],
            estimator=cell["estimator"],
            outcome_model=cell["outcome_model"],
            seed=cell["seed"],
            ate_hat=float(ate),
            ate_true=s["tau_true"],
            ate_bias=abs(float(ate) - s["tau_true"]),
            wall_s=dt,
        ))
        pbar.update(1)

    pbar.close()
    print(f"[gridworld] results -> {RESULTS_PATH}")


def plot_figure() -> None:
    if not RESULTS_PATH.exists():
        return
    df = pd.read_csv(RESULTS_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))

    panel = df[
        (df["tau_regime"] == "homogeneous")
        & (df["gamma"] == 0.3)
        & (df["estimator"] == "dr")
        & (df["outcome_model"] == "lgbm")
    ]
    for rep in DEFAULT_REPS:
        rp = panel[panel["rep"] == rep]
        if rp.empty:
            continue
        agg = rp.groupby("N")["ate_bias"].agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).sort_index()
        Ns = agg.index.to_numpy(dtype=float)
        axes[0].plot(Ns, agg["median"], marker=REP_MARKERS[rep], color=REP_COLORS[rep],
                     linewidth=1.6, markersize=5, label=REP_LABELS[rep])
        axes[0].fill_between(Ns, agg["q25"], agg["q75"], color=REP_COLORS[rep], alpha=0.15)
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    axes[0].set_xlabel("Sample size $N$")
    axes[0].set_ylabel("ATE Bias  $|\\hat{\\tau} - \\tau^*|$")
    axes[0].set_title("(a) ATE bias vs $N$ at $\\gamma=0.3$", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    rob = df[
        (df["tau_regime"] == "homogeneous")
        & (df["gamma"] == 0.3)
        & (df["N"] == 1000)
        & (df["estimator"] == "dr")
    ]
    outcomes = ["lgbm", "linear"]
    x = np.arange(len(DEFAULT_REPS))
    width = 0.35
    for i, outcome_model in enumerate(outcomes):
        vals, lo, hi = [], [], []
        for rep in DEFAULT_REPS:
            cell = rob[(rob["rep"] == rep) & (rob["outcome_model"] == outcome_model)]["ate_bias"]
            med = cell.median() if len(cell) else np.nan
            vals.append(med)
            lo.append(med - cell.quantile(0.25) if len(cell) else 0.0)
            hi.append(cell.quantile(0.75) - med if len(cell) else 0.0)
        axes[1].bar(x + i * width - width / 2, vals, width,
                    label="LightGBM" if outcome_model == "lgbm" else "Linear (Ridge)",
                    yerr=[lo, hi], capsize=3, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([REP_LABELS[r] for r in DEFAULT_REPS], fontsize=9)
    axes[1].set_ylabel("ATE Bias  $|\\hat{\\tau} - \\tau^*|$")
    axes[1].set_title("(b) Nuisance-swap robustness at $N=1{,}000$", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Fig Gridworld — Secondary Environment Replication", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "figGridworld.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[gridworld] figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--ns", nargs="+", type=int, default=DEFAULT_NS)
    ap.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    ap.add_argument("--tau-regimes", nargs="+", default=DEFAULT_TAU_REGIMES,
                    choices=["homogeneous"])
    ap.add_argument("--reps", nargs="+", default=DEFAULT_REPS, choices=DEFAULT_REPS)
    ap.add_argument("--estimators", nargs="+", default=None,
                    help="subset of: dr-lgbm dr-linear")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.estimators:
        mapping = {"dr-lgbm": ("dr", "lgbm"), "dr-linear": ("dr", "linear")}
        estimators = [mapping[e] for e in args.estimators]
    else:
        estimators = DEFAULT_ESTIMATORS

    df_pool = build_pool(force=args.rebuild_pool)
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
    if not args.dry_run:
        plot_figure()


if __name__ == "__main__":
    main()
