"""
Sanity ablation: game-outcome Y (Oracle-correlation defense).

Per docs/experimental_notes.md §4, this re-runs the primary grid with
Y_obs = game outcome from the side-to-move perspective:
    Win  → 1.0
    Draw → 0.5
    Loss → 0.0

Because Y is now fully engine-independent, if the concept-rep advantage
persists, it validates H1 on an outcome with zero Oracle-correlation risk.
If the advantage vanishes, the primary results may be contaminated.

Since there is no planted τ*, we report:
    - ATE estimate per rep (magnitude and sign of aggressive-play effect)
    - ATE standard error / CI width per rep (tighter = better adjustment)
    - Relative efficiency: SE(raw) / SE(concept) across seeds

Step 1: Re-extract decisions from the cached PGN with game_result added.
Step 2: Run the ablation grid (same reps, estimators, seeds; no γ/α sweep).
Step 3: Save results to outputs/ablation_game_outcome.csv.

Run:
    python ablation_game_outcome.py
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd
from tqdm import tqdm

from pilot import (
    CACHE, OUT, GAMES_PATH,
    AGGRESSION_THRESHOLD, MIN_PROPENSITY,
    GROUP_A_FEATURES, GROUP_B_FEATURES,
    CONCEPT_FEATURES, CONTEXT_FEATURES, S_FEATURES,
    board_to_planes, concepts, aggression_score,
    _safe_int, _make_propensity_model,
    MIN_PLY, MAX_PLY, POSITION_SAMPLE_PER_GAME,
)
from primary import fit_learned_Z, dr_ate, t_learner_ate

ABLATION_DECISIONS = CACHE / "decisions_with_result.parquet"
ABLATION_RESULTS = OUT / "ablation_game_outcome.csv"

# Ablation grid: no gamma/alpha sweep — just N × rep × estimator × seed
ABLATION_NS = [1000, 2000, 5000, 10000]
ABLATION_SEEDS = list(range(10))
ABLATION_REPS = ["raw", "concept", "learned"]
ABLATION_ESTIMATORS = [("dr", "lgbm"), ("dr", "linear")]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Re-extract decisions with game_result
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_result(force: bool = False, max_decisions: int = 25000) -> pd.DataFrame:
    if ABLATION_DECISIONS.exists() and not force:
        df = pd.read_parquet(ABLATION_DECISIONS)
        print(f"[ablation] cached decisions with result ({len(df)} rows); skipping extraction.")
        return df

    if not GAMES_PATH.exists():
        raise SystemExit(f"No PGN at {GAMES_PATH}. Run pilot.py first.")

    print(f"[ablation] extracting decisions with game_result from {GAMES_PATH}...")
    rows = []
    rng = np.random.default_rng(0)  # same seed as pilot.py extraction
    with open(GAMES_PATH) as fin:
        pbar = tqdm(total=max_decisions, desc="decisions+result")
        while len(rows) < max_decisions:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            headers = game.headers
            w_elo = _safe_int(headers.get("WhiteElo"))
            b_elo = _safe_int(headers.get("BlackElo"))
            tc = headers.get("TimeControl", "0+0")
            try:
                base, inc = tc.split("+")
                tc_seconds = int(base) + 40 * int(inc)
            except Exception:
                tc_seconds = 0
            if tc_seconds < 1500:
                continue
            try:
                eco = headers.get("ECO", "A00")
                eco_num = (ord(eco[0]) - ord("A")) * 100 + int(eco[1:])
            except Exception:
                eco_num = 0

            # Parse game result
            result_str = headers.get("Result", "*")
            if result_str == "1-0":
                white_score = 1.0
            elif result_str == "0-1":
                white_score = 0.0
            elif result_str == "1/2-1/2":
                white_score = 0.5
            else:
                continue  # skip unfinished games

            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLY + 4:
                continue
            eligible = [i for i in range(MIN_PLY, min(len(moves), MAX_PLY))]
            if not eligible:
                continue
            sample_idxs = rng.choice(eligible, size=min(POSITION_SAMPLE_PER_GAME, len(eligible)), replace=False)
            sample_idxs.sort()

            ply = 0
            for i, move in enumerate(moves):
                if ply in sample_idxs:
                    stm = board.turn
                    elo_self = w_elo if stm == chess.WHITE else b_elo
                    elo_opp = b_elo if stm == chess.WHITE else w_elo
                    delta_elo = elo_self - elo_opp

                    # Game result from side-to-move perspective
                    if stm == chess.WHITE:
                        game_result = white_score
                    else:
                        game_result = 1.0 - white_score

                    try:
                        s_vec = board_to_planes(board)
                        c_feats = concepts(board)
                        agg = aggression_score(board, move)
                    except Exception:
                        ply += 1
                        board.push(move)
                        continue

                    row = dict(
                        ply=ply,
                        phase=(0 if ply < 24 else (1 if ply < 60 else 2)),
                        elo_self=elo_self,
                        elo_opp=elo_opp,
                        delta_elo=delta_elo,
                        tc_seconds=tc_seconds,
                        eco_num=eco_num,
                        color=int(stm == chess.WHITE),
                        agg=agg,
                        T=int(agg >= AGGRESSION_THRESHOLD),
                        game_result=game_result,
                        **c_feats,
                    )
                    for k in range(len(s_vec)):
                        row[f"s{k}"] = s_vec[k]
                    rows.append(row)
                    pbar.update(1)
                    if len(rows) >= max_decisions:
                        break
                board.push(move)
                ply += 1
        pbar.close()

    df = pd.DataFrame(rows)
    print(f"[ablation] extracted {len(df)} decisions; P(T=1) = {df['T'].mean():.3f}")
    print(f"[ablation] game_result dist: W={float((df['game_result']==1.0).mean()):.2f}, "
          f"D={float((df['game_result']==0.5).mean()):.2f}, L={float((df['game_result']==0.0).mean()):.2f}")
    df.to_parquet(ABLATION_DECISIONS)
    print(f"[ablation] saved to {ABLATION_DECISIONS}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run ablation grid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationRun:
    N: int
    rep: str
    estimator: str
    outcome_model: str
    seed: int
    ate_hat: float
    ate_se: float
    wall_s: float


CELL_KEYS = ("N", "rep", "estimator", "outcome_model", "seed")


def load_existing() -> set[tuple]:
    if not ABLATION_RESULTS.exists():
        return set()
    df = pd.read_csv(ABLATION_RESULTS)
    if df.empty:
        return set()
    return set(map(tuple, df[list(CELL_KEYS)].itertuples(index=False, name=None)))


def append_row(row: AblationRun) -> None:
    new_file = not ABLATION_RESULTS.exists()
    with open(ABLATION_RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([fld.name for fld in fields(AblationRun)])
        w.writerow([getattr(row, fld.name) for fld in fields(AblationRun)])


def dr_ate_with_se(X, T, Y, seed, outcome):
    """DR-Learner ATE + bootstrap SE."""
    from econml.dr import LinearDRLearner
    from primary import _make_outcome_model
    est = LinearDRLearner(
        model_propensity=_make_propensity_model(X.shape[1], seed),
        model_regression=_make_outcome_model(outcome, seed),
        min_propensity=MIN_PROPENSITY,
        cv=5,
        random_state=seed,
    )
    est.fit(Y=Y, T=T, X=None, W=X)
    ate = float(est.ate(X=None, T0=0, T1=1))
    # Use inference for SE
    try:
        inf = est.ate_inference(X=None, T0=0, T1=1)
        se = float(inf.stderr_mean)
    except Exception:
        se = float("nan")
    return ate, se


def run_ablation(df_pool: pd.DataFrame) -> None:
    done = load_existing()
    cells = []
    for N in ABLATION_NS:
        if N > len(df_pool):
            continue
        for seed in ABLATION_SEEDS:
            for rep in ABLATION_REPS:
                for est_name, out_model in ABLATION_ESTIMATORS:
                    cell = dict(N=N, rep=rep, estimator=est_name, outcome_model=out_model, seed=seed)
                    cells.append(cell)

    remaining = [c for c in cells if tuple(c[k] for k in CELL_KEYS) not in done]
    print(f"[ablation] grid total = {len(cells)}; done = {len(cells) - len(remaining)}; remaining = {len(remaining)}")

    pbar = tqdm(total=len(cells), initial=len(cells) - len(remaining), desc="ablation fits")
    sub_cache: dict = {}

    for cell in remaining:
        sub_key = (cell["N"], cell["seed"])
        if sub_key not in sub_cache:
            rng = np.random.default_rng(1000 + cell["seed"])
            idx = rng.choice(len(df_pool), size=cell["N"], replace=False)
            sub = df_pool.iloc[idx].reset_index(drop=True)
            X_ctx = sub[CONTEXT_FEATURES].to_numpy(dtype=np.float32)
            X_raw = np.concatenate([sub[S_FEATURES].to_numpy(dtype=np.float32), X_ctx], axis=1)
            X_con = np.concatenate([sub[CONCEPT_FEATURES].to_numpy(dtype=np.float32), X_ctx], axis=1)
            T_obs = sub["T"].to_numpy()  # rule-based T, no latent confounding
            X_lrn = fit_learned_Z(X_raw, T_obs, dim=32, seed=cell["seed"])
            sub_cache[sub_key] = dict(
                Y=sub["game_result"].to_numpy(),
                T=T_obs,
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
                ate, se = dr_ate_with_se(X_adj, s["T"], s["Y"],
                                          seed=cell["seed"], outcome=cell["outcome_model"])
            else:
                ate = t_learner_ate(X_adj, s["T"], s["Y"],
                                     seed=cell["seed"], outcome=cell["outcome_model"])
                se = float("nan")
        except Exception as e:
            tqdm.write(f"FAIL {cell}: {e}")
            pbar.update(1)
            continue
        dt = time.time() - t0

        row = AblationRun(
            N=cell["N"], rep=cell["rep"], estimator=cell["estimator"],
            outcome_model=cell["outcome_model"], seed=cell["seed"],
            ate_hat=ate, ate_se=se, wall_s=dt,
        )
        append_row(row)
        pbar.update(1)

    pbar.close()
    print(f"[ablation] done. results -> {ABLATION_RESULTS}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Report summary
# ─────────────────────────────────────────────────────────────────────────────

def report() -> None:
    if not ABLATION_RESULTS.exists():
        print("[ablation] no results to report.")
        return
    df = pd.read_csv(ABLATION_RESULTS)
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY: Game-Outcome Y (Oracle-Correlation Defense)")
    print("=" * 60)

    dr_lgbm = df[(df["estimator"] == "dr") & (df["outcome_model"] == "lgbm")]
    for N in sorted(dr_lgbm["N"].unique()):
        print(f"\n  N = {N:,}:")
        for rep in ABLATION_REPS:
            cell = dr_lgbm[(dr_lgbm["N"] == N) & (dr_lgbm["rep"] == rep)]
            if cell.empty:
                continue
            ate_med = cell["ate_hat"].median()
            se_med = cell["ate_se"].median()
            print(f"    {rep:>8s}: ATE = {ate_med:+.4f}  median SE = {se_med:.4f}  (n_seeds={len(cell)})")

    # Relative efficiency: SE(raw) / SE(concept) at each N
    print("\n  Relative efficiency SE(raw) / SE(concept):")
    for N in sorted(dr_lgbm["N"].unique()):
        raw_se = dr_lgbm[(dr_lgbm["N"] == N) & (dr_lgbm["rep"] == "raw")]["ate_se"].median()
        con_se = dr_lgbm[(dr_lgbm["N"] == N) & (dr_lgbm["rep"] == "concept")]["ate_se"].median()
        if con_se > 0:
            print(f"    N={N:>6,}: {raw_se / con_se:.2f}x")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    df_pool = extract_with_result()
    run_ablation(df_pool)
    report()


if __name__ == "__main__":
    main()
