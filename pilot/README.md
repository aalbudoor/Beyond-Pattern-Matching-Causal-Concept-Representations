# Pilot Runner

Minimum pilot for the Beyond Pattern Matching chess causality study.
Spec: `../experimental_notes.md`, section "Minimum pilot".

## What it runs

- Grid: `N in {2000, 5000}` x `gamma in {0.0, 0.3}` x homogeneous `tau* = 0.3` x 3 seeds
- Representations: raw `S` and expert `C` (learned `Z` deferred to full run)
- Estimator: `LinearDRLearner` (EconML) with LogisticRegression propensity + LightGBM outcome, 5-fold cross-fit
- 24 total fits

## Simplifications vs. the full design

1. **No Stockfish in feature extraction.** Both Group A and Group B concepts are computed with python-chess only. Group B uses structural surrogates (king-ring attacker count, legal-move count, passed-pawn bitboard, etc.). This eliminates the Oracle-correlation path entirely in the pilot.
2. **No feasibility filter.** The full run will reinstate a Stockfish depth-10 top-4 check; the pilot uses all sampled positions.
3. **Decision pool: ~15k positions** sampled from ~2000 classical Lichess games pulled from strong players via the public `/api/games/user/{user}` endpoint. ~100 KB/user, one-shot download, cached.

## Setup

```bash
cd pilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python pilot.py                   # full pipeline, reuses caches
python pilot.py --refetch         # re-download Lichess games
python pilot.py --rebuild         # re-extract decisions from cached PGN
python pilot.py --max-decisions 8000   # smaller pool for a dry run
```

Outputs:
- `cache/games.pgn`            — raw Lichess PGN
- `cache/decisions.parquet`    — extracted decision rows with S, C, X, T, agg score
- `outputs/pilot_results.csv`  — one row per (N, gamma, rep, seed)

## Exit criteria (checked automatically in the summary)

1. Concept-rep mean ATE bias < raw-rep mean ATE bias at `N=5000, gamma=0.3`.
2. All fits have propensity min/max inside `[0.02, 0.98]`.
3. Wall-clock per fit at `N=5000` under 60 seconds.

If all three pass, scale to the primary grid. If any fail, inspect the CSV before proceeding.

## Knob checks also printed

- Marginal observed `P(T=1)` after the rule-based aggression score + `alpha_U` logit nudge — target `[0.3, 0.6]`.
- `Var(psi(C))` on a 1k sample — should be non-trivial.
