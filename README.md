# Beyond Pattern Matching: Causal Concept Representations for Strategic Decision-Making in Chess

Course project for **ML808 — Causality & Machine Learning** at the Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI).

**Authors:** Abdulla Alfalasi, Mohammed AlBalooshi

## What this project is

A semi-synthetic case study that asks whether human-interpretable chess concepts (material, king safety, pawn structure, piece activity, space, development) form a more *sample-efficient* covariate encoding for causal effect estimation than raw board states or learned low-dimensional embeddings.

We frame strategic decision-making at individual chess positions as an interventional problem:

- **Treatment `T`** — a rule-based binary label for "aggressive" vs. "conservative" moves, defined from captures, checks, king-zone pressure, and sacrifices. Engine evaluations are explicitly excluded from the treatment definition.
- **Outcome `Y`** — a semi-synthetic payoff with a planted ground-truth average treatment effect, built from pre-treatment context and structural board features only. Stockfish total evaluation is never used in `Y`.
- **Three covariate encodings** of the same back-door adjustment problem:
  1. `X_raw = (S, X)` — an 18×8×8 bitboard-style encoding of the position plus player/game context (~1158-d).
  2. `X_concept = (C, X)` — a 12-dim expert concept vector plus context.
  3. `X_learned = (Z, X)` — a 32-dim linear deconfounding embedding (propensity-direction removal + PCA).

We estimate the ATE under each encoding using a cross-fitted `LinearDRLearner`, with a `T-Learner` baseline and a nuisance-model robustness swap, then compare ATE bias as a function of sample size and latent-confounding strength. The core hypothesis is that under DR-style estimators that depend on stable propensity estimation, low-dimensional deterministic encodings of `S` dominate raw bitboards at small `N`.

## Causal framing

The concept vector `C = f(S)` is treated as a **deterministic encoding of `S`**, not a separate causal node. The underlying DAG is:

```
X -> S,  X -> T,  X -> Y
S -> T,  S -> Y
T -> Y
U -> T,  U -> Y          (latent, never adjusted for)
```

`X -> S` is a selection mechanism — players with different Elo, style, or opening preferences systematically reach different positions, inducing observational dependence between `X` and `S`. The three encodings (`S`, `C`, `Z`) are three choices of covariate basis for the same back-door adjustment problem; the paper tests which choice is most sample-efficient rather than which is uniquely valid.

See [`main.tex`](main.tex) for the full specification and [`experimental_notes.md`](experimental_notes.md) for the locked experimental setup.

## Repository layout

```
.
├── main.tex                   LaTeX source of the paper
├── refs.bib                   Bibliography
├── experimental_notes.md      Locked experimental design (DGP, estimators, grid, exit criteria)
├── pilot/
│   ├── pilot.py               Minimum pilot runner (2 N × 2 γ × 3 reps, 24 fits)
│   ├── primary.py             Full grid runner (5 N × 3 γ × 2 τ × 3 reps × 3 estimators × 10 seeds)
│   ├── requirements.txt       Python dependencies
│   ├── README.md              Pilot-specific run notes
│   ├── cache/                 Downloaded PGNs and extracted decision table (git-ignored)
│   └── outputs/               pilot_results.csv and primary_results.csv (git-ignored)
└── README.md                  This file
```

## Build the paper

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Reproduce the experiments

All runners live in [`pilot/`](pilot/). They reuse the same feature extraction, treatment assignment, and data-generating process.

### Environment

Python 3.11 (Conda is the tested path; a plain venv works too):

```bash
conda create -n chess-pilot python=3.11
conda activate chess-pilot
pip install -r pilot/requirements.txt
```

### Data

Games are fetched at runtime from the Lichess public API — classical-only, from a live pool of the top rated classical players (with a static fallback list for when the `/api/player/top` endpoint is unavailable). The fetcher writes `pilot/cache/games.pgn` and is cached between runs.

Positions are sampled from each game, filtered to rated classical games (≥ 25 minute time control), and extracted into `pilot/cache/decisions.parquet` along with board-encoding tensors, concept features, context variables, and the rule-based aggression score.

### Pilot (24 fits, ~5 minutes)

Validates the pipeline end-to-end and checks three exit criteria before scaling:
1. Concept advantage visible at `N=2000, γ=0.3` (the sample-efficiency regime).
2. DR-Learner clipped propensities stay inside `[0.02, 0.98]`.
3. Wall-clock per fit at `N=5000` under 60 seconds.

```bash
cd pilot
python pilot.py                    # full pipeline, reuses caches
python pilot.py --refetch          # force re-download from Lichess
python pilot.py --rebuild          # re-extract decisions from cached PGN
```

Writes `outputs/pilot_results.csv` and prints a pass/fail summary for the three exit criteria.

### Primary grid (~2700 fits, several hours)

```bash
cd pilot
python primary.py --dry-run        # confirm grid loads and count remaining cells
python primary.py                  # run the full grid; resumable
```

The runner is **resumable**: it appends each fit's result to `outputs/primary_results.csv` as it goes and skips cells already present on startup. Safe to CTRL+C and re-launch. Useful subsetting flags:

```bash
python primary.py --reps raw concept --gammas 0.3 --seeds 0 1 2
python primary.py --tau-regimes homogeneous
python primary.py --estimators dr-lgbm tlearner-lgbm
```

If the decision cache is smaller than `N=20000 + buffer`, rebuild with:

```bash
python pilot.py --rebuild --max-decisions 30000
```

## Key design choices

- **Oracle-correlation defense.** `Y` is built only from pre-treatment context and Group A structural features computed with `python-chess`. Stockfish total evaluation does not appear in the outcome path. Concept features are themselves computed without Stockfish in the pilot codepath, eliminating the leakage channel entirely.
- **Linear deconfounding `Z`.** The learned encoding is deliberately light: standardize, fit a regularized propensity logistic, project out the treatment-discriminative direction, PCA-compress the residual to 32 dims. Trains in seconds, no PyTorch dependency.
- **Feasibility filter disabled in the pilot.** The paper specification calls for a Stockfish top-k feasibility filter on `F=1`; the pilot uses all sampled positions to keep the runtime laptop-friendly. Reinstating the filter is an open task for the full run.
- **Cross-fit discipline.** All DR fits use K=5 cross-fitting. The learned `Z` map is fit once per `(N, seed)` sub-sample and reused across estimators to avoid double work; this is a documented compromise relative to per-fold `Z` fitting.

## Status

- Minimum pilot: passing all three exit criteria.
- Small-scope primary slice (108 fits): validates the DR-vs-T-Learner divergence and the H1 sample-efficiency shape.
- Full primary grid (2700 fits): queued for a laptop overnight run.
- Figures and final paper results: pending full-grid completion.

## Dependencies

See [`pilot/requirements.txt`](pilot/requirements.txt). Main packages:
`python-chess`, `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `econml`, `tqdm`, `requests`.

## Citation

If you reference this work, please cite the paper (see `main.tex` for the final citation once a venue is chosen).
