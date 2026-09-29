# Beyond Pattern Matching: Causal Concept Representations for Strategic Decision-Making in Chess

Course project for **ML808: Causality & Machine Learning** at MBZUAI.

**Authors:** Abdulla Alfalasi, Mohammed AlBalooshi

**Paper:** [paper/main.pdf](paper/main.pdf)

## Summary

This repository contains **ChessCausalBench**, a semi-synthetic causal testbed for strategic decision-making, and every experiment behind the paper. The testbed fixes a structural causal model with a planted average treatment effect and adjustable confounding strength `gamma`. It then asks which covariate encoding gives the most accurate doubly robust ATE estimate for the same back-door adjustment problem:

| Representation | Adjustment set | Dimension |
| --- | --- | --- |
| Raw | `X_raw = (S, X)`: bitboard state plus context | 1152 + 6 |
| Concept | `X_concept = (C, X)`: expert chess concepts plus context | 12 + 6 |
| Learned | `X_learned = (Z, X)`: linear deconfounding embedding plus context | 32 + 6 |

Two environments use the same SCM/DGP protocol:

1. **Chess** (primary): about 30k decision points from rated classical Lichess games. The treatment is a rule-based aggressive-vs-conservative label, and the outcome is semi-synthetic, built from structural board features.
2. **Gridworld** (secondary): a 10×10 grid with obstacles and a goal. A hidden checkerboard confounder is recoverable from raw `S` but absent from the concepts `C`.

## Main findings

- Concept representations win at small `N` under non-trivial confounding, mainly because the outcome model is easier to fit.
- Raw representations overtake at large `N` once the concept basis's sufficiency gap dominates. The crossover appears in both environments.
- Concept adjustment is stable when the outcome model is swapped (LightGBM to Ridge); raw bias rises by 58%.
- The learned linear embedding improves overlap but has chance-level outcome R², so it lands between the other two.
- Replacing the engine-derived outcome with the actual game result keeps the concept advantage (SE ratio raw/concept of 7.6 at `N = 1000`), so the advantage is not an oracle-correlation artifact.
- Adding a PCA residual of `S` to the concepts does not help; a treatment-supervised PLS residual helps in the mid-range only.

## Repository layout

```text
.
├── paper/
│   ├── main.tex                   Manuscript source
│   ├── main.pdf                   Compiled manuscript
│   ├── refs.bib                   Bibliography
│   └── neurips_2026.sty           NeurIPS 2026 style file
├── pilot/
│   ├── pilot.py                   Data pipeline (Lichess download, feature extraction) and pilot grid
│   ├── primary.py                 Full chess grid (2,700 fits)
│   ├── plan_v2_analysis.py        Sample-efficiency, bias-variance, regime, and crossover analyses
│   ├── concept_queries.py         Concept-level CATE / conditional ATE queries
│   ├── conplus.py                 PCA residual augmentation
│   ├── conplus_supervised.py      T-supervised PLS residual augmentation
│   ├── nuisance_diagnostics.py    Propensity AUC and outcome R² diagnostics
│   ├── ablation_game_outcome.py   Game-outcome (oracle-correlation) ablation
│   ├── plot_figures.py            Main figures
│   ├── gridworld/                 Secondary environment
│   ├── cache/                     Cached games and decision tables
│   ├── outputs/                   Result CSVs and figures
│   └── requirements.txt
└── docs/
    ├── experimental_notes.md      Original experimental design
    ├── experimental_plan_v2.md    Extension plan
    └── finding_1_2_summary_report.md  Notes on residual augmentation and the gridworld confounder
```

## Setup

```bash
conda create -n chess-pilot python=3.11
conda activate chess-pilot
pip install -r pilot/requirements.txt
```

## Reproducing the results

Caches, result CSVs, and figures are committed, so every number in the paper can be checked without rerunning anything. To regenerate them, run from `pilot/`:

| Step | Command | Writes |
| --- | --- | --- |
| Pilot and data pipeline | `python pilot.py` (`--refetch` to re-download, `--rebuild` to re-extract) | `outputs/pilot_results.csv` |
| Primary chess grid | `python primary.py` (resumable; `--dry-run` to preview) | `outputs/primary_results.csv` |
| Analyses | `python plan_v2_analysis.py` | `t3_threshold.csv`, `bias_variance.csv`, `regime_diagram.csv`, `crossover.csv`, figures |
| Concept queries | `python concept_queries.py --force` | `concept_queries.csv`, `figQ_concept_queries.pdf` |
| Residual augmentation | `python conplus.py`, then `python conplus_supervised.py` | `conplus_*results.csv`, `figConplus.pdf` |
| Nuisance diagnostics | `python nuisance_diagnostics.py` | `nuisance_diagnostics.csv`, `figNuisance.pdf` |
| Game-outcome ablation | `python ablation_game_outcome.py` | `ablation_game_outcome.csv` |
| Main figures | `python plot_figures.py` (after the primary grid and ablation) | `outputs/figures/fig2`–`fig6` |
| Gridworld | `python gridworld/runner.py --rebuild-pool` | `gridworld_results.csv`, `figGridworld.pdf` |

All runners use seeds 0–9. Hyperparameters and seeding are listed in the paper's reproducibility appendix. The full chess grid takes about 2 hours on a laptop CPU; every other grid finishes in under 30 minutes.

Use `conda run -n chess-pilot python ...` rather than `python3`, which may resolve to a system interpreter.

## Building the paper

```bash
cd paper
latexmk -pdf main.tex      # or: tectonic main.tex
```

Figures are read from `pilot/outputs/figures/`.
