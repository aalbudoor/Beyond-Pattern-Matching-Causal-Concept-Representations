# Beyond Pattern Matching: Causal Concept Representations for Strategic Decision-Making in Chess

Course project for **ML808 — Causality & Machine Learning** at **MBZUAI**.

**Authors:** Abdulla Alfalasi, Mohammed AlBalooshi

## Project Summary

This repository contains a semi-synthetic causal testbed for strategic decision-making in chess. The core question is whether expert-defined, human-interpretable chess concepts form a more sample-efficient adjustment basis for ATE estimation than:

- raw board states
- a learned low-dimensional deconfounding embedding

The project compares three covariate encodings of the same back-door adjustment problem:

1. `X_raw = (S, X)` — raw bitboard-style board encoding plus context
2. `X_concept = (C, X)` — expert concept vector plus context
3. `X_learned = (Z, X)` — learned linear deconfounding embedding plus context

The estimand is the ATE of an aggressive-vs-conservative strategic treatment under a semi-synthetic DGP with planted ground-truth effects.

## Current Status

This repo is no longer at the proposal/pilot stage. The following are complete and committed:

- pilot run: `24` rows in [pilot/outputs/pilot_results.csv](pilot/outputs/pilot_results.csv)
- full primary grid: `2700` rows in [pilot/outputs/primary_results.csv](pilot/outputs/primary_results.csv)
- phase-1 v2 analyses:
  - [pilot/outputs/t3_threshold.csv](pilot/outputs/t3_threshold.csv)
  - [pilot/outputs/bias_variance.csv](pilot/outputs/bias_variance.csv)
  - [pilot/outputs/regime_diagram.csv](pilot/outputs/regime_diagram.csv)
  - [pilot/outputs/crossover.csv](pilot/outputs/crossover.csv)
- concept queries: `210` rows in [pilot/outputs/concept_queries.csv](pilot/outputs/concept_queries.csv)
- residual augmentation:
  - PCA residual baseline in [pilot/outputs/conplus_results.csv](pilot/outputs/conplus_results.csv)
  - supervised T-PLS residual variant in [pilot/outputs/conplus_supervised_results.csv](pilot/outputs/conplus_supervised_results.csv)
- nuisance diagnostics: `150` rows in [pilot/outputs/nuisance_diagnostics.csv](pilot/outputs/nuisance_diagnostics.csv)
- gridworld secondary environment: `900` rows in [pilot/outputs/gridworld_results.csv](pilot/outputs/gridworld_results.csv)
- NeurIPS-format manuscript source in [neurips.tex](neurips.tex) and compiled PDF in [neurips.pdf](neurips.pdf)

## Main Findings Snapshot

The committed results support the following broad picture:

- **Concept representations win at small `N`** because they make nuisance estimation easier.
- **Raw representations can overtake at large `N`** when the expert concept basis is only approximately sufficient.
- **The learned linear embedding is mixed**: it improves overlap but can be weak on outcome fit.
- **PCA residual augmentation fails** to close the large-`N` gap.
- **T-supervised residual compression helps** in the mid-range but does not fully recover raw's asymptotic advantage.
- **The gridworld checkerboard fix creates a real raw-vs-concept crossover**, though the final tuned version is stronger than the original target regime.

For the detailed handoff on the two follow-up findings, see [finding_1_2_summary_report.md](finding_1_2_summary_report.md).

## Manuscript Files

- [neurips.tex](neurips.tex): primary manuscript source going forward
- [neurips.pdf](neurips.pdf): compiled NeurIPS-format PDF
- [main.tex](main.tex): working/full manuscript source retained for project history
- [neurips_2026.sty](neurips_2026.sty): official NeurIPS style file committed locally
- [checklist.tex](checklist.tex): NeurIPS checklist template

## Build the Paper

The tested compile path in this repo is `tectonic`, not raw `pdflatex`.

```bash
conda activate chess-pilot
tectonic neurips.tex
```

Equivalent one-shot command:

```bash
conda run -n chess-pilot tectonic neurips.tex
```

This writes `neurips.pdf`.

## Environment Setup

Tested Python environment:

```bash
conda create -n chess-pilot python=3.11
conda activate chess-pilot
pip install -r pilot/requirements.txt
conda install -c conda-forge tectonic
```

Important note:

- use `conda run -n chess-pilot python ...`
- do **not** rely on `conda run -n chess-pilot python3 ...` on this machine, because `python3` may resolve to Homebrew Python instead of the Conda env interpreter

## Repository Layout

```text
.
├── neurips.tex                    NeurIPS manuscript source
├── neurips.pdf                    Compiled NeurIPS manuscript
├── main.tex                       Working/full manuscript source
├── refs.bib                       Bibliography
├── experimental_notes.md          Original locked setup
├── experimental_plan_v2.md        v2 extension plan
├── finding_1_2_summary_report.md  Handoff note for Findings 1 and 2
├── pilot/
│   ├── pilot.py                   Minimum pilot runner
│   ├── primary.py                 Full chess grid runner
│   ├── plan_v2_analysis.py        T3 / bias-variance / regime / crossover analyses
│   ├── concept_queries.py         Concept-level CATE / conditional ATE queries
│   ├── conplus.py                 PCA residual augmentation
│   ├── conplus_supervised.py      T-supervised PLS residual augmentation
│   ├── nuisance_diagnostics.py    Propensity AUC / outcome R² diagnostics
│   ├── ablation_game_outcome.py   Oracle-correlation sanity check
│   ├── plot_figures.py            Figure generation
│   ├── gridworld/                 Secondary environment
│   ├── cache/                     Chess caches committed in this snapshot
│   └── outputs/                   Result CSVs and figures committed in this snapshot
└── README.md                      This file
```

## Core Experimental Design

### Treatment

`T` is a rule-based binary label for aggressive vs. conservative moves. It is built from captures, checks, king-zone pressure, sacrifices, and related move features. Engine evaluations are explicitly excluded from treatment assignment.

### Outcome

`Y` is semi-synthetic with a planted ground-truth treatment effect. The DGP is anchored in pre-treatment context and structural board features. Stockfish total evaluation is not used in the outcome path.

### Causal Framing

The concept vector `C = f(S)` is treated as a deterministic encoding of the raw board state `S`, not a separate causal parent. The project studies which covariate basis gives the best adjustment behavior under the same underlying SCM.

## Reproduce the Experiments

All runners live in [pilot/](pilot/).

### 1. Pilot

```bash
cd pilot
python pilot.py
```

Useful flags:

```bash
python pilot.py --refetch
python pilot.py --rebuild
```

Writes [pilot/outputs/pilot_results.csv](pilot/outputs/pilot_results.csv).

### 2. Full Primary Chess Grid

```bash
cd pilot
python primary.py --dry-run
python primary.py
```

Writes [pilot/outputs/primary_results.csv](pilot/outputs/primary_results.csv). The runner is resumable.

### 3. v2 Analysis Pass

```bash
cd pilot
python plan_v2_analysis.py
```

Writes:

- [pilot/outputs/t3_threshold.csv](pilot/outputs/t3_threshold.csv)
- [pilot/outputs/bias_variance.csv](pilot/outputs/bias_variance.csv)
- [pilot/outputs/regime_diagram.csv](pilot/outputs/regime_diagram.csv)
- [pilot/outputs/crossover.csv](pilot/outputs/crossover.csv)

and figures:

- [pilot/outputs/figures/figT3.pdf](pilot/outputs/figures/figT3.pdf)
- [pilot/outputs/figures/figBV.pdf](pilot/outputs/figures/figBV.pdf)
- [pilot/outputs/figures/figRegime.pdf](pilot/outputs/figures/figRegime.pdf)
- [pilot/outputs/figures/figCross.pdf](pilot/outputs/figures/figCross.pdf)

### 4. Concept Queries

```bash
cd pilot
python concept_queries.py --force
```

Writes [pilot/outputs/concept_queries.csv](pilot/outputs/concept_queries.csv) and [pilot/outputs/figures/figQ_concept_queries.pdf](pilot/outputs/figures/figQ_concept_queries.pdf).

### 5. Residual Augmentation

PCA residual baseline:

```bash
cd pilot
python conplus.py
```

Supervised T-PLS residual variant:

```bash
cd pilot
python conplus_supervised.py
```

Writes:

- [pilot/outputs/conplus_results.csv](pilot/outputs/conplus_results.csv)
- [pilot/outputs/conplus_supervised_results.csv](pilot/outputs/conplus_supervised_results.csv)
- [pilot/outputs/figures/figConplus.pdf](pilot/outputs/figures/figConplus.pdf)

### 6. Nuisance Diagnostics

```bash
cd pilot
python nuisance_diagnostics.py
```

Writes [pilot/outputs/nuisance_diagnostics.csv](pilot/outputs/nuisance_diagnostics.csv) and [pilot/outputs/figures/figNuisance.pdf](pilot/outputs/figures/figNuisance.pdf).

### 7. Gridworld Secondary Environment

```bash
cd pilot
python gridworld/runner.py --rebuild-pool
```

Writes:

- [pilot/gridworld/cache/gridworld_pool.parquet](pilot/gridworld/cache/gridworld_pool.parquet)
- [pilot/outputs/gridworld_results.csv](pilot/outputs/gridworld_results.csv)
- [pilot/outputs/figures/figGridworld.pdf](pilot/outputs/figures/figGridworld.pdf)

## Committed Artifacts

This repository currently includes a committed snapshot of:

- chess caches in `pilot/cache/`
- gridworld cache in `pilot/gridworld/cache/`
- result CSVs in `pilot/outputs/`
- figure PDFs in `pilot/outputs/figures/`

That snapshot is intended to make the manuscript and empirical claims inspectable without rerunning the full experiment suite.

## Notes on the Two Follow-up Findings

### Finding 1: Supervised Residual Compression

- PCA residual augmentation is a negative result.
- T-supervised PLS residual augmentation improves on concept in the mid-range.
- It does **not** fully close the asymptotic raw gap.

Primary files:

- [pilot/conplus.py](pilot/conplus.py)
- [pilot/conplus_supervised.py](pilot/conplus_supervised.py)
- [pilot/outputs/conplus_results.csv](pilot/outputs/conplus_results.csv)
- [pilot/outputs/conplus_supervised_results.csv](pilot/outputs/conplus_supervised_results.csv)

### Finding 2: Gridworld Checkerboard Fix

- Gridworld now includes an `S`-only hidden checkerboard confounder.
- The final checked-in coefficients are stronger than the originally requested ones.
- The result is a genuine raw-vs-concept crossover, but more aggressive than the target regime sketch.

Primary files:

- [pilot/gridworld/features.py](pilot/gridworld/features.py)
- [pilot/gridworld/treatment.py](pilot/gridworld/treatment.py)
- [pilot/gridworld/dgp.py](pilot/gridworld/dgp.py)
- [pilot/outputs/gridworld_results.csv](pilot/outputs/gridworld_results.csv)

## Citation

If you reference this work, cite the manuscript in [neurips.tex](neurips.tex).
