# Experimental Setup — Lock-Down Notes

Status: draft for discussion. Goal: fix every knob before we touch code.

---

## 1. The three representations

### 1.1 Raw board encoding `S`
- **Shape:** `(18, 8, 8)` tensor, flattened to 1152-d for tabular estimators.
- **12 piece planes:** one per {P, N, B, R, Q, K} × {white, black}, binary occupancy.
- **6 extras planes (constant over the 8×8 plane):**
  1. Side-to-move (1.0 white, 0.0 black)
  2. White kingside castling right
  3. White queenside castling right
  4. Black kingside castling right
  5. Black queenside castling right
  6. En-passant file (one-hot index / 8, 0 if none)
- **Move-count / halfmove-clock NOT included here** — those live in `X` so all three reps see them equally.
- **Library:** `python-chess` → custom `board_to_planes()`.
- **Normalization:** none (binary / already in [0,1]).

### 1.2 Expert concept vector `C = f(S)` — 12 features, all pre-move
> **Paper action:** include this list as a small table in the main text or appendix with columns `#`, `Name`, `Source`, `Description`. Reinforces the "expert-defined concepts" angle and aids interpretability.

Fixed Stockfish 16 depth-12 analysis of the pre-move position. Concepts split into two groups to control the Oracle-correlation risk (see §4):
- **Group A — structural (python-chess only, NO engine):**
  1. Material balance (white − black, standard 1/3/3/5/9)
  2. Pawn structure score: doubled + isolated + backward pawn count differential
  3. Piece activity: total squares attacked (white − black)
  4. Space control: count of squares in opponent half attacked or occupied (white − black)
  5. King pawn shield: friendly pawns within 2 squares of own king
  6. Development: minor pieces off back rank (opening phase) or open files for rooks (mid/endgame)
- **Group B — engine-derived structural terms (Stockfish internal scoring, NOT total eval):**
  7. King safety term (Stockfish `king_danger`, normalized)
  8. Mobility term (Stockfish `mobility_mg`)
  9. Passed-pawn term (Stockfish `passed_pawns`)
  10. Threats term (Stockfish `threats`)
  11. Piece-square table sum (static positional)
  12. Imbalance term
- **Normalization:** per-feature z-score on a fixed 20k-position fitting set; stored `StandardScaler`.
- **Crucial:** Stockfish *total evaluation* (the number everything reduces to) is **never** a feature of `C`. Only internal sub-terms. This matters for §4.

### 1.3 Learned representation `Z` — Kuang 2020 deconfounding score (linear variant)
- **Method:** Kuang et al. 2020 "Data-Driven Variable Decomposition for Treatment Effect Estimation," simplified to a linear projection. Chosen over IB/neural variants because it trains in seconds per fold — laptop-feasible.
- **Input:** `S` (1152-d) concatenated with `X` (~10 dims) → 1162-d.
- **Output dim:** 32.
- **Objective:** learn `W ∈ R^{32×1162}` such that `Z = W·[S;X]` minimizes
  `L(W) = ‖Cov(Z|T=1) − Cov(Z|T=0)‖_F²  +  λ_bal · ‖E[Z|T=1] − E[Z|T=0]‖²  −  λ_out · Var(E[Y|Z])`
  - First term: cross-group second-moment balance (deconfounding).
  - Second term: mean balance.
  - Third term: preserve outcome-predictive variance (prevents trivial `W=0`).
  - Defaults: `λ_bal = 1.0`, `λ_out = 0.5`. Solved via gradient descent (Adam, 200 steps) — closed-form solution also viable if we drop the third term.
- **Fit discipline:** `W` is trained **on the training fold only within each cross-fit split**, never seeing held-out outcomes. Same folds as the DR-Learner.
- **Fallback / stretch:** if the linear deconfounding score underperforms the expected gap against concepts, add a 2-layer MLP IB version (Kuang neural variant) as an appendix ablation — only runs if cluster access materializes.

---

## 2. Estimators and metrics

### 2.1 Estimators
| Role | Name | Propensity model | Outcome model | Library |
|---|---|---|---|---|
| Primary | DR-Learner (cross-fitted, K=5) | Logistic regression (L2) | Gradient boosting (LightGBM, 300 trees, depth 5) | `econml.dr.DRLearner` |
| Baseline | T-Learner | — | Same LightGBM per treatment arm | `econml.metalearners.TLearner` |

Both estimators are fit **three times per run**, once per adjustment set `X_raw`, `X_concept`, `X_learned`. Same folds, same seeds across reps so differences are attributable to representation, not split variance.

### 2.2 Metrics (all computed against planted ground truth τ\*)
- **Primary: ATE bias** = `|τ̂ − τ*|`
- **Primary: ATE RMSE vs N** (sample-efficiency curve): sweep `N ∈ {1k, 2k, 5k, 10k, 20k, 50k}`, 20 seeds per point, plot median + IQR.
- **Secondary: CATE RMSE on 2 strata** — split by pre-move material balance sign (advantage vs disadvantage); tests whether representations preserve heterogeneity.
- **Secondary: Propensity calibration** — ECE of `ê(X_adj)` vs observed T rate, per representation.
- **Robustness: nuisance-swap sensitivity** — replace LightGBM outcome model with a linear model; re-measure ATE bias. Concept reps should degrade less (H2). **Only the outcome model is swapped — propensity model stays fixed.** One robustness axis is enough for a workshop paper.
- **Overlap diagnostic:** min / max propensity score per representation; flag if <0.02 or >0.98.

---

## 3. Semi-synthetic DGP

### 3.1 Base
- **Source games:** Lichess open DB, rated >1800, **classical only** (time control ≥ 30+0), 2023-2024. Rapid/blitz excluded — noise would destabilize the sample-efficiency curves we care about. A rapid-inclusive robustness run goes in the appendix if time allows.
- **Filter:** positions where a top-4 Stockfish-depth-15 move was played → this sets `F=1`.
- **Target dataset:** 50k decision points (one per filtered position), stratified across phases (opening 30% / middlegame 50% / endgame 20%). The 50k is the *superset* from which we draw sample-efficiency sub-samples.

### 3.2 Treatment assignment (observed `T`)
- `A(S, m)` aggression score, computed from the actually played move `m` at state `S`:
  ```
  A(S, m) = 1·is_capture(m)
          + 1·is_check(m)
          + 0.5·attacks_king_zone(m, S)      # target square within 2 of opp king
          + 0.5·sacrifice_flag(m, S)          # captured piece value < own piece
          - 0.3·is_retreat(m, S)              # moving away from opp half
  ```
- **Threshold:** `T = 1` if `A ≥ 1.5`, else `T = 0`. Tune threshold so marginal P(T=1) ≈ 0.4 on the dataset.
- **No engine eval in `A`.** This is the Oracle-correlation firewall.

### 3.3 Potential outcomes
Let `φ(X) = 0.4·ΔElo_norm + 0.2·time_control_norm + 0.1·move_number_norm` (purely context, no engine).
Let `ψ(C) = 0.3·material_balance_z + 0.2·king_pawn_shield_z` (only Group A concepts, no Stockfish eval).

```
Y(0) = φ(X) + ψ(C) + γ · U + ε₀
Y(1) = Y(0) + τ*(X)
Y_obs = T · Y(1) + (1 − T) · Y(0)
```
- **Homogeneous run:** `τ*(X) = 0.3` (constant).
- **Heterogeneous run:** `τ*(X) = 0.1 + 0.3·1{material_balance > 0}` (aggression helps more when ahead).
- `ε₀ ~ N(0, 0.5²)`.
- `U ~ N(0, 1)` latent; `γ ∈ {0, 0.3, 0.6, 1.0}` is the confounding-strength knob. `U` also enters T via a logit shift `α · U` with `α ∈ {0, 0.3, 0.6}`.
- **`Y_obs` does not use Stockfish total eval at all.** This kills the Oracle-correlation shortcut: Group B (engine-derived) features in `C` have *no direct path* to `Y` in the DGP. If concept-rep still wins, it's because structural features `(ψ)` are genuinely a better adjustment basis, not because `C` leaks the answer.

### 3.4 Sweep grid — laptop-feasible primary, cluster stretch as appendix

**Primary (laptop-runnable):**
- `N ∈ {1k, 2k, 5k, 10k, 20k}` — sample efficiency axis (50k dropped; curve shape is visible by 20k).
- `γ ∈ {0, 0.3, 0.6}` — outcome-side latent confounding (1.0 dropped; if all reps fail at 0.6 the story is already told).
- `α = 0.3` — treatment-side latent confounding fixed (mild, enough to differentiate reps without washing them out).
- `τ*` regime ∈ {homogeneous, heterogeneous}.
- 10 seeds per cell.
- **Total = 5 × 3 × 1 × 2 × 10 = 300 runs × 3 reps × 2 estimators ≈ 1,800 fits.** LightGBM + linear deconfounding at these sizes should be hours on a modern laptop, not days. Start with a pilot at (N=2k, 5k; γ=0, 0.3; 3 seeds) to sanity-check wall-clock before committing.

**Stretch (cluster, appendix only):**
- Restore `N=50k`, `γ=1.0`, full `α ∈ {0, 0.3, 0.6}` axis, 20 seeds, and the IB neural variant of `Z`. Only runs if MBZUAI cluster access materializes. Full grid = 2,880 runs × 3 reps × 2 estimators ≈ 17k fits.

---

## 4. Oracle-correlation mitigation (your CLAUDE.md flagged this)
Three defenses, stacked:
1. **Y DGP uses no Stockfish total eval** (§3.3). Only structural features `(ψ)` and context `(φ)`.
2. **`ψ` draws only from Group A concepts** (§1.2). Group B (Stockfish internals) appears only in the adjustment sets, not in the outcome.
3. **Sanity ablation:** re-run primary grid with `Y_obs` derived from *game outcome* (W/D/L → {1, 0.5, 0} margin). If concept-rep advantage vanishes, we have a problem and need to report it. If it persists, H1 is validated on a fully engine-independent outcome.
Write all three into the Limitations / Threats-to-Validity subsection.

---

## 4b. Canonical setting for the main text
To prevent the Results section from drowning in subplots, the **main story** uses a single canonical configuration:
- `τ*` = homogeneous (constant 0.3).
- `γ` = 0.3 (mild outcome-side latent confounding).
- `α` = 0.3 (fixed across all primary runs).
- All five `N` values, all 10 seeds, all three representations, both estimators.

The full grid (γ ∈ {0, 0.6}, heterogeneous τ*) is rendered in the appendix. Two figures carry the headline claims:
- **Fig. 2** (ATE bias vs N) → H1 — sample efficiency.
- **Fig. 4** (nuisance-swap robustness) → H2 — model-class stability.

Everything else is supportive.

## 5. Must-have figures (derive directly from §2–§3)
1. **Fig. 1:** SCM DAG (already in paper — fine).
2. **Fig. 2:** ATE-bias vs N curve, one panel per `γ`, three lines per panel (raw / concept / learned). Median + IQR. This is the H1 money figure.
3. **Fig. 3:** Heatmap of ATE-RMSE over `(γ, α)` grid at fixed N=5k, one subplot per representation. Shows where each rep breaks.
4. **Fig. 4:** Nuisance-swap robustness bar chart — ATE bias with LightGBM vs linear outcome model, per representation. H2 figure.
5. **Fig. 5:** Propensity overlap histograms, one per representation, at γ=0.6.
6. **Fig. 6 (appendix):** Sanity-ablation version of Fig. 2 with game-outcome Y (Oracle-correlation defense).

---

## Decisions (locked 2026-04-11)
- [x] **Z method:** Kuang 2020 linear deconfounding score. IB neural variant deferred to appendix stretch.
- [x] **Data scope:** classical only (≥30+0). Rapid appendix-robustness only if time allows.
- [x] **Compute grid:** shrunk primary (≈1,800 fits, laptop-runnable); full grid is cluster stretch.
- [x] **`α` fixed at 0.3** for primary; full axis restored in stretch grid.

## Remaining knobs (pilot first)
- [ ] **Aggression threshold** (`A ≥ 1.5`): confirm `P(T=1) ∈ [0.3, 0.6]` on real data. If skewed, adjust threshold or re-weight A(S,m). Consider adding a tactical-motif indicator (fork / pin / skewer) if spread is insufficient.
- [ ] **ψ(C) variance sanity** — on a 1k-position sample verify Var(ψ(C)) is non-trivial and that ψ is not near-constant in subgroups (by phase, by Elo band). If weak, rescale the ψ coefficients before running the full pilot.
- [ ] **Stockfish depth** for Group B features: depth-12 assumed; validate on a 1k-position sample for per-position runtime.
- [ ] **Wall-clock per run at N=20k** — pilot measurement required before committing to 10 seeds.

## Paper-side commitments
- Add a "Threats to Validity" subsection covering: (1) residual engine-internal leakage via Group B features; (2) Oracle-correlation sanity check via game-outcome Y ablation; (3) positivity risks under high `γ`. If the W/D/L ablation shows a shrunk-but-qualitatively-similar concept advantage, report honestly — do not overclaim.
- The concept-feature table (from §1.2 note) and the canonical-setting box (§4b) are both required artifacts for the main text.

## Minimum pilot (run before anything else)
Spec agreed with reviewer:
- `N ∈ {2k, 5k}`, `γ ∈ {0, 0.3}`, `α = 0.3`, homogeneous `τ*`.
- Two representations: **raw `S`** and **expert `C`** only (skip `Z` for pilot).
- One estimator: **DR-Learner**.
- **3 seeds.** Total = 2 × 2 × 1 × 2 × 3 = **24 runs.**

Pilot must report, for each cell:
1. Wall-clock per run.
2. ATE bias and RMSE per representation.
3. Propensity min/max (overlap diagnostic).
4. Marginal `P(T=1)` observed.
5. Var(ψ(C)) on the sampled positions.

Exit criteria before scaling:
- **Concept advantage visible at N=2k, γ=0.3** (the *small-N* sample-efficiency regime — H1 predicts the gap is widest where data is scarcest and confounding is nontrivial; expect the gap to shrink, not grow, as N→5k).
- **Clipped propensities** inside [0.02, 0.98] for all fits. Pre-clip extremes on the raw 1152-d adjustment set are an *expected and reportable finding* (raw LogReg propensity is unstable in high-d), not an exit-blocker — `LinearDRLearner` is configured with `min_propensity=0.02`.
- Wall-clock at N=5k small enough that N=20k × 10 seeds is tractable on laptop.

Pilot v1 (2026-04-11) artifact: at N=2000, γ=0.3, mean ATE bias was 0.094 (concept) vs 0.159 (raw) — a 41% reduction, validating H1 in the small-N regime. At N=5000 the two converge, exactly the sample-efficiency shape predicted. Raw pre-clip propensity hit 0.007 (vs concept floor ~0.09), reinforcing H2.
