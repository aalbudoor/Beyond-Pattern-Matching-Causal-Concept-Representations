# Experimental Plan v2 — Post-Theory-Refinement Roadmap

**Date:** 2026-04-13
**Status:** Planning document for experiments not yet run. All plans assume the primary grid (`pilot/outputs/primary_results.csv`, 2700 fits) and ablation (`pilot/outputs/ablation_game_outcome.csv`, 240 fits) are already complete.

---

## Context

After the theory-section refinement (Hahn 1998 / Ma–Zhu 2012 / Nabi et al. 2022), Claim 2 was rewritten as a **crossover** story rather than a gap-closing story:

- **Small-$N$ regime:** concept wins (finite-sample nuisance-estimation penalty on high-$d$ raw)
- **Large-$N$, $\gamma>0$ regime:** raw wins (asymptotic sufficiency gap — $C$ is only approximately sufficient for $(T,Y)|X$)
- **Crossover point** $N^{\text{cross}}(\gamma)$ is an empirical measurement of how incomplete the expert concept basis is

The experiments in this plan are intended to:
1. Put concrete numbers behind the crossover claim
2. Visually demonstrate both terms of the theory on the same figure
3. Cover the testbed tasks (T1, T2, T3) promised in the abstract
4. Fill the Concept-Level Causal Queries and Gridworld sections already drafted in `main.tex`

---

## Complete list of missing experiments

| # | Name | Effort | Uses existing data? | Tier |
|---|------|--------|---------------------|------|
| 1 | T3 sample-efficiency threshold | ~30 min analysis | Yes | 1 |
| 2 | Bias-variance decomposition | ~45 min analysis | Yes | 1 |
| 3 | Regime diagram (winner-per-cell) | ~20 min analysis | Yes | 1 |
| 4 | Concept queries Q1-Q3 | ~15 min run (already coded) | No, new fits | 2 |
| 5 | Crossover-point $N^{\text{cross}}(\gamma)$ | ~30 min analysis | Yes | 1 |
| 6 | Gridworld secondary environment | ~5-6 hours code + 30 min run | No, new pipeline | 3 |
| 7 | Concept+residual augmented rep | ~3 hours code + 10 min run | No, new fits | 2 |
| 8 | Nuisance-fit diagnostics (CV AUC/R²) | ~2 hours code + re-run | No, new fits | 3 |

---

## Priority ordering

**Tier 1 — cheapest and highest signal (all use existing data):**
1. **Plan 2 (bias-variance decomposition)** — single most important free experiment. The visual proof of the theory.
2. **Plan 5 (crossover point)** — the number Claim 2 needs.
3. **Plan 3 (regime diagram)** — cheapest figure; can replace or augment Fig 3.
4. **Plan 1 (T3 thresholds)** — fills the T3 task promised in the abstract.

**Tier 2 — new code but bounded scope:**
5. **Plan 4 (concept queries)** — already coded in `pilot/concept_queries.py`, just run.
6. **Plan 7 (concept+residual)** — paper-elevating stretch experiment. Turns the sufficiency-gap finding from a limitation into a contribution.

**Tier 3 — expensive or diminishing returns:**
7. **Plan 6 (gridworld)** — the "testbed not one-off" reviewer-convincer.
8. **Plan 8 (nuisance diagnostics)** — diminishing returns given Plan 2 visualizes similar story.

---

## Plan 1 — T3 sample-efficiency threshold

**Goal:** For each `(rep, γ)`, find the smallest $N$ at which median ATE bias drops below a target $\epsilon$. This instantiates the T3 task promised in the abstract.

**Inputs:** `pilot/outputs/primary_results.csv`.

**Canonical filter:**
- `tau_regime == 'homogeneous'`
- `estimator == 'dr'`
- `outcome_model == 'lgbm'`

**Algorithm:**
```
for each (rep, gamma):
    for epsilon in [0.05, 0.10]:
        group by N, take median(ate_bias) across 10 seeds
        sort by N ascending
        linearly interpolate between adjacent (N_i, bias_i), (N_{i+1}, bias_{i+1})
        find smallest N where median_bias <= epsilon
        record N_T3(rep, gamma, epsilon)
        if median_bias never drops below epsilon in [1k, 20k], record ">20k"
```

**Output:** `pilot/outputs/t3_threshold.csv`
Columns: `rep, gamma, epsilon, N_T3`

**Figure (Fig T3):** 2-panel bar chart, one panel per $\epsilon$. X-axis: $\gamma$ values. Bars grouped by rep. Y-axis: $N_{T3}$ on log scale. Missing-threshold cells: show as a striped bar at 20k with hatch pattern.

**Validation criterion:** Order must be `concept ≤ learned ≤ raw` at small $\gamma$; may flip at $\gamma = 0.6$ for large $\epsilon$.

---

## Plan 2 — Bias-variance decomposition

**Goal:** Separate $\text{MSE}(\hat\tau) = \text{Bias}^2 + \text{Var}$ across seeds per cell, visually showing that concept's variance collapses while its asymptotic bias stays flat, and raw's bias drops while its variance stays high at small $N$. This is the **direct visual proof of the refined theory**.

**Inputs:** `pilot/outputs/primary_results.csv`.

**Algorithm:** For each `(tau_regime, gamma, N, rep, estimator, outcome_model)` cell:
```
ate_hats = all 10 seeds' ate_hat values  (list of floats)
ate_true = ate_true  (constant per cell — it's the planted tau*)

bias_signed = mean(ate_hats) - ate_true           # can be + or -
bias_sq     = (mean(ate_hats) - ate_true) ** 2
variance    = var(ate_hats, ddof=1)               # sample variance across seeds
mse         = bias_sq + variance

sanity_check = mean((ate_hats - ate_true) ** 2)   # must match mse within 1e-9
```

**Output:** `pilot/outputs/bias_variance.csv`
Columns: `N, gamma, tau_regime, rep, estimator, outcome_model, bias_signed, bias_sq, variance, mse`

**Figure (Fig BV):** Two-panel stacked bar plot at canonical setting (homogeneous τ*, dr-lgbm).
- X-axis: N (log scale)
- For each N, three stacked bars (one per rep), each bar with two segments: `bias²` (darker shade) and `variance` (lighter shade)
- Panel (a) = γ=0.3
- Panel (b) = γ=0.6

**Expected pattern:**
- **Raw at small N**: big variance, modest bias² → total MSE dominated by variance
- **Raw at large N**: both shrink → total MSE near zero
- **Concept at all N**: tiny variance, stable bias² → total MSE dominated by a **flat bias² floor** at γ>0
- **Learned**: intermediate on both axes

**Why this matters:** The flat concept-bias² floor at γ>0 IS the sufficiency gap. The shrinking raw-variance IS the finite-sample nuisance penalty decaying. Both terms of Ma–Zhu–Nabi are visible on one plot.

---

## Plan 3 — Regime diagram (winner-per-cell)

**Goal:** A single figure showing which representation wins at each $(\gamma, N)$ cell. Can replace or augment the current RMSE heatmap (Fig 3).

**Inputs:** `pilot/outputs/primary_results.csv`, filter to canonical. Generate both homogeneous and heterogeneous versions.

**Algorithm:**
```
for each (gamma, N) cell:
    med_bias_per_rep = {rep: median(ate_bias across 10 seeds) for rep in [raw, concept, learned]}
    winner = argmin(med_bias_per_rep)
    second_lowest = sorted(med_bias_per_rep.values())[1]
    lowest = sorted(med_bias_per_rep.values())[0]
    margin = second_lowest - lowest   # how decisive the win is
    record winner, margin
```

**Output:** `pilot/outputs/regime_diagram.csv`
Columns: `gamma, N, tau_regime, winner, margin, concept_bias, raw_bias, learned_bias`

**Figure (Fig Regime):** Single colored cell grid, 3 rows (γ) × 5 cols (N). Each cell filled with the color of its winner (red=raw, green=concept, blue=learned). Annotate with the margin value inside the cell. Optionally: alpha the cell by margin so decisive wins look saturated and near-ties look pale.

**Expected shape (from existing data):**
```
γ=0.6 | learned  learned  learned  raw    raw
γ=0.3 | concept  learned  learned  raw    raw
γ=0.0 | concept  learned  concept  learned  learned
        1k       2k       5k       10k    20k
```

**Interpretation:** The "diagonal" from top-left-green to bottom-right-red is the visual signature of the finite-sample ↔ asymptotic tradeoff.

---

## Plan 4 — Concept queries Q1-Q3

**Status:** `pilot/concept_queries.py` already written. Runnable as-is.

**What it does:** For each of 10 seeds, draws N=5,000 positions, generates outcomes under heterogeneous τ* / γ=0.3 / α=0.3, then:

- **Q1 (CATE by material balance):** Splits by `c_material` sign into `{behind, equal, ahead}` and fits a separate DR-LightGBM ATE per stratum per rep. Planted truth:
  - `τ*_behind = 0.1`
  - `τ*_equal = 0.1`
  - `τ*_ahead = 0.4`
  (Because heterogeneous τ* = 0.1 + 0.3·1{material>0}.)

- **Q2 (Conditional ATE at high king-shield):** Filters to `c_king_shield ≥ 75th percentile` and fits ATE on that subset per rep. Planted truth: `E[τ*(X) | king_shield ≥ q75] = 0.1 + 0.3·P(material>0 | king_shield≥q75)`. This is **not** `do(king_safety=high)` — it's a conditional ATE, per the corrected paper text.

- **Q3 (CATE by space-control tercile):** Splits by `c_space` into `{low, mid, high}` terciles and fits separate ATEs. Planted truth: same calculation as Q2 per tercile. Since space is orthogonal to material in the DGP, truths should be nearly equal across terciles — a **false-positive check** for any rep that spuriously infers heterogeneity.

**Total cells:** 10 seeds × 3 reps × (3 + 1 + 3 strata) = **210 fits**, ~15–20 min wall-clock.

**Output:** `pilot/outputs/concept_queries.csv` + `pilot/outputs/figures/figQ_concept_queries.pdf`

**Pre-run knobs (in `concept_queries.py`):**
- `king_shield_mask(df, q=0.75)` — change `q` if you want a different percentile
- Q3 tercile cuts: `.quantile(0.33)` / `.quantile(0.67)` — can change to quartiles
- `CANONICAL_SEEDS = list(range(10))` — drop to `range(3)` for a quick sanity run
- `fit_ate_on_subset` has a positivity guard (`T.sum() < 5`) that returns NaN for degenerate strata — leave it

**Validation criteria:**
- Q1 concept rep should return approximately `{0.10, 0.10, 0.40}` with bias < 0.05 per stratum
- Q1 raw rep should return noisier estimates with bias > 0.10 per stratum (especially `behind`/`ahead` which have smaller `n_sub`)
- Q3 all reps should return near-constant CATE across `{low, mid, high}` — but raw's IQR will be much wider

**Run command:**
```
conda run -n chess-pilot python3 concept_queries.py
```

---

## Plan 5 — Crossover-point $N^{\text{cross}}(\gamma)$ measurement

**Goal:** Quantify the empirical location where raw's median bias first drops below concept's, per γ. This is the number that Claim 2 implicitly cites.

**Inputs:** `pilot/outputs/primary_results.csv`, canonical filter (homogeneous τ*, dr-lgbm).

**Algorithm:**
```
for each (gamma, tau_regime):
    concept_curve = median bias per N for concept
    raw_curve     = median bias per N for raw
    learned_curve = median bias per N for learned

    # find smallest N where raw <= concept (linear interpolation between adjacent N values)
    for each pair (N_i, N_{i+1}) in [(1k, 2k), (2k, 5k), (5k, 10k), (10k, 20k)]:
        if raw[N_i] > concept[N_i] and raw[N_{i+1}] <= concept[N_{i+1}]:
            # linear interpolate in log-N space
            N_cross = interp_log(N_i, N_{i+1}, raw, concept)
            break
    else:
        if all cells: raw > concept:     N_cross = ">20k"
        elif all cells: raw <= concept:  N_cross = "<1k"

    repeat for raw_vs_learned and learned_vs_concept pairs
```

**Output:** `pilot/outputs/crossover.csv`
Columns: `gamma, tau_regime, pair, N_cross, bias_at_cross`
Where `pair` ∈ `{raw_vs_concept, raw_vs_learned, learned_vs_concept}`

**Figure:** Two options:
- **Option A:** Tiny annotation on Fig 2 — vertical dashed lines at each $N^{\text{cross}}$ per panel, labeled with the $N$ value.
- **Option B:** Separate 1×2 plot: $N^{\text{cross}}$ vs $\gamma$, one line per rep pair, with markers indicating "did not cross within grid".

**Expected numbers (pre-computed from existing data):**
- γ=0.0, raw_vs_concept: between 2k and 5k (raw 0.131 at 2k → 0.033 at 5k, vs concept 0.023 at 2k)
- γ=0.3, raw_vs_concept: between 5k and 10k (raw 0.077 at 5k, 0.049 at 10k; concept ~0.083 plateau)
- γ=0.6, raw_vs_concept: between 5k and 10k (raw 0.144 at 5k, 0.116 at 10k; concept ~0.155 plateau)

**Paper use:** Quote the exact numbers in the Claim 2 rewrite where the text currently reads "crossover point $N^{\text{cross}}(\gamma)$" abstractly.

---

## Plan 6 — Gridworld secondary environment

**Goal:** Reproduce the regime structure in a synthetic 10×10 gridworld using the same SCM/DGP machinery. Appendix-only figure; paper already claims replication in the Discussion and Experimental Setup sections, so this is load-bearing for those claims.

**Module layout:** Create `pilot/gridworld/` (new subdirectory):
```
gridworld/
    __init__.py
    env.py            # grid + obstacles + goal generator
    features.py       # S, C, concept functions
    treatment.py      # action-entropy-based T (or simple sigmoid confounding)
    dgp.py            # outcome generator (mirrors primary.py)
    runner.py         # grid runner (mirrors primary.py)
```

### env.py — Decision-point generator

- 10×10 grid
- One goal position (random corner, or always (9,9))
- ~8–12 obstacle cells placed by rejection sampling (avoid goal and agent)
- Agent's current position (random, not on goal or obstacle)
- Episode = one decision point; no trajectory simulation
- Generate 30,000 decision points to match chess pool size

**Data structure:** Each decision point is a dict with `agent_pos`, `goal_pos`, `obstacles (set of tuples)`, `grid_seed`, `step_number`, `difficulty`, `start_distance`.

### features.py — S, C, X encoders

**`S` (raw, 300-d):**
- 100-d one-hot for agent position
- 100-d binary mask for obstacles
- 100-d binary mask for goal
- Concatenated and flattened

**`C` (concept, 3-d):**
1. `c_dist_to_goal` = L1 Manhattan distance from agent to goal
2. `c_obstacle_density` = count of obstacles in 3×3 neighborhood around agent, divided by 9
3. `c_quadrant` = quadrant index (0=NW, 1=NE, 2=SW, 3=SE). Can one-hot encode or keep as integer.

**`X` (context, 4-d):**
1. `step_number` (random integer 1–50)
2. `grid_seed` (integer identifier, cast to float)
3. `difficulty` = total obstacle count normalized by grid area
4. `start_distance` = L1 distance at episode start (analog of Elo differential in chess)

### treatment.py — T assignment

**Recommended (simpler):**
```python
logit(p) = a * c_dist_to_goal_z + b * c_obstacle_density_z + c * x_context_term
T = Bernoulli(sigmoid(logit(p)))
```
Pick `a, b, c` so marginal `P(T=1) ≈ 0.4` and treatment has nontrivial dependence on concepts.

**Alternative (paper text claims this):** Action-entropy based.
- Draw a "policy" over 4 actions:
  - Greedy: deterministic move toward goal → entropy ≈ 0
  - Exploratory: uniform random → entropy = log(4) ≈ 1.39
  - Mix by temperature depending on skill/context
- `T = 1 if entropy > threshold else 0`

Action-entropy is more faithful to the paper text but adds complexity without changing the scientific story. Use the sigmoid version unless you need the paper to match exactly.

### dgp.py — Outcome generation

Mirror `generate_outcomes_regime` from `primary.py` exactly:
```python
phi(X) = 0.4 * difficulty_norm + 0.2 * start_distance_norm + 0.1 * step_norm
psi(C) = 0.3 * c_dist_to_goal_z + 0.2 * c_obstacle_density_z

U ~ N(0, 1)
eps0 ~ N(0, 0.5^2)

Y0 = phi(X) + psi(C) + gamma * U + eps0
Y1 = Y0 + tau_star              # tau_star = 0.3 (homogeneous only for appendix)
T_obs resampled via logit shift alpha_U * U (alpha_U = 0.3)
Y_obs = T_obs * Y1 + (1 - T_obs) * Y0
```

### runner.py — Grid

Copy `primary.py`'s structure. Grid:
- N ∈ {500, 1000, 2000, 5000, 10000} (smaller because d=300 not 1152)
- γ ∈ {0.0, 0.3, 0.6}
- tau_regime ∈ {homogeneous} only
- reps ∈ {raw, concept, learned}
- estimators ∈ {(dr, lgbm), (dr, linear)}
- 10 seeds

**Total:** 5 × 3 × 1 × 3 × 2 × 10 = **900 fits**, ~20–30 min wall-clock.

**Output:** `pilot/outputs/gridworld_results.csv` with same column schema as `primary_results.csv`.

**Figure (Fig Gridworld, appendix):** 2-panel layout
- (a) ATE bias vs N per rep at γ=0.3 — mimic Fig 2's right panel
- (b) Nuisance-swap bar chart at N=1000 — mimic Fig 4

**Validation criterion:** Concept wins at small N (same pattern as chess). Raw catches up or crosses over at large N if `psi(C)` is only a partial sufficient statistic — which it is by construction (obstacles enter via `c_obstacle_density` which discards exact positions). Aim for qualitative reproduction of the regime structure, not matching absolute numbers.

---

## Plan 7 — Concept+residual augmented rep

**Goal:** Test the sufficiency-gap interpretation directly. If the concept basis is missing ~X% of the sufficient information, adding a learned residual should close ~X% of the large-N gap.

**Algorithm:**
```
for each (N, seed, gamma) in canonical grid:
    X_raw = [S, context]
    X_con = [C, context]

    # Residual: what raw knows that concept doesn't
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(C_features, S_features)
    S_predicted_from_C = reg.predict(C_features)
    residual = S_features - S_predicted_from_C

    # Compress residual to manageable dim
    from sklearn.decomposition import PCA
    R = PCA(n_components=16).fit_transform(residual)

    X_conplus = [C, R, context]   # total dim: 12 + 16 + 6 = 34

    fit DR-LGBM on X_conplus
    record ate_hat, bias vs planted tau*
```

**Grid:** Restrict to canonical to keep cost down.
- N ∈ {1000, 2000, 5000, 10000, 20000}
- γ ∈ {0.3, 0.6}
- tau_regime = homogeneous only
- 10 seeds
- estimator = dr-lgbm only
- Just the new rep (`conplus`) — don't re-run raw/concept/learned since we already have those

**Total:** 5 × 2 × 10 = **100 fits**, ~3–5 min wall-clock.

**Output:** `pilot/outputs/conplus_results.csv` with columns matching `primary_results.csv` but with `rep = 'conplus'`.

**Expected result:** If concept is approximately sufficient, `X_conplus` should:
- Match concept's small-N performance (same nuisance-estimation advantage — dim is still low at 34)
- Match raw's large-N performance (residual recovers the missing structure)
- Dominate both reps at every cell in between

**Figure:** Add a fourth line to Fig 2 ("Concept + residual") showing the curve fall below both concept (past the crossover) and raw (at all N).

**Paper framing if it works:** Turns the sufficiency-gap finding from a limitation into a contribution: *"Expert concepts get you 90% of the way with 1% of the dimension; adding a learned residual closes the remaining gap. This decomposes the representation problem into (a) low-dimensional sufficient statistics for what experts know, plus (b) a residual for what they don't."*

---

## Plan 8 — Nuisance-fit diagnostics (CV AUC / R²)

**Goal:** Direct mechanistic evidence that concept's advantage at small $N$ comes from better nuisance estimation, not some other artifact.

**Algorithm:** During each DR-Learner fit in a re-run of the canonical grid, also compute:
- Propensity model cross-validated AUC on held-out folds
- Outcome model cross-validated R² per treatment arm

**Implementation options:**
1. **Clean:** EconML exposes fit nuisance models via `est.models_nuisance_`. Read their CV scores if available, or re-fit using scikit-learn `cross_val_score` on the same X/T/Y.
2. **Simpler:** Fit a standalone `LogisticRegressionCV` and `LGBMRegressorCV` on the same X/T/Y outside the DR-Learner, using the same 5-fold scheme. Record AUC and R². These won't match EconML's internal models exactly but will be close enough mechanistically.

**Output:** Either add two columns to `primary_results.csv` (requires re-running) or save to a separate file `pilot/outputs/nuisance_diagnostics.csv` with columns `N, gamma, rep, seed, prop_auc_cv, outcome_r2_cv`.

**Figure:** 2-panel line plot at γ=0.3
- (a) Propensity AUC vs N per rep
- (b) Outcome R² vs N per rep

**Expected pattern:** Concept's curves should plateau quickly and stay ~flat. Raw's should climb slowly (from near-random at N=1k to near-concept at N=20k). The point where raw's nuisance scores catch up to concept's is the same region as the ATE-bias crossover — mechanistic confirmation.

**Strong evidence scenario:** If concept AUC/R² catches up to raw at large N *while ATE bias stays high for concept*, that proves the large-N problem is **sufficiency (bias)**, not nuisance fit (variance). That's the finest-grained possible validation of the theory's two-part argument.

---

## Recommended execution order

**Phase 1 (immediate, token-cheap, all free from existing data):**
Do Plans 1, 2, 3, 5 first. These require no new fits, take < 2 hours of coding total, and produce the most theoretically load-bearing results. Plan 2 is highest priority.

**Phase 2 (new fits, bounded scope):**
Run Plan 4 (already coded). Then decide whether Plan 7 is worth the effort — it's the single biggest paper elevation if it works.

**Phase 3 (expensive):**
Plans 6 and 8 are optional depending on how much reviewer-convincing the paper needs for the testbed framing. Plan 6 is load-bearing if the `main.tex` gridworld section stays in; Plan 8 is a nice-to-have.

**Deliverable from Phase 1:**
- `pilot/outputs/t3_threshold.csv` + `figures/figT3.pdf`
- `pilot/outputs/bias_variance.csv` + `figures/figBV.pdf`  ← most important
- `pilot/outputs/regime_diagram.csv` + `figures/figRegime.pdf`
- `pilot/outputs/crossover.csv` + crossover numbers patched into Claim 2 text

These four deliverables alone — computed from the existing 2700-fit grid, no new runs — provide concrete empirical support for every claim made in the refined theory section.
