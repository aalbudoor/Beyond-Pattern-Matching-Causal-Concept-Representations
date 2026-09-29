# Summary Report: Finding 1 and Finding 2

This note scopes only to the two follow-up findings that were explicitly requested after the main v2 experiment pass:

- Finding 1: supervised `conplus` residual compression
- Finding 2: gridworld checkerboard fix

It is written as a handoff for a research assistant who needs to know what was changed, what was run, what the outputs mean, and what caveats remain.

## Scope

This report does **not** summarize the full Plan 1-8 run. It only covers:

- the supervised residual augmentation follow-up built on top of the existing PCA `conplus` experiment
- the gridworld hidden-confounder follow-up intended to create a genuine raw-vs-concept crossover

## Finding 1: Supervised `conplus` Residual Compression

### Objective

The goal was to test whether the large-`N` raw advantage could be recovered by augmenting the concept basis with a low-dimensional residual of `S` that is **supervised by treatment** rather than chosen by variance alone.

The requested interpretation was:

- keep the existing PCA residual result as a negative finding
- add a new `T`-supervised PLS variant
- compare both against `raw`, `concept`, and `learned`

### Code Changes

Baseline PCA variant already existed in [pilot/conplus.py](../pilot/conplus.py). That file still defines the original residual pipeline:

- regress `S` on `C`
- take the residual
- compress with PCA
- adjust on `[C, PCA residual, context]`

The new supervised variant was added in [pilot/conplus_supervised.py](../pilot/conplus_supervised.py).

Key implementation details:

- `fit_conplus_tpls(...)` was added at [pilot/conplus_supervised.py](../pilot/conplus_supervised.py)
- it regresses `S` on `C`, standardizes the residual, then fits `PLSRegression` against observed `T`
- the runner writes `rep="conplus_tpls"` and saves to [pilot/outputs/conplus_supervised_results.csv](../pilot/outputs/conplus_supervised_results.csv)
- the plotting path was extended so [figConplus.pdf](../pilot/outputs/figures/figConplus.pdf) now overlays:
  - `raw`
  - `concept`
  - `learned`
  - PCA residual (`conplus_pca`, loaded from [pilot/outputs/conplus_results.csv](../pilot/outputs/conplus_results.csv))
  - T-PLS residual (`conplus_tpls`, loaded from [pilot/outputs/conplus_supervised_results.csv](../pilot/outputs/conplus_supervised_results.csv))

Paper text for this finding was added in [paper/main.tex](../paper/main.tex).

### What Was Run

The supervised variant was run end-to-end with:

```bash
conda run -n chess-pilot python conplus_supervised.py
```

Important environment note:

- `conda run -n chess-pilot python3 ...` resolved to Homebrew Python on this machine, not the Conda env interpreter
- `conda run -n chess-pilot python ...` was the correct invocation

The run completed successfully and wrote:

- [pilot/outputs/conplus_supervised_results.csv](../pilot/outputs/conplus_supervised_results.csv) with 100 result rows
- [pilot/outputs/figures/figConplus.pdf](../pilot/outputs/figures/figConplus.pdf)

The runner is resume-safe: rerunning `--dry-run` reported `done = 100; remaining = 0`.

### Results

The comparison uses the homogeneous-`tau*`, DR-Learner, LightGBM slice from [pilot/outputs/primary_results.csv](../pilot/outputs/primary_results.csv), plus the two residual result files:

- [pilot/outputs/conplus_results.csv](../pilot/outputs/conplus_results.csv)
- [pilot/outputs/conplus_supervised_results.csv](../pilot/outputs/conplus_supervised_results.csv)

Median ATE bias highlights:

- At `gamma = 0.3, N = 1000`:
  - `concept = 0.0787`
  - `learned = 0.0827`
  - `conplus_pca = 0.0902`
  - `conplus_tpls = 0.1037`
  - `raw = 0.4317`
- At `gamma = 0.3, N = 5000`:
  - `learned = 0.0682`
  - `conplus_tpls = 0.0747`
  - `raw = 0.0773`
  - `conplus_pca = 0.0827`
  - `concept = 0.0830`
- At `gamma = 0.3, N = 20000`:
  - `raw = 0.0336`
  - `learned = 0.0673`
  - `conplus_pca = 0.0776`
  - `concept = 0.0778`
  - `conplus_tpls = 0.0798`
- At `gamma = 0.6, N = 5000`:
  - `learned = 0.1428`
  - `raw = 0.1443`
  - `conplus_tpls = 0.1492`
  - `concept = 0.1555`
  - `conplus_pca = 0.1563`
- At `gamma = 0.6, N = 20000`:
  - `raw = 0.1067`
  - `learned = 0.1377`
  - `concept = 0.1490`
  - `conplus_pca = 0.1493`
  - `conplus_tpls = 0.1509`

### Interpretation

The supervised residual experiment produced a **partial success**, not a full closure of the gap.

What worked:

- The PCA variant is a clear negative control. It stays near concept and does **not** recover the raw large-`N` advantage.
- The T-PLS variant improves over concept and over PCA in the middle of the grid, especially at:
  - `gamma = 0.3, N = 2000` (`0.0794` vs `0.0832` for concept)
  - `gamma = 0.3, N = 5000` (`0.0747` vs `0.0830` for concept)
  - `gamma = 0.6, N = 2000` (`0.1498` vs `0.1587` for concept)
  - `gamma = 0.6, N = 5000` (`0.1492` vs `0.1555` for concept)

What did **not** happen:

- `conplus_tpls` did not match raw at the largest `N`
- `conplus_tpls` did not dominate learned
- raw still wins asymptotically at:
  - `gamma = 0.3, N = 10000, 20000`
  - `gamma = 0.6, N = 10000, 20000`

Bottom line:

- the sufficiency gap is **not** located in high-variance residual directions, because PCA fails
- treatment-discriminative residual directions **do** matter, because T-PLS helps
- but this 16-dimensional linear T-PLS residual is **not enough** to recover the full asymptotic raw advantage

Recommended interpretation for the assistant:

- keep PCA as a negative result
- describe T-PLS as evidence that part of the missing information is treatment-discriminative
- do **not** claim that the sufficiency gap was closed
- if this line is extended later, the natural next step is an outcome-aware or cross-fitted supervised residual variant

## Finding 2: Gridworld Checkerboard Fix

### Objective

The goal was to modify the gridworld secondary environment so that raw `S` contains a genuine hidden confounder that concept `C` does not, thereby producing a real large-`N` crossover rather than persistent concept dominance.

The proposed hidden confounder was checkerboard parity:

`chi(S) = (r + c) mod 2`

where `(r, c)` are the agent cell coordinates.

### Code Changes

The implementation touched three gridworld data-generation files:

- [pilot/gridworld/features.py](../pilot/gridworld/features.py)
- [pilot/gridworld/treatment.py](../pilot/gridworld/treatment.py)
- [pilot/gridworld/dgp.py](../pilot/gridworld/dgp.py)

Specific changes:

- [pilot/gridworld/features.py](../pilot/gridworld/features.py) now adds `hidden_checker` to each cached row
- `hidden_checker` was **not** added to `CONCEPT_FEATURES` or `CONTEXT_FEATURES`; it remains outside the concept adjustment set
- [pilot/gridworld/treatment.py](../pilot/gridworld/treatment.py) now includes a checker term in the treatment score
- [pilot/gridworld/dgp.py](../pilot/gridworld/dgp.py) now includes a checker term in `psi`

Paper text was updated in [paper/main.tex](../paper/main.tex).

### Important Deviation From the Initial Requested Coefficients

The initial requested coefficients were:

- treatment checker term: `+0.6`
- outcome checker term: `+0.25`

Those settings were implemented first, but they did **not** produce the desired raw-vs-concept crossover. Per the requested failure rule, the checker signal was then strengthened.

The **final locked-in setting** that was actually rerun and left in code is:

- treatment checker coefficient: `+1.0` in [pilot/gridworld/treatment.py](../pilot/gridworld/treatment.py)
- outcome checker coefficient: `+0.5` in [pilot/gridworld/dgp.py](../pilot/gridworld/dgp.py)

This means the paper text was also updated to reflect the stronger final DGP:

- [paper/main.tex](../paper/main.tex) now says the outcome DGP adds a `$0.5\\,\\chi$` term

### What Was Run

The stale pool and stale grid results were removed, then the full grid was rerun with pool rebuild:

```bash
conda run -n chess-pilot python pilot/gridworld/runner.py --rebuild-pool
```

The rerun completed successfully:

- full grid size: 900 fits
- wall clock: about 20 minutes

Artifacts written:

- rebuilt pool: [pilot/gridworld/cache/gridworld_pool.parquet](../pilot/gridworld/cache/gridworld_pool.parquet)
- results: [pilot/outputs/gridworld_results.csv](../pilot/outputs/gridworld_results.csv)
- figure: [pilot/outputs/figures/figGridworld.pdf](../pilot/outputs/figures/figGridworld.pdf)

### Results

The key summary below uses the `dr + lgbm` slice of [pilot/outputs/gridworld_results.csv](../pilot/outputs/gridworld_results.csv), because that is the directly comparable regime panel.

Winner-by-cell pattern:

- `gamma = 0.0`:
  - `N = 500`: `concept` wins
  - `N = 1000`: `concept` wins
  - `N = 2000, 5000, 10000`: `raw` wins
- `gamma = 0.3`:
  - `N = 500`: `concept` wins
  - `N = 1000, 2000, 5000, 10000`: `raw` wins
- `gamma = 0.6`:
  - `N = 500`: `concept` wins
  - `N = 1000, 2000, 5000, 10000`: `raw` wins

Representative median ATE biases:

- `gamma = 0.3, N = 500`:
  - `concept = 0.2797`
  - `raw = 0.6145`
  - `learned = 0.4994`
- `gamma = 0.3, N = 2000`:
  - `raw = 0.1062`
  - `concept = 0.2632`
  - `learned = 0.4302`
- `gamma = 0.6, N = 500`:
  - `concept = 0.3539`
  - `raw = 0.5799`
  - `learned = 0.5180`
- `gamma = 0.6, N = 10000`:
  - `raw = 0.2713`
  - `concept = 0.3211`
  - `learned = 0.4968`

### Interpretation

This finding is also a **partial success with a caveat**.

What worked:

- the checkerboard fix did create a genuine raw-vs-concept crossover
- concept still wins in the small-`N` corner at `gamma >= 0.3`
- raw wins at larger `N` once it has enough data to exploit the extra `S`-only confounder information
- learned remains consistently weak, which is directionally aligned with the chess narrative

What did **not** line up cleanly with the original target regime:

- the crossover happens **too early**
  - raw already wins by `N = 1000` for `gamma = 0.3` and `gamma = 0.6`
- raw also wins at large `N` even for `gamma = 0.0`
  - the intended story was that `gamma = 0.0` should show no sufficiency-gap-driven raw advantage

Bottom line:

- the gridworld environment now does demonstrate the qualitative existence of an `S`-only hidden confounder
- it does reproduce the basic “concept small-`N`, raw large-`N`” crossover behavior
- but the final tuned version is **not** a perfect structural replica of the desired chess-style regime map

Recommended interpretation for the assistant:

- describe the checkerboard fix as having produced a real crossover
- be explicit that the exact requested `0.6 / 0.25` coefficients failed and the final rerun used `1.0 / 0.5`
- flag that the final gridworld is stronger than intended because raw also wins at large `N` for `gamma = 0.0`
- do not overstate this as an exact replication of the chess crossover surface

## Net Assessment Across the Two Findings

These two follow-ups sharpened the paper’s story, but both came back as **qualified** rather than clean wins:

- Finding 1 supports the claim that the missing information is not in high-variance residual directions, and that treatment-supervised residual structure matters
- Finding 1 does **not** justify saying that supervised residual compression fully closes the large-`N` gap
- Finding 2 succeeds in making gridworld exhibit a true crossover
- Finding 2 does **not** justify saying the gridworld now reproduces the chess regime map exactly

If the assistant needs a one-sentence takeaway:

> Finding 1 shows that PCA residual augmentation fails while T-supervised residual augmentation helps but does not fully recover raw’s asymptotic edge; Finding 2 injects a real hidden confounder into gridworld and creates a crossover, but only after strengthening the parity signal beyond the originally requested coefficients, yielding a crossover that is qualitatively right but too aggressive.
