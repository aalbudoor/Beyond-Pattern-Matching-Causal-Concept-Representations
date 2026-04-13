from __future__ import annotations

import numpy as np
import pandas as pd


def _z(x: pd.Series) -> np.ndarray:
    arr = x.to_numpy(dtype=float)
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def _calibrate_intercept(score: np.ndarray, target: float = 0.4) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(mid + score)))
        if p.mean() > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def assign_treatment_rule(df: pd.DataFrame, seed: int = 0, target_rate: float = 0.4) -> tuple[pd.DataFrame, float]:
    dist_z = _z(df["c_dist_to_goal"])
    dens_z = _z(df["c_obstacle_density"])
    diff_z = _z(df["difficulty"])
    step_z = _z(df["step_number"])
    start_z = _z(df["start_distance"])
    checker_z = _z(df["hidden_checker"])

    score = (
        0.9 * dist_z
        + 0.8 * dens_z
        + 0.35 * diff_z
        + 0.15 * start_z
        - 0.10 * step_z
        + 1.0 * checker_z
    )
    intercept = _calibrate_intercept(score, target=target_rate)
    logit = intercept + score
    p = 1.0 / (1.0 + np.exp(-logit))

    rng = np.random.default_rng(seed)
    out = df.copy()
    out["p_rule"] = p
    out["T"] = (rng.random(len(df)) < p).astype(int)
    return out, float(intercept)
