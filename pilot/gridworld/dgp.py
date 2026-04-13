from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_A_FEATURES = ["c_dist_to_goal", "c_obstacle_density"]
CONCEPT_FEATURES = ["c_dist_to_goal", "c_obstacle_density", "c_quadrant"]
CONTEXT_FEATURES = ["step_number", "grid_seed", "difficulty", "start_distance"]
S_FEATURES = [f"s{k}" for k in range(300)]

ALPHA_U = 0.3
TAU_HOMO = 0.3


def _standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu, sd = out[c].mean(), out[c].std()
        out[c] = (out[c] - mu) / (sd + 1e-8)
    return out


def generate_outcomes_regime(df: pd.DataFrame, gamma: float,
                             tau_regime: str = "homogeneous",
                             alpha_U: float = ALPHA_U,
                             seed: int = 42) -> pd.DataFrame:
    if tau_regime != "homogeneous":
        raise ValueError("gridworld runner only supports homogeneous tau* in v2")

    rng = np.random.default_rng(seed)
    d = _standardize(
        df,
        ["difficulty", "start_distance", "step_number"] + GROUP_A_FEATURES + ["hidden_checker"],
    )
    n = len(d)

    phi = (
        0.4 * d["difficulty"].to_numpy()
        + 0.2 * d["start_distance"].to_numpy()
        + 0.1 * d["step_number"].to_numpy()
    )
    psi = (
        0.3 * d["c_dist_to_goal"].to_numpy()
        + 0.2 * d["c_obstacle_density"].to_numpy()
        + 0.5 * d["hidden_checker"].to_numpy()
    )
    U = rng.standard_normal(n)
    eps0 = rng.standard_normal(n) * 0.5

    Y0 = phi + psi + gamma * U + eps0
    tau = np.full(n, TAU_HOMO, dtype=np.float64)
    Y1 = Y0 + tau

    base_logit = np.where(df["T"].to_numpy() == 1, 1.2, -1.2)
    logit = base_logit + alpha_U * U
    p = 1.0 / (1.0 + np.exp(-logit))
    T_obs = (rng.random(n) < p).astype(int)
    Y_obs = T_obs * Y1 + (1 - T_obs) * Y0

    out = df.copy()
    out["Y_obs"] = Y_obs
    out["T_obs"] = T_obs
    out["_tau_star_mean"] = float(tau.mean())
    return out
