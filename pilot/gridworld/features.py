from __future__ import annotations

import numpy as np

from .env import DecisionPoint, GRID_SIZE, N_CELLS, cell_to_coord, manhattan

S_FEATURES = [f"s{k}" for k in range(3 * N_CELLS)]
CONCEPT_FEATURES = ["c_dist_to_goal", "c_obstacle_density", "c_quadrant"]
CONTEXT_FEATURES = ["step_number", "grid_seed", "difficulty", "start_distance"]


def raw_vector(point: DecisionPoint) -> np.ndarray:
    vec = np.zeros(3 * N_CELLS, dtype=np.float32)
    vec[point.agent_cell] = 1.0
    for obstacle in point.obstacles:
        vec[N_CELLS + obstacle] = 1.0
    vec[2 * N_CELLS + point.goal_cell] = 1.0
    return vec


def concept_vector(point: DecisionPoint) -> dict[str, float]:
    r, c = cell_to_coord(point.agent_cell)
    obstacle_cells = set(point.obstacles)

    local = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                if rr * GRID_SIZE + cc in obstacle_cells:
                    local += 1

    if r < GRID_SIZE // 2 and c < GRID_SIZE // 2:
        quadrant = 0
    elif r < GRID_SIZE // 2 and c >= GRID_SIZE // 2:
        quadrant = 1
    elif r >= GRID_SIZE // 2 and c < GRID_SIZE // 2:
        quadrant = 2
    else:
        quadrant = 3

    return dict(
        c_dist_to_goal=float(manhattan(point.agent_cell, point.goal_cell)),
        c_obstacle_density=float(local / 9.0),
        c_quadrant=float(quadrant),
    )


def context_vector(point: DecisionPoint) -> dict[str, float]:
    return dict(
        step_number=float(point.step_number),
        grid_seed=float(point.grid_seed),
        difficulty=float(point.difficulty),
        start_distance=float(point.start_distance),
    )


def row_from_point(point: DecisionPoint) -> dict[str, float]:
    r, c = cell_to_coord(point.agent_cell)
    hidden_checker = float((r + c) % 2)
    row = dict(
        agent_cell=int(point.agent_cell),
        goal_cell=int(point.goal_cell),
        obstacle_count=int(len(point.obstacles)),
        hidden_checker=hidden_checker,
        **context_vector(point),
        **concept_vector(point),
    )
    s = raw_vector(point)
    for i, val in enumerate(s):
        row[S_FEATURES[i]] = float(val)
    return row
