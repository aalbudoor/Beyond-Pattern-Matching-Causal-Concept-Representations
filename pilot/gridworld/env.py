from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID_SIZE = 10
N_CELLS = GRID_SIZE * GRID_SIZE
DEFAULT_POOL_SIZE = 30_000
CORNER_CELLS = (0, GRID_SIZE - 1, N_CELLS - GRID_SIZE, N_CELLS - 1)


@dataclass(frozen=True)
class DecisionPoint:
    agent_cell: int
    goal_cell: int
    obstacles: tuple[int, ...]
    grid_seed: int
    step_number: int
    difficulty: float
    start_distance: int


def cell_to_coord(cell: int) -> tuple[int, int]:
    return divmod(cell, GRID_SIZE)


def manhattan(cell_a: int, cell_b: int) -> int:
    ra, ca = cell_to_coord(cell_a)
    rb, cb = cell_to_coord(cell_b)
    return abs(ra - rb) + abs(ca - cb)


def _sample_free_cell(rng: np.random.Generator, forbidden: set[int]) -> int:
    while True:
        cell = int(rng.integers(0, N_CELLS))
        if cell not in forbidden:
            return cell


def _sample_obstacles(rng: np.random.Generator, count: int, forbidden: set[int]) -> tuple[int, ...]:
    obstacles: set[int] = set()
    while len(obstacles) < count:
        cell = _sample_free_cell(rng, forbidden | obstacles)
        obstacles.add(cell)
    return tuple(sorted(obstacles))


def sample_decision_points(n: int = DEFAULT_POOL_SIZE, seed: int = 0) -> list[DecisionPoint]:
    rng = np.random.default_rng(seed)
    points: list[DecisionPoint] = []

    while len(points) < n:
        grid_seed = int(rng.integers(0, 1_000_000_000))
        goal_cell = int(rng.choice(CORNER_CELLS))
        agent_cell = _sample_free_cell(rng, {goal_cell})
        obstacle_count = int(rng.integers(8, 13))
        obstacles = _sample_obstacles(rng, obstacle_count, {goal_cell, agent_cell})
        start_cell = _sample_free_cell(rng, {goal_cell, agent_cell, *obstacles})

        points.append(DecisionPoint(
            agent_cell=agent_cell,
            goal_cell=goal_cell,
            obstacles=obstacles,
            grid_seed=grid_seed,
            step_number=int(rng.integers(1, 51)),
            difficulty=float(obstacle_count / N_CELLS),
            start_distance=manhattan(start_cell, goal_cell),
        ))

    return points

