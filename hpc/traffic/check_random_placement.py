"""Offline geometry check mirroring TrafficScenarioManager's ten-car sampler.

This checks map feasibility only; the Unity runtime smoke test is still required.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_trajectories import asphalt_components, load_map_grid  # noqa: E402

MAPS = ROOT / "Assets/Resources/Maps/VoronoiVal"
POOL = 10
CELL_SIZE = 1.5
MIN_SEPARATION = 8.0
MIN_OWN = 20.0


def select(cells: list[tuple[int, int]], seed: int) -> tuple[float, float] | None:
    rng = random.Random(seed)
    min_sep2 = (MIN_SEPARATION / CELL_SIZE) ** 2
    min_own2 = (MIN_OWN / CELL_SIZE) ** 2
    for _ in range(50):
        chosen = [cells[rng.randrange(len(cells))]]
        nearest2 = [math.inf] * len(cells)
        while len(chosen) < 2 * POOL:
            last = chosen[-1]
            best_index, best_score = -1, -1.0
            for i, cell in enumerate(cells):
                d2 = (cell[0] - last[0]) ** 2 + (cell[1] - last[1]) ** 2
                nearest2[i] = min(nearest2[i], d2)
                if nearest2[i] < min_sep2:
                    continue
                score = nearest2[i] * (0.9 + 0.1 * rng.random())
                if score > best_score:
                    best_index, best_score = i, score
            if best_index < 0:
                break
            chosen.append(cells[best_index])
        if len(chosen) != 2 * POOL:
            continue
        center = (sum(p[0] for p in chosen) / len(chosen),
                  sum(p[1] for p in chosen) / len(chosen))
        chosen.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
        offset = rng.randrange(POOL)
        own = []
        for seat in range(POOL):
            start = chosen[(seat + offset) % (2 * POOL)]
            goal = chosen[(seat + offset + POOL) % (2 * POOL)]
            own.append((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
        if min(own) >= min_own2:
            pairwise = min((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                           for i, a in enumerate(chosen) for b in chosen[i + 1:])
            return math.sqrt(pairwise) * CELL_SIZE, math.sqrt(min(own)) * CELL_SIZE
    return None


def main() -> None:
    failures = []
    minima = []
    for map_index, path in enumerate(sorted(MAPS.glob("*.txt"))):
        grid = load_map_grid(path)
        components = asphalt_components(grid)
        ids, counts = np.unique(components[components >= 0], return_counts=True)
        if not len(ids):
            failures.append(path.name + ": no Asphalt")
            continue
        largest_id = ids[np.argmax(counts)]
        size = grid.shape[0]
        cells = [(int(x), int(z)) for z, x in zip(*np.where(components == largest_id))
                 if 2 <= x < size - 2 and 2 <= z < size - 2]
        result = select(cells, 3401 + map_index) if len(cells) >= 2 * POOL else None
        if result is None:
            failures.append(path.name + ": no valid placement")
        else:
            minima.append(result)
    print(f"Valid ten-car randomized placements: {len(minima)}/100 maps")
    if minima:
        print(f"Minimum endpoint distance: {min(v[0] for v in minima):.2f} m")
        print(f"Minimum own spawn-goal distance: {min(v[1] for v in minima):.2f} m")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
