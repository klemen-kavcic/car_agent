"""Build two-car fixed scenarios from the project's ten trajectory candidates.

The second shortest asphalt route must overlap the first away from both routes'
endpoints. This establishes a possible encounter, not a guaranteed simultaneous one.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_trajectories import (  # noqa: E402
    asphalt_components,
    bfs_distances,
    cell_to_world,
    compute_road_heading,
    load_map_grid,
)

CANDIDATES = ROOT / "hpc/0816/trajectories/moved/candidates_pushed.json"
MAPS = ROOT / "Assets/Resources/Maps/VoronoiVal"
OUTPUT = ROOT / "Assets/Resources/Traffic/traffic_pairs.json"
PREVIEW = ROOT / "hpc/traffic/traffic_pairs_preview.png"
CELL_SIZE_M = 1.5
MIN_ENDPOINT_SEPARATION_M = 15.0
MIN_ROUTE_LENGTH_CELLS = 24


def reconstruct(parent: dict, end: tuple[int, int]) -> list[tuple[int, int]]:
    path = []
    while end is not None:
        path.append(end)
        end = parent[end]
    return path[::-1]


def separation_m(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.dist(a, b) * CELL_SIZE_M


def route_dict(grid, start: tuple[int, int], goal: tuple[int, int]) -> dict:
    sx, sz = start[1], start[0]
    gx, gz = goal[1], goal[0]
    wx, wz = cell_to_world(sx, sz, grid.shape[0], CELL_SIZE_M)
    tx, tz = cell_to_world(gx, gz, grid.shape[0], CELL_SIZE_M)
    return dict(
        spawn_x=wx, spawn_z=wz, goal_x=tx, goal_z=tz,
        heading_deg=compute_road_heading(grid, sx, sz, gx, gz),
        spawn_cell_x=sx, spawn_cell_z=sz, goal_cell_x=gx, goal_cell_z=gz,
    )


def build() -> list[dict]:
    scenarios = []
    for base in json.loads(CANDIDATES.read_text(encoding="utf-8")):
        grid = load_map_grid(MAPS / base["map_file"])
        comp = asphalt_components(grid)
        start0 = (base["z_spawn_cell"], base["x_spawn_cell"])
        goal0 = (base["z_goal_cell"], base["x_goal_cell"])
        comp_id = int(comp[start0])
        dist0, parent0 = bfs_distances(grid, comp, comp_id, start0)
        assert goal0 in dist0, base["map_file"]
        path0 = reconstruct(parent0, goal0)
        middle0 = set(path0[max(5, len(path0) // 5):min(len(path0) - 5, len(path0) * 4 // 5)])
        cells = [tuple(map(int, cell)) for cell in zip(*((comp == comp_id).nonzero()))]
        endpoints0 = (start0, goal0)
        eligible = [cell for cell in cells if all(
            separation_m(cell, endpoint) >= MIN_ENDPOINT_SEPARATION_M for endpoint in endpoints0)]
        rng = random.Random(24000 + int(base["candidate_id"]))
        rng.shuffle(eligible)
        best = None
        best_score = -float("inf")
        for start1 in eligible[:120]:
            dist1, parent1 = bfs_distances(grid, comp, comp_id, start1)
            goals = [cell for cell in eligible if cell in dist1 and
                     dist1[cell] >= MIN_ROUTE_LENGTH_CELLS and
                     separation_m(start1, cell) >= 25]
            rng.shuffle(goals)
            for goal1 in goals[:35]:
                path1 = reconstruct(parent1, goal1)
                middle1 = set(path1[max(5, len(path1) // 5):min(len(path1) - 5, len(path1) * 4 // 5)])
                overlap = middle0 & middle1
                if not overlap:
                    continue
                # Prefer a potential meeting near both routes' middles, with similar
                # driving distances from their starts. No route following is forced.
                crossing = min(overlap, key=lambda c: abs(path0.index(c) - path1.index(c)))
                arrival_gap = abs(path0.index(crossing) - path1.index(crossing))
                score = len(overlap) * 0.02 - arrival_gap - abs(len(path0) - len(path1)) * 0.15
                if score > best_score:
                    best_score = score
                    best = (start1, goal1, path1, crossing, len(overlap), arrival_gap)
        if best is None:
            raise RuntimeError(f"No intersecting second route found for {base['map_file']}")
        start1, goal1, path1, crossing, overlap_count, arrival_gap = best
        scenario = dict(
            scenario_id=int(base["candidate_id"]),
            map_file=base["map_file"],
            map_index=int(base["map_index"]),
            crossing_cell_x=crossing[1], crossing_cell_z=crossing[0],
            overlap_cells=overlap_count, arrival_gap_cells=arrival_gap,
            routes=[route_dict(grid, start0, goal0), route_dict(grid, start1, goal1)],
        )
        scenarios.append(scenario)
        print(f"{scenario['scenario_id']:02d} {scenario['map_file']}: "
              f"route lengths {len(path0) - 1}/{len(path1) - 1} cells; "
              f"overlap {overlap_count}; arrival gap {arrival_gap} cells")
    return scenarios


def write_preview(scenarios: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = ListedColormap(["#9e9b90", "#434343", "#31a332", "#65b4e6", "#cf3030"])
    fig, axes = plt.subplots(2, 5, figsize=(19, 8), constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenarios):
        grid = load_map_grid(MAPS / scenario["map_file"])
        comp = asphalt_components(grid)
        ax.imshow(grid, origin="lower", cmap=colors, vmin=0, vmax=4)
        for seat, (route, color) in enumerate(zip(scenario["routes"], ("#ffb000", "#c982ff"))):
            start = (route["spawn_cell_z"], route["spawn_cell_x"])
            goal = (route["goal_cell_z"], route["goal_cell_x"])
            _, parent = bfs_distances(grid, comp, int(comp[start]), start)
            path = reconstruct(parent, goal)
            ax.plot([p[1] for p in path], [p[0] for p in path], color=color, lw=1.6)
            ax.scatter(start[1], start[0], c=color, marker="o", s=48, edgecolor="black")
            ax.scatter(goal[1], goal[0], c=color, marker="*", s=100, edgecolor="black")
            ax.annotate(str(seat), (start[1], start[0]), color="black", fontsize=7)
        ax.scatter(scenario["crossing_cell_x"], scenario["crossing_cell_z"],
                   c="white", marker="x", s=60, linewidth=2)
        ax.set_title(f"{scenario['scenario_id']:02d} {scenario['map_file']} | "
                     f"overlap {scenario['overlap_cells']}", fontsize=9)
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(-0.5, grid.shape[0] - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW, dpi=150)
    plt.close(fig)
    print(f"Wrote {PREVIEW}")


if __name__ == "__main__":
    scenarios = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"scenarios": scenarios}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    write_preview(scenarios)
