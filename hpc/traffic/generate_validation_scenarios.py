"""Generate reproducible, separated traffic routes on every validation map.

The first N routes of each scenario can be used with N active cars (2..max-cars),
so car-count comparisons share the same maps and route prefixes. Every endpoint
is on one connected asphalt component and is separated from every other endpoint.
Shortest-path overlap is preferred, but not required on maps without a crossing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_trajectories import (  # noqa: E402
    asphalt_components,
    bfs_distances,
    cell_to_world,
    compute_road_heading,
    load_map_grid,
)

MAPS = ROOT / "Assets/Resources/Maps/VoronoiVal"
DEFAULT_OUTPUT = ROOT / "Assets/Resources/Traffic/traffic_validation_4cars.json"


def path_to(parent: dict, goal: tuple[int, int]) -> list[tuple[int, int]]:
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent[goal]
    return path[::-1]


def endpoint_distance_m(a: tuple[int, int], b: tuple[int, int], cell_size: float) -> float:
    return math.dist(a, b) * cell_size


def route_record(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
                 path_cells: int, cell_size: float) -> dict:
    sx, sz = start[1], start[0]
    gx, gz = goal[1], goal[0]
    wx, wz = cell_to_world(sx, sz, grid.shape[0], cell_size)
    tx, tz = cell_to_world(gx, gz, grid.shape[0], cell_size)
    return {
        "spawn_x": wx, "spawn_z": wz, "goal_x": tx, "goal_z": tz,
        "heading_deg": compute_road_heading(grid, sx, sz, gx, gz),
        "spawn_cell_x": sx, "spawn_cell_z": sz,
        "goal_cell_x": gx, "goal_cell_z": gz,
        "path_len_cells": path_cells,
    }


def choose_endpoints(cells: np.ndarray, count: int,
                     rng: random.Random) -> list[tuple[int, int]]:
    """Randomised farthest-point sampling keeps starts and goals spread out."""
    selected = [rng.randrange(len(cells))]
    nearest_squared = np.full(len(cells), np.iinfo(np.int32).max, dtype=np.int32)
    while len(selected) < count:
        delta = cells - cells[selected[-1]]
        nearest_squared = np.minimum(nearest_squared, np.sum(delta * delta, axis=1))
        nearest_squared[selected] = -1
        top_count = min(12, len(cells) - len(selected))
        top = np.argpartition(nearest_squared, -top_count)[-top_count:]
        selected.append(rng.choice(top.tolist()))
    return [tuple(map(int, cells[index])) for index in selected]


def routes_for_map(grid: np.ndarray, max_cars: int, min_endpoint_m: float,
                   min_path_cells: int, cell_size: float, attempts: int,
                   seed: int) -> tuple[list[dict], dict]:
    components = asphalt_components(grid)
    ids, counts = np.unique(components[components >= 0], return_counts=True)
    if len(ids) == 0:
        raise ValueError("map contains no asphalt")
    component = int(ids[np.argmax(counts)])
    z_size, x_size = grid.shape
    margin = 2  # Keep the vehicle body away from the edge of the map.
    cells = np.asarray([(int(z), int(x)) for z, x in zip(*np.where(components == component))
                        if margin <= z < z_size - margin and margin <= x < x_size - margin],
                       dtype=np.int32)
    if len(cells) < 2 * max_cars:
        raise ValueError("largest asphalt component has too few interior endpoints")

    rng = random.Random(seed)
    best = None
    best_score = -float("inf")
    for _ in range(attempts):
        endpoints = choose_endpoints(cells, 2 * max_cars, rng)
        minimum = min(endpoint_distance_m(a, b, cell_size)
                      for i, a in enumerate(endpoints) for b in endpoints[i + 1:])
        if minimum < min_endpoint_m:
            continue
        center_z = sum(z for z, _ in endpoints) / len(endpoints)
        center_x = sum(x for _, x in endpoints) / len(endpoints)
        ordered = sorted(endpoints, key=lambda c: math.atan2(c[0] - center_z, c[1] - center_x))
        # Rotations only reverse/reorder these same opposite endpoint pairs; they do
        # not change the paths or encounter score, so evaluate one rotation per try.
        rotation = rng.randrange(max_cars)
        rotated = ordered[rotation:] + ordered[:rotation]
        pairs = [(rotated[i], rotated[i + max_cars]) for i in range(max_cars)]
        paths = []
        for start, goal in pairs:
            distances, parents = bfs_distances(grid, components, component, start)
            if distances.get(goal, -1) < min_path_cells:
                break
            paths.append(path_to(parents, goal))
        if len(paths) != max_cars:
            continue
        interiors = [set(path[max(3, len(path) // 6):min(len(path) - 3, len(path) * 5 // 6)])
                     for path in paths]
        overlapping_pairs = sum(bool(interiors[i] & interiors[j])
                                for i in range(max_cars) for j in range(i + 1, max_cars))
        first_pair_overlaps = bool(interiors[0] & interiors[1])
        # First two seats are also the 2-car test. Ensure they have a potential
        # encounter before optimising extra-seat encounters.
        score = (1000 * first_pair_overlaps + 100 * overlapping_pairs + minimum +
                 0.1 * min(len(p) for p in paths))
        if score > best_score:
            best_score = score
            best = (pairs, paths, minimum, overlapping_pairs, first_pair_overlaps)
    if best is None:
        raise ValueError(f"no {max_cars}-car route set meets {min_endpoint_m:g} m endpoint "
                         f"and {min_path_cells}-cell path minima after {attempts} tries")
    pairs, paths, minimum, overlapping_pairs, first_pair_overlaps = best
    routes = [route_record(grid, start, goal, len(path) - 1, cell_size)
              for (start, goal), path in zip(pairs, paths)]
    return routes, {
        "minimum_endpoint_separation_m": round(minimum, 3),
        "overlapping_route_pairs": overlapping_pairs,
        "first_two_routes_overlap": first_pair_overlaps,
    }


def generate(args: argparse.Namespace) -> dict:
    files = sorted(args.maps_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"no validation .txt maps in {args.maps_dir}")
    scenarios = []
    failures = []
    for map_index, path in enumerate(files):
        grid = load_map_grid(path)
        try:
            routes, quality = routes_for_map(
                grid, args.max_cars, args.min_endpoint_m, args.min_path_cells,
                args.cell_size_m, args.attempts, args.seed + map_index,
            )
        except ValueError as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        scenarios.append({
            "scenario_id": map_index, "map_index": map_index, "map_file": path.name,
            **quality, "routes": routes,
        })
        if (map_index + 1) % 10 == 0:
            print(f"Generated {map_index + 1}/{len(files)} validation maps", flush=True)
    if failures:
        raise ValueError(f"Only {len(scenarios)}/{len(files)} maps have valid routes:\n" +
                         "\n".join(failures))
    return {
        "generator": "generate_validation_scenarios.py",
        "max_cars": args.max_cars,
        "min_endpoint_m": args.min_endpoint_m,
        "min_path_cells": args.min_path_cells,
        "cell_size_m": args.cell_size_m,
        "seed": args.seed,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=MAPS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cars", type=int, default=4)
    parser.add_argument("--min-endpoint-m", type=float, default=12.0)
    parser.add_argument("--min-path-cells", type=int, default=20)
    parser.add_argument("--cell-size-m", type=float, default=1.5)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=48151)
    args = parser.parse_args()
    if args.max_cars < 2 or args.min_endpoint_m <= 0 or args.min_path_cells < 1 or args.attempts < 1:
        parser.error("require max-cars >= 2, positive distance/path limits, and attempts >= 1")
    manifest = generate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['scenarios'])} validation maps, {args.max_cars} routes each, "
          f"to {args.out}")
    print(f"Route-overlap opportunities: "
          f"{sum(s['overlapping_route_pairs'] > 0 for s in manifest['scenarios'])}/"
          f"{len(manifest['scenarios'])} maps")
    print(f"First two routes overlap: "
          f"{sum(s['first_two_routes_overlap'] for s in manifest['scenarios'])}/"
          f"{len(manifest['scenarios'])} maps")


if __name__ == "__main__":
    main()
