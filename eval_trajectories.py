"""Fixed-map deterministic-vs-stochastic trajectory comparison.

Every metric in eval_checkpoints.py is a statistical aggregate over many randomized episodes -
useful for "which combo is better on average," but it can't show HOW the deterministic and
stochastic policies actually differ in behavior on the same situation. This script instead picks
a handful of maps with a FIXED (non-randomized) start point, heading, and goal point, drives the
same checkpoint over each one twice - once greedy/deterministic, once sampled/stochastic (several
times, to see the sampled policy's spread) - and records the car's (x, z) position at every
decision step, so the actual paths can be plotted and visually compared.

Requires a rebuilt Unity binary (Assets/carAgent.cs's fixed_eval_enabled / Assets/grid_manager.cs's
forcedMapIndex) - existing checkpoints are still valid, only the built player needs the new code,
no retraining. See the plan doc / carAgent.cs comments for the full design.

Three subcommands, meant to be run in order:

    1) generate - auto-picks --num-maps candidate (map, spawn, goal, heading) triples from
       --maps-dir, each verified reachable via a real BFS flood-fill over 4-connected Asphalt
       cells (stronger than what the game itself checks at runtime - IsSpawnableTile only verifies
       a single point is Asphalt, not that two points are actually connected). Writes
       candidates.json.

           python eval_trajectories.py generate \\
               --maps-dir "Assets/Resources/Maps/VoronoiVal" \\
               --num-maps 10 --out candidates.json

    2) plot-candidates - renders one preview PNG per candidate (map + spawn/goal markers + heading
       arrow) for a quick visual review pass before spending any Unity time on them. Bad/boring
       candidates can be deleted from candidates.json by hand before the next step.

           python eval_trajectories.py plot-candidates \\
               --candidates candidates.json --maps-dir "Assets/Resources/Maps/VoronoiVal" \\
               --out-dir candidate_review

    3) run - connects ONE Unity environment (no --num-envs pooling: StatsSideChannel has no
       per-agent/episode correlation key, so parallel agents' Custom/PosX/Custom/PosZ streams
       would interleave unrecoverably - this only works because it's one agent, one episode, at a
       time), drives 1 deterministic + --stochastic-rollouts stochastic episodes per candidate
       against a single checkpoint, and writes a long-format trajectories.csv.

           python eval_trajectories.py run \\
               --binary /path/to/Linux/car.x86_64 \\
               --run-dir /path/to/results/<run_id> \\
               --config /path/to/configs/<run_id>.yaml \\
               --candidates candidates.json --stochastic-rollouts 5 \\
               --out trajectories.csv

--gridsize/--cellsize MUST match the built binary's GridManager Inspector values (project default:
66 / 1.5, per Assets/build_info.txt) - used to convert between grid cells and world coordinates
(mirrors GridManager.GenerateGrid's exact placement formula, see cell_to_world below). Wrong values
here won't error, they'll just silently spawn the car in the wrong place.
"""
from __future__ import annotations  # lets `run`-only type hints (ort.InferenceSession, EvalWorker)
                                     # reference lazily-imported names without failing at module load

import argparse
import csv
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# matplotlib/scipy are only needed by generate/from-points (scipy.ndimage, inside
# snap_to_road_center) and plot-candidates/render-maps (matplotlib, inside render_map_grid/
# cmd_plot_candidates/cmd_render_maps) - imported lazily inside those functions instead of here so
# `run` works in an environment that has onnxruntime/mlagents_envs but not matplotlib/scipy (e.g.
# the HPC training container, which has no need for either otherwise). Conversely,
# onnxruntime/mlagents_envs/eval_checkpoints (which itself imports mlagents_envs) are only needed
# by `run` and imported lazily inside cmd_run()/run_one_rollout() instead, so generate/plot-
# candidates/render-maps work on a plain local machine that lacks the ML-Agents stack.

logging.basicConfig(level=logging.INFO, format="[traj] %(message)s")
log = logging.getLogger(__name__)

# Index = the integer code Map Gen/voronoi_curved.py's classify_grid()/export_unity_grid() writes
# per cell - must match Assets/grid_manager.cs's VoronoiCodeToType exactly.
CODE_TO_TILE = ["Gravel", "Asphalt", "SpeedLimited", "Slippery", "Terminal"]
ASPHALT_CODE = CODE_TO_TILE.index("Asphalt")

# Same tile-color convention as hpc/0816/goalrate_log_vs_eval.ipynb's TILE_COLORS, kept consistent
# across the project's visual language. 8-digit hex = RGBA (matplotlib accepts this directly).
TILE_COLORS = {
    "Asphalt": "#1d1d1d68",
    "Gravel": "#a09a8fb8",
    "Slippery": "#56b4e9",
    "SpeedLimited": "#15a715",
    "Terminal": "#d62728",
}

# Deliberately NOT green/red/yellow - those are reserved for driven-path outcome coloring
# (goal/terminated/maxstep) once trajectories.csv exists (2.11 in the notebook).
SPAWN_COLOR = "#00008b"  # dark blue
GOAL_COLOR = "#800080"   # purple


# =================================================================== map grid parsing ==========

def load_map_grid(path: Path) -> np.ndarray:
    """Parses a Voronoi .txt map (space-separated int codes, one row per line) into a
    (gridsize, gridsize) int array. rows=z, cols=x - same orientation as grid_manager.cs's
    BuildVoronoiPlan (`cols = rows[z].Split(' '); plan[x, z] = ...cols[x]...`)."""
    text = path.read_text().strip()
    rows = [row.split() for row in text.splitlines()]
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int32)


def cell_to_world(x_cell: int, z_cell: int, gridsize: int, cellsize: float) -> Tuple[float, float]:
    """Mirrors GridManager.GenerateGrid's exact tile-placement formula:
    gridOrigin = -(gridsize*cellsize)/2 on each axis; pos = gridOrigin + idx*cellsize + cellsize*0.5
    (cell CENTER, not corner)."""
    origin = -(gridsize * cellsize) / 2.0
    x_world = origin + x_cell * cellsize + cellsize * 0.5
    z_world = origin + z_cell * cellsize + cellsize * 0.5
    return x_world, z_world


# =================================================================== candidate generation ======

def asphalt_components(grid: np.ndarray) -> np.ndarray:
    """4-connected flood-fill over Asphalt cells only. Returns a same-shape int array of
    component ids, -1 for non-Asphalt cells. This is new logic - nothing in the codebase already
    checks spawn/goal REACHABILITY (IsSpawnableTile only checks a single point is Asphalt)."""
    gridsize_z, gridsize_x = grid.shape
    comp = np.full(grid.shape, -1, dtype=np.int32)
    next_id = 0
    for z0 in range(gridsize_z):
        for x0 in range(gridsize_x):
            if grid[z0, x0] != ASPHALT_CODE or comp[z0, x0] != -1:
                continue
            stack = [(z0, x0)]
            comp[z0, x0] = next_id
            while stack:
                z, x = stack.pop()
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nz, nx = z + dz, x + dx
                    if (0 <= nz < gridsize_z and 0 <= nx < gridsize_x
                            and grid[nz, nx] == ASPHALT_CODE and comp[nz, nx] == -1):
                        comp[nz, nx] = next_id
                        stack.append((nz, nx))
            next_id += 1
    return comp


def bfs_distances(grid: np.ndarray, comp: np.ndarray, comp_id: int, start: Tuple[int, int]):
    """BFS shortest-path CELL distance (not straight-line) from `start` to every other cell in
    the same component. Returns (dist dict, parent dict) for path reconstruction."""
    gridsize_z, gridsize_x = grid.shape
    dist = {start: 0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    q = deque([start])
    while q:
        z, x = q.popleft()
        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nz, nx = z + dz, x + dx
            if (0 <= nz < gridsize_z and 0 <= nx < gridsize_x
                    and comp[nz, nx] == comp_id and (nz, nx) not in dist):
                dist[(nz, nx)] = dist[(z, x)] + 1
                parent[(nz, nx)] = (z, x)
                q.append((nz, nx))
    return dist, parent


def pick_candidate(grid: np.ndarray, comp: np.ndarray, rng: np.random.Generator,
                    min_path_cells: int, max_attempts: int = 50) -> Optional[dict]:
    """Picks a random large-enough Asphalt component, then a random start cell within it, then
    the FARTHEST (by real BFS graph distance, not straight-line) reachable cell as the goal - for
    a long, hopefully fork-rich route. Returns None if no component/start yields a path
    >= min_path_cells within max_attempts."""
    comp_ids, counts = np.unique(comp[comp >= 0], return_counts=True)
    eligible = comp_ids[counts >= min_path_cells]
    if len(eligible) == 0:
        return None
    comp_id = int(rng.choice(eligible))
    cells = list(zip(*np.where(comp == comp_id)))

    for _ in range(max_attempts):
        start = cells[int(rng.integers(len(cells)))]
        dist, parent = bfs_distances(grid, comp, comp_id, start)
        far_enough = [c for c, d in dist.items() if d >= min_path_cells]
        if not far_enough:
            continue
        goal = max(far_enough, key=lambda c: dist[c])
        z0, x0 = start
        z1, x1 = goal
        heading_deg = compute_road_heading(grid, x0, z0, x1, z1)
        return dict(
            x_spawn_cell=int(x0), z_spawn_cell=int(z0),
            x_goal_cell=int(x1), z_goal_cell=int(z1),
            heading_deg=heading_deg,
            path_len_cells=int(dist[goal]),
        )
    return None


def generate_candidates(args) -> List[dict]:
    maps_dir = Path(args.maps_dir)
    map_files = sorted(maps_dir.glob("*.txt"))
    if not map_files:
        raise FileNotFoundError(f"No .txt maps found under {maps_dir}")

    rng_top = np.random.default_rng(args.seed)
    n = min(args.num_maps, len(map_files))
    chosen_indices = sorted(rng_top.choice(len(map_files), size=n, replace=False).tolist())

    candidates = []
    for candidate_id, map_index in enumerate(chosen_indices):
        map_file = map_files[map_index]
        grid = load_map_grid(map_file)
        comp = asphalt_components(grid)
        rng = np.random.default_rng(args.seed + 1000 + map_index)
        result = pick_candidate(grid, comp, rng, args.min_path_cells)
        if result is None:
            log.warning("No valid spawn/goal pair found on %s (component too small?) - skipping.",
                        map_file.name)
            continue

        world_spawn_x, world_spawn_z = cell_to_world(
            result["x_spawn_cell"], result["z_spawn_cell"], args.gridsize, args.cellsize)
        world_goal_x, world_goal_z = cell_to_world(
            result["x_goal_cell"], result["z_goal_cell"], args.gridsize, args.cellsize)

        candidates.append(dict(
            candidate_id=candidate_id,
            map_file=map_file.name,
            map_index=map_index,
            x_spawn_cell=result["x_spawn_cell"], z_spawn_cell=result["z_spawn_cell"],
            world_spawn_x=world_spawn_x, world_spawn_z=world_spawn_z,
            x_goal_cell=result["x_goal_cell"], z_goal_cell=result["z_goal_cell"],
            world_goal_x=world_goal_x, world_goal_z=world_goal_z,
            heading_deg=result["heading_deg"],
            path_len_cells=result["path_len_cells"],
        ))
    return candidates


def cmd_generate(args):
    candidates = generate_candidates(args)
    Path(args.out).write_text(json.dumps(candidates, indent=2))
    log.info("Wrote %d candidates -> %s", len(candidates), args.out)


# =================================================================== hand-picked candidates ====

def world_to_cell(x_world: float, z_world: float, gridsize: int, cellsize: float) -> Tuple[int, int]:
    """Inverse of cell_to_world - nearest cell to a given world coordinate."""
    origin = -(gridsize * cellsize) / 2.0
    x_cell = int(round((x_world - origin) / cellsize - 0.5))
    z_cell = int(round((z_world - origin) / cellsize - 0.5))
    return x_cell, z_cell


def snap_to_road_center(grid: np.ndarray, x_cell: int, z_cell: int, search_radius: int = 6) -> Optional[Tuple[int, int]]:
    """Finds the most 'centered' Asphalt cell within search_radius of (x_cell, z_cell) - i.e. the
    one locally maximizing distance to the nearest non-Asphalt cell (via a distance transform on
    the whole map's Asphalt mask) - so a hand-picked point lands in the middle of the road rather
    than wherever was closest, which could be right at the shoulder. Ties broken by proximity to
    the original point. Returns None if no Asphalt cell exists within search_radius at all."""
    from scipy import ndimage
    asphalt_mask = grid == ASPHALT_CODE
    # Pad with a 1-cell False border before the distance transform, then crop back - otherwise
    # scipy's distance_transform_edt only ever sees what's inside the array, so a road that runs
    # right up to the map's physical edge looks just as "centered" there as it would in the middle
    # of a wide-open stretch (nothing beyond the array boundary counts as non-Asphalt to be close
    # to). Without this, centering could - and did - pick cells sitting right at the map edge.
    padded_dist = ndimage.distance_transform_edt(np.pad(asphalt_mask, 1, constant_values=False))
    dist_to_edge = padded_dist[1:-1, 1:-1]
    gridsize_z, gridsize_x = grid.shape

    best = None
    best_key = None
    for dz in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            nz, nx = z_cell + dz, x_cell + dx
            if not (0 <= nz < gridsize_z and 0 <= nx < gridsize_x) or not asphalt_mask[nz, nx]:
                continue
            key = (dist_to_edge[nz, nx], -(dx * dx + dz * dz))  # most-centered first, then closest
            if best_key is None or key > best_key:
                best_key = key
                best = (nx, nz)
    return best


def push_from_edge(grid: np.ndarray, comp: np.ndarray, x_cell: int, z_cell: int, gridsize: int,
                    cellsize: float, edge_limit_world: float) -> Tuple[int, int]:
    """If (x_cell, z_cell) sits outside the box normal TRAINING actually samples spawn/goal points
    from (Assets/Scenes/SampleScene.unity's spawnAreaMin/Max/goalAreaMin/Max - +/-45 world units by
    default, well inside the true map edge at +/-49.5, since a uniform random point in that box
    still has to pass IsSpawnableTile's Asphalt-only retry loop), walks along the SAME connected
    road (BFS graph distance, not straight-line - so it can't jump to a different segment) to the
    NEAREST cell whose world coordinates fall back inside +/-edge_limit_world on both axes. Returns
    the original cell unchanged if it's already within bounds, or if no in-bounds cell exists on
    this component at all. Ties (multiple cells at the same BFS distance) broken by straight-line
    proximity to the original point, matching snap_to_road_center's convention."""
    def in_bounds(xc, zc):
        wx, wz = cell_to_world(xc, zc, gridsize, cellsize)
        return abs(wx) <= edge_limit_world and abs(wz) <= edge_limit_world

    if in_bounds(x_cell, z_cell):
        return x_cell, z_cell

    comp_id = int(comp[z_cell, x_cell])
    dist, _ = bfs_distances(grid, comp, comp_id, (z_cell, x_cell))
    best = None
    best_key = None
    for (nz, nx), d in dist.items():
        if not in_bounds(nx, nz):
            continue
        key = (d, (nx - x_cell) ** 2 + (nz - z_cell) ** 2)
        if best_key is None or key < best_key:
            best_key = key
            best = (nx, nz)
    return best if best is not None else (x_cell, z_cell)


def compute_road_heading(grid: np.ndarray, x_cell: int, z_cell: int, x_goal_cell: int,
                          z_goal_cell: int, radius: int = 2) -> float:
    """Mirrors Assets/carAgent.cs's ComputeRoadAlignedHeading: samples the local
    (2*radius+1)x(2*radius+1) neighborhood of Asphalt cells and finds the dominant LINE
    orientation via the doubled-angle circular-mean trick (a road is a line, not an arrow - 180-
    degree ambiguous, so a plain vector average of opposite-pointing offsets along a straight road
    would wrongly cancel to ~zero). Unlike the C# version (which coin-flips which of the two
    directions to face), resolves the ambiguity DETERMINISTICALLY toward the goal, so the car
    starts already facing generally the right way down the road, not away from it."""
    gridsize_z, gridsize_x = grid.shape
    sum_sin = sum_cos = 0.0
    count = 0
    for dz in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dz == 0:
                continue
            nz, nx = z_cell + dz, x_cell + dx
            if 0 <= nz < gridsize_z and 0 <= nx < gridsize_x and grid[nz, nx] == ASPHALT_CODE:
                angle = np.arctan2(dx, dz)  # matches Unity's yaw convention (0 deg = +Z)
                sum_sin += np.sin(2 * angle)
                sum_cos += np.cos(2 * angle)
                count += 1

    to_goal_deg = float(np.degrees(np.arctan2(x_goal_cell - x_cell, z_goal_cell - z_cell)))
    if count < 2:
        return to_goal_deg  # too few road neighbours for a reliable line fit - just face the goal

    road_heading_deg = float(np.degrees(np.arctan2(sum_sin, sum_cos)) * 0.5)
    diff = (road_heading_deg - to_goal_deg + 180) % 360 - 180  # signed diff in [-180, 180)
    if abs(diff) > 90:
        road_heading_deg += 180
    return road_heading_deg % 360


def build_candidate_from_points(map_file: Path, candidate_id: int, x_spawn_world: float,
                                 z_spawn_world: float, x_goal_world: float, z_goal_world: float,
                                 gridsize: int, cellsize: float) -> Optional[dict]:
    """Turns a hand-picked (spawn, goal) world-coordinate pair into a full candidate dict: snaps
    each point to the most CENTERED nearby Asphalt cell (points read off a preview image are
    rarely pixel-exact, and even an on-road pick could be right at the shoulder), confirms they're
    on the SAME connected road network (fails loudly, not silently, if not - a route with no real
    path would otherwise fail mysteriously at Unity-drive time), and derives path_len_cells/
    heading_deg (road-aligned, oriented toward the goal) via the same real BFS/neighborhood
    analysis `generate` uses."""
    grid = load_map_grid(map_file)
    x_spawn_cell, z_spawn_cell = world_to_cell(x_spawn_world, z_spawn_world, gridsize, cellsize)
    x_goal_cell, z_goal_cell = world_to_cell(x_goal_world, z_goal_world, gridsize, cellsize)

    snapped_spawn = snap_to_road_center(grid, x_spawn_cell, z_spawn_cell)
    snapped_goal = snap_to_road_center(grid, x_goal_cell, z_goal_cell)
    if snapped_spawn is None or snapped_goal is None:
        log.error("%s: no Asphalt cell found near the given %s point - pick closer to a road.",
                   map_file.name, "spawn" if snapped_spawn is None else "goal")
        return None
    if snapped_spawn != (x_spawn_cell, z_spawn_cell):
        log.info("%s: spawn snapped from cell (%d, %d) to road-centered cell (%d, %d).",
                  map_file.name, x_spawn_cell, z_spawn_cell, *snapped_spawn)
    if snapped_goal != (x_goal_cell, z_goal_cell):
        log.info("%s: goal snapped from cell (%d, %d) to road-centered cell (%d, %d).",
                  map_file.name, x_goal_cell, z_goal_cell, *snapped_goal)
    x_spawn_cell, z_spawn_cell = snapped_spawn
    x_goal_cell, z_goal_cell = snapped_goal

    comp = asphalt_components(grid)
    comp_id = int(comp[z_spawn_cell, x_spawn_cell])
    if comp[z_goal_cell, x_goal_cell] != comp_id:
        log.error("%s: spawn and goal are on DIFFERENT disconnected road segments - no path "
                   "exists between them. Pick a goal on the same connected road network as spawn.",
                   map_file.name)
        return None

    dist, _ = bfs_distances(grid, comp, comp_id, (z_spawn_cell, x_spawn_cell))
    goal_key = (z_goal_cell, x_goal_cell)
    heading_deg = compute_road_heading(grid, x_spawn_cell, z_spawn_cell, x_goal_cell, z_goal_cell)

    world_spawn_x, world_spawn_z = cell_to_world(x_spawn_cell, z_spawn_cell, gridsize, cellsize)
    world_goal_x, world_goal_z = cell_to_world(x_goal_cell, z_goal_cell, gridsize, cellsize)
    return dict(
        candidate_id=candidate_id,
        map_file=map_file.name,
        map_index=None,  # filled in by cmd_from_points, relative to --maps-dir's sorted listing
        x_spawn_cell=x_spawn_cell, z_spawn_cell=z_spawn_cell,
        world_spawn_x=world_spawn_x, world_spawn_z=world_spawn_z,
        x_goal_cell=x_goal_cell, z_goal_cell=z_goal_cell,
        world_goal_x=world_goal_x, world_goal_z=world_goal_z,
        heading_deg=heading_deg,
        path_len_cells=int(dist[goal_key]),
    )


def cmd_from_points(args):
    """--points is a JSON list of {"map_file": "curved_401.txt", "spawn": [x, z], "goal": [x, z]}
    entries, world coordinates read off a render-maps/plot-candidates preview image. --maps-dir
    MUST be the same folder Unity will actually load from (e.g. Assets/Resources/Maps/VoronoiVal),
    not a shortlist copy - map_index is computed from ITS sorted listing, and that's what
    GridManager.forcedMapIndex actually indexes into at runtime."""
    maps_dir = Path(args.maps_dir)
    map_files_sorted = sorted(maps_dir.glob("*.txt"))
    index_by_name = {f.name: i for i, f in enumerate(map_files_sorted)}

    points = json.loads(Path(args.points).read_text())
    candidates = []
    for point in points:
        if point["map_file"] not in index_by_name:
            log.error("%s not found in %s - skipping.", point["map_file"], maps_dir)
            continue
        candidate = build_candidate_from_points(
            maps_dir / point["map_file"], len(candidates),
            point["spawn"][0], point["spawn"][1], point["goal"][0], point["goal"][1],
            args.gridsize, args.cellsize,
        )
        if candidate is None:
            continue
        candidate["map_index"] = index_by_name[point["map_file"]]
        candidates.append(candidate)

    Path(args.out).write_text(json.dumps(candidates, indent=2))
    log.info("Wrote %d/%d candidates -> %s", len(candidates), len(points), args.out)


def cmd_push_from_edge(args):
    import matplotlib.pyplot as plt
    candidates = json.loads(Path(args.candidates).read_text())
    maps_dir = Path(args.maps_dir)
    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    pushed = []
    for candidate in candidates:
        grid = load_map_grid(maps_dir / candidate["map_file"])
        comp = asphalt_components(grid)

        old_spawn_cell = (candidate["x_spawn_cell"], candidate["z_spawn_cell"])
        old_goal_cell = (candidate["x_goal_cell"], candidate["z_goal_cell"])
        new_spawn_cell = push_from_edge(grid, comp, *old_spawn_cell, args.gridsize, args.cellsize, args.edge_limit)
        new_goal_cell = push_from_edge(grid, comp, *old_goal_cell, args.gridsize, args.cellsize, args.edge_limit)

        # Re-center on the road after the edge push (same technique build_candidate_from_points
        # uses for hand-picked points) - this can nudge a point back outside +/-edge_limit by a
        # cell or two, which is fine, centering wins; snap_to_road_center returns None only if no
        # Asphalt cell exists within its search_radius at all, which can't happen here since the
        # pushed cell is itself already Asphalt (it came from the component's own BFS).
        new_spawn_cell = snap_to_road_center(grid, *new_spawn_cell) or new_spawn_cell
        new_goal_cell = snap_to_road_center(grid, *new_goal_cell) or new_goal_cell

        comp_id = int(comp[new_spawn_cell[1], new_spawn_cell[0]])
        if comp[new_goal_cell[1], new_goal_cell[0]] != comp_id:
            log.error("%s (candidate %d): spawn and goal ended up on DIFFERENT road segments after "
                       "the edge push - keeping this candidate's original points unchanged.",
                       candidate["map_file"], candidate["candidate_id"])
            pushed.append(candidate)
            continue

        dist, _ = bfs_distances(grid, comp, comp_id, (new_spawn_cell[1], new_spawn_cell[0]))
        heading_deg = compute_road_heading(grid, new_spawn_cell[0], new_spawn_cell[1],
                                            new_goal_cell[0], new_goal_cell[1])
        world_spawn_x, world_spawn_z = cell_to_world(*new_spawn_cell, args.gridsize, args.cellsize)
        world_goal_x, world_goal_z = cell_to_world(*new_goal_cell, args.gridsize, args.cellsize)

        new_candidate = dict(candidate)
        new_candidate.update(
            x_spawn_cell=new_spawn_cell[0], z_spawn_cell=new_spawn_cell[1],
            world_spawn_x=world_spawn_x, world_spawn_z=world_spawn_z,
            x_goal_cell=new_goal_cell[0], z_goal_cell=new_goal_cell[1],
            world_goal_x=world_goal_x, world_goal_z=world_goal_z,
            heading_deg=heading_deg,
            path_len_cells=int(dist[(new_goal_cell[1], new_goal_cell[0])]),
        )
        pushed.append(new_candidate)

        moved_spawn = old_spawn_cell != new_spawn_cell
        moved_goal = old_goal_cell != new_goal_cell
        log.info("%s (candidate %d): spawn %s, goal %s", candidate["map_file"], candidate["candidate_id"],
                  "moved" if moved_spawn else "unchanged", "moved" if moved_goal else "unchanged")

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_candidate_before_after(ax, grid, args.gridsize, args.cellsize, candidate, new_candidate)
        out_path = preview_dir / f"candidate_{candidate['candidate_id']:02d}_{Path(candidate['map_file']).stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved preview -> %s", out_path)

    Path(args.out).write_text(json.dumps(pushed, indent=2))
    log.info("Wrote %d candidates -> %s", len(pushed), args.out)


# =================================================================== candidate review render ===

def render_map_grid(ax, grid: np.ndarray, gridsize: int, cellsize: float):
    import matplotlib.colors as mcolors
    origin = -(gridsize * cellsize) / 2.0
    rgba = np.zeros((*grid.shape, 4))
    for code, tile_name in enumerate(CODE_TO_TILE):
        rgba[grid == code] = mcolors.to_rgba(TILE_COLORS[tile_name])
    extent = [origin, origin + gridsize * cellsize, origin, origin + gridsize * cellsize]
    # imshow's row axis (grid's z) increases "up" the image with origin="lower" - matches world Z
    # increasing upward, grid's col axis (x) increasing right - matches world X, no transpose needed.
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
    ax.set_aspect("equal")
    # Gridlines every 10 world units (~6.7 cells) - makes it practical to read off approximate
    # spawn/goal coordinates by eye for hand-picking routes (see `from-points`), without needing
    # pixel-perfect clicking on the image.
    ax.set_xticks(np.arange(np.ceil(origin / 10) * 10, origin + gridsize * cellsize, 10))
    ax.set_yticks(np.arange(np.ceil(origin / 10) * 10, origin + gridsize * cellsize, 10))
    ax.grid(True, color="white", alpha=0.5, linewidth=0.6)
    ax.tick_params(labelsize=7)


def plot_candidate(ax, grid: np.ndarray, gridsize: int, cellsize: float, candidate: dict):
    render_map_grid(ax, grid, gridsize, cellsize)
    ax.plot(candidate["world_spawn_x"], candidate["world_spawn_z"], marker="o", color=SPAWN_COLOR,
             markersize=10, linestyle="none", label="spawn", zorder=5)
    ax.plot(candidate["world_goal_x"], candidate["world_goal_z"], marker="*", color=GOAL_COLOR,
             markersize=16, linestyle="none", label="goal", zorder=5)

    heading_rad = np.radians(candidate["heading_deg"])
    arrow_len = cellsize * 3
    dx = arrow_len * np.sin(heading_rad)  # matches atan2(dx, dz): heading 0 deg = +Z
    dz = arrow_len * np.cos(heading_rad)
    ax.annotate("", xy=(candidate["world_spawn_x"] + dx, candidate["world_spawn_z"] + dz),
                xytext=(candidate["world_spawn_x"], candidate["world_spawn_z"]),
                arrowprops=dict(arrowstyle="->", color=SPAWN_COLOR, lw=2), zorder=6)

    ax.set_title(f"{candidate['map_file']}  (candidate {candidate['candidate_id']}, "
                 f"path~{candidate['path_len_cells']} cells)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def plot_candidate_before_after(ax, grid: np.ndarray, gridsize: int, cellsize: float,
                                 old_candidate: dict, new_candidate: dict):
    """Same as plot_candidate (map + new spawn/goal/heading arrow), plus - only for whichever of
    spawn/goal actually moved - a hollow marker at the OLD position and a dotted line to the NEW
    one, so an edge-push (or any other automated re-pick) can be reviewed before trusting it."""
    plot_candidate(ax, grid, gridsize, cellsize, new_candidate)

    old_sx, old_sz = old_candidate["world_spawn_x"], old_candidate["world_spawn_z"]
    old_gx, old_gz = old_candidate["world_goal_x"], old_candidate["world_goal_z"]
    new_sx, new_sz = new_candidate["world_spawn_x"], new_candidate["world_spawn_z"]
    new_gx, new_gz = new_candidate["world_goal_x"], new_candidate["world_goal_z"]

    if (old_sx, old_sz) != (new_sx, new_sz):
        ax.plot(old_sx, old_sz, marker="o", markerfacecolor="none", markeredgecolor=SPAWN_COLOR,
                 markersize=10, markeredgewidth=2, linestyle="none", zorder=4, label="old spawn")
        ax.plot([old_sx, new_sx], [old_sz, new_sz], linestyle=":", color=SPAWN_COLOR, linewidth=1.5, zorder=4)
    if (old_gx, old_gz) != (new_gx, new_gz):
        ax.plot(old_gx, old_gz, marker="*", markerfacecolor="none", markeredgecolor=GOAL_COLOR,
                 markersize=16, markeredgewidth=2, linestyle="none", zorder=4, label="old goal")
        ax.plot([old_gx, new_gx], [old_gz, new_gz], linestyle=":", color=GOAL_COLOR, linewidth=1.5, zorder=4)
    ax.legend(fontsize=7, loc="upper right")


def cmd_plot_candidates(args):
    import matplotlib.pyplot as plt
    candidates = json.loads(Path(args.candidates).read_text())
    maps_dir = Path(args.maps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        grid = load_map_grid(maps_dir / candidate["map_file"])
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_candidate(ax, grid, args.gridsize, args.cellsize, candidate)
        out_path = out_dir / f"candidate_{candidate['candidate_id']:02d}_{Path(candidate['map_file']).stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved -> %s", out_path)


def cmd_render_maps(args):
    """Plain map renders (no spawn/goal/heading) for browsing a shortlist of candidate maps
    BEFORE committing to routes on any of them - e.g. a hand-picked pool to choose the final N
    from, as opposed to `generate`'s fully automatic pick. --map-files limits to specific
    filenames (as many as you like); omit it to render every .txt map in --maps-dir."""
    import matplotlib.pyplot as plt
    maps_dir = Path(args.maps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.map_files:
        map_files = [maps_dir / name for name in args.map_files]
    else:
        map_files = sorted(maps_dir.glob("*.txt"))

    for map_file in map_files:
        if not map_file.is_file():
            log.warning("Map file not found, skipping: %s", map_file)
            continue
        grid = load_map_grid(map_file)
        fig, ax = plt.subplots(figsize=(6, 6))
        render_map_grid(ax, grid, args.gridsize, args.cellsize)
        ax.set_title(map_file.stem, fontsize=10)
        out_path = out_dir / f"{map_file.stem}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved -> %s", out_path)


# =================================================================== rollout execution =========

def run_one_rollout(worker: EvalWorker, session: ort.InferenceSession, cont_out, disc_out,
                     continuous_size: int, discrete_size: int, candidate: dict, action_mode: str,
                     max_steps: int) -> dict:
    """Runs exactly ONE episode with the given candidate's fixed spawn/goal/heading pinned via
    env params, and reads back its full (x, z) position stream. Safe to read the raw
    Custom/PosX/Custom/PosZ lists directly (no per-episode disambiguation needed) specifically
    because this is one worker/one agent/one episode at a time - see the module docstring."""
    from mlagents_envs.base_env import ActionTuple
    from eval_checkpoints import build_feed

    worker.env_params_channel.set_float_parameter("fixed_eval_enabled", 1.0)
    worker.env_params_channel.set_float_parameter("fixed_map_index", float(candidate["map_index"]))
    worker.env_params_channel.set_float_parameter("fixed_spawn_x", float(candidate["world_spawn_x"]))
    worker.env_params_channel.set_float_parameter("fixed_spawn_z", float(candidate["world_spawn_z"]))
    worker.env_params_channel.set_float_parameter("fixed_goal_x", float(candidate["world_goal_x"]))
    worker.env_params_channel.set_float_parameter("fixed_goal_z", float(candidate["world_goal_z"]))
    worker.env_params_channel.set_float_parameter("fixed_spawn_heading_deg", float(candidate["heading_deg"]))

    worker.stats_channel.get_and_reset_stats()  # clear anything left over from the previous rollout
    worker.env.reset()

    agent_cum_reward: Dict[int, float] = {}
    agent_len: Dict[int, int] = {}
    episode_length: Optional[int] = None
    steps_taken = 0

    while episode_length is None and steps_taken < max_steps:
        decision_steps, terminal_steps = worker.env.get_steps(worker.behavior_name)

        for agent_id, r in zip(decision_steps.agent_id, decision_steps.reward):
            agent_cum_reward[agent_id] = agent_cum_reward.get(agent_id, 0.0) + float(r)
            agent_len[agent_id] = agent_len.get(agent_id, 0) + 1
        for agent_id, r in zip(terminal_steps.agent_id, terminal_steps.reward):
            agent_cum_reward.pop(agent_id, 0.0)
            episode_length = agent_len.pop(agent_id, 0)

        if episode_length is not None:
            # ML-Agents auto-resets the agent the instant its episode ends, and single-env auto-
            # reset can complete the new episode's first CollectObservations call (which logs its
            # own Custom/PosX/PosZ for the RESET spawn point) within this SAME env.step() call,
            # before Python ever regains control - so decision_steps here, if non-empty, belongs
            # to the NEXT episode, not this one. Stop immediately without acting on it or
            # stepping again: doing so would (a) leak that reset-episode's spawn position onto the
            # end of this rollout's recorded path (this rollout's LAST point stays whatever the
            # true terminal position was) and (b) take one wasted extra env.step() advancing an
            # episode we're about to throw away anyway.
            break

        if len(decision_steps) > 0:
            feed = build_feed(session, decision_steps, continuous_size, discrete_size, action_mode)
            outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed)))
            n_agents = len(decision_steps)
            cont_actions = (outputs[cont_out].astype(np.float32) if cont_out
                             else np.zeros((n_agents, 0), dtype=np.float32))
            disc_actions = (outputs[disc_out].astype(np.int32) if disc_out
                             else np.zeros((n_agents, 0), dtype=np.int32))
            worker.env.set_actions(worker.behavior_name, ActionTuple(continuous=cont_actions, discrete=disc_actions))

        worker.env.step()
        steps_taken += 1

    if episode_length is None:
        log.warning("Rollout hit max_steps=%d without a terminal step (map=%s, mode=%s) - path "
                    "will be truncated.", max_steps, candidate["map_file"], action_mode)
        episode_length = steps_taken

    stats = worker.stats_channel.get_and_reset_stats()
    xs = [v for v, _ in stats.get("Custom/PosX", [])]
    zs = [v for v, _ in stats.get("Custom/PosZ", [])]
    reverse_flags = [bool(v) for v, _ in stats.get("Custom/ReverseGear", [])]
    speeds = [v for v, _ in stats.get("Custom/SpeedMS", [])]
    tile_types = [int(v) for v, _ in stats.get("Custom/TileType", [])]
    tile_targets = [v for v, _ in stats.get("Custom/TileTargetSpeedMS", [])]
    effective_tops = [v for v, _ in stats.get("Custom/EffectiveTopSpeedMS", [])]

    # These are emitted by the current Unity build.  Failing explicitly is
    # safer than writing an apparently valid but empty trajectory CSV when an
    # older executable is used for evaluation.
    if xs and any(len(values) == 0 for values in (speeds, tile_types, tile_targets, effective_tops)):
        raise RuntimeError(
            "The Unity executable did not report speed telemetry. Rebuild the "
            "player with the current carAgent.cs before running trajectories."
        )
    lengths = [len(xs), len(zs), len(reverse_flags), len(speeds), len(tile_types), len(tile_targets), len(effective_tops)]
    if len(set(lengths)) != 1:
        raise RuntimeError(
            "Inconsistent trajectory statistic lengths "
            f"{lengths}; aborting rather than writing misaligned samples."
        )
    # episode_length counts this episode's regular decision steps (agent_len, incremented once per
    # decision_steps appearance) - that prefix is always genuine, never leaked (leaking can only
    # happen strictly after termination is detected). Whatever comes after it is one of two things:
    # a genuine terminal observation (collision/goal - EndEpisode() is called explicitly from game
    # logic, which does trigger one more real CollectObservations call at the terminal position)
    # and/or a leaked auto-reset echo at the candidate's own spawn point (ML-Agents auto-resets the
    # instant the episode ends, and single-env auto-reset can complete the NEW episode's first
    # CollectObservations - logging ITS spawn position - within the same env.step() call that
    # reported the old episode's end, before Python ever regains control to stop; verified this can
    # leak 1 OR 2 such echoes depending on outcome type - e.g. built-in MaxStepReached appears not
    # to trigger its own fresh terminal observation the way an explicit EndEpisode() call does, so
    # both trailing entries are leaked echoes there). Strip trailing entries that exactly match
    # spawn - safe because that prefix is never touched, so a genuinely-real terminal position is
    # never at risk of being stripped, only true leaked echoes are (which are bit-exact spawn
    # coordinates, not just nearby).
    n_real = min(episode_length, *lengths)
    full_path = list(zip(xs, zs, reverse_flags, speeds, tile_types, tile_targets, effective_tops))
    tail = full_path[n_real:]
    spawn_x, spawn_z = candidate["world_spawn_x"], candidate["world_spawn_z"]
    while tail and (n_real + len(tail) > 1) \
            and abs(tail[-1][0] - spawn_x) < 1e-3 and abs(tail[-1][1] - spawn_z) < 1e-3:
        tail.pop()  # (n_real + len(tail) > 1) guards against an episode that genuinely ends on
                    # its very first step right at spawn - never strip the last remaining point
    path = full_path[:n_real] + tail

    goals = sum(v for v, _ in stats.get("Custom/GoalReached", []))
    terminated = sum(v for v, _ in stats.get("Custom/Terminated", []))
    maxstep = sum(v for v, _ in stats.get("Custom/MaxStepReached", []))
    outcome = "Goal" if goals else "Terminated" if terminated else "MaxStep" if maxstep else "Unknown"

    return dict(path=path, outcome=outcome, episode_length=episode_length)


def cmd_run(args):
    import onnxruntime as ort
    from eval_checkpoints import (
        CHECKPOINT_RE, EvalWorker, environment_parameters_from_config,
        resolve_action_outputs, step_from_path,
    )

    if not args.checkpoint and not args.run_dir:
        log.error("Provide --checkpoint or --run-dir.")
        sys.exit(1)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.run_dir) / "CarAgent.onnx"
    if not checkpoint.is_file():
        log.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)
    checkpoint_step = step_from_path(checkpoint) if CHECKPOINT_RE.search(checkpoint.name) else None
    run_id = Path(args.run_dir).name if args.run_dir else checkpoint.stem

    candidates = json.loads(Path(args.candidates).read_text())
    if not candidates:
        log.error("No candidates in %s.", args.candidates)
        sys.exit(1)

    args.environment_parameters = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            log.error("--config does not exist: %s", config_path)
            sys.exit(1)
        args.environment_parameters = environment_parameters_from_config(config_path)
        log.info("Loaded %d constant environment parameter(s) from %s.",
                 len(args.environment_parameters), config_path)
        for name in (
            "vehicle_speed_cap_enabled", "vehicle_speed_cap_ms", "regen_brake_torque",
            "sensor_observations_enabled",
        ):
            if name in args.environment_parameters:
                log.info("Evaluation parameter %s=%s", name, args.environment_parameters[name])

    worker = EvalWorker(args, args.worker_id, args.seed)
    rows: List[dict] = []
    try:
        worker.connect()
        session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
        spec = worker.spec
        continuous_size = spec.action_spec.continuous_size
        discrete_branches = spec.action_spec.discrete_branches
        discrete_size = sum(discrete_branches) if discrete_branches else 0
        action_output_names = {
            mode: resolve_action_outputs(session, continuous_size, discrete_size, mode)
            for mode in ("deterministic", "stochastic")
        }

        for candidate in candidates:
            rollouts = [("deterministic", 0)] + [("stochastic", k) for k in range(args.stochastic_rollouts)]
            for action_mode, rollout_index in rollouts:
                cont_out, disc_out = action_output_names[action_mode]
                t0 = time.time()
                result = run_one_rollout(worker, session, cont_out, disc_out, continuous_size,
                                          discrete_size, candidate, action_mode, args.max_steps_per_episode)
                requested_cap_enabled = args.environment_parameters.get("vehicle_speed_cap_enabled", 0.0) >= 0.5
                requested_cap = args.environment_parameters.get("vehicle_speed_cap_ms")
                if requested_cap_enabled and requested_cap is not None:
                    observed_top_speeds = [point[6] for point in result["path"]]
                    if observed_top_speeds and max(observed_top_speeds) > requested_cap + 0.05:
                        raise RuntimeError(
                            "Unity did not apply the requested vehicle speed cap before trajectory "
                            f"evaluation: config requested {requested_cap:.3f} m/s but "
                            f"EffectiveTopSpeedMS reported {max(observed_top_speeds):.3f} m/s."
                        )
                log.info("map=%s candidate=%d mode=%s rollout=%d outcome=%s steps=%d (%.1fs)",
                          candidate["map_file"], candidate["candidate_id"], action_mode, rollout_index,
                          result["outcome"], len(result["path"]), time.time() - t0)
                for step_index, (x, z, reverse, speed, tile_type, tile_target, effective_top) in enumerate(result["path"]):
                    rows.append(dict(
                        run_id=run_id, checkpoint_step=checkpoint_step,
                        map_file=candidate["map_file"], map_index=candidate["map_index"],
                        candidate_id=candidate["candidate_id"], action_mode=action_mode,
                        rollout_index=rollout_index, step_index=step_index, x=x, z=z,
                        reverse_gear=int(reverse), speed_mps=speed, tile_type=tile_type,
                        tile_target_speed_mps=tile_target, effective_top_speed_mps=effective_top,
                        outcome=result["outcome"], episode_length=result["episode_length"],
                    ))
    finally:
        worker.close()

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "checkpoint_step", "map_file", "map_index", "candidate_id", "action_mode",
            "rollout_index", "step_index", "x", "z", "reverse_gear", "speed_mps", "tile_type",
            "tile_target_speed_mps", "effective_top_speed_mps", "outcome", "episode_length",
        ])
        writer.writeheader()
        writer.writerows(rows)
    log.info("Done. %d rows written to %s", len(rows), out_path)


# =================================================================== CLI =======================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Auto-generate N fixed (map, spawn, goal, heading) candidates.")
    gen.add_argument("--maps-dir", required=True,
                      help="Folder of Voronoi .txt maps, e.g. Assets/Resources/Maps/VoronoiVal.")
    gen.add_argument("--gridsize", type=int, default=66, help="Must match GridManager.gridSize.")
    gen.add_argument("--cellsize", type=float, default=1.5, help="Must match GridManager.cellSize.")
    gen.add_argument("--num-maps", type=int, default=10)
    gen.add_argument("--min-path-cells", type=int, default=15,
                      help="Reject spawn/goal pairs whose real BFS path is shorter than this many cells.")
    gen.add_argument("--seed", type=int, default=12345, help="Reproducible candidate selection.")
    gen.add_argument("--out", default="candidates.json")
    gen.set_defaults(func=cmd_generate)

    pts = sub.add_parser("from-points",
                          help="Build candidates.json from hand-picked spawn/goal world coordinates.")
    pts.add_argument("--points", required=True,
                      help='JSON list of {"map_file": ..., "spawn": [x, z], "goal": [x, z]}.')
    pts.add_argument("--maps-dir", required=True,
                      help="MUST be the folder Unity actually loads from (e.g. "
                           "Assets/Resources/Maps/VoronoiVal), not a shortlist copy - map_index "
                           "is computed from this folder's sorted listing.")
    pts.add_argument("--gridsize", type=int, default=66)
    pts.add_argument("--cellsize", type=float, default=1.5)
    pts.add_argument("--out", default="candidates.json")
    pts.set_defaults(func=cmd_from_points)

    push = sub.add_parser("push-from-edge",
                           help="Move any spawn/goal outside +/-edge-limit (world units) inward "
                                "along its own road, re-center it on the road (may end up slightly "
                                "past edge-limit - centering wins), re-derive heading, and render "
                                "a before/after preview per candidate.")
    push.add_argument("--candidates", default="candidates.json")
    push.add_argument("--maps-dir", required=True,
                       help="MUST be the folder Unity actually loads from - same requirement as "
                            "from-points/generate.")
    push.add_argument("--edge-limit", type=float, required=True,
                       help="World-unit box half-extent to push points back inside - e.g. 40, or "
                            "45 to match Assets/Scenes/SampleScene.unity's spawnAreaMin/Max/"
                            "goalAreaMin/Max (the box normal random training actually samples "
                            "from; the true map edge is +/-49.5 for gridsize=66/cellsize=1.5).")
    push.add_argument("--gridsize", type=int, default=66)
    push.add_argument("--cellsize", type=float, default=1.5)
    push.add_argument("--out", default="candidates_pushed.json",
                       help="Deliberately NOT candidates.json by default - review the preview "
                            "images first, then rename/copy this over candidates.json yourself.")
    push.add_argument("--preview-dir", default="candidate_review_pushed")
    push.set_defaults(func=cmd_push_from_edge)

    plot = sub.add_parser("plot-candidates",
                           help="Render one preview PNG per candidate for a visual review pass.")
    plot.add_argument("--candidates", default="candidates.json")
    plot.add_argument("--maps-dir", required=True)
    plot.add_argument("--gridsize", type=int, default=66)
    plot.add_argument("--cellsize", type=float, default=1.5)
    plot.add_argument("--out-dir", default="candidate_review")
    plot.set_defaults(func=cmd_plot_candidates)

    render = sub.add_parser("render-maps",
                             help="Plain map renders (no spawn/goal) for browsing a shortlist before picking routes.")
    render.add_argument("--maps-dir", required=True)
    render.add_argument("--map-files", nargs="+", default=None,
                         help="Specific filenames to render (e.g. curved_401.txt curved_406.txt ...). "
                              "Omit to render every .txt map in --maps-dir.")
    render.add_argument("--gridsize", type=int, default=66)
    render.add_argument("--cellsize", type=float, default=1.5)
    render.add_argument("--out-dir", default="map_preview")
    render.set_defaults(func=cmd_render_maps)

    run = sub.add_parser("run",
                          help="Drive 1 deterministic + K stochastic rollouts per candidate, write trajectories.csv.")
    run.add_argument("--binary", required=True, help="Path to the headless Linux build executable.")
    run.add_argument("--run-dir", default=None,
                      help="results/<run_id> dir - defaults --checkpoint to <run-dir>/CarAgent.onnx "
                           "and the CSV's run_id column to <run-dir>'s name.")
    run.add_argument("--checkpoint", default=None, help="Explicit .onnx path - overrides --run-dir's default.")
    run.add_argument("--config", default=None,
                     help="Exact training YAML for this run. Its constant environment_parameters "
                          "are applied before the first episode so trajectory dynamics match "
                          "training (speed cap, regen, sensor mask, rewards, etc.).")
    run.add_argument("--candidates", default="candidates.json")
    run.add_argument("--stochastic-rollouts", type=int, default=5)
    run.add_argument("--maps", choices=["train", "val"], default="val",
                      help="Which map pool to load fixed_map_index from (use_validation_maps).")
    run.add_argument("--max-step-budget", type=int, default=None,
                      help="Overrides carAgent.cs's Agent.MaxStep for the duration of eval only - "
                           "see eval_checkpoints.py's --max-step-budget for the full reasoning.")
    run.add_argument("--max-steps-per-episode", type=int, default=5000,
                      help="Safety cap on env steps for a single rollout, in case of a stuck episode.")
    run.add_argument("--base-port", type=int, default=7005)
    run.add_argument("--worker-id", type=int, default=0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--time-scale", type=float, default=20.0)
    run.add_argument("--unity-job-worker-count", type=int, default=1,
                      help="Unity internal job-worker threads for the executable (default: 1).")
    run.add_argument("--no-graphics", action="store_true", default=True)
    run.add_argument("--graphics", dest="no_graphics", action="store_false",
                      help="Run with graphics on (local debugging only).")
    run.add_argument("--out", default="trajectories.csv")
    run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
