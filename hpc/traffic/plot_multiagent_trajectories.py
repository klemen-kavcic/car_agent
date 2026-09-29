"""Plot the ten held-out traffic trajectory maps for one model/car-count result.

Run after both modes finish, for example:
  python3 plot_multiagent_trajectories.py \
    traffic_results/multiagent_checkpoints_123/train2/finetune_s0/step_20/cars2 \
    --maps-dir Assets/Resources/Maps/VoronoiVal

All cars' paths are shown. Solid lines are deterministic; faint lines are five
stochastic rollouts on the same start/goal positions. Colors identify seats.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

MAP_COLORS = ["#aaa69e", "#444444", "#16a516", "#59b5e8", "#cf3030"]
END_MARKERS = {"Goal": "*", "VehicleCollision": "X", "Terminated": "x", "MaxStep": "s"}


def read_mode(path: Path, mode: str) -> dict:
    csv_path = path / mode / "trajectories" / "trajectories.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    grouped = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] != mode:
                raise ValueError(f"Incorrect mode in {csv_path}: {row['mode']}")
            key = (int(row["scenario"]), int(row["rollout"]), int(row["seat"]))
            grouped[key].append(row)
    if not grouped:
        raise ValueError(f"Empty trajectory file: {csv_path}")
    for key, rows in grouped.items():
        if [int(row["sample"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous samples for {mode} {key}")
    return grouped


def route(row: dict) -> tuple[float, ...]:
    return tuple(float(row[name]) for name in ("spawn_x", "spawn_z", "goal_x", "goal_z"))


def plot(result_dir: Path, maps_dir: Path | None, output: Path) -> None:
    paths = {mode: read_mode(result_dir, mode)
             for mode in ("deterministic", "stochastic")}
    scenarios = sorted({key[0] for mode in paths.values() for key in mode})
    if len(scenarios) != 10:
        raise ValueError(f"Expected ten trajectory maps, found {scenarios}")
    num_cars = int(next(iter(paths["deterministic"].values()))[0]["num_cars"])
    cmap = plt.get_cmap("tab10", num_cars)
    fig, axes = plt.subplots(2, 5, figsize=(23, 10), constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenarios):
        example = paths["deterministic"][(scenario, 0, 0)][0]
        map_file = example["map_file"]
        map_path = maps_dir / map_file if maps_dir is not None else None
        if map_path is not None and not map_path.is_file():
            raise FileNotFoundError(map_path)
        if map_path is not None and map_path.is_file():
            grid = np.loadtxt(map_path, dtype=int)
            if grid.ndim != 2 or not np.isin(grid, range(5)).all():
                raise ValueError(f"Invalid map grid: {map_path}")
            half_x = grid.shape[1] * 1.5 / 2
            half_z = grid.shape[0] * 1.5 / 2
            ax.imshow(grid, origin="lower", interpolation="nearest",
                      cmap=ListedColormap(MAP_COLORS), vmin=-0.5, vmax=4.5,
                      extent=(-half_x, half_x, -half_z, half_z), alpha=0.72)
        for seat in range(num_cars):
            reference = route(paths["deterministic"][(scenario, 0, seat)][0])
            for mode, collection in paths.items():
                rollouts = sorted(key[1] for key in collection
                                  if key[0] == scenario and key[2] == seat)
                expected = [0] if mode == "deterministic" else list(range(5))
                if rollouts != expected:
                    raise ValueError(f"Unexpected {mode} rollouts: map {scenario}, seat {seat}: {rollouts}")
                for rollout in rollouts:
                    rows = collection[(scenario, rollout, seat)]
                    if not np.allclose(route(rows[0]), reference, atol=1e-4, rtol=0):
                        raise ValueError(f"Spawn/goal changed on map {scenario}, seat {seat}")
                    x = [float(row["x"]) for row in rows]
                    z = [float(row["z"]) for row in rows]
                    ax.plot(x, z, color=cmap(seat),
                            lw=1.8 if mode == "deterministic" else 0.8,
                            alpha=0.95 if mode == "deterministic" else 0.22,
                            zorder=3 if mode == "deterministic" else 2)
                    if mode == "deterministic":
                        ax.plot(x[-1], z[-1], marker=END_MARKERS[rows[-1]["outcome"]],
                                color=cmap(seat), markersize=8, linestyle="None", zorder=5)
            sx, sz, gx, gz = reference
            ax.plot(sx, sz, "o", color=cmap(seat), markeredgecolor="black", markersize=5)
            ax.plot(gx, gz, "*", color=cmap(seat), markeredgecolor="black", markersize=10)
        ax.set_title(f"{map_file} · map {scenario}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.grid(alpha=0.16)
    fig.suptitle(f"{result_dir.parent.name} / {result_dir.name} · {num_cars} cars · "
                 "solid=deterministic, faint=5 stochastic; circle=start, star=goal; "
                 "end: *=goal, X=collision, x=terminal, square=max-step")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path,
                        help="Directory containing deterministic/ and stochastic/")
    parser.add_argument("--maps-dir", type=Path,
                        help="VoronoiVal text maps; omit for a path-only plot")
    parser.add_argument("--out", type=Path,
                        help="Output PNG; defaults to result_dir/trajectory_grid.png")
    args = parser.parse_args()
    plot(args.result_dir, args.maps_dir, args.out or args.result_dir / "trajectory_grid.png")


if __name__ == "__main__":
    main()
