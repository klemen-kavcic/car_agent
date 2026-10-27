"""Generate simple, interpretable road layouts for the traffic-recovery curriculum.

The maps deliberately live in a separate Resources folder.  They therefore do
not alter the sorted VoronoiTrain/VoronoiVal map indices used by existing
experiments.  A later recovery-curriculum scene/manager can load this catalog
explicitly.

Text-map codes match ``grid_manager.cs``'s ``VoronoiCodeToType`` mapping:
0 Gravel, 1 Asphalt, 2 SpeedLimited, 3 Slippery, 4 Terminal.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.interpolate import splev, splprep

SIZE = 66
# These are map-file codes, deliberately not the numeric TileType enum values.
GRAVEL, ASPHALT, TERMINAL = 0, 1, 4
ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "Assets" / "Resources" / "Maps" / "TrafficRecovery"
PREVIEW_DIR = ROOT / "docs" / "recovery_maps"


def segment_distance(x: np.ndarray, z: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """Distance from every cell centre to a finite centre-line segment."""
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    length_sq = dx * dx + dz * dz
    t = np.clip(((x - ax) * dx + (z - az) * dz) / length_sq, 0.0, 1.0)
    return np.hypot(x - (ax + t * dx), z - (az + t * dz))


def road(points: list[tuple[float, float]], asphalt_radius: float = 1.55, shoulder: float = 1.5) -> np.ndarray:
    """Return a terminal map with a wide asphalt centre and gravel shoulders."""
    z, x = np.mgrid[0:SIZE, 0:SIZE]
    # Put centre lines between grid-cell centres.  With the 1.55-cell radius
    # this rasterises a straight segment as a four-cell asphalt lane, matching
    # the modal road width of the 66x66 curved Voronoi maps.
    points = [(px + 0.5, pz + 0.5) for px, pz in points]
    nearest = np.full((SIZE, SIZE), np.inf)
    for a, b in zip(points, points[1:]):
        nearest = np.minimum(nearest, segment_distance(x, z, a, b))
    grid = np.full((SIZE, SIZE), TERMINAL, dtype=np.uint8)
    grid[nearest <= asphalt_radius + shoulder] = GRAVEL
    grid[nearest <= asphalt_radius] = ASPHALT
    return grid


def spline_path(control_points: list[tuple[float, float]], samples: int = 100) -> list[tuple[float, float]]:
    """Use the same interpolating B-spline approach as voronoi_curved.py.

    A dense smooth centre line lets ``road`` rasterise the S-bend as a natural
    road rather than a set of visibly straight diagonal segments.
    """
    controls = np.asarray(control_points, dtype=float)
    degree = min(3, len(controls) - 1)
    try:
        knots, _ = splprep([controls[:, 0], controls[:, 1]], s=0, k=degree)
        xs, zs = splev(np.linspace(0, 1, samples), knots)
        return list(zip(xs, zs))
    except (TypeError, ValueError):
        return control_points


def combine(*grids: np.ndarray) -> np.ndarray:
    """Merge road pieces; Asphalt takes precedence over Gravel then Terminal."""
    stacked = np.stack(grids)
    merged = np.full((SIZE, SIZE), TERMINAL, dtype=np.uint8)
    merged[np.any(stacked == GRAVEL, axis=0)] = GRAVEL
    merged[np.any(stacked == ASPHALT, axis=0)] = ASPHALT
    return merged


def terminal_wall_corridor(grid: np.ndarray, x0: int, x1: int, z0: int, z1: int) -> np.ndarray:
    """Remove forgiving gravel shoulders from a narrow recovery corridor.

    The four-cell asphalt remains driveable, but every surrounding cell in the
    supplied rectangle becomes Terminal. This prevents a normal-sized car from
    making a wide U-turn on the shoulder and makes reversing the only safe way
    back from a capped branch.
    """
    result = grid.copy()
    region = result[z0:z1 + 1, x0:x1 + 1]
    region[region != ASPHALT] = TERMINAL
    return result


def narrow_horizontal_terminal_corridor(grid: np.ndarray, x0: int, x1: int, z_center: int) -> np.ndarray:
    """Constrain an existing horizontal road to a 2-cell asphalt reverse lane.

    The source road determines where asphalt may remain; no new road is drawn.
    This prevents the cap from accidentally extending across a terminal gap.
    """
    result = grid.copy()
    original = grid.copy()
    # Remove shoulders and the outer asphalt rows. The two central cells give
    # a 3 m lane for a 2 m wide car: feasible to reverse, impractical to turn.
    result[z_center - 3:z_center + 5, x0:x1 + 1] = TERMINAL
    source_strip = original[z_center:z_center + 2, x0:x1 + 1]
    target_strip = result[z_center:z_center + 2, x0:x1 + 1]
    target_strip[source_strip == ASPHALT] = ASPHALT
    return result


def asphalt_plaza(x0: int, x1: int, z0: int, z1: int, shoulder: int = 2) -> np.ndarray:
    """Open asphalt square with a gravel shoulder and terminal exterior."""
    grid = np.full((SIZE, SIZE), TERMINAL, dtype=np.uint8)
    grid[z0 - shoulder:z1 + shoulder + 1, x0 - shoulder:x1 + shoulder + 1] = GRAVEL
    grid[z0:z1 + 1, x0:x1 + 1] = ASPHALT
    return grid


MAPS: dict[str, dict] = {
    "recovery_01_straight": {
        "title": "Straight lane",
        "purpose": "Basic forward/reverse steering with forgiving gravel shoulders and terminal outside.",
        "grid": lambda: road([(33, 6), (33, 59)]),
    },
    "recovery_02_turn": {
        "title": "Right-angle turn",
        "purpose": "Learn to turn into a new corridor rather than continue into the terminal boundary.",
        "grid": lambda: road([(18, 7), (18, 39), (49, 39)]),
    },
    "recovery_04_crossroads": {
        "title": "Four-way intersection",
        "purpose": "Introduce crossing paths with initially generous approach distances.",
        "grid": lambda: combine(road([(33, 7), (33, 58)]), road([(7, 33), (58, 33)])),
    },
    "recovery_05_passing_bay": {
        "title": "Single lane with passing bay",
        "purpose": "A yielding/reversing car can retreat into the bay to let another car pass.",
        "grid": lambda: combine(
            road([(33, 7), (33, 58)]),
            road([(33, 26), (21, 26), (21, 40), (33, 40)]),
        ),
    },
    "recovery_06_blind_alley": {
        "title": "Blind alley",
        "purpose": "A 3 m terminal-walled capped branch is too narrow for a U-turn, so the car must reverse to the main road.",
        "grid": lambda: narrow_horizontal_terminal_corridor(combine(
            road([(18, 7), (18, 57)]),
            road([(18, 45), (48, 45)]),
        ), 33, 52, 45),
    },
    "recovery_08_dead_end_detour": {
        "title": "Dead end with detour",
        "purpose": "The direct branch ends at Terminal; reaching the forward-side goal requires reversing to the main road and taking the lower bypass.",
        "grid": lambda: narrow_horizontal_terminal_corridor(combine(
            road([(18, 7), (18, 57)]),
            # The tempting upper branch is a genuine dead end.
            road([(18, 45), (40, 45)]),
            # A separate lower branch reconnects on the far side of the cap.
            # Keep a terminal gap between the upper cap and this connector;
            # gravel shoulders must not make an accidental shortcut.
            road([(18, 28), (50, 28), (50, 39)]),
        ), 31, 45, 45),
    },
    "recovery_07_s_curve": {
        "title": "S-curve",
        "purpose": "Follow continuous road geometry through alternating bends rather than memorising a single corner.",
        "grid": lambda: road(spline_path([(20, 7), (20, 18), (46, 27), (46, 39), (20, 48), (20, 59)])),
    },
    "recovery_09_circulation_loop": {
        "title": "Close-range traffic plaza",
        "purpose": "Four nearby cars have forward goals on open asphalt, isolating vehicle negotiation from road-boundary following.",
        "grid": lambda: asphalt_plaza(18, 48, 18, 48),
    },
}


def write_map(name: str, grid: np.ndarray) -> None:
    text = "\n".join(" ".join(map(str, row)) for row in grid) + "\n"
    (MAP_DIR / f"{name}.txt").write_text(text, encoding="utf-8")


def main() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    catalog = []
    cmap = ListedColormap(["#9a907e", "#414141", "#ffe680", "#d7f2ff", "#cf2525"])
    labels = ["Gravel", "Asphalt", "Speed limited", "Slippery", "Terminal"]
    rendered = []
    for name, spec in MAPS.items():
        grid = spec["grid"]()
        write_map(name, grid)
        catalog.append({"file": f"{name}.txt", "title": spec["title"], "purpose": spec["purpose"],
                        "grid_size": SIZE, "tile_legend": {"0": "Gravel", "1": "Asphalt", "4": "Terminal"}})
        rendered.append((name, spec, grid))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, origin="lower", interpolation="nearest")
        ax.set_title(f"{spec['title']}\n{spec['purpose']}", fontsize=9)
        ax.set_xlabel("x cell"); ax.set_ylabel("z cell")
        fig.tight_layout()
        fig.savefig(PREVIEW_DIR / f"{name}.png", dpi=180, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    for ax, (name, spec, grid) in zip(axes.flat, rendered):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, origin="lower", interpolation="nearest")
        ax.set_title(spec["title"])
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(rendered):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(code)) for code in (4, 1, 0)]
    fig.legend(handles, [labels[4], labels[1], labels[0]], loc="lower center", ncol=3)
    fig.suptitle("Traffic recovery curriculum: proposed basic maps", fontsize=16)
    fig.savefig(PREVIEW_DIR / "recovery_map_overview.png", dpi=200, facecolor="white")
    plt.close(fig)

    (MAP_DIR / "recovery_map_catalog.json").write_text(json.dumps({"maps": catalog}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rendered)} maps to {MAP_DIR}")
    print(f"Wrote previews to {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
