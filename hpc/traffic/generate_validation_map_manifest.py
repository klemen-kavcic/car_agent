"""List the held-out Voronoi maps for runtime-randomized traffic placement.

Only the map is predetermined. TrafficScenarioManager samples new, separated
Asphalt spawn/goal locations on that map for each episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / "Assets/Resources/Maps/VoronoiVal"
OUTPUT = ROOT / "Assets/Resources/Traffic/traffic_validation_maps.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, default=MAPS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()
    files = sorted(args.maps_dir.glob("*.txt"))
    if len(files) != args.expected_count:
        parser.error(f"expected {args.expected_count} maps, found {len(files)} in {args.maps_dir}")
    manifest = {
        "placement_mode": "random_connected_asphalt",
        "scenarios": [
            {"scenario_id": index, "map_index": index, "map_file": path.name}
            for index, path in enumerate(files)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} map-only cases to {args.out}")


if __name__ == "__main__":
    main()
