"""Collect the full 1M-step traffic checkpoint sweep into learning curves.

Reads the evaluator's small traffic_summary.csv files rather than loading all
per-episode traffic.csv files. Requires all expected files unless the caller
passes a smaller --expected-episodes for a deliberately shortened evaluation.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

TRAIN_DENSITIES = (2, 5, 10)
EVAL_DENSITIES = (2, 5, 10)
MODES = ("deterministic", "stochastic")
METHOD_STEPS_M = {"finetune": 20, "scratch": 40}
COUNTS = {
    "goal_seat_rate": "goal_seats",
    "vehicle_collision_seat_rate": "vehicle_collision_seats",
    "termination_seat_rate": "non_vehicle_termination_seats",
    "maxstep_seat_rate": "maxstep_seats",
    "all_goal_case_rate": "cases_all_goal",
    "any_collision_case_rate": "cases_with_vehicle_collision",
}
CHECKPOINT_NAME = re.compile(r"CarAgent-(\d+)\.onnx$")


def read_rates(path: Path, cars: int, expected: int) -> dict[str, float]:
    if not path.is_file():
        raise ValueError(f"missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one summary row, found {len(rows)}")
    row = rows[0]
    if int(row["cases"]) != expected or int(row["cars_per_case"]) != cars:
        raise ValueError(f"{path}: cases/cars do not match {expected}/{cars}")
    if sum(int(row[key]) for key in (
        "goal_seats", "vehicle_collision_seats", "non_vehicle_termination_seats", "maxstep_seats"
    )) != expected * cars:
        raise ValueError(f"{path}: seat outcomes do not sum to {expected * cars}")
    return {name: int(row[field]) / (expected * cars if "seat_rate" in name else expected)
            for name, field in COUNTS.items()}


def actual_step(directory: Path) -> int:
    path = directory / "input_hashes.sha256"
    if not path.is_file():
        raise ValueError(f"missing input hashes: {path}")
    matches = [int(match.group(1)) for line in path.read_text(encoding="utf-8").splitlines()
               if (match := CHECKPOINT_NAME.search(line))]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one checkpoint ONNX path")
    return matches[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="traffic_results/multiagent_checkpoints_TRAIN_JOB_ID")
    parser.add_argument("--expected-episodes", type=int, default=250)
    args = parser.parse_args()
    if args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")
    by_seed: list[dict] = []
    paired: list[dict] = []
    for eval_cars in EVAL_DENSITIES:
        for mode in MODES:
            for seed in range(5):
                baseline_dir = (args.root / "train0" / f"original_s{seed}" / "step_final" /
                                f"cars{eval_cars}" / mode)
                baseline = read_rates(baseline_dir / "traffic_summary.csv", eval_cars,
                                      args.expected_episodes)
                by_seed.append({"train_cars": 0, "method": "original", "seed": seed,
                                "checkpoint_million": 0, "lifetime_million": 20,
                                "actual_train_steps": "",
                                "eval_cars": eval_cars, "mode": mode, **baseline})
                for train_cars in TRAIN_DENSITIES:
                    for method, last_million in METHOD_STEPS_M.items():
                        for milestone in range(1, last_million + 1):
                            directory = (args.root / f"train{train_cars}" /
                                         f"{method}_s{seed}" / f"step_{milestone}" /
                                         f"cars{eval_cars}" / mode)
                            rates = read_rates(directory / "traffic_summary.csv", eval_cars,
                                               args.expected_episodes)
                            row = {"train_cars": train_cars, "method": method, "seed": seed,
                                   "checkpoint_million": milestone,
                                   "lifetime_million": milestone + (20 if method == "finetune" else 0),
                                   "actual_train_steps": actual_step(directory),
                                   "eval_cars": eval_cars, "mode": mode, **rates}
                            by_seed.append(row)
                            paired.append({"train_cars": train_cars, "method": method,
                                           "seed": seed, "checkpoint_million": milestone,
                                           "lifetime_million": row["lifetime_million"],
                                           "eval_cars": eval_cars, "mode": mode,
                                           **{f"delta_{key}": rates[key] - baseline[key]
                                              for key in COUNTS}})
    grouped: dict[tuple, list[dict]] = {}
    for row in by_seed:
        key = (row["train_cars"], row["method"], row["checkpoint_million"],
               row["eval_cars"], row["mode"])
        grouped.setdefault(key, []).append(row)
    means = []
    for (train_cars, method, milestone, eval_cars, mode), rows in sorted(grouped.items()):
        if len(rows) != 5:
            raise ValueError(f"incomplete seed set for {train_cars}/{method}/{milestone}/{eval_cars}/{mode}")
        means.append({"train_cars": train_cars, "method": method,
                      "checkpoint_million": milestone,
                      "lifetime_million": rows[0]["lifetime_million"],
                      "eval_cars": eval_cars,
                      "mode": mode, "seeds": 5,
                      **{key: sum(row[key] for row in rows) / 5 for key in COUNTS}})
    write_csv(args.root / "checkpoint_curve_by_seed.csv", by_seed)
    write_csv(args.root / "paired_delta_vs_original_by_seed.csv", paired)
    write_csv(args.root / "checkpoint_curve_means.csv", means)
    print(f"Validated {len(by_seed)} summary files: 900 checkpoints x 3 car counts x 2 modes "
          f"plus 5 baselines x 3 x 2, {args.expected_episodes} episodes each.")
    print("Wrote checkpoint_curve_by_seed.csv, checkpoint_curve_means.csv, "
          "and paired_delta_vs_original_by_seed.csv")


if __name__ == "__main__":
    main()
