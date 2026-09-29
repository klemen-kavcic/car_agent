"""Validate and compare the paired original/fine-tune/scratch traffic evaluations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TRAIN_CAR_COUNTS = (2, 5, 10)
EVAL_CAR_COUNTS = (2, 5, 10)
TRAINED_ARMS = ("finetune", "scratch")
MODES = ("deterministic", "stochastic")
OUTCOMES = ("Goal", "VehicleCollision", "Terminated", "MaxStep")


def read_rows(path: Path, expected: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing evaluation: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} episodes, got {len(rows)}")
    return rows


def route_key(row: dict[str, str], cars: int) -> tuple:
    fields = ("scenario", "rollout", "map_index", "placement_seed")
    route = [row[field] for field in fields]
    for seat in range(cars):
        for field in ("spawnx", "spawnz", "goalx", "goalz"):
            route.append(round(float(row[f"{field}{seat}"]), 4))
    return tuple(route)


def summarize(rows: list[dict[str, str]], cars: int) -> dict[str, float]:
    seat_total = len(rows) * cars
    summary = {f"{outcome.lower()}_seat_rate": 0.0 for outcome in OUTCOMES}
    all_goal = any_collision = 0
    for row in rows:
        outcomes = [row[f"outcome{seat}"] for seat in range(cars)]
        for outcome in outcomes:
            if outcome not in OUTCOMES:
                raise ValueError(f"unknown outcome {outcome}")
            summary[f"{outcome.lower()}_seat_rate"] += 1 / seat_total
        all_goal += all(outcome == "Goal" for outcome in outcomes)
        any_collision += "VehicleCollision" in outcomes
    summary["all_goal_case_rate"] = all_goal / len(rows)
    summary["any_collision_case_rate"] = any_collision / len(rows)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="traffic_results/multiagent_EVAL_JOB_ID")
    parser.add_argument("--expected-episodes", type=int, default=250)
    args = parser.parse_args()
    summaries: list[dict] = []
    paired: list[dict] = []
    metrics = ("goal_seat_rate", "vehiclecollision_seat_rate", "terminated_seat_rate",
               "maxstep_seat_rate", "all_goal_case_rate", "any_collision_case_rate")
    for cars in EVAL_CAR_COUNTS:
        for mode in MODES:
            for seed in range(5):
                baseline_path = (args.root / "train0" / f"original_s{seed}" /
                                 f"cars{cars}" / mode / "traffic.csv")
                baseline_rows = read_rows(baseline_path, args.expected_episodes)
                baseline_keys = [route_key(row, cars) for row in baseline_rows]
                baseline = summarize(baseline_rows, cars)
                summaries.append({"train_cars": 0, "eval_cars": cars, "mode": mode,
                                  "seed": seed, "arm": "original",
                                  "episodes": len(baseline_rows), **baseline})
                for train_cars in TRAIN_CAR_COUNTS:
                    for arm in TRAINED_ARMS:
                        path = (args.root / f"train{train_cars}" / f"{arm}_s{seed}" /
                                f"cars{cars}" / mode / "traffic.csv")
                        rows = read_rows(path, args.expected_episodes)
                        if [route_key(row, cars) for row in rows] != baseline_keys:
                            raise ValueError(f"route/map pairing differs from original for {path}")
                        values = summarize(rows, cars)
                        summaries.append({"train_cars": train_cars, "eval_cars": cars,
                                          "mode": mode, "seed": seed, "arm": arm,
                                          "episodes": len(rows), **values})
                        paired.append({"train_cars": train_cars, "eval_cars": cars,
                                       "mode": mode, "seed": seed, "arm": arm,
                                       **{f"delta_{name}": values[name] - baseline[name]
                                          for name in metrics}})
    means = []
    conditions = [(0, "original")] + [
        (density, method) for density in TRAIN_CAR_COUNTS for method in TRAINED_ARMS
    ]
    for cars in EVAL_CAR_COUNTS:
        for mode in MODES:
            for train_cars, arm in conditions:
                group = [row for row in summaries if row["eval_cars"] == cars and
                         row["mode"] == mode and row["train_cars"] == train_cars and
                         row["arm"] == arm]
                means.append({"train_cars": train_cars, "eval_cars": cars, "mode": mode,
                              "arm": arm, "seeds": len(group),
                              **{name: sum(row[name] for row in group) / len(group)
                                 for name in metrics}})
    write_csv(args.root / "comparison_by_seed.csv", summaries)
    write_csv(args.root / "paired_delta_vs_original.csv", paired)
    write_csv(args.root / "comparison_means.csv", means)
    print(f"Validated {len(summaries)} evaluations and their paired routes.")
    for cars in EVAL_CAR_COUNTS:
        for mode in MODES:
            print(f"Evaluation with {cars} cars, {mode}:")
            for row in means:
                if row["eval_cars"] == cars and row["mode"] == mode:
                    print(f"  train={row['train_cars']:2d} {row['arm']:8s} "
                          f"goal={row['goal_seat_rate']:.1%} "
                          f"collision seats={row['vehiclecollision_seat_rate']:.1%}")
    print(f"Wrote comparison_by_seed.csv, paired_delta_vs_original.csv, "
          f"and comparison_means.csv under {args.root}")


if __name__ == "__main__":
    main()
