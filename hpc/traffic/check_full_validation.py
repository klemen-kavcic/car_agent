"""Read-only completeness and sanity check for traffic full-validation jobs.

Copy this file to car1/check_full_validation.py on HPC, then run for example:
    python3 check_full_validation.py --expected smoke --job-ids 19375063 19375064
    python3 check_full_validation.py --expected full --job-ids 19375100
No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

JOB_PATTERN = re.compile(r"full_validation_(\d+)$")
CONDITION_PATTERN = re.compile(r"cars(2|5|10)_(same|mixed)_seeds.*$")
MODES = ("deterministic", "stochastic")
EXPECTED_CONDITIONS = {(cars, strategy) for cars in (2, 5, 10)
                       for strategy in ("same", "mixed")}
OUTCOMES = {"Goal", "VehicleCollision", "Terminated", "MaxStep"}
MATCH_FIELDS = ("scenario", "rollout", "map_index", "map_file", "num_cars",
                "placement_seed", "assignment_seed", "map_schedule_seed")
SEAT_MATCH_FIELDS = ("model", "model_seed", "spawnx", "spawnz", "goalx", "goalz",
                     "headingdeg", "spawn_tile", "goal_tile")
FLOAT_FIELDS = {"spawnx", "spawnz", "goalx", "goalz", "headingdeg"}


def load_mode(condition: Path, mode: str, problems: list[str]) -> tuple[list[dict], int, list[int], int]:
    files = sorted((condition / mode).glob("shard*/traffic.csv"))
    rows = []
    sizes = []
    summaries = 0
    for path in files:
        summaries += path.with_name("traffic_summary.csv").is_file()
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    problems.append(f"{mode}/{path.parent.name}: empty CSV")
                    sizes.append(0)
                    continue
                shard_rows = list(reader)
        except (OSError, csv.Error) as exc:
            problems.append(f"{mode}/{path.parent.name}: unreadable CSV ({exc})")
            sizes.append(0)
            continue
        sizes.append(len(shard_rows))
        rows.extend(shard_rows)
    return rows, len(files), sizes, summaries


def case_key(row: dict) -> tuple[str, str]:
    return row["scenario"], row["rollout"]


def validate_rows(rows: list[dict], cars: int, strategy: str, mode: str,
                  problems: list[str]) -> Counter:
    outcomes = Counter()
    seen = set()
    for row_number, row in enumerate(rows, 1):
        label = f"{mode} row {row_number}"
        try:
            key = case_key(row)
            if key in seen:
                problems.append(f"{label}: duplicate scenario/rollout {key}")
            seen.add(key)
            if int(row["num_cars"]) != cars or row["mode"] != mode or row["seed_strategy"] != strategy:
                problems.append(f"{label}: condition columns do not match the directory")
            if row["random_placement"] != "1":
                problems.append(f"{label}: random_placement is not 1")
            if row["scenario"] != row["map_index"] or not row["map_file"]:
                problems.append(f"{label}: wrong or missing validation map")
            if float(row["min_endpoint_distance_m"]) < 8 - 1e-3:
                problems.append(f"{label}: endpoint spacing is below 8 m")
            if float(row["min_own_route_distance_m"]) < 20 - 1e-3:
                problems.append(f"{label}: a car's start and goal are less than 20 m apart")
            seeds = [int(row[f"model_seed{seat}"]) for seat in range(cars)]
            if strategy == "same" and len(set(seeds)) != 1:
                problems.append(f"{label}: same condition has different model seeds {seeds}")
            if strategy == "mixed":
                expected = ([0, 1, 2, 3, 4] if cars == 5 else
                            [seed for seed in range(5) for _ in range(2)] if cars == 10 else None)
                if cars == 2 and len(set(seeds)) != 2:
                    problems.append(f"{label}: mixed two-car seeds are not distinct {seeds}")
                elif expected is not None and sorted(seeds) != expected:
                    problems.append(f"{label}: mixed {cars}-car seed counts are wrong {seeds}")
            for seat in range(cars):
                if row[f"spawn_tile{seat}"] != "Asphalt" or row[f"goal_tile{seat}"] != "Asphalt":
                    problems.append(f"{label}: seat {seat} spawn/goal is not on Asphalt")
                outcome = row[f"outcome{seat}"]
                if outcome not in OUTCOMES:
                    problems.append(f"{label}: seat {seat} has invalid outcome {outcome!r}")
                else:
                    outcomes[outcome] += 1
                float(row[f"reward{seat}"])
        except (KeyError, ValueError, TypeError) as exc:
            problems.append(f"{label}: missing or invalid CSV value ({exc})")
        if len(problems) >= 20:
            break
    return outcomes


def compare_modes(det: list[dict], stoch: list[dict], cars: int,
                  problems: list[str]) -> int:
    det_by_case = {case_key(row): row for row in det if row.get("scenario") is not None
                   and row.get("rollout") is not None}
    stoch_by_case = {case_key(row): row for row in stoch if row.get("scenario") is not None
                     and row.get("rollout") is not None}
    paired = set(det_by_case) & set(stoch_by_case)
    for key in sorted(paired):
        a, b = det_by_case[key], stoch_by_case[key]
        for field in MATCH_FIELDS:
            if a.get(field) != b.get(field):
                problems.append(f"case {key}: deterministic/stochastic {field} differs")
        for seat in range(cars):
            for suffix in SEAT_MATCH_FIELDS:
                field = f"{suffix}{seat}"
                if suffix in FLOAT_FIELDS:
                    try:
                        same = math.isclose(float(a[field]), float(b[field]), abs_tol=1e-4)
                    except (KeyError, ValueError, TypeError):
                        same = False
                else:
                    same = a.get(field) == b.get(field)
                if not same:
                    problems.append(f"case {key}: deterministic/stochastic {field} differs")
        if len(problems) >= 20:
            break
    return len(paired)


def outcome_text(counts: Counter) -> str:
    return " ".join(f"{short}={counts[name]}" for short, name in
                    (("G", "Goal"), ("C", "VehicleCollision"),
                     ("T", "Terminated"), ("M", "MaxStep")))


def inspect_condition(job: Path, condition: Path, expected: str) -> str:
    match = CONDITION_PATTERN.fullmatch(condition.name)
    if match is None:
        return ""
    cars, strategy = int(match.group(1)), match.group(2)
    problems: list[str] = []
    loaded = {mode: load_mode(condition, mode, problems) for mode in MODES}
    det, det_files, det_sizes, det_summaries = loaded["deterministic"]
    stoch, stoch_files, stoch_sizes, stoch_summaries = loaded["stochastic"]
    det_outcomes = validate_rows(det, cars, strategy, "deterministic", problems)
    stoch_outcomes = validate_rows(stoch, cars, strategy, "stochastic", problems)
    paired = compare_modes(det, stoch, cars, problems)
    wanted_files, wanted_rows, wanted_size = (1, 1, 1) if expected == "smoke" else (10, 250, 25)
    complete = (det_files == wanted_files and stoch_files == wanted_files and
                det_summaries == wanted_files and stoch_summaries == wanted_files and
                len(det) == wanted_rows and len(stoch) == wanted_rows and
                all(size == wanted_size for size in det_sizes + stoch_sizes) and
                paired == wanted_rows)
    status = "FAIL" if problems else "PASS" if complete else "INCOMPLETE"
    lines = [f"{job.name} cars={cars} strategy={strategy}: {status}",
             f"  deterministic: {len(det)} rows / {det_files} CSVs / {det_summaries} summaries; seats {outcome_text(det_outcomes)}",
             f"  stochastic:    {len(stoch)} rows / {stoch_files} CSVs / {stoch_summaries} summaries; seats {outcome_text(stoch_outcomes)}",
             f"  paired maps/placements/models: {paired}; expected {wanted_rows}"]
    lines.extend(f"  ERROR: {problem}" for problem in problems[:8])
    if len(problems) > 8:
        lines.append(f"  ... {len(problems) - 8} more problems")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("traffic_results"),
                        help="Directory containing full_validation_<job-id> folders")
    parser.add_argument("--expected", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--job-ids", nargs="*", help="Only check these Slurm job IDs")
    parser.add_argument("--latest", type=int, default=12,
                        help="Without --job-ids, inspect the newest N matching folders")
    parser.add_argument("--require-six-conditions", action="store_true",
                        help="For the one-job sweep, require all 2/5/10-car same/mixed conditions")
    args = parser.parse_args()
    if args.job_ids:
        jobs = [args.root / f"full_validation_{job_id}" for job_id in args.job_ids]
    else:
        jobs = sorted((path for path in args.root.glob("full_validation_*")
                       if JOB_PATTERN.fullmatch(path.name)),
                      key=lambda path: int(JOB_PATTERN.fullmatch(path.name).group(1)))[-args.latest:]
    if not jobs:
        parser.error(f"no full-validation job folders found under {args.root}")
    if not args.job_ids and len(jobs) < args.latest:
        print(f"WARNING: requested the latest {args.latest} jobs, but found only "
              f"{len(jobs)} full-validation result folders under {args.root}.")
    for job in jobs:
        if not job.is_dir():
            print(f"{job.name}: INCOMPLETE (results folder not present yet)")
            continue
        conditions = [path for path in sorted(job.iterdir())
                      if path.is_dir() and CONDITION_PATTERN.fullmatch(path.name)]
        if not conditions:
            print(f"{job.name}: INCOMPLETE (no same/mixed condition folder yet)")
        for condition in conditions:
            print(inspect_condition(job, condition, args.expected))
        if args.require_six_conditions:
            found = {(int(match.group(1)), match.group(2))
                     for condition in conditions
                     if (match := CONDITION_PATTERN.fullmatch(condition.name))}
            missing = sorted(EXPECTED_CONDITIONS - found)
            extra = sorted(found - EXPECTED_CONDITIONS)
            status = "PASS" if not missing and not extra else "INCOMPLETE"
            print(f"{job.name} six-condition coverage: {status}"
                  f" (found {len(found)}/6; missing={missing}; extra={extra})")
    print("G=Goal C=VehicleCollision T=Terminated M=MaxStep; "
          "PASS validates CSV content only—confirm Slurm COMPLETED/0:0 separately.")


if __name__ == "__main__":
    main()
