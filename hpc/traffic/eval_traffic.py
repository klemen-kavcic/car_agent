"""Evaluate 2..N laser77 ONNX policies on shared-map traffic cases.

The original ten-case, two-car pilot remains the default. Use --manifest with a
TrafficAllValidation build to evaluate all 100 held-out maps with randomized
Asphalt starts/goals and 2..pool-size cars.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

HERE = Path(__file__).resolve()
ROOT = next((parent for parent in HERE.parents if (parent / "eval_checkpoints.py").exists()),
            HERE.parents[2])
sys.path.insert(0, str(ROOT))
from eval_checkpoints import (  # noqa: E402
    build_feed,
    environment_parameters_from_config,
    resolve_action_outputs,
)

DEFAULT_MODEL = ROOT / "hpc/0909/results/car_vgrid_19016936_laser12_cap3_s0/CarAgent.onnx"
PORT_BLOCK_SIZE = 64  # Matches the HPC scripts' local port reservation.
PORT_ATTEMPTS = 6
OUTCOMES = ("Goal", "Terminated", "VehicleCollision", "MaxStep")
TILE_TYPES = (
    ("Slippery", "slippery"),
    ("SpeedLimited", "speed_limited"),
    ("Terminal", "terminal"),
    ("Asphalt", "asphalt"),
    ("Gravel", "gravel"),
)


def placement_seed_for_case(args, scenario: int, rollout: int) -> int:
    offset = 0 if args.fixed_placement_across_rollouts else 1000 * rollout
    return args.placement_seed + scenario + offset


def model_run_name(model: Path) -> str:
    return model.parent.parent.name if model.parent.name == "CarAgent" else model.parent.name


def write_summary(rows: list[dict], path: Path) -> Path:
    """Count both affected cars and cases; one crash may affect multiple seats."""
    car_count = int(rows[0]["num_cars"])
    summary = {
        "mode": rows[0]["mode"],
        "seed_strategy": rows[0]["seed_strategy"],
        "assignment_seed": rows[0]["assignment_seed"],
        "cases": len(rows),
        "cars_per_case": car_count,
        "goal_seats": 0,
        "vehicle_collision_seats": 0,
        "non_vehicle_termination_seats": 0,
        "maxstep_seats": 0,
        "cases_with_vehicle_collision": 0,
        "cases_with_non_vehicle_termination": 0,
        "cases_with_maxstep": 0,
        "cases_both_goal": 0,
        "cases_all_goal": 0,
    }
    field_by_outcome = {
        "Goal": "goal_seats",
        "VehicleCollision": "vehicle_collision_seats",
        "Terminated": "non_vehicle_termination_seats",
        "MaxStep": "maxstep_seats",
    }
    for row in rows:
        outcomes = tuple(row[f"outcome{seat}"] for seat in range(car_count))
        for outcome in outcomes:
            summary[field_by_outcome[outcome]] += 1
        summary["cases_with_vehicle_collision"] += int("VehicleCollision" in outcomes)
        summary["cases_with_non_vehicle_termination"] += int("Terminated" in outcomes)
        summary["cases_with_maxstep"] += int("MaxStep" in outcomes)
        all_goal = all(outcome == "Goal" for outcome in outcomes)
        summary["cases_all_goal"] += int(all_goal)
        if car_count == 2:
            summary["cases_both_goal"] += int(all_goal)
    timed_out = [(row, seat) for row in rows for seat in range(car_count)
                 if row[f"outcome{seat}"] == "MaxStep"]
    for tile_name, column_name in TILE_TYPES:
        summary[f"maxstep_finish_on_{column_name}_seats"] = sum(
            row.get(f"finish_tile{seat}") == tile_name for row, seat in timed_out)
        durations = [float(row[f"tile_seconds{seat}_{column_name}"])
                     for row, seat in timed_out
                     if row.get(f"tile_seconds{seat}_{column_name}") != ""]
        summary[f"mean_maxstep_seconds_on_{column_name}"] = (
            sum(durations) / len(durations) if durations else "")
    summary_path = path.with_name(f"{path.stem}_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary_path


def latest_stat(stats: dict, name: str) -> float:
    values = stats.get(name, [])
    if not values:
        raise RuntimeError(f"Traffic build did not report {name}; available keys: {sorted(stats)}")
    return float(values[-1][0])


def model_paths_for_case(args, scenario: int) -> tuple[list[Path], list[int | str]]:
    if args.seed_strategy == "fixed":
        return args.models, [""] * args.num_cars
    rng = random.Random(args.assignment_seed + scenario * 1009)
    anchor = args.assignment_anchors[scenario]
    if args.seed_strategy == "same":
        indices = [anchor] * args.num_cars
    else:
        others = [index for index in range(5) if index != anchor]
        rng.shuffle(others)
        if args.num_cars == 2:
            indices = [anchor, args.assignment_partners[scenario]]
        elif args.num_cars == 5:
            indices = [anchor] + others
        elif args.num_cars == 10:
            remaining = [anchor] + [index for index in others for _ in range(2)]
            rng.shuffle(remaining)
            indices = [anchor] + remaining
        else:
            raise ValueError("mixed strategy supports 2, 5, or 10 cars")
    return [args.seed_pool[index] for index in indices], indices


def run_case(args, scenario: int, rollout: int,
             session_cache: dict[Path, ort.InferenceSession],
             worker_id: int) -> tuple[dict, list[dict]]:
    np.random.seed(args.policy_seed + scenario + 1000 * rollout)
    model_paths, seed_indices = model_paths_for_case(args, scenario)
    sessions = [session_cache[path] for path in model_paths]
    scenario_data = args.manifest_data["scenarios"][scenario] if args.manifest_data else None
    params = EnvironmentParametersChannel()
    for name, value in args.environment_parameters.items():
        params.set_float_parameter(name, value)
    params.set_float_parameter("traffic_scenario_index", float(scenario))
    params.set_float_parameter("traffic_placement_seed",
                               float(placement_seed_for_case(args, scenario, rollout)))
    params.set_float_parameter("bootstrap_curriculum_enabled", 0.0)
    params.set_float_parameter("voronoi_near_goal_enabled", 0.0)
    if args.trajectories_out is not None:
        params.set_float_parameter("traffic_record_trajectories", 1.0)
        params.set_float_parameter("traffic_trajectory_stride_steps",
                                   float(args.trajectory_stride_steps))
    engine = EngineConfigurationChannel()
    stats = StatsSideChannel()
    env = UnityEnvironment(
        file_name=str(args.build), worker_id=worker_id,
        base_port=args.base_port,
        seed=args.unity_seed + scenario + 1000 * rollout, no_graphics=args.no_graphics,
        additional_args=["-job-worker-count", str(args.unity_job_worker_count),
                         "-traffic-car-count", str(args.num_cars)],
        side_channels=[params, engine, stats],
    )
    try:
        engine.set_configuration_parameters(time_scale=args.time_scale, target_frame_rate=-1)
        env.reset()
        behavior_by_seat = {}
        for behavior in env.behavior_specs:
            for seat in range(args.num_cars):
                if behavior.split("?")[0] == f"TrafficCar{seat}":
                    behavior_by_seat[seat] = behavior
        if len(behavior_by_seat) != args.num_cars or len(env.behavior_specs) != args.num_cars:
            raise RuntimeError(f"Expected exactly {args.num_cars} TrafficCar behaviors, "
                               f"found {list(env.behavior_specs)}. Rebuild the scene with "
                               "a large enough car pool, and check -traffic-car-count.")

        outputs = {}
        for seat in range(args.num_cars):
            spec = env.behavior_specs[behavior_by_seat[seat]]
            if sum(int(np.prod(obs.shape)) for obs in spec.observation_specs) != args.expected_observations:
                raise RuntimeError(f"TrafficCar{seat} does not have "
                                   f"{args.expected_observations} laser observations")
            cont_size = spec.action_spec.continuous_size
            disc_size = sum(spec.action_spec.discrete_branches)
            outputs[seat] = (cont_size, disc_size,
                             *resolve_action_outputs(sessions[seat], cont_size, disc_size, args.mode))

        rewards = [0.0] * args.num_cars
        ended = [False] * args.num_cars
        steps = 0
        while not all(ended) and steps <= args.max_engine_steps:
            for seat in range(args.num_cars):
                behavior = behavior_by_seat[seat]
                decision, terminal = env.get_steps(behavior)
                rewards[seat] += float(np.sum(decision.reward)) + float(np.sum(terminal.reward))
                if len(terminal) > 0:
                    ended[seat] = True
                if len(decision) == 0 or ended[seat]:
                    continue
                cont_size, disc_size, cont_name, disc_name = outputs[seat]
                session = sessions[seat]
                feed = build_feed(session, decision, cont_size, disc_size, args.mode)
                values = dict(zip([item.name for item in session.get_outputs()], session.run(None, feed)))
                continuous = values[cont_name].astype(np.float32) if cont_name else np.zeros((len(decision), 0), np.float32)
                discrete = values[disc_name].astype(np.int32) if disc_name else np.zeros((len(decision), 0), np.int32)
                env.set_actions(behavior, ActionTuple(continuous=continuous, discrete=discrete))
            if not all(ended):
                env.step()
                steps += 1
        if not all(ended):
            raise RuntimeError(f"Scenario {scenario} did not complete by {args.max_engine_steps} steps")

        recorded = stats.get_and_reset_stats()
        reported_count = recorded.get("Custom/TrafficActiveCars")
        reported_map = recorded.get("Custom/TrafficMapIndex")
        if args.require_map_metrics and (not reported_count or not reported_map):
            raise RuntimeError("Traffic build lacks car-count/map-index measurements; rebuild it")
        random_stat = recorded.get("Custom/TrafficRandomPlacement")
        if args.require_random_placement and (
            not random_stat or round(latest_stat(recorded, "Custom/TrafficRandomPlacement")) != 1
        ):
            raise RuntimeError("Traffic build did not report runtime-randomized placement")
        if reported_count and round(latest_stat(recorded, "Custom/TrafficActiveCars")) != args.num_cars:
            raise RuntimeError(f"Traffic build reported the wrong active car count in case {scenario}")
        map_index = round(latest_stat(recorded, "Custom/TrafficMapIndex")) if reported_map else ""
        if scenario_data and reported_map and map_index != scenario_data["map_index"]:
            raise RuntimeError(f"Case {scenario} loaded map index {map_index}; "
                               f"manifest expected {scenario_data['map_index']}")
        row = {"scenario": scenario, "rollout": rollout, "map_index": map_index,
               "map_file": scenario_data["map_file"] if scenario_data else "",
               "num_cars": args.num_cars, "mode": args.mode,
               "seed_strategy": args.seed_strategy,
               "assignment_seed": args.assignment_seed if args.seed_strategy != "fixed" else "",
               "map_schedule_seed": args.map_schedule_seed if args.episodes is not None else "",
               "random_placement": round(latest_stat(recorded, "Custom/TrafficRandomPlacement"))
               if random_stat else "",
               "placement_seed": round(latest_stat(recorded, "Custom/TrafficPlacementSeed"))
               if recorded.get("Custom/TrafficPlacementSeed") else "",
               "engine_steps": steps,
               "shared_steps": latest_stat(recorded, "Custom/TrafficSharedSteps"),
               "min_car_center_distance_m": latest_stat(recorded, "Custom/TrafficMinCenterDistanceM"),
               "near_10m_steps": latest_stat(recorded, "Custom/TrafficNearEncounterSteps")}
        if args.require_random_placement:
            expected_seed = placement_seed_for_case(args, scenario, rollout)
            if row["placement_seed"] != expected_seed:
                raise RuntimeError(f"Placement seed was {row['placement_seed']}; "
                                   f"expected {expected_seed}. Check the traffic build/side channel.")
        for seat in range(args.num_cars):
            row[f"model{seat}"] = model_run_name(model_paths[seat])
            row[f"model_seed{seat}"] = seed_indices[seat]
            row[f"reward{seat}"] = rewards[seat]
            for coordinate in ("SpawnX", "SpawnZ", "GoalX", "GoalZ", "HeadingDeg"):
                key = f"Custom/TrafficSeat{seat}{coordinate}"
                if args.require_random_placement and not recorded.get(key):
                    raise RuntimeError(f"Traffic build did not report {key}")
                row[f"{coordinate.lower()}{seat}"] = latest_stat(recorded, key) if recorded.get(key) else ""
            for point in ("Spawn", "Goal"):
                key = f"Custom/TrafficSeat{seat}{point}TileCode"
                if args.require_random_placement and not recorded.get(key):
                    raise RuntimeError(f"Traffic build did not report {key}")
                tile_code = round(latest_stat(recorded, key)) if recorded.get(key) else None
                if args.require_random_placement and tile_code != 3:  # TileType.Asphalt
                    raise RuntimeError(f"{point} for seat {seat} was not on Asphalt: {tile_code}")
                row[f"{point.lower()}_tile{seat}"] = TILE_TYPES[tile_code][0] if tile_code is not None else ""
        for seat in range(args.num_cars):
            found = [outcome for outcome in OUTCOMES
                     if recorded.get(f"Custom/TrafficSeat{seat}{outcome}")]
            if len(found) != 1:
                raise RuntimeError(f"Expected one outcome for seat {seat}, found {found}; "
                                   f"available stats: {sorted(recorded)}")
            row[f"outcome{seat}"] = found[0]
            row[f"finish_step{seat}"] = latest_stat(recorded, f"Custom/TrafficSeat{seat}FinishStep")
            finish_tile_key = f"Custom/TrafficSeat{seat}FinishTileCode"
            if recorded.get(finish_tile_key):
                tile_code = round(latest_stat(recorded, finish_tile_key))
                if not 0 <= tile_code < len(TILE_TYPES):
                    raise RuntimeError(f"Invalid finish tile code {tile_code} for seat {seat}")
                row[f"finish_tile{seat}"] = TILE_TYPES[tile_code][0]
            elif args.require_tile_metrics:
                raise RuntimeError(f"Traffic build is missing {finish_tile_key}; rebuild it")
            else:
                row[f"finish_tile{seat}"] = ""
            for tile_name, column_name in TILE_TYPES:
                key = f"Custom/TrafficSeat{seat}TileSeconds{tile_name}"
                if recorded.get(key):
                    row[f"tile_seconds{seat}_{column_name}"] = latest_stat(recorded, key)
                elif args.require_tile_metrics:
                    raise RuntimeError(f"Traffic build is missing {key}; rebuild it")
                else:
                    row[f"tile_seconds{seat}_{column_name}"] = ""
        if args.require_random_placement:
            endpoints = [(row[f"spawnx{seat}"], row[f"spawnz{seat}"])
                         for seat in range(args.num_cars)] + [
                (row[f"goalx{seat}"], row[f"goalz{seat}"])
                for seat in range(args.num_cars)]
            row["min_endpoint_distance_m"] = min(
                math.dist(a, b) for i, a in enumerate(endpoints) for b in endpoints[i + 1:])
            row["min_own_route_distance_m"] = min(
                math.dist(endpoints[seat], endpoints[seat + args.num_cars])
                for seat in range(args.num_cars))
        traces = []
        if args.trajectories_out is not None:
            step_key = "Custom/TrafficTraceStep"
            steps = [round(value) for value, _ in recorded.get(step_key, [])]
            if not steps:
                raise RuntimeError("Traffic build did not report trajectory steps; rebuild it")
            # ML-Agents may reset the next episode in the same env.step() as this
            # episode's terminal decisions. Ignore any following step-zero samples.
            stop = next((i for i in range(1, len(steps)) if steps[i] < steps[i - 1]),
                        len(steps))
            if steps[stop - 1] != round(row["shared_steps"]):
                raise RuntimeError(f"Trajectory did not reach step {row['shared_steps']} "
                                   f"in scenario {scenario}: {steps[:2]}..{steps[stop - 2:stop]}")
            for seat in range(args.num_cars):
                xs = [float(value) for value, _ in
                      recorded.get(f"Custom/TrafficSeat{seat}TraceX", [])]
                zs = [float(value) for value, _ in
                      recorded.get(f"Custom/TrafficSeat{seat}TraceZ", [])]
                if len(xs) != len(steps) or len(zs) != len(steps):
                    raise RuntimeError(f"Trajectory sample mismatch for seat {seat}: "
                                       f"steps={len(steps)}, x={len(xs)}, z={len(zs)}")
                points = list(zip(steps[:stop], xs[:stop], zs[:stop]))
                if steps[0] > 0:  # Some Unity versions flush the first stat only after env.step().
                    points.insert(0, (0, float(row[f"spawnx{seat}"]),
                                      float(row[f"spawnz{seat}"])))
                for sample, (step, x, z) in enumerate(points):
                    traces.append({
                        "scenario": scenario, "rollout": rollout,
                        "map_file": row["map_file"], "mode": args.mode,
                        "num_cars": args.num_cars, "seat": seat,
                        "sample": sample, "shared_step": step, "x": x, "z": z,
                        "spawn_x": row[f"spawnx{seat}"],
                        "spawn_z": row[f"spawnz{seat}"],
                        "goal_x": row[f"goalx{seat}"],
                        "goal_z": row[f"goalz{seat}"],
                        "finish_step": row[f"finish_step{seat}"],
                        "outcome": row[f"outcome{seat}"],
                    })
        return row, traces
    finally:
        env.close()


def run_case_with_port_retry(args, scenario: int, rollout: int,
                             session_cache: dict[Path, ort.InferenceSession]) -> tuple[dict, list[dict]]:
    """Try another port in the reserved block if Unity cannot bind this one."""
    for attempt in range(PORT_ATTEMPTS):
        worker_id = args.worker_id_base + (scenario + rollout + 10 * attempt) % PORT_BLOCK_SIZE
        try:
            return run_case(args, scenario, rollout, session_cache, worker_id)
        except UnityWorkerInUseException:
            port = args.base_port + worker_id
            if attempt + 1 == PORT_ATTEMPTS:
                print(f"Scenario {scenario}: all {PORT_ATTEMPTS} ports tried; "
                      f"last busy port was {port}", file=sys.stderr, flush=True)
                raise
            print(f"Scenario {scenario}: port {port} is busy; trying another "
                  "port in the reserved block", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="Path to the new traffic Unity executable")
    parser.add_argument("--model0", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model1", type=Path, help="Another laser77 seed; default uses model0")
    parser.add_argument("--models", nargs="+", type=Path,
                        help="One model shared by all cars, or exactly one model per car")
    parser.add_argument("--seed-pool", nargs=5, type=Path,
                        help="Five trained seed ONNX files, used by same/mixed seed strategies")
    parser.add_argument("--seed-strategy", choices=("fixed", "same", "mixed"), default="fixed",
                        help="Fixed --models, one random seed shared by all cars per map, "
                             "or different/balanced seeds per map")
    parser.add_argument("--assignment-seed", type=int, default=7319,
                        help="Seed for assigning trained model seeds to maps")
    parser.add_argument("--num-cars", type=int, default=2,
                        help="Active cars (2..pool size embedded in the Unity build)")
    parser.add_argument("--expected-observations", type=int, choices=(77, 89), default=77,
                        help="Laser observation size: legacy car-as-Terminal=77, separate vehicle distance=89")
    parser.add_argument("--config", type=Path, help="Saved YAML; defaults to model0's run directory")
    parser.add_argument("--manifest", type=Path,
                        help="Scenario JSON; if omitted, use the original ten-case pilot")
    parser.add_argument("--scenario", type=int, help="Run just this scenario index")
    parser.add_argument("--scenario-indices", nargs="+", type=int,
                        help="Run this ordered subset of scenario indices (e.g. ten trajectory maps)")
    parser.add_argument("--scenario-start", type=int, default=0,
                        help="First scenario index in a contiguous shard (inclusive)")
    parser.add_argument("--scenario-stop", type=int,
                        help="Last scenario index in a contiguous shard (exclusive)")
    parser.add_argument("--rollouts-per-map", type=int, default=1,
                        help="Independent randomized placements per map (default: 1)")
    parser.add_argument("--episodes", type=int,
                        help="Total episodes across the selected maps; rotates maps evenly "
                             "(e.g. 25 episodes across a ten-map shard)")
    parser.add_argument("--map-schedule-seed", type=int, default=4199,
                        help="Shuffle map order when using --episodes, identically across conditions")
    parser.add_argument("--placement-seed", type=int, default=3401,
                        help="Base placement seed; same seed gives matching routes in both action modes")
    parser.add_argument("--fixed-placement-across-rollouts", action="store_true",
                        help="Keep each map's sampled routes fixed across stochastic rollouts")
    parser.add_argument("--mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--unity-seed", type=int, default=3401)
    parser.add_argument("--policy-seed", type=int, default=3401)
    parser.add_argument("--base-port", type=int, default=5005)
    parser.add_argument("--worker-id-base", type=int, default=100)
    parser.add_argument("--unity-job-worker-count", type=int, default=1)
    parser.add_argument("--max-engine-steps", type=int, default=5020)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--require-tile-metrics", action="store_true",
                        help="Fail if the traffic build lacks per-seat tile-time measurements")
    parser.add_argument("--require-map-metrics", action="store_true",
                        help="Fail unless the build reports its map index and active car count")
    parser.add_argument("--require-random-placement", action="store_true",
                        help="Fail unless the build samples new connected Asphalt routes")
    parser.add_argument("--out", type=Path, default=ROOT / "hpc/traffic/traffic_eval.csv")
    parser.add_argument("--trajectories-out", type=Path,
                        help="Also save every car's sampled x/z path in a long-format CSV")
    parser.add_argument("--trajectory-stride-steps", type=int, default=5,
                        help="Physics-step spacing between path samples (default: 5)")
    args = parser.parse_args()
    if args.num_cars < 2:
        parser.error("--num-cars must be at least 2; use the original scene for one car")
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.trajectory_stride_steps < 1:
        parser.error("--trajectory-stride-steps must be positive")
    if args.rollouts_per_map < 1 or args.placement_seed < 0 or (
        args.placement_seed + 100 * args.rollouts_per_map * 1000 >= 16000000
    ):
        parser.error("rollouts must be >= 1 and all placement seeds must stay below 16 million")
    if args.seed_strategy != "fixed":
        if args.models or args.model1 or not args.seed_pool or args.num_cars not in (2, 5, 10):
            parser.error("same/mixed needs --seed-pool with five models and 2, 5 or 10 cars; "
                         "do not combine it with --models/--model1")
        if len(set(args.seed_pool)) != 5:
            parser.error("--seed-pool must contain five distinct model paths")
        args.models = []
        args.available_models = args.seed_pool
    else:
        if args.seed_pool or (args.models and args.model1):
            parser.error("fixed strategy uses --models or --model0/--model1, not --seed-pool")
        if args.models:
            args.models = args.models * args.num_cars if len(args.models) == 1 else args.models
            if len(args.models) != args.num_cars:
                parser.error("--models needs one shared model or one model per car")
        elif args.model1:
            if args.num_cars != 2:
                parser.error("--model1 is only for the two-car pilot; use --models for more cars")
            args.models = [args.model0, args.model1]
        else:
            args.models = [args.model0] * args.num_cars
        args.available_models = args.models
    for path in (args.build, *args.available_models):
        if not path.exists():
            parser.error(f"Missing: {path}")
    args.manifest_data = None
    if args.manifest:
        if not args.manifest.exists():
            parser.error(f"Missing scenario manifest: {args.manifest}")
        args.manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
        scenarios = args.manifest_data.get("scenarios", [])
        random_manifest = args.manifest_data.get("placement_mode") == "random_connected_asphalt"
        if not scenarios or any("map_index" not in s or "map_file" not in s for s in scenarios):
            parser.error("scenario manifest has no valid map cases")
        if random_manifest and not args.require_random_placement:
            parser.error("map-only manifest requires --require-random-placement")
        if not random_manifest and any(len(s.get("routes", [])) < args.num_cars for s in scenarios):
            parser.error("fixed-route manifest has too few routes for --num-cars")
        scenario_count = len(scenarios)
    else:
        if args.num_cars != 2:
            parser.error("--manifest is required for more than two cars")
        scenario_count = 10
    # Two maps per seed in each ten-map shard. This makes both the 100-map set
    # and the 25-episode/shard experiment balanced without fixing a seed to a map.
    args.assignment_anchors = []
    for block_start in range(0, scenario_count, 10):
        labels = [seed for seed in range(5) for _ in range(2)]
        random.Random(args.assignment_seed + block_start).shuffle(labels)
        args.assignment_anchors.extend(labels[:min(10, scenario_count - block_start)])
    args.assignment_partners = [-1] * scenario_count
    for anchor in range(5):
        maps = [index for index, label in enumerate(args.assignment_anchors) if label == anchor]
        random.Random(args.assignment_seed + 10000 + anchor).shuffle(maps)
        partners = [seed for seed in range(5) if seed != anchor]
        for i, scenario in enumerate(maps):
            args.assignment_partners[scenario] = partners[i % len(partners)]
    if args.scenario_indices is not None:
        if (args.scenario is not None or args.scenario_start != 0 or
                args.scenario_stop is not None or args.episodes is not None):
            parser.error("--scenario-indices cannot be combined with scenario ranges or --episodes")
        if len(set(args.scenario_indices)) != len(args.scenario_indices) or any(
                not 0 <= index < scenario_count for index in args.scenario_indices):
            parser.error("--scenario-indices must be unique indices in the manifest")
        selected_scenarios = args.scenario_indices
    elif args.scenario is not None:
        if not 0 <= args.scenario < scenario_count:
            parser.error(f"--scenario must be 0..{scenario_count - 1}")
        if args.scenario_start != 0 or args.scenario_stop is not None:
            parser.error("--scenario cannot be combined with --scenario-start/--scenario-stop")
        selected_scenarios = [args.scenario]
    else:
        stop = args.scenario_stop if args.scenario_stop is not None else scenario_count
        if not 0 <= args.scenario_start < stop <= scenario_count:
            parser.error(f"scenario shard must satisfy 0 <= start < stop <= {scenario_count}")
        selected_scenarios = range(args.scenario_start, stop)
    config_path = args.config or args.available_models[0].parent / "configuration.yaml"
    if not config_path.exists():
        parser.error(f"Missing training configuration: {config_path}")
    args.environment_parameters = environment_parameters_from_config(config_path)
    for model in set(args.available_models[1:]):
        if model == args.available_models[0]:
            continue
        other_config = model.parent / "configuration.yaml"
        if not other_config.exists():
            parser.error(f"Missing model training configuration: {other_config}")
        other_parameters = environment_parameters_from_config(other_config)
        comparable = ("vehicle_speed_cap_enabled", "vehicle_speed_cap_ms", "regen_brake_torque",
                      "laser_special_distance_scale_m", "laser_road_boundary_distance_scale_m",
                      "laser_vehicle_distance_observation_enabled", "laser_vehicle_distance_scale_m")
        differences = [key for key in comparable
                       if args.environment_parameters.get(key) != other_parameters.get(key)]
        if differences:
            parser.error(f"Models were trained with different dynamics/laser scales: {differences}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_cache = {path: ort.InferenceSession(str(path), sess_options=session_options,
                                               providers=["CPUExecutionProvider"])
                     for path in set(args.available_models)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.episodes is None:
        schedule = [(scenario, rollout) for scenario in selected_scenarios
                    for rollout in range(args.rollouts_per_map)]
    else:
        map_order = list(selected_scenarios)
        random.Random(args.map_schedule_seed + map_order[0]).shuffle(map_order)
        if args.seed_strategy != "fixed" and args.episodes == 25 and len(map_order) == 10:
            extras = []
            for seed in range(5):
                candidates = [scenario for scenario in map_order
                              if args.assignment_anchors[scenario] == seed]
                extras.append(random.Random(args.map_schedule_seed + map_order[0] + seed)
                              .choice(candidates))
            schedule = ([(scenario, 0) for scenario in map_order] +
                        [(scenario, 1) for scenario in map_order] +
                        [(scenario, 2) for scenario in extras])
        else:
            schedule = [(map_order[i % len(map_order)], i // len(map_order))
                        for i in range(args.episodes)]
    if args.trajectories_out is not None:
        args.trajectories_out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_handle = (args.trajectories_out.open("w", newline="", encoding="utf-8")
                         if args.trajectories_out is not None else None)
    try:
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = None
            trajectory_writer = None
            for episode_index, (scenario, rollout) in enumerate(schedule):
                case_row, traces = run_case_with_port_retry(args, scenario, rollout, session_cache)
                row = {"episode_index": episode_index, **case_row}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
                handle.flush()  # Preserve completed cases if a later Unity process fails.
                if trajectory_handle is not None:
                    if not traces:
                        raise RuntimeError(f"No trajectory samples for scenario {scenario}")
                    if trajectory_writer is None:
                        trajectory_writer = csv.DictWriter(
                            trajectory_handle, fieldnames=["episode_index", *traces[0]])
                        trajectory_writer.writeheader()
                    for trace in traces:
                        trajectory_writer.writerow({"episode_index": episode_index, **trace})
                    trajectory_handle.flush()
                outcomes = "/".join(row[f"outcome{seat}"] for seat in range(args.num_cars))
                print(f"case {row['scenario']:02d} rollout {rollout}: {outcomes} "
                       f"at step {row['shared_steps']:.0f}; "
                       f"minimum separation {row['min_car_center_distance_m']:.2f} m", flush=True)
    finally:
        if trajectory_handle is not None:
            trajectory_handle.close()
    summary_path = write_summary(rows, args.out)
    print(f"Wrote {args.out} and {summary_path}")


if __name__ == "__main__":
    main()
