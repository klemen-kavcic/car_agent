"""Make a matched multi-car PPO config from a completed laser77 seed run.

The saved configuration is the source of truth for both scratch and fine-tune.
Only initialization and the new sample budget differ between these two arms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def constant_value(name: str, setting: dict | int | float) -> float:
    if isinstance(setting, (int, float)):
        return float(setting)
    stages = setting.get("curriculum", [])
    if len(stages) != 1 or stages[0]["value"]["sampler_type"] != "constant":
        raise ValueError(f"{name} is not a single constant; inspect the source config")
    return stages[0]["value"]["sampler_parameters"]["value"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path,
                        help="Original .pt for fine-tuning; omit for scratch")
    parser.add_argument("--vehicle-distance", action="store_true",
                        help="Enable the separate 12-value vehicle-distance channel (89 observations)")
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        parser.error("--steps and --num-envs must be positive")
    if not args.source.is_file():
        parser.error(f"missing source configuration: {args.source}")
    if args.checkpoint is not None and (
        not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0
    ):
        parser.error(f"missing or empty .pt checkpoint: {args.checkpoint}")

    source = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    behaviors = source["behaviors"]
    if set(behaviors) != {"CarAgent"} or behaviors["CarAgent"]["trainer_type"] != "ppo":
        parser.error("expected one PPO behavior named CarAgent")
    trainer = behaviors["CarAgent"]
    network = trainer["network_settings"]
    if (network["hidden_units"], network["num_layers"], network["normalize"]) != (128, 2, True):
        parser.error("source policy architecture differs from the laser77 experiment")
    environment_parameters = source["environment_parameters"]
    params = {name: constant_value(name, setting)
              for name, setting in environment_parameters.items()}
    expected = {
        "vehicle_speed_cap_enabled": 1,
        "vehicle_speed_cap_ms": 3.0,
        "regen_brake_torque": 450.0,
        "laser_special_distance_scale_m": 10.0,
        "laser_road_boundary_distance_scale_m": 3.0,
        "sensor_observations_enabled": 1,
    }
    if args.vehicle_distance:
        expected["laser_vehicle_distance_observation_enabled"] = 1
        expected["laser_vehicle_distance_scale_m"] = 10.0
        # Older 77-observation source configs predate these two parameters.
        params.setdefault("laser_vehicle_distance_observation_enabled", 1.0)
        params.setdefault("laser_vehicle_distance_scale_m", 10.0)
    for name, value in expected.items():
        if params.get(name) != value:
            parser.error(f"{name}={params.get(name)}; expected {value}")
    trainer["max_steps"] = args.steps
    # Full runs save every million and retain every checkpoint, including an
    # occasional extra final export. Short smoke runs still save at completion.
    trainer["checkpoint_interval"] = min(1_000_000, args.steps)
    trainer["keep_checkpoints"] = max(3, args.steps // 1_000_000 + 2)
    trainer["init_path"] = str(args.checkpoint.resolve()) if args.checkpoint else None
    def set_constant(name: str, value: float) -> None:
        if name not in environment_parameters:
            environment_parameters[name] = value
        elif isinstance(environment_parameters[name], dict):
            environment_parameters[name]["curriculum"][0]["value"]["sampler_parameters"]["value"] = value
        else:
            environment_parameters[name] = value

    set_constant("total_training_steps", args.steps)
    set_constant("training_num_envs", args.num_envs)
    if args.vehicle_distance:
        set_constant("laser_vehicle_distance_observation_enabled", 1)
        set_constant("laser_vehicle_distance_scale_m", 10.0)
    source.setdefault("env_settings", {})["num_envs"] = args.num_envs
    source.setdefault("checkpoint_settings", {})["run_id"] = args.out.stem
    # Keep all saved PPO/reward/trainer settings, but do not leave the old run
    # ID/build path in the generated YAML. The launch still sets these on CLI.
    source["env_settings"]["env_path"] = str(
        args.out.parent.parent / "Linux_traffic_training" / "car.x86_64")
    source["checkpoint_settings"]["run_id"] = args.out.stem
    config = source
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.out}; init={trainer['init_path'] or 'scratch'}, "
          f"steps={args.steps}, environments={args.num_envs}")


if __name__ == "__main__":
    main()
