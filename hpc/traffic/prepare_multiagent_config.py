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
    parser.add_argument("--hidden-units", type=int, choices=(128, 256), default=128,
                        help="PPO hidden-layer width; preserve 128 unless deliberately testing capacity")
    parser.add_argument("--batch-size", type=int,
                        help="Optional PPO minibatch size; otherwise retain the source configuration")
    parser.add_argument("--buffer-size", type=int,
                        help="Optional PPO rollout buffer size; otherwise retain the source configuration")
    parser.add_argument("--learning-rate", type=float,
                        help="Optional PPO learning rate; otherwise retain the source configuration")
    parser.add_argument("--checkpoint", type=Path,
                        help="Original .pt for fine-tuning; omit for scratch")
    parser.add_argument("--vehicle-distance", action="store_true",
                        help="Enable the separate 12-value vehicle-distance channel (89 observations)")
    parser.add_argument("--arcade-reverse-controls", action="store_true",
                        help="Use the matching one-branch arcade build: brake/reverse and accelerator/forward")
    parser.add_argument("--signed-pedal-arcade-controls", action="store_true",
                        help="Use the matching zero-branch build: one signed pedal, -1 reverse through +1 forward")
    parser.add_argument("--idle-penalty", action="store_true",
                        help="Enable escalating low-speed penalty with the supplied idle parameters")
    parser.add_argument("--idle-speed-threshold", type=float, default=0.15,
                        help="Speed below which the optional idle penalty accumulates (m/s)")
    parser.add_argument("--idle-grace-steps", type=int, default=20,
                        help="Decisions below the idle threshold before the optional penalty begins")
    parser.add_argument("--idle-penalty-base", type=float, default=-0.02,
                        help="First optional idle penalty after grace; must be negative")
    parser.add_argument("--idle-penalty-growth", type=float, default=0.05,
                        help="Linear per-decision growth of the optional idle penalty")
    parser.add_argument("--exponential-checkpoint-rewards", action="store_true",
                        help="Redistribute a fixed total checkpoint budget exponentially toward later route progress")
    parser.add_argument("--checkpoint-exponential-total-reward", type=float, default=2.0,
                        help="Total reward across all 100 checkpoints in exponential mode")
    parser.add_argument("--checkpoint-exponential-beta", type=float, default=0.04,
                        help="Non-negative exponential steepness for checkpoint rewards")
    parser.add_argument("--remove-goal-cars", action="store_true",
                        help="Legacy alias: remove finished traffic cars from physics/laser hits until shared reset")
    parser.add_argument("--remove-finished-cars", action="store_true",
                        help="Remove every finished traffic car, regardless of outcome, until shared reset")
    parser.add_argument("--crash-lagrangian", action="store_true",
                        help="Adapt the collision/terminal penalty to a target crash rate after bootstrap")
    parser.add_argument("--crash-lagrangian-target", type=float, default=0.08,
                        help="Target VehicleCollision rate when --crash-lagrangian is enabled")
    parser.add_argument("--crash-lagrangian-init", type=float, default=10.0,
                        help="Initial collision-penalty magnitude; 10 preserves the current -10 penalty")
    parser.add_argument("--crash-lagrangian-lr", type=float, default=1.0,
                        help="Dual-ascent update rate for the crash constraint")
    parser.add_argument("--crash-lagrangian-max", type=float, default=20.0,
                        help="Maximum collision-penalty magnitude for the crash constraint")
    parser.add_argument("--constrained-ppo", action="store_true",
                        help="Emit the terminal collision-cost signal consumed by the patched constrained PPO trainer")
    parser.add_argument("--speed-limited-speed-fraction", type=float,
                        help="Optional SpeedLimited-tile target-speed fraction of the effective vehicle cap")
    parser.add_argument("--signed-pedal-reverse-torque-fraction", type=float,
                        help="Optional negative-pedal motor-torque fraction (signed-pedal mode; not a hard speed cap)")
    parser.add_argument("--vehicle-speed-cap-ms", type=float,
                        help="Optional override for the vehicle top-speed cap in m/s")
    parser.add_argument("--vehicle-ttc-risk-penalty", type=float, default=0.0,
                        help="Optional negative dense reward at unsafe forward vehicle time-to-collision")
    parser.add_argument("--vehicle-ttc-risk-threshold-s", type=float, default=1.5,
                        help="TTC horizon in seconds for the optional vehicle-risk penalty")
    parser.add_argument("--vehicle-ttc-risk-min-closing-speed-ms", type=float, default=0.25,
                        help="Ignore TTC shaping below this closing speed in m/s")
    parser.add_argument("--gravel-speed-fraction", type=float,
                        help="Optional Gravel-tile target-speed fraction of the effective vehicle cap")
    parser.add_argument("--gravel-friction-multiplier", type=float,
                        help="Optional Gravel wheel-friction multiplier")
    parser.add_argument("--smooth-car-count-curriculum", action="store_true",
                        help="Enable the per-episode 2--10-car moving Gaussian curriculum")
    parser.add_argument("--smooth-route-length-curriculum", action="store_true",
                        help="Enable the 10-car short-to-long route-distance curriculum")
    parser.add_argument("--recovery-curriculum", action="store_true",
                        help="Enable the staged TrafficRecovery curriculum before normal traffic maps")
    parser.add_argument("--recovery-curriculum-end-steps", type=int, default=13_000_000,
                        help="Trainer-step boundary at which recovery lessons return to ordinary traffic maps")
    parser.add_argument("--recovery-blocker-clearance-reward", type=float, default=0.0,
                        help="Optional one-off reward for backing away from a parked blocker; zero disables it")
    parser.add_argument("--summary-freq", type=int,
                        help="Override PPO/TensorBoard summary interval in trainer steps")
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        parser.error("--steps and --num-envs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.buffer_size is not None and args.buffer_size <= 0:
        parser.error("--buffer-size must be positive")
    if args.batch_size is not None and args.buffer_size is not None:
        if args.batch_size > args.buffer_size or args.buffer_size % args.batch_size != 0:
            parser.error("--buffer-size must be a multiple of --batch-size and no smaller than it")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.crash_lagrangian_target <= 1:
        parser.error("--crash-lagrangian-target must be within [0, 1]")
    if args.crash_lagrangian_init < 0 or args.crash_lagrangian_lr < 0 or args.crash_lagrangian_max < 0:
        parser.error("crash Lagrangian coefficients must be non-negative")
    if args.recovery_curriculum_end_steps <= 0:
        parser.error("--recovery-curriculum-end-steps must be positive")
    if args.recovery_blocker_clearance_reward < 0:
        parser.error("--recovery-blocker-clearance-reward must be non-negative")
    if args.arcade_reverse_controls and args.signed_pedal_arcade_controls:
        parser.error("choose either --arcade-reverse-controls or --signed-pedal-arcade-controls")
    if args.idle_speed_threshold < 0 or args.idle_grace_steps < 0:
        parser.error("idle speed threshold and grace steps must be non-negative")
    if args.idle_penalty_base >= 0 or args.idle_penalty_growth < 0:
        parser.error("idle penalty base must be negative and growth non-negative")
    if args.checkpoint_exponential_total_reward < 0 or args.checkpoint_exponential_beta < 0:
        parser.error("exponential checkpoint total reward and beta must be non-negative")
    for option, value in (
        ("speed-limited speed fraction", args.speed_limited_speed_fraction),
        ("signed-pedal reverse torque fraction", args.signed_pedal_reverse_torque_fraction),
        ("gravel speed fraction", args.gravel_speed_fraction),
        ("gravel friction multiplier", args.gravel_friction_multiplier),
    ):
        if value is not None and not 0 <= value <= 1:
            parser.error(f"{option} must be within [0, 1]")
    if args.summary_freq is not None and args.summary_freq <= 0:
        parser.error("--summary-freq must be positive")
    if args.vehicle_speed_cap_ms is not None and args.vehicle_speed_cap_ms <= 0:
        parser.error("--vehicle-speed-cap-ms must be positive")
    if args.vehicle_ttc_risk_penalty > 0:
        parser.error("--vehicle-ttc-risk-penalty must be zero or negative")
    if args.vehicle_ttc_risk_threshold_s <= 0 or args.vehicle_ttc_risk_min_closing_speed_ms < 0:
        parser.error("TTC threshold must be positive and minimum closing speed non-negative")
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
    hyperparameters = trainer["hyperparameters"]
    if args.batch_size is not None:
        hyperparameters["batch_size"] = args.batch_size
    if args.buffer_size is not None:
        hyperparameters["buffer_size"] = args.buffer_size
    if args.learning_rate is not None:
        hyperparameters["learning_rate"] = args.learning_rate
    # Keep the source architecture as the validated reference, then apply the
    # deliberate width-only ablation requested for a new output config.
    network["hidden_units"] = args.hidden_units
    if args.summary_freq is not None:
        trainer["summary_freq"] = args.summary_freq
    # Full runs save every million and retain every checkpoint, including an
    # occasional extra final export. Short smoke runs still save at completion.
    trainer["checkpoint_interval"] = min(1_000_000, args.steps)
    trainer["keep_checkpoints"] = max(3, args.steps // 1_000_000 + 2)
    trainer["init_path"] = str(args.checkpoint.resolve()) if args.checkpoint else None

    def constant_sampler(name: str, value: float) -> dict:
        """Return the only environment-parameter representation ML-Agents accepts.

        Some older hand-authored templates contain bare YAML scalars.  They are
        convenient to read but ML-Agents 1.1 cannot structure them as sampler
        settings (and aborts with ``Unsupported type: float``).  Always emit
        the canonical sampler/curriculum shape instead.
        """
        return {
            "curriculum": [{
                "value": {
                    "sampler_type": "constant",
                    "sampler_parameters": {"seed": 0, "value": value},
                },
                "name": name,
                "completion_criteria": None,
            }]
        }

    # Canonicalise *all* raw source parameters before any option changes a
    # value. This also repairs templates that were previously generated while
    # a parameter was accidentally written as a bare float.
    for parameter_name, parameter_setting in list(environment_parameters.items()):
        if isinstance(parameter_setting, (int, float)):
            environment_parameters[parameter_name] = constant_sampler(
                parameter_name, float(parameter_setting)
            )

    def set_constant(name: str, value: float) -> None:
        if name not in environment_parameters:
            # ML-Agents 1.1 parses environment parameters as sampler
            # definitions, not raw YAML scalars. New parameters must therefore
            # use the same one-item constant curriculum shape as the template's
            # existing parameters. A raw `foo: 1` reaches cattrs as a float and
            # fails before Unity starts ("Unsupported type: float").
            environment_parameters[name] = constant_sampler(name, value)
        elif isinstance(environment_parameters[name], dict):
            environment_parameters[name]["curriculum"][0]["value"]["sampler_parameters"]["value"] = value
        else:  # Defensive fallback for an unusual source parameter type.
            environment_parameters[name] = constant_sampler(name, value)

    set_constant("total_training_steps", args.steps)
    set_constant("training_num_envs", args.num_envs)
    if args.vehicle_distance:
        set_constant("laser_vehicle_distance_observation_enabled", 1)
        set_constant("laser_vehicle_distance_scale_m", 10.0)
    if args.arcade_reverse_controls:
        set_constant("arcade_reverse_controls_enabled", 1)
        set_constant("signed_pedal_arcade_controls_enabled", 0)
    if args.signed_pedal_arcade_controls:
        set_constant("arcade_reverse_controls_enabled", 0)
        set_constant("signed_pedal_arcade_controls_enabled", 1)
    if args.idle_penalty:
        set_constant("idle_penalty_enabled", 1)
        set_constant("idle_speed_threshold", args.idle_speed_threshold)
        set_constant("idle_grace_steps", args.idle_grace_steps)
        set_constant("idle_penalty_base", args.idle_penalty_base)
        set_constant("idle_penalty_growth", args.idle_penalty_growth)
    if args.exponential_checkpoint_rewards:
        set_constant("checkpoint_exponential_rewards_enabled", 1)
        set_constant("checkpoint_exponential_total_reward", args.checkpoint_exponential_total_reward)
        set_constant("checkpoint_exponential_beta", args.checkpoint_exponential_beta)
    if args.remove_goal_cars or args.remove_finished_cars:
        # The legacy option deliberately now has the same all-outcomes
        # behavior, so existing job scripts receive the current traffic rule.
        set_constant("traffic_remove_finished_cars", 1)
    if args.crash_lagrangian:
        set_constant("lagrangian_penalty_enabled", 1)
        set_constant("lagrangian_target_crash_rate", args.crash_lagrangian_target)
        set_constant("lagrangian_lambda_init", args.crash_lagrangian_init)
        set_constant("lagrangian_lambda_lr", args.crash_lagrangian_lr)
        set_constant("lagrangian_lambda_max", args.crash_lagrangian_max)
    if args.constrained_ppo:
        # This does not alter task reward. It only enables carAgent.cs's
        # terminal-only collision sentinel, consumed by ppo_constrained_trainer.
        set_constant("constrained_ppo_cost_signal_enabled", 1)
    if args.speed_limited_speed_fraction is not None:
        set_constant("speed_limited_speed_fraction", args.speed_limited_speed_fraction)
    if args.signed_pedal_reverse_torque_fraction is not None:
        set_constant("signed_pedal_reverse_torque_fraction", args.signed_pedal_reverse_torque_fraction)
    if args.vehicle_speed_cap_ms is not None:
        set_constant("vehicle_speed_cap_enabled", 1)
        set_constant("vehicle_speed_cap_ms", args.vehicle_speed_cap_ms)
    if args.vehicle_ttc_risk_penalty < 0:
        set_constant("vehicle_ttc_risk_penalty_enabled", 1)
        set_constant("reward_vehicle_ttc_risk_penalty", args.vehicle_ttc_risk_penalty)
        set_constant("vehicle_ttc_risk_threshold_s", args.vehicle_ttc_risk_threshold_s)
        set_constant("vehicle_ttc_risk_min_closing_speed_ms", args.vehicle_ttc_risk_min_closing_speed_ms)
    if args.gravel_speed_fraction is not None:
        set_constant("gravel_speed_fraction", args.gravel_speed_fraction)
    if args.gravel_friction_multiplier is not None:
        set_constant("gravel_friction_multiplier", args.gravel_friction_multiplier)
    if args.smooth_car_count_curriculum:
        # Hold a low-density distribution for 5M steps, move its discrete
        # Gaussian centre to 9.5 during 5--35M, then consolidate at high
        # density through 40M. A 9% uniform mixture retains every count.
        set_constant("traffic_smooth_car_count_curriculum_enabled", 1)
        set_constant("traffic_car_count_min", 2)
        set_constant("traffic_car_count_max", 10)
        set_constant("traffic_car_count_centre_start", 2.5)
        set_constant("traffic_car_count_centre_end", 9.5)
        set_constant("traffic_car_count_gaussian_sigma", 1.4)
        set_constant("traffic_car_count_uniform_mix", 0.09)
        set_constant("traffic_curriculum_warmup_steps", 5_000_000)
        set_constant("traffic_curriculum_ramp_end_steps", 35_000_000)
    if args.smooth_route_length_curriculum:
        # This is deliberately separate from the car-count curriculum. The
        # manager starts with an approximately 8--10m band (the normal 8m
        # endpoint spacing remains active), expands its upper bound to 70m by
        # 35M, then uses ordinary unlimited routes for the final 5M.
        set_constant("traffic_smooth_route_length_curriculum_enabled", 1)
        set_constant("traffic_route_distance_min_m", 0.1)
        set_constant("traffic_route_distance_start_m", 10.0)
        set_constant("traffic_route_distance_end_m", 70.0)
        set_constant("traffic_curriculum_warmup_steps", 5_000_000)
        set_constant("traffic_curriculum_ramp_end_steps", 35_000_000)
    if args.recovery_curriculum:
        # TrafficScenarioManager selects the compact one/two/four-car lessons
        # until this boundary, then restores the command-line normal traffic
        # count and the usual VoronoiTrain random placement.
        set_constant("traffic_recovery_curriculum_enabled", 1)
        set_constant("traffic_recovery_curriculum_end_steps", args.recovery_curriculum_end_steps)
        set_constant("traffic_recovery_blocker_clearance_reward", args.recovery_blocker_clearance_reward)
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
