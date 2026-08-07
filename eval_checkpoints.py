"""Deterministic (inference-only, no exploration) evaluation of a sweep of ML-Agents
checkpoints from a single training run.

Connects directly to the built Unity binary via the mlagents_envs low-level API (the
same binary/apptainer setup used for training - no C# changes or rebuild needed), loads
each CarAgent-<step>.onnx checkpoint in turn with onnxruntime, and drives the agent using
the model's *deterministic* action outputs (no sampling noise), so results reflect the
greedy policy at that checkpoint rather than the exploring training-time policy.

Per-episode outcome stats (Custom/GoalReached, Custom/Terminated, Custom/MaxStepReached)
are read back from the same StatsSideChannel mechanism the trainer uses - carAgent.cs
reports them unconditionally on every episode end, regardless of what is connected and
driving the agent's actions, so no extra Unity-side instrumentation is needed for eval.

One CSV row is written per checkpoint:
    run_id, step, episodes, mean_reward, std_reward,
    goal_rate, terminated_rate, maxstep_rate, mean_episode_length

Usage:
    python eval_checkpoints.py \
        --binary /path/to/Linux/car.x86_64 \
        --run-dir /path/to/results/<run_id> \
        --episodes 50 \
        --out /path/to/results/<run_id>/eval_results.csv
"""
import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from mlagents_envs.base_env import ActionTuple, BehaviorSpec, DecisionSteps
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

logging.basicConfig(level=logging.INFO, format="[eval] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"-(\d+)\.onnx$")

# Names ML-Agents 4.x exports for greedy/no-exploration action selection. If your
# onnxruntime session doesn't have these, the script will print the available output
# names it found so you can adjust these two constants.
DETERMINISTIC_CONTINUOUS_OUTPUT = "deterministic_continuous_actions"
DETERMINISTIC_DISCRETE_OUTPUT = "deterministic_discrete_actions"


def discover_checkpoints(run_dir: Path, behavior_name: str) -> List[Path]:
    ckpt_dir = run_dir / behavior_name
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No checkpoint dir at {ckpt_dir}")
    checkpoints = []
    for f in ckpt_dir.glob(f"{behavior_name}-*.onnx"):
        m = CHECKPOINT_RE.search(f.name)
        if m:
            checkpoints.append((int(m.group(1)), f))
    checkpoints.sort(key=lambda t: t[0])
    return [f for _, f in checkpoints]


def step_from_path(p: Path) -> int:
    m = CHECKPOINT_RE.search(p.name)
    assert m, f"Could not parse step from checkpoint filename: {p.name}"
    return int(m.group(1))


def build_feed(session: ort.InferenceSession, decision_steps: DecisionSteps,
                continuous_size: int, discrete_size: int) -> Dict[str, np.ndarray]:
    n_agents = len(decision_steps)
    feed: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        if name.startswith("obs_"):
            idx = int(name.split("_")[1])
            feed[name] = decision_steps.obs[idx].astype(np.float32)
        elif name == "action_masks":
            # All-zero = nothing masked out (matches the unmasked default this project uses).
            feed[name] = np.zeros((n_agents, max(discrete_size, 0)), dtype=np.float32)
        elif name == "epsilon":
            # Fed but irrelevant when reading the deterministic_* outputs (mean action).
            feed[name] = np.zeros((n_agents, max(continuous_size, 0)), dtype=np.float32)
        elif name == "sequence_length":
            feed[name] = np.array(1, dtype=np.int64)
        else:
            log.warning("Unhandled model input '%s' - onnxruntime will raise if it's required.", name)
    return feed


def resolve_action_outputs(session: ort.InferenceSession, continuous_size: int, discrete_size: int):
    output_names = {o.name for o in session.get_outputs()}
    cont_name = DETERMINISTIC_CONTINUOUS_OUTPUT if continuous_size > 0 else None
    disc_name = DETERMINISTIC_DISCRETE_OUTPUT if discrete_size > 0 else None
    missing = [n for n in (cont_name, disc_name) if n and n not in output_names]
    if missing:
        raise RuntimeError(
            f"Expected deterministic output(s) {missing} not found in ONNX model. "
            f"Available outputs: {sorted(output_names)}. "
            "Adjust DETERMINISTIC_CONTINUOUS_OUTPUT/DETERMINISTIC_DISCRETE_OUTPUT at the "
            "top of this script to match your ML-Agents export version."
        )
    return cont_name, disc_name


def run_checkpoint(env: UnityEnvironment, stats_channel: StatsSideChannel, behavior_name: str,
                    spec: BehaviorSpec, onnx_path: Path, n_episodes: int,
                    max_steps: int) -> Optional[dict]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    continuous_size = spec.action_spec.continuous_size
    discrete_branches = spec.action_spec.discrete_branches
    discrete_size = sum(discrete_branches) if discrete_branches else 0
    cont_out, disc_out = resolve_action_outputs(session, continuous_size, discrete_size)

    stats_channel.get_and_reset_stats()  # clear anything left over from the previous checkpoint

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    agent_cum_reward: Dict[int, float] = {}
    agent_len: Dict[int, int] = {}

    env.reset()
    steps_taken = 0
    while len(episode_rewards) < n_episodes and steps_taken < max_steps:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        for agent_id, r in zip(decision_steps.agent_id, decision_steps.reward):
            agent_cum_reward[agent_id] = agent_cum_reward.get(agent_id, 0.0) + float(r)
            agent_len[agent_id] = agent_len.get(agent_id, 0) + 1
        for agent_id, r in zip(terminal_steps.agent_id, terminal_steps.reward):
            episode_rewards.append(agent_cum_reward.pop(agent_id, 0.0) + float(r))
            episode_lengths.append(agent_len.pop(agent_id, 0))

        if len(decision_steps) > 0:
            feed = build_feed(session, decision_steps, continuous_size, discrete_size)
            outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed)))
            n_agents = len(decision_steps)
            cont_actions = (outputs[cont_out].astype(np.float32) if cont_out
                             else np.zeros((n_agents, 0), dtype=np.float32))
            disc_actions = (outputs[disc_out].astype(np.int32) if disc_out
                             else np.zeros((n_agents, 0), dtype=np.int32))
            env.set_actions(behavior_name, ActionTuple(continuous=cont_actions, discrete=disc_actions))

        env.step()
        steps_taken += 1

    if steps_taken >= max_steps:
        log.warning("Hit max_steps=%d for %s with only %d/%d episodes collected.",
                    max_steps, onnx_path.name, len(episode_rewards), n_episodes)

    stats = stats_channel.get_and_reset_stats()
    goals = sum(v for v, _ in stats.get("Custom/GoalReached", []))
    terminated = sum(v for v, _ in stats.get("Custom/Terminated", []))
    maxstep = sum(v for v, _ in stats.get("Custom/MaxStepReached", []))
    outcomes = goals + terminated + maxstep

    return {
        "step": step_from_path(onnx_path),
        "episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else float("nan"),
        "goal_rate": goals / outcomes if outcomes else float("nan"),
        "terminated_rate": terminated / outcomes if outcomes else float("nan"),
        "maxstep_rate": maxstep / outcomes if outcomes else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="Path to the headless Linux build executable.")
    ap.add_argument("--run-dir", required=True, help="results/<run_id> directory to evaluate.")
    ap.add_argument("--behavior-name", default="CarAgent")
    ap.add_argument("--episodes", type=int, default=50, help="Episodes to average per checkpoint.")
    ap.add_argument("--max-steps-per-checkpoint", type=int, default=200_000,
                     help="Safety cap on env steps per checkpoint, in case of a stuck episode.")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--base-port", type=int, default=7005)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graphics", action="store_true", default=True)
    ap.add_argument("--graphics", dest="no_graphics", action="store_false",
                     help="Run with graphics on (local debugging only).")
    ap.add_argument("--every-nth-checkpoint", type=int, default=1,
                     help="Evaluate only every Nth discovered checkpoint (subsample without retraining).")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N checkpoints (testing).")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <run-dir>/eval_results.csv).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_id = run_dir.name
    out_path = Path(args.out) if args.out else run_dir / "eval_results.csv"

    checkpoints = discover_checkpoints(run_dir, args.behavior_name)
    checkpoints = checkpoints[:: args.every_nth_checkpoint]
    if args.limit:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        log.error("No checkpoints found under %s/%s", run_dir, args.behavior_name)
        sys.exit(1)
    log.info("Found %d checkpoints to evaluate for run '%s'.", len(checkpoints), run_id)

    engine_channel = EngineConfigurationChannel()
    stats_channel = StatsSideChannel()

    env = UnityEnvironment(
        file_name=args.binary,
        worker_id=args.worker_id,
        base_port=args.base_port,
        seed=args.seed,
        no_graphics=args.no_graphics,
        side_channels=[engine_channel, stats_channel],
    )
    engine_channel.set_configuration_parameters(time_scale=args.time_scale, target_frame_rate=-1)

    fieldnames = ["run_id", "step", "episodes", "mean_reward", "std_reward",
                  "goal_rate", "terminated_rate", "maxstep_rate", "mean_episode_length"]
    write_header = not out_path.exists()

    try:
        env.reset()
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]

        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for i, ckpt in enumerate(checkpoints):
                t0 = time.time()
                row = run_checkpoint(env, stats_channel, behavior_name, spec, ckpt,
                                      args.episodes, args.max_steps_per_checkpoint)
                row["run_id"] = run_id
                writer.writerow(row)
                f.flush()
                log.info(
                    "[%d/%d] step=%d episodes=%d mean_reward=%.3f goal_rate=%.3f (%.1fs)",
                    i + 1, len(checkpoints), row["step"], row["episodes"],
                    row["mean_reward"], row["goal_rate"], time.time() - t0,
                )
    finally:
        env.close()

    log.info("Done. Results written to %s", out_path)


if __name__ == "__main__":
    main()
