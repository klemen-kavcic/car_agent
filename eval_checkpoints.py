"""Deterministic (inference-only, no exploration) evaluation of a sweep of ML-Agents
checkpoints from a single training run.

Connects directly to the built Unity binary via the mlagents_envs low-level API (the
same binary/apptainer setup used for training - no C# changes or rebuild needed), loads
each CarAgent-<step>.onnx checkpoint in turn with onnxruntime, and drives the agent using
the model's *deterministic* action outputs (no sampling noise), so results reflect the
greedy policy at that checkpoint rather than the exploring training-time policy.

Per-episode outcome stats (Custom/GoalReached, Custom/Terminated, Custom/MaxStepReached) and
per-decision-step tile occupancy (Custom/TileTime<Type>) are read back from the same
StatsSideChannel mechanism the trainer uses for TensorBoard - carAgent.cs reports them
unconditionally every episode/step, regardless of what's connected and driving the agent's
actions, so the same numbers are also visible live in TensorBoard during ordinary training.

One CSV row is written per checkpoint (per --action-mode, if --action-mode=both):
    run_id, maps, action_mode, step, episodes, mean_reward, std_reward,
    goal_rate, terminated_rate, maxstep_rate,
    goal_rate_std, terminated_rate_std, maxstep_rate_std,
    mean_episode_length, std_episode_length,
    time_frac_slippery, time_frac_speedlimited, time_frac_terminal,
    time_frac_asphalt, time_frac_gravel,
    visits_per_episode_slippery, visits_per_episode_speedlimited, visits_per_episode_terminal,
    visits_per_episode_asphalt, visits_per_episode_gravel,
    episode_tile_time_std_slippery, episode_tile_time_std_speedlimited,
    episode_tile_time_std_terminal, episode_tile_time_std_asphalt, episode_tile_time_std_gravel,
    episode_tile_visits_std_slippery, episode_tile_visits_std_speedlimited,
    episode_tile_visits_std_terminal, episode_tile_visits_std_asphalt, episode_tile_visits_std_gravel
The time_frac_* columns are the fraction of decision steps spent on each tile type - e.g. a
policy that's actually learned to favour roads should show most of its time on asphalt rather
than gravel/grass/ice, regardless of what's directly ahead of it. time_frac_* is biased by how
fast the car moves on each surface, though - Slippery/Gravel inherently take longer to cross than
Asphalt at the same crossing frequency, so a slow surface racks up more time just by being slow.
visits_per_episode_* counts genuine surface *changes* instead (including the episode's starting
tile) divided by episode count - how many times each surface was crossed onto per episode,
independent of how long the car then lingered there.

*_std columns (all additive, none change what any existing column measures):
    - goal_rate_std/terminated_rate_std/maxstep_rate_std: sqrt(p*(1-p)) for rate p - the std of a
      per-episode 0/1 outcome indicator, the binomial analogue of std_reward/std_episode_length.
    - std_episode_length: same population-std treatment as std_reward, just for episode length.
    - episode_tile_time_std_*/episode_tile_visits_std_*: std ACROSS EPISODES of the raw per-episode
      step-count/visit-count on that tile (same units as std_episode_length - NOT a std of the
      time_frac_*/visits_per_episode_* fraction/mean, which stay pooled-window quantities exactly
      as before). Sourced from carAgent.cs's Custom/EpisodeTileTime<Type>/
      Custom/EpisodeTileVisit<Type>, reported once per episode (see OnEpisodeBegin there) alongside
      the existing per-step Custom/TileTime<Type>/Custom/TileVisit<Type> this script already read -
      requires a rebuilt binary to appear (new StatsRecorder keys), same as visits_per_episode_*.

Under mapSource=Voronoi, --maps (default 'val') selects Resources/Maps/VoronoiVal - a set of
maps never seen during training - via the use_validation_maps environment parameter, so results
reflect generalisation rather than memorisation of the training pool. Pass --maps train to
re-evaluate the training pool itself instead, for a direct train-vs-val comparison. No effect
under mapSource=Perlin.

Also explicitly disables the bootstrap curriculum and the Voronoi near-goal ramp (both of which
default to their training-time-relevant behavior in carAgent.cs if left unset), since neither
means anything relative to this standalone eval session's own irrelevant local step count - see
EvalWorker.__init__ for the full reasoning. Without this, most evaluation runs would silently
spend all their episodes inside the Front/Behind/DeadEnd/Side bootstrap drills (forced blank
all-Asphalt grid) instead of testing the real map.

--num-envs > 1 runs that many independent Unity environment processes in parallel (same idea as
training's --num-envs, one full copy of the environment per worker) and pools their episode
outcomes for each checkpoint - since the checkpoint's policy is frozen, this changes nothing about
what's measured, it just collects the same total episode count in roughly 1/num_envs the wall
time. Each worker gets its own worker_id/port (base_port + worker_id, mlagents_envs' own
convention) and its own seed (--seed + worker_id) so their episode/map/spawn randomization isn't
correlated.

Usage:
    python eval_checkpoints.py \
        --binary /path/to/Linux/car.x86_64 \
        --run-dir /path/to/results/<run_id> \
        --episodes 50 \
        --num-envs 32 \
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
import yaml

from mlagents_envs.base_env import ActionTuple, BehaviorSpec
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

logging.basicConfig(level=logging.INFO, format="[eval] %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"-(\d+)\.onnx$")

# Must match Assets/TileType.cs exactly - used to read back the Custom/TileTime<Type> stats
# carAgent.cs reports (CollectObservations), one per enum value.
TILE_TYPES = ["Slippery", "SpeedLimited", "Terminal", "Asphalt", "Gravel"]

# Names ML-Agents 4.x exports for greedy/no-exploration action selection. If your
# onnxruntime session doesn't have these, the script will print the available output
# names it found so you can adjust these two constants.
DETERMINISTIC_CONTINUOUS_OUTPUT = "deterministic_continuous_actions"
DETERMINISTIC_DISCRETE_OUTPUT = "deterministic_discrete_actions"
# The "regular" (sampled, matches training-time behavior) counterparts - continuous sampling is
# the standard ML-Agents reparameterization trick (action = mean + std * epsilon), so genuine
# per-call randomness requires feeding real noise into the "epsilon" input (see build_feed) rather
# than the all-zero epsilon used for deterministic reads. Discrete sampling is expected to be
# self-contained in the exported graph (no extra external randomness needed). If your onnxruntime
# session doesn't have these two names, the script prints available outputs, same safety net as
# the deterministic pair above.
STOCHASTIC_CONTINUOUS_OUTPUT = "continuous_actions"
STOCHASTIC_DISCRETE_OUTPUT = "discrete_actions"

# Matches CarAgent's Inspector-set default MaxStep (carAgent.cs UpdateRewardWeights,
# max_step_budget defaults to 5000 when unset) - used only to size the default
# --max-steps-per-checkpoint safety cap generously enough that it's never hit in normal
# operation, even if every single requested episode happened to run to a full timeout.
ASSUMED_MAX_EPISODE_LENGTH = 5000


def load_completed_rows(out_path: Path, maps: str) -> set:
    """Reads an existing --out CSV from a previous (possibly SLURM-timeout-killed) invocation
    and returns the (step, action_mode) pairs already written for THIS --maps pool, so a re-run
    with the same --out path resumes instead of re-evaluating every checkpoint from scratch.

    Keyed on --maps too, not just (step, action_mode): the sbatch scripts run this script twice
    per run (once for the train pool, once for val), both appending to the same eval_results.csv -
    without the maps filter, a completed train-pool row would wrongly make the val-pool pass skip
    that same step.

    Safe to resume: each row is one independent, complete, inference-only evaluation of a frozen
    checkpoint - nothing is carried or averaged across rows, so which process invocation wrote a
    given row doesn't affect what it measures. The only difference a resumed process introduces is
    that its Unity workers reseed from --seed fresh rather than continuing the original process's
    RNG stream partway through - episodes still come from the same underlying random distribution
    (map/spawn draws), so this changes which specific samples are drawn, not their distribution -
    no systematic bias, i.e. no skew.

    A SLURM kill can land mid-write and leave a torn final line (killed after some bytes were
    flushed but before the row/newline completed). Detected here (a row that fails to parse into
    every expected field) and the file truncated to drop it, so that checkpoint gets cleanly
    re-evaluated rather than silently skipped or counted as done.
    """
    if not out_path.exists():
        return set()
    with open(out_path, newline="") as f:
        lines = f.readlines()
    if not lines:
        return set()
    completed = set()
    good_line_count = 1  # header line is always kept as-is
    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(None) is not None or row.get("step") is None or row.get("action_mode") is None \
                    or row.get("maps") is None:
                log.warning(
                    "Dropping malformed trailing row in %s (likely torn by a killed process) - "
                    "that checkpoint will be re-evaluated.", out_path,
                )
                break
            if row["maps"] == maps:
                completed.add((int(row["step"]), row["action_mode"]))
            good_line_count += 1
    if good_line_count < len(lines):
        with open(out_path, "w", newline="") as f:
            f.writelines(lines[:good_line_count])
    return completed


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


def training_num_envs_from_config(config_path: Path) -> Optional[int]:
    """Reads environment_parameters.training_num_envs from the same yaml passed to
    mlagents-learn - that value is already required to match the real --num-envs used for
    training (see car_agent.yaml's own comment on the key), so it doubles as a sensible
    default for how many parallel eval workers to run. Returns None if the file/key is
    missing so the caller can fall back to a plain default instead of guessing."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    value = config.get("environment_parameters", {}).get("training_num_envs")
    return int(value) if value is not None else None


def build_feed(session: ort.InferenceSession, decision_steps, continuous_size: int,
                discrete_size: int, action_mode: str = "deterministic") -> Dict[str, np.ndarray]:
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
            # Zero (irrelevant) for a deterministic/mean read; for stochastic, real standard-normal
            # noise, matching the reparameterization trick (action = mean + std * epsilon) training
            # itself samples with - this is what actually makes the "stochastic" mode genuinely
            # sampled rather than a second copy of the mean action.
            if action_mode == "stochastic":
                feed[name] = np.random.randn(n_agents, max(continuous_size, 0)).astype(np.float32)
            else:
                feed[name] = np.zeros((n_agents, max(continuous_size, 0)), dtype=np.float32)
        elif name == "sequence_length":
            feed[name] = np.array(1, dtype=np.int64)
        else:
            log.warning("Unhandled model input '%s' - onnxruntime will raise if it's required.", name)
    return feed


def resolve_action_outputs(session: ort.InferenceSession, continuous_size: int, discrete_size: int,
                            action_mode: str = "deterministic"):
    output_names = {o.name for o in session.get_outputs()}
    if action_mode == "stochastic":
        cont_default, disc_default = STOCHASTIC_CONTINUOUS_OUTPUT, STOCHASTIC_DISCRETE_OUTPUT
    else:
        cont_default, disc_default = DETERMINISTIC_CONTINUOUS_OUTPUT, DETERMINISTIC_DISCRETE_OUTPUT
    cont_name = cont_default if continuous_size > 0 else None
    disc_name = disc_default if discrete_size > 0 else None
    missing = [n for n in (cont_name, disc_name) if n and n not in output_names]
    if missing:
        raise RuntimeError(
            f"Expected {action_mode} output(s) {missing} not found in ONNX model. "
            f"Available outputs: {sorted(output_names)}. "
            "Adjust DETERMINISTIC_*/STOCHASTIC_* constants at the top of this script to match "
            "your ML-Agents export version."
        )
    return cont_name, disc_name


class EvalWorker:
    """One independent Unity environment process/connection. Multiple of these run in
    parallel (see --num-envs) and get pooled together for each checkpoint - since the
    checkpoint's policy is frozen and each worker is fully independent, this only affects
    wall-clock time, never the results."""

    def __init__(self, args, worker_id: int, seed: int):
        self.worker_id = worker_id
        self.engine_channel = EngineConfigurationChannel()
        self.stats_channel = StatsSideChannel()
        self.env_params_channel = EnvironmentParametersChannel()
        # Queued before the environment connects, so CarAgent.OnEpisodeBegin already sees these on
        # the very first episode - no separate reset needed to make them take effect.
        self.env_params_channel.set_float_parameter("use_validation_maps", 1.0 if args.maps == "val" else 0.0)
        # Without this, bootstrap_curriculum_enabled defaults to 1 (carAgent.cs
        # DetermineBootstrapStage) and training_num_envs defaults to 1 instead of the real
        # training value (e.g. 32) - so eval computes bootstrap stage lengths ~32x longer than
        # during training, and most/all of an eval run gets stuck inside the Front/Behind/DeadEnd/
        # Side bootstrap drills (forceAllNormal=true, i.e. a blank all-Asphalt grid, not the real
        # Voronoi map) instead of evaluating full-random placement on the actual map like a mature
        # checkpoint was trained against. Forcing this off always evaluates the "graduated"
        # full-random regime, regardless of this session's own (irrelevant) local step count.
        self.env_params_channel.set_float_parameter("bootstrap_curriculum_enabled", 0.0)
        # Same reasoning: voronoi_near_goal_enabled ramps the goal-distance cap based on the
        # session's own local TotalStepCount, meaningless outside of a real training run. Forced
        # off so eval always uses the fully-unconstrained goal placement a mature checkpoint was
        # ultimately trained against, regardless of what car_agent.yaml has it set to.
        self.env_params_channel.set_float_parameter("voronoi_near_goal_enabled", 0.0)
        # Unset, this stays at carAgent.cs's hardcoded C# default (5000) regardless of what the
        # training yaml's max_step_budget says, and regardless of whatever value
        # adaptive_step_budget_enabled had actually converged to during that specific training run
        # - eval never pushes this parameter unless told to. A degenerate/still-training policy
        # that times out on most episodes then costs up to 5000 ticks PER episode to evaluate,
        # across up to 1000 episodes x ~100 checkpoints x 2 map pools - see --max-step-budget to
        # cap that cost (a healthy policy's episodes end in ~100-300 ticks regardless, so a lower
        # cap only bites on the runs that were already blowing the eval walltime budget).
        if args.max_step_budget is not None:
            self.env_params_channel.set_float_parameter("max_step_budget", float(args.max_step_budget))

        self.env = UnityEnvironment(
            file_name=args.binary,
            worker_id=worker_id,
            base_port=args.base_port,
            seed=seed,
            no_graphics=args.no_graphics,
            side_channels=[self.engine_channel, self.stats_channel, self.env_params_channel],
        )
        self.engine_channel.set_configuration_parameters(time_scale=args.time_scale, target_frame_rate=-1)
        self.behavior_name: Optional[str] = None
        self.spec: Optional[BehaviorSpec] = None

    def connect(self):
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

    def close(self):
        self.env.close()


def run_checkpoint(workers: List[EvalWorker], onnx_path: Path, n_episodes: int,
                    max_steps_per_env: int, action_mode: str = "deterministic") -> Optional[dict]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    spec = workers[0].spec
    continuous_size = spec.action_spec.continuous_size
    discrete_branches = spec.action_spec.discrete_branches
    discrete_size = sum(discrete_branches) if discrete_branches else 0
    cont_out, disc_out = resolve_action_outputs(session, continuous_size, discrete_size, action_mode)

    for w in workers:
        w.stats_channel.get_and_reset_stats()  # clear anything left over from the previous checkpoint
        w.env.reset()

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    # Per-worker agent-id bookkeeping - agent ids are only unique within a single environment
    # connection, so each worker needs its own dict (kept separate rather than merged into one).
    agent_cum_reward: List[Dict[int, float]] = [{} for _ in workers]
    agent_len: List[Dict[int, int]] = [{} for _ in workers]
    steps_taken = [0] * len(workers)
    active = list(range(len(workers)))

    while len(episode_rewards) < n_episodes and active:
        for i in list(active):
            if len(episode_rewards) >= n_episodes:
                break
            w = workers[i]
            if steps_taken[i] >= max_steps_per_env:
                active.remove(i)
                continue

            decision_steps, terminal_steps = w.env.get_steps(w.behavior_name)

            for agent_id, r in zip(decision_steps.agent_id, decision_steps.reward):
                agent_cum_reward[i][agent_id] = agent_cum_reward[i].get(agent_id, 0.0) + float(r)
                agent_len[i][agent_id] = agent_len[i].get(agent_id, 0) + 1
            for agent_id, r in zip(terminal_steps.agent_id, terminal_steps.reward):
                episode_rewards.append(agent_cum_reward[i].pop(agent_id, 0.0) + float(r))
                episode_lengths.append(agent_len[i].pop(agent_id, 0))

            if len(decision_steps) > 0:
                feed = build_feed(session, decision_steps, continuous_size, discrete_size, action_mode)
                outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed)))
                n_agents = len(decision_steps)
                cont_actions = (outputs[cont_out].astype(np.float32) if cont_out
                                 else np.zeros((n_agents, 0), dtype=np.float32))
                disc_actions = (outputs[disc_out].astype(np.int32) if disc_out
                                 else np.zeros((n_agents, 0), dtype=np.int32))
                w.env.set_actions(w.behavior_name, ActionTuple(continuous=cont_actions, discrete=disc_actions))

            w.env.step()
            steps_taken[i] += 1

    if not active and len(episode_rewards) < n_episodes:
        log.warning(
            "All %d worker(s) hit max_steps_per_env=%d for %s with only %d/%d episodes collected.",
            len(workers), max_steps_per_env, onnx_path.name, len(episode_rewards), n_episodes,
        )

    # Pool stats across every worker - each is an independent environment, so summing their raw
    # counts (rather than averaging pre-computed per-worker rates) is the correct aggregation.
    goals = terminated = maxstep = 0
    tile_time = {t: 0.0 for t in TILE_TYPES}
    tile_visits = {t: 0.0 for t in TILE_TYPES}
    # Per-episode breakdowns (one list entry per episode per type, from carAgent.cs's
    # Custom/EpisodeTileTime<Type>/Custom/EpisodeTileVisit<Type> - see the field comments in
    # OnEpisodeBegin there) - used only to compute a std-across-episodes below; the pooled
    # tile_time/tile_visits totals above (and everything derived from them) are unaffected.
    episode_tile_time = {t: [] for t in TILE_TYPES}
    episode_tile_visits = {t: [] for t in TILE_TYPES}
    for w in workers:
        stats = w.stats_channel.get_and_reset_stats()
        goals += sum(v for v, _ in stats.get("Custom/GoalReached", []))
        terminated += sum(v for v, _ in stats.get("Custom/Terminated", []))
        maxstep += sum(v for v, _ in stats.get("Custom/MaxStepReached", []))
        for t in TILE_TYPES:
            tile_time[t] += sum(v for v, _ in stats.get(f"Custom/TileTime{t}", []))
            tile_visits[t] += sum(v for v, _ in stats.get(f"Custom/TileVisit{t}", []))
            episode_tile_time[t] += [v for v, _ in stats.get(f"Custom/EpisodeTileTime{t}", [])]
            episode_tile_visits[t] += [v for v, _ in stats.get(f"Custom/EpisodeTileVisit{t}", [])]
    outcomes = goals + terminated + maxstep
    tile_time_total = sum(tile_time.values())

    goal_rate = goals / outcomes if outcomes else float("nan")
    terminated_rate = terminated / outcomes if outcomes else float("nan")
    maxstep_rate = maxstep / outcomes if outcomes else float("nan")

    row = {
        "step": step_from_path(onnx_path),
        "action_mode": action_mode,
        "episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else float("nan"),
        "goal_rate": goal_rate,
        "terminated_rate": terminated_rate,
        "maxstep_rate": maxstep_rate,
        # Std of a per-episode 0/1 outcome indicator, i.e. sqrt(p*(1-p)) for rate p - the binomial
        # analogue of std_reward/std_episode_length above (population std, not a mean's standard
        # error), computed straight from the pooled rate rather than needing per-episode outcome
        # tracking.
        "goal_rate_std": float(np.sqrt(goal_rate * (1 - goal_rate))) if outcomes else float("nan"),
        "terminated_rate_std": float(np.sqrt(terminated_rate * (1 - terminated_rate))) if outcomes else float("nan"),
        "maxstep_rate_std": float(np.sqrt(maxstep_rate * (1 - maxstep_rate))) if outcomes else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        "std_episode_length": float(np.std(episode_lengths)) if episode_lengths else float("nan"),
    }
    n_episodes = len(episode_rewards)
    for t in TILE_TYPES:
        row[f"time_frac_{t.lower()}"] = (
            tile_time[t] / tile_time_total if tile_time_total else float("nan")
        )
        row[f"visits_per_episode_{t.lower()}"] = (
            tile_visits[t] / n_episodes if n_episodes else float("nan")
        )
        # Std across episodes of the RAW per-episode step-count/visit-count on this tile (same
        # units as std_episode_length, i.e. not a std of the time_frac_*/visits_per_episode_*
        # fraction/mean above - those stay pooled-window quantities exactly as before).
        row[f"episode_tile_time_std_{t.lower()}"] = (
            float(np.std(episode_tile_time[t])) if episode_tile_time[t] else float("nan")
        )
        row[f"episode_tile_visits_std_{t.lower()}"] = (
            float(np.std(episode_tile_visits[t])) if episode_tile_visits[t] else float("nan")
        )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="Path to the headless Linux build executable.")
    ap.add_argument("--run-dir", required=True, help="results/<run_id> directory to evaluate.")
    ap.add_argument("--behavior-name", default="CarAgent")
    ap.add_argument("--episodes", type=int, default=50, help="Episodes to average per checkpoint (pooled across all --num-envs workers).")
    ap.add_argument("--max-step-budget", type=int, default=None,
                     help="Overrides carAgent.cs's Agent.MaxStep (max_step_budget environment "
                          "parameter, default 5000 if left unset) for the DURATION OF EVAL ONLY - "
                          "has no effect on the checkpoint's actual trained behavior, just how long "
                          "a stalled/timing-out episode is allowed to run before being cut off. "
                          "Lowering this (e.g. to 2000) mainly speeds up evaluating a still-broken "
                          "or degenerate policy whose episodes mostly run to the timeout (which is "
                          "also what was blowing eval's own walltime budget on those runs) - a "
                          "healthy policy's episodes already end in ~100-300 ticks regardless, so "
                          "this doesn't change its results. Leave unset to use whatever "
                          "car_component.cs/carAgent.cs's own default (5000) already is.")
    ap.add_argument("--max-steps-per-checkpoint", type=int, default=None,
                     help="Safety cap on env steps per checkpoint PER WORKER, in case of a stuck "
                          "episode (default: --episodes * effective max-step-budget, generous "
                          "enough to never hit in normal operation even if every requested episode "
                          "timed out).")
    ap.add_argument("--num-envs", type=int, default=None,
                     help="Number of parallel Unity environment processes to run and pool episodes "
                          "across per checkpoint (same idea as training's --num-envs). Each gets its "
                          "own worker_id/port and seed. Purely a wall-clock speedup - doesn't change "
                          "what's measured, since the checkpoint's policy is frozen and each worker "
                          "is fully independent. Default: read environment_parameters."
                          "training_num_envs from --config if given, else 1.")
    ap.add_argument("--config", default=None,
                     help="Path to the yaml passed to mlagents-learn for this run, used only to "
                          "default --num-envs to the same environment_parameters.training_num_envs "
                          "value training used (that key is already required to match the real "
                          "--num-envs training was launched with - see the comment on it in "
                          "car_agent.yaml). Purely a convenience; has no other effect on eval.")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--base-port", type=int, default=7005)
    ap.add_argument("--worker-id", type=int, default=0,
                     help="Starting worker id - workers use worker_id..worker_id+num_envs-1. Bump "
                          "this if running multiple eval processes on the same node at once.")
    ap.add_argument("--seed", type=int, default=0,
                     help="Base seed - worker i uses seed+i, so parallel workers aren't correlated.")
    ap.add_argument("--no-graphics", action="store_true", default=True)
    ap.add_argument("--graphics", dest="no_graphics", action="store_false",
                     help="Run with graphics on (local debugging only).")
    ap.add_argument("--every-nth-checkpoint", type=int, default=1,
                     help="Evaluate only every Nth discovered checkpoint (subsample without retraining).")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N checkpoints (testing).")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <run-dir>/eval_results.csv).")
    ap.add_argument("--maps", choices=["train", "val"], default="val",
                     help="Which map pool to evaluate against under mapSource=Voronoi, via the "
                          "use_validation_maps environment parameter (see grid_manager.cs/"
                          "carAgent.cs) - 'val' (default) measures generalisation to the held-out "
                          "set never seen during training, 'train' re-evaluates the training pool "
                          "itself for comparison. Has no effect under mapSource=Perlin.")
    ap.add_argument("--action-mode", choices=["deterministic", "stochastic", "both"], default="deterministic",
                     help="'deterministic' (default, backward compatible) reads the greedy/mean "
                          "action - what a deployed checkpoint would actually do. 'stochastic' "
                          "samples from the policy's output distribution instead, same as training "
                          "rollouts. 'both' runs a full separate pass of each per checkpoint (roughly "
                          "doubles eval time) and writes one row per mode, distinguished by the new "
                          "'action_mode' CSV column - useful for directly measuring how much of any "
                          "train-vs-eval gap is explained by sampling noise vs. a real difference.")
    args = ap.parse_args()

    if args.max_steps_per_checkpoint is None:
        effective_max_episode_length = (
            args.max_step_budget if args.max_step_budget is not None else ASSUMED_MAX_EPISODE_LENGTH
        )
        args.max_steps_per_checkpoint = args.episodes * effective_max_episode_length

    if args.num_envs is None:
        args.num_envs = 1
        if args.config:
            found = training_num_envs_from_config(Path(args.config))
            if found is not None:
                args.num_envs = found
                log.info("Read training_num_envs=%d from %s - using --num-envs=%d.",
                          found, args.config, found)
            else:
                log.warning("environment_parameters.training_num_envs not found in %s - "
                             "defaulting --num-envs=1.", args.config)

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
    log.info("Map pool: %s | num_envs=%d | max_steps_per_checkpoint(per env)=%d",
              args.maps, args.num_envs, args.max_steps_per_checkpoint)

    workers = [EvalWorker(args, args.worker_id + i, args.seed + i) for i in range(args.num_envs)]

    action_modes = ["deterministic", "stochastic"] if args.action_mode == "both" else [args.action_mode]

    fieldnames = ["run_id", "maps", "action_mode", "step", "episodes", "mean_reward", "std_reward",
                  "goal_rate", "terminated_rate", "maxstep_rate",
                  "goal_rate_std", "terminated_rate_std", "maxstep_rate_std",
                  "mean_episode_length", "std_episode_length"] + \
                 [f"time_frac_{t.lower()}" for t in TILE_TYPES] + \
                 [f"visits_per_episode_{t.lower()}" for t in TILE_TYPES] + \
                 [f"episode_tile_time_std_{t.lower()}" for t in TILE_TYPES] + \
                 [f"episode_tile_visits_std_{t.lower()}" for t in TILE_TYPES]
    completed = load_completed_rows(out_path, args.maps)
    # size==0 case: a previous invocation was killed before even writing the header row.
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    if completed:
        log.info(
            "Resuming %s - %d (step, action_mode) row(s) already present for maps=%s, will be "
            "skipped.", out_path, len(completed), args.maps,
        )

    try:
        for w in workers:
            w.connect()

        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for i, ckpt in enumerate(checkpoints):
                step = step_from_path(ckpt)
                for mode in action_modes:
                    if (step, mode) in completed:
                        log.info("[%d/%d] mode=%s step=%d - already done, skipping.",
                                  i + 1, len(checkpoints), mode, step)
                        continue
                    t0 = time.time()
                    row = run_checkpoint(workers, ckpt, args.episodes, args.max_steps_per_checkpoint, mode)
                    row["run_id"] = run_id
                    row["maps"] = args.maps
                    writer.writerow(row)
                    f.flush()
                    log.info(
                        "[%d/%d] mode=%s step=%d episodes=%d mean_reward=%.3f goal_rate=%.3f "
                        "time_on_asphalt=%.2f (%.1fs)",
                        i + 1, len(checkpoints), mode, row["step"], row["episodes"],
                        row["mean_reward"], row["goal_rate"], row["time_frac_asphalt"],
                        time.time() - t0,
                    )
    finally:
        for w in workers:
            w.close()

    log.info("Done. Results written to %s", out_path)


if __name__ == "__main__":
    main()
