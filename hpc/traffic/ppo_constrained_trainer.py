"""PPO trainer with an explicit completed-car collision constraint.

Bind this over ML-Agents 1.1.0's ``ppo/trainer.py`` together with
``ppo_constrained_optimizer.py``. Unity writes a terminal-only -1 sentinel in
the existing remaining-time input for VehicleCollision. This trainer turns that
marker into a separate binary cost; it never folds it into the task reward.
"""

from __future__ import annotations

import os
from typing import cast, Type, Union, Dict, Any

import numpy as np
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer, PPOSettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic

from mlagents.trainers.ppo.optimizer_torch import (
    COST_KEY,
    COST_VALUE_KEY,
    COST_RETURN_KEY,
    COST_ADVANTAGE_KEY,
    COST_COMPLETED_KEY,
    COST_RAW_KEY,
)

logger = get_logger(__name__)
TRAINER_NAME = "ppo"

# ``CollectObservations`` in carAgent.cs writes the 90-observation vector in
# this order: 12 kinematic/goal/control values, remaining-time fraction,
# current-tile one-hot, then the laser/tile inputs.  The collision sentinel is
# therefore not the final value of the vector.  Keeping this explicit makes a
# future observation-layout change fail loudly instead of silently disabling
# the safety constraint again.
REMAINING_STEPS_OBSERVATION_INDEX = 12


class PPOTrainer(OnPolicyTrainer):
    """PPO where reward and collision-cost learning are separate signals."""

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(
            behavior_name, reward_buff_cap, trainer_settings, training, load, seed, artifact_path
        )
        self.hyperparameters: PPOSettings = cast(PPOSettings, self.trainer_settings.hyperparameters)
        self.seed = seed
        self.shared_critic = self.hyperparameters.shared_critic
        self.policy: TorchPolicy = None  # type: ignore
        # Deliberately test-only: a one-seed 1M smoke run can force every
        # completed trajectory to cost one, proving that the dual update and
        # TensorBoard reporting are live. The original Unity marker is still
        # retained and reported separately.
        self.force_smoke_cost = os.environ.get("CPO_SMOKE_FORCE_COST", "0") == "1"

    @staticmethod
    def _terminal_collision_cost(trajectory: Trajectory) -> float:
        """Read the collision sentinel from the terminal observation.

        The remaining-time scalar is index 12 in the 90-observation layout.
        During a decision it is always [0,1]. C# writes -1 only after a
        completed VehicleCollision, immediately before EndEpisode. Partial
        time-horizon trajectories never contribute an episode cost or
        denominator.
        """
        if not trajectory.done_reached:
            return 0.0
        if not trajectory.next_obs:
            raise RuntimeError("CPO needs the terminal vector observation")
        terminal_vector = np.asarray(trajectory.next_obs[0], dtype=np.float32).reshape(-1)
        if terminal_vector.size <= REMAINING_STEPS_OBSERVATION_INDEX:
            raise RuntimeError(
                "CPO needs the 90-observation vector with its remaining-time "
                f"slot at index {REMAINING_STEPS_OBSERVATION_INDEX}, but got "
                f"only {terminal_vector.size} values"
            )
        return float(terminal_vector[REMAINING_STEPS_OBSERVATION_INDEX] < -0.5)

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        # Preserve RLTrainer's lifecycle bookkeeping before doing CPO-specific
        # processing.  This advances the global trainer step and is what
        # schedules summary writes, checkpoint exports, and the 40M stop
        # condition.  Omitting it lets Unity keep simulating while PPO appears
        # permanently at step zero (no TensorBoard scalars/checkpoints).
        super()._process_trajectory(trajectory)

        agent_id = trajectory.agent_id
        agent_buffer_trajectory = trajectory.to_agentbuffer()
        self._warn_if_group_reward(agent_buffer_trajectory)

        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)
            self.optimizer.cost_critic.update_normalization(agent_buffer_trajectory)

        value_estimates, value_next, value_memories = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)
        for name, values in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(values)
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(values),
            )

        # Unchanged task-reward path.
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluated = reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(evaluated)
            self.collected_rewards[name][agent_id] += np.sum(evaluated)

        reward_advantages = []
        reward_returns = []
        for name in self.optimizer.reward_signals:
            local_rewards = agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].get_batch()
            local_values = agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].get_batch()
            local_advantage = get_gae(
                rewards=local_rewards,
                value_estimates=local_values,
                value_next=value_next[name],
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
            )
            local_return = local_advantage + local_values
            agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(local_return)
            agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(local_advantage)
            reward_advantages.append(local_advantage)
            reward_returns.append(local_return)
        agent_buffer_trajectory[BufferKey.ADVANTAGES].set(
            list(np.mean(np.asarray(reward_advantages, dtype=np.float32), axis=0))
        )
        agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(
            list(np.mean(np.asarray(reward_returns, dtype=np.float32), axis=0))
        )

        # Cost is a binary completed-car outcome, never a discounted reward.
        # gamma_cost = lambda_cost = 1 makes its return the probability of a
        # collision before the car completes, conditional on the current state.
        raw_terminal_cost = self._terminal_collision_cost(trajectory)
        terminal_cost = 1.0 if self.force_smoke_cost and trajectory.done_reached else raw_terminal_cost
        cost_values, cost_next = self.optimizer.get_cost_value_estimates(
            agent_buffer_trajectory, trajectory.next_obs, trajectory.done_reached
        )
        local_costs = np.zeros(len(cost_values), dtype=np.float32)
        if len(local_costs) > 0 and trajectory.done_reached:
            local_costs[-1] = terminal_cost
        # Exact Monte-Carlo return within each completed trajectory; on a
        # horizon cut the learned cost value supplies the bootstrap instead.
        cost_advantage = get_gae(
            rewards=local_costs,
            value_estimates=cost_values,
            value_next=cost_next,
            gamma=1.0,
            lambd=1.0,
        )
        cost_return = cost_advantage + cost_values
        agent_buffer_trajectory[COST_VALUE_KEY].set(cost_values)
        agent_buffer_trajectory[COST_RETURN_KEY].set(cost_return)
        agent_buffer_trajectory[COST_ADVANTAGE_KEY].set(cost_advantage)
        agent_buffer_trajectory[COST_KEY].set(local_costs)
        raw_costs = np.zeros(len(cost_values), dtype=np.float32)
        if len(raw_costs) > 0 and trajectory.done_reached:
            raw_costs[-1] = raw_terminal_cost
        agent_buffer_trajectory[COST_RAW_KEY].set(raw_costs)
        completed = np.zeros(len(cost_values), dtype=np.float32)
        if len(completed) > 0 and trajectory.done_reached:
            completed[-1] = 1.0
        agent_buffer_trajectory[COST_COMPLETED_KEY].set(completed)

        self._append_to_update_buffer(agent_buffer_trajectory)
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def _update_policy(self):
        # A dual update uses each completed car exactly once. In particular,
        # rollout chunks that merely hit time_horizon never enter its
        # denominator, so a long car and short car each count as one outcome.
        completed = np.asarray(self.update_buffer[COST_COMPLETED_KEY], dtype=np.float32)
        completed_count = float(completed.sum())
        if completed_count > 0.0:
            costs = np.asarray(self.update_buffer[COST_KEY], dtype=np.float32)
            observed_rate = float(costs.sum() / completed_count)
            raw_costs = np.asarray(self.update_buffer[COST_RAW_KEY], dtype=np.float32)
            raw_observed_rate = float(raw_costs.sum() / completed_count)
            for key, value in self.optimizer.update_lagrange_multiplier(observed_rate).items():
                self._stats_reporter.add_stat(key, value)
            self._stats_reporter.add_stat("Safety/Raw collision-sentinel rate", raw_observed_rate)
            self._stats_reporter.add_stat(
                "Safety/Forced smoke cost enabled", 1.0 if self.force_smoke_cost else 0.0
            )
        return super()._update_policy()

    def create_optimizer(self) -> TorchOptimizer:
        return TorchPPOOptimizer(cast(TorchPolicy, self.policy), self.trainer_settings)  # type: ignore

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
        actor_kwargs: Dict[str, Any] = {"conditional_sigma": False, "tanh_squash": False}
        if self.shared_critic:
            actor_cls = SharedActorCritic
            actor_kwargs.update({
                "stream_names": [key.value for key, _ in self.trainer_settings.reward_signals.items()]
            })
        return TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )

    def get_policy(self, name_behavior_id: str) -> Policy:
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
