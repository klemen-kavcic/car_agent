"""PPO optimizer with a separate, undiscounted collision-cost critic.

This file is bind-mounted over ML-Agents 1.1.0's
``mlagents/trainers/ppo/optimizer_torch.py`` only for constrained-PPO runs.
It intentionally keeps the ordinary reward critic and reward PPO loss intact.
The additional critic estimates P(VehicleCollision before episode end), while
the actor receives a Lagrangian-weighted cost surrogate in addition to the
normal reward surrogate.
"""

import os
from typing import Dict, cast

import attr

from mlagents.torch_utils import torch, default_device
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.networks import ValueNetwork
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


COST_KEY = "cpo_collision_cost"
COST_VALUE_KEY = "cpo_cost_value_estimate"
COST_RETURN_KEY = "cpo_cost_return"
COST_ADVANTAGE_KEY = "cpo_cost_advantage"
COST_COMPLETED_KEY = "cpo_completed_car"
# Retains the unmodified Unity terminal marker when a deliberately forced-cost
# smoke test is active.  This is diagnostic data only; normal CPO runs have
# COST_RAW_KEY == COST_KEY.
COST_RAW_KEY = "cpo_raw_collision_cost"


def _environment_flag(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc


@attr.s(auto_attribs=True)
class PPOSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR


class TorchPPOOptimizer(TorchOptimizer):
    """Ordinary PPO plus one collision-probability value network."""

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: PPOSettings = cast(PPOSettings, trainer_settings.hyperparameters)
        params = list(self.policy.actor.parameters())
        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            self._critic.to(default_device())
            params += list(self._critic.parameters())

        # The cost network is deliberately separate even when reward uses a
        # shared critic. A collision probability is not a reward value and we do
        # not want its regression target to distort the reward/value streams.
        self._cost_critic = ValueNetwork(
            [COST_KEY],
            policy.behavior_spec.observation_specs,
            network_settings=trainer_settings.network_settings,
        )
        self._cost_critic.to(default_device())
        params += list(self._cost_critic.parameters())

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )
        self.optimizer = torch.optim.Adam(params, lr=self.hyperparameters.learning_rate)
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }
        self.stream_names = list(self.reward_signals.keys())

        self.cost_target = _environment_flag("CPO_TARGET_CRASH_RATE", 0.08)
        self.dual_learning_rate = _environment_flag("CPO_DUAL_LR", 0.2)
        self.lambda_max = _environment_flag("CPO_LAMBDA_MAX", 10.0)
        self.lagrange_multiplier = max(
            0.0,
            min(_environment_flag("CPO_LAMBDA_INIT", 0.0), self.lambda_max),
        )
        if not 0.0 <= self.cost_target <= 1.0:
            raise RuntimeError("CPO_TARGET_CRASH_RATE must lie in [0, 1]")
        if self.dual_learning_rate < 0.0 or self.lambda_max < 0.0:
            raise RuntimeError("CPO_DUAL_LR and CPO_LAMBDA_MAX must be non-negative")

    @property
    def critic(self):
        return self._critic

    @property
    def cost_critic(self):
        return self._cost_critic

    def get_cost_value_estimates(self, batch: AgentBuffer, next_obs, done: bool):
        """Return V_cost for a trajectory and its bootstrap value.

        Cost uses no recurrence in this first implementation. The traffic PPO
        configurations are feed-forward; fail loudly instead of silently using
        incorrect LSTM memories in a future configuration.
        """
        if self.policy.use_recurrent:
            raise RuntimeError("Constrained PPO currently supports feed-forward policies only")
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = [
            ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)
        ]
        next_obs_t = [ModelUtils.list_to_tensor(obs).unsqueeze(0) for obs in next_obs]
        with torch.no_grad():
            values, _ = self.cost_critic.critic_pass(
                current_obs, None, sequence_length=batch.num_experiences
            )
            next_values, _ = self.cost_critic.critic_pass(next_obs_t, None, sequence_length=1)
        value = ModelUtils.to_numpy(values[COST_KEY])
        next_value = float(ModelUtils.to_numpy(next_values[COST_KEY])[0])
        # Every completed car has an explicit terminal cost (1 collision, 0
        # otherwise), so no bootstrap is allowed on a completed trajectory.
        if done:
            next_value = 0.0
        return value, next_value

    def update_lagrange_multiplier(self, observed_cost_rate: float) -> Dict[str, float]:
        """One projected dual-ascent step per rollout, not per minibatch."""
        self.lagrange_multiplier = min(
            self.lambda_max,
            max(
                0.0,
                self.lagrange_multiplier
                + self.dual_learning_rate * (observed_cost_rate - self.cost_target),
            ),
        )
        return {
            "Safety/Completed-car collision rate": observed_cost_rate,
            "Safety/Collision target": self.cost_target,
            "Safety/Lagrange multiplier": self.lagrange_multiplier,
        }

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(batch[RewardSignalUtil.returns_key(name)])

        old_cost_values = {COST_KEY: ModelUtils.list_to_tensor(batch[COST_VALUE_KEY])}
        cost_returns = {COST_KEY: ModelUtils.list_to_tensor(batch[COST_RETURN_KEY])}
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)]
        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)
        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if memories:
            memories = torch.stack(memories).unsqueeze(0)
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length)
        ]
        if value_memories:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        run_out = self.policy.actor.get_stats(
            current_obs, actions, masks=act_masks, memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        log_probs = run_out["log_probs"].flatten()
        entropy = run_out["entropy"]
        values, _ = self.critic.critic_pass(
            current_obs, memories=value_memories, sequence_length=self.policy.sequence_length
        )
        cost_values, _ = self.cost_critic.critic_pass(
            current_obs, memories=None, sequence_length=self.policy.sequence_length
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        cost_value_loss = ModelUtils.trust_region_value_loss(
            cost_values, old_cost_values, cost_returns, decay_eps, loss_masks
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs, old_log_probs, loss_masks, decay_eps,
        )

        # PPO's reward surrogate is a minimisation loss (-min surrogate). For
        # the cost objective we minimise cost, hence +max(ratio*A_cost,
        # clipped_ratio*A_cost): the conservative counterpart to reward PPO.
        cost_advantages = ModelUtils.list_to_tensor(batch[COST_ADVANTAGE_KEY])
        # ``log_probs`` has one column per continuous action.  Collision cost
        # is one scalar per state/action *vector*, not one cost per pedal or
        # steering component.  PPO therefore needs the joint action ratio:
        # exp(sum_a(log pi(a) - log pi_old(a))).  Using the uncollapsed
        # [batch, 2] tensor here tries to multiply it by [batch] advantages,
        # which both fails broadcasting and would apply the same trajectory
        # cost independently to each action dimension.
        joint_log_ratio = torch.sum(log_probs - old_log_probs, dim=1)
        ratio = torch.exp(joint_log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - decay_eps, 1.0 + decay_eps)
        cost_policy_loss = ModelUtils.masked_mean(
            torch.maximum(ratio * cost_advantages, clipped_ratio * cost_advantages),
            loss_masks,
        )
        loss = (
            policy_loss
            + self.lagrange_multiplier * cost_policy_loss
            + 0.5 * (value_loss + cost_value_loss)
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Safety/Cost critic loss": cost_value_loss.item(),
            "Safety/Cost policy loss": cost_policy_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
            "Optimizer:cpo_cost_critic": self._cost_critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
