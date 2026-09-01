# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import threading
from typing import Dict, Optional, Set, List
from collections import defaultdict

import numpy as np

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from mlagents_envs.timers import (
    hierarchical_timer,
    timed,
    get_timer_stack_for_thread,
    merge_gauges,
)
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.agent_processor import AgentManager
from mlagents.trainers.settings import ConstantSettings
from mlagents import torch_utils
from mlagents.torch_utils.globals import get_rank

# Module reference (not `from mlagents.trainers.stats import LATEST_CRASH_RATE`) - the latter
# would only snapshot the value at import time and never see stats_patched.py's later updates,
# since module attribute access is the only way to see a name rebound after import.
import mlagents.trainers.stats as stats_module


class TrainerController:
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        output_path: str,
        run_id: str,
        param_manager: EnvironmentParameterManager,
        train: bool,
        training_seed: int,
    ):
        """
        :param output_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param param_manager: EnvironmentParameterManager object which stores information about all
        environment parameters.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Torch random number generation.
        :param threaded: Whether or not to run trainers in a separate thread. Disable for testing/debugging.
        """
        self.trainers: Dict[str, Trainer] = {}
        self.brain_name_to_identifier: Dict[str, Set] = defaultdict(set)
        self.trainer_factory = trainer_factory
        self.output_path = output_path
        self.logger = get_logger(__name__)
        self.run_id = run_id
        self.train_model = train
        self.param_manager = param_manager
        self.ghost_controller = self.trainer_factory.ghost_controller
        self.registered_behavior_ids: Set[str] = set()

        self.trainer_threads: List[threading.Thread] = []
        self.kill_trainers = False
        np.random.seed(training_seed)
        torch_utils.torch.manual_seed(training_seed)
        self.rank = get_rank()

        # Adaptive Lagrangian-style penalty state (see car_agent.yaml's lagrangian_*/
        # sensor_terminal_lagrangian_* keys) - one independent (lambda, last-consumed-rate) pair
        # per constraint, keyed by constraint name. lambda starts uninitialised (None) so the
        # first update seeds it from that constraint's lambda_init rather than some arbitrary
        # default; last-consumed-rate is tracked so a lesson-change-free, data-free loop iteration
        # doesn't redundantly recompute and re-push the same value every single iteration between
        # summary windows. "maxstep" (lagrangian_maxstep_*) was removed, it never converged to a
        # useful policy. "sensor_terminal" drives carAgent.cs's dense forward-sensor terminal-
        # avoidance shaping (see CarSensor.isForwardLineSensor) - deliberately its OWN constraint,
        # not sharing "crash"'s lambda, since it targets a different metric (mean per-step sensor
        # exposure, not episode-outcome crash rate - the shaping reward itself telescopes to ~0
        # over a full approach/retreat, so it can't be used as its own target).
        self._lagrangian_lambda: Dict[str, Optional[float]] = {
            "crash": None, "sensor_terminal": None, "maxstep_timepenalty": None,
        }
        self._lagrangian_last_consumed_rate: Dict[str, Optional[float]] = {
            "crash": None, "sensor_terminal": None, "maxstep_timepenalty": None,
        }
        # One-time "just activated" log per constraint, once curr_step first clears the bootstrap
        # gate (see _bootstrap_total_steps) - lets a run confirm from the log alone that Lagrangian
        # actually started, without having to cross-reference step counts by hand.
        self._lagrangian_gate_logged: Dict[str, bool] = {
            "crash": False, "sensor_terminal": False, "maxstep_timepenalty": False,
        }

        # Adaptive step-time budget state (car_agent.yaml's adaptive_step_budget_* keys) - kept
        # separate from _lagrangian_lambda/_lagrangian_last_consumed_rate above despite the
        # similar shape: this is a self-paced curriculum on Agent.MaxStep (task structure), not a
        # Lagrangian-style reward penalty - see _update_adaptive_step_budget.
        self._step_budget: Optional[float] = None
        self._step_budget_last_consumed_rate: Optional[float] = None
        self._step_budget_gate_logged: bool = False

    @timed
    def _save_models(self):
        """
        Saves current model to checkpoint folder.
        """
        if self.rank is not None and self.rank != 0:
            return

        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.debug("Saved Model")

    @staticmethod
    def _create_output_path(output_path):
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except Exception:
            raise UnityEnvironmentException(
                f"The folder {output_path} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly."
            )

    @timed
    def _reset_env(self, env_manager: EnvManager) -> None:
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        new_config = self.param_manager.get_current_samplers()
        env_manager.reset(config=new_config)
        # Register any new behavior ids that were generated on the reset.
        self._register_new_behaviors(env_manager, env_manager.first_step_infos)

    def _not_done_training(self) -> bool:
        return (
            any(t.should_still_train for t in self.trainers.values())
            or not self.train_model
        ) or len(self.trainers) == 0

    def _create_trainer_and_manager(
        self, env_manager: EnvManager, name_behavior_id: str
    ) -> None:

        parsed_behavior_id = BehaviorIdentifiers.from_name_behavior_id(name_behavior_id)
        brain_name = parsed_behavior_id.brain_name
        trainerthread = None
        if brain_name in self.trainers:
            trainer = self.trainers[brain_name]
        else:
            trainer = self.trainer_factory.generate(brain_name)
            self.trainers[brain_name] = trainer
            if trainer.threaded:
                # Only create trainer thread for new trainers
                trainerthread = threading.Thread(
                    target=self.trainer_update_func, args=(trainer,), daemon=True
                )
                self.trainer_threads.append(trainerthread)
            env_manager.on_training_started(
                brain_name, self.trainer_factory.trainer_config[brain_name]
            )

        policy = trainer.create_policy(
            parsed_behavior_id,
            env_manager.training_behaviors[name_behavior_id],
        )
        trainer.add_policy(parsed_behavior_id, policy)

        agent_manager = AgentManager(
            policy,
            name_behavior_id,
            trainer.stats_reporter,
            trainer.parameters.time_horizon,
            threaded=trainer.threaded,
        )
        env_manager.set_agent_manager(name_behavior_id, agent_manager)
        env_manager.set_policy(name_behavior_id, policy)
        self.brain_name_to_identifier[brain_name].add(name_behavior_id)

        trainer.publish_policy_queue(agent_manager.policy_queue)
        trainer.subscribe_trajectory_queue(agent_manager.trajectory_queue)

        # Only start new trainers
        if trainerthread is not None:
            trainerthread.start()

    def _create_trainers_and_managers(
        self, env_manager: EnvManager, behavior_ids: Set[str]
    ) -> None:
        for behavior_id in behavior_ids:
            self._create_trainer_and_manager(env_manager, behavior_id)

    @timed
    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_output_path(self.output_path)
        try:
            # Initial reset
            self._reset_env(env_manager)
            self.param_manager.log_current_lesson()
            while self._not_done_training():
                n_steps = self.advance(env_manager)
                for _ in range(n_steps):
                    self.reset_env_if_ready(env_manager)
            # Stop advancing trainers
            self.join_threads()
        except (
            KeyboardInterrupt,
            UnityCommunicationException,
            UnityEnvironmentException,
            UnityCommunicatorStoppedException,
        ) as ex:
            self.join_threads()
            self.logger.info(
                "Learning was interrupted. Please wait while the graph is generated."
            )
            if isinstance(ex, KeyboardInterrupt) or isinstance(
                ex, UnityCommunicatorStoppedException
            ):
                pass
            else:
                # If the environment failed, we want to make sure to raise
                # the exception so we exit the process with an return code of 1.
                raise ex
        finally:
            if self.train_model:
                self._save_models()

    def end_trainer_episodes(self) -> None:
        # Reward buffers reset takes place only for curriculum learning
        # else no reset.
        for trainer in self.trainers.values():
            trainer.end_episode()

    # Global-step point (same units as curr_step below - summed decisions across all workers,
    # matching the trainer's own "Step:" counter and total_steps/max_steps) at which the bootstrap
    # curriculum finishes - carAgent.cs's bootstrap_stage_*_steps keys are already defined in this
    # same GLOBAL convention (see car_agent.yaml's own comment on them), so no /training_num_envs
    # conversion is needed here the way carAgent.cs itself has to do it (that conversion is only
    # needed there because Academy.Instance.TotalStepCount is a per-process LOCAL tick count, not
    # this GLOBAL decision count). Returns 0 (no gating) if bootstrap is disabled.
    def _bootstrap_total_steps(self, samplers) -> float:
        enabled = getattr(samplers.get("bootstrap_curriculum_enabled"), "value", 1.0)
        if enabled < 0.5:
            return 0.0
        stage_keys = [
            "bootstrap_stage_front_steps",
            "bootstrap_stage_behind_steps",
            "bootstrap_stage_deadend_steps",
            "bootstrap_stage_side_steps",
        ]
        return sum(getattr(samplers.get(k), "value", 0.0) for k in stage_keys)

    # Shared by both Lagrangian-style constraints (crash and maxstep-timeout - see
    # reset_env_if_ready below). Reads its config straight from the same environment_parameters
    # yaml block as everything else (car_agent.yaml), via self.param_manager - no new
    # config-loading mechanism. Reads whichever rate stats_patched.py's ConsoleWriter caches each
    # summary window (stats_module.LATEST_CRASH_RATE / LATEST_MAXSTEP_RATE) and, only when that's
    # a genuinely new value (not just this method being called again on a data-free iteration),
    # performs one dual-ascent step and pushes the result to the running environments as the given
    # penalty parameter. No-ops entirely if this constraint's *_enabled key isn't set - ordinary
    # training (these keys absent from the yaml) is completely unaffected.
    #
    # Gated until curr_step clears _bootstrap_total_steps: during bootstrap the crash/maxstep rate
    # reflects artificial drills (e.g. DeadEnd's walled reverse-out pocket), not the real task, and
    # is naturally far higher than any realistic target - letting lambda react to that noise let it
    # start climbing before training even reached the real task, and (given the missing clamp below
    # was the other half of the bug) never recover.
    #
    # lambda is now clamped both directions (lambda_max_key), matching the pattern
    # _update_adaptive_step_budget already used - previously only floored at 0, so a persistently
    # unmet target (measured_rate never converging to target_key, e.g. a target_crash_rate far
    # below what any functional policy can achieve) let lambda climb unboundedly every update
    # instead of saturating, which is what actually happened in the first Lagrangian grid runs
    # (target 5% crash rate vs. baseline's own natural ~60%+ crash rate -> lambda diverged past
    # 100-500, dwarfing reward_goal and collapsing the policy into "never attempt anything").
    def _update_lagrangian_constraint(
        self,
        env_manager: EnvManager,
        *,
        constraint_name: str,
        enabled_key: str,
        target_key: str,
        lambda_init_key: str,
        lambda_lr_key: str,
        lambda_max_key: str,
        latest_rate: Optional[float],
        penalty_param_name: str,
        curr_step: float,
    ) -> None:
        samplers = self.param_manager.get_current_samplers()
        enabled_sampler = samplers.get(enabled_key)
        if enabled_sampler is None or getattr(enabled_sampler, "value", 0.0) < 0.5:
            return

        bootstrap_total = self._bootstrap_total_steps(samplers)
        if curr_step < bootstrap_total:
            return
        if not self._lagrangian_gate_logged[constraint_name]:
            self._lagrangian_gate_logged[constraint_name] = True
            self.logger.info(
                f"[Lagrangian:{constraint_name}] bootstrap finished at step {bootstrap_total:.0f} - "
                f"now active (curr_step={curr_step:.0f})."
            )

        if latest_rate is None or latest_rate == self._lagrangian_last_consumed_rate[constraint_name]:
            return
        self._lagrangian_last_consumed_rate[constraint_name] = latest_rate

        target = samplers[target_key].value
        lr = samplers[lambda_lr_key].value
        lambda_max = getattr(samplers.get(lambda_max_key), "value", float("inf"))
        if self._lagrangian_lambda[constraint_name] is None:
            self._lagrangian_lambda[constraint_name] = samplers[lambda_init_key].value

        self._lagrangian_lambda[constraint_name] = min(
            lambda_max,
            max(0.0, self._lagrangian_lambda[constraint_name] + lr * (latest_rate - target)),
        )
        lam = self._lagrangian_lambda[constraint_name]

        self.logger.info(
            f"[Lagrangian:{constraint_name}] rate={latest_rate:.3f} target={target:.3f} "
            f"lambda={lam:.3f} (max={lambda_max:.3f}) -> {penalty_param_name}={-lam:.3f}"
        )
        # Same StatsReporter.set_stat pattern already used for Environment/Lesson Number/* in
        # advance() below - flows straight into ML-Agents' existing TensorboardWriter/CSVWriter
        # (no further patching needed), so lambda is graphable in TensorBoard under
        # "Environment/Lagrangian Lambda/<crash|maxstep>" alongside everything else. Pushed once
        # per constraint per summary window (whenever this method doesn't early-return above),
        # same cadence as the underlying crash/maxstep rate it's reacting to.
        for trainer in self.trainers.values():
            trainer.stats_reporter.set_stat(f"Environment/Lagrangian Lambda/{constraint_name}", lam)
        env_manager.set_env_parameters({penalty_param_name: ConstantSettings(value=-lam)})

    def _update_lagrangian_penalties(self, env_manager: EnvManager, curr_step: float) -> None:
        self._update_lagrangian_constraint(
            env_manager,
            constraint_name="crash",
            enabled_key="lagrangian_penalty_enabled",
            target_key="lagrangian_target_crash_rate",
            lambda_init_key="lagrangian_lambda_init",
            lambda_lr_key="lagrangian_lambda_lr",
            lambda_max_key="lagrangian_lambda_max",
            latest_rate=stats_module.LATEST_CRASH_RATE,
            penalty_param_name="reward_terminal_penalty",
            curr_step=curr_step,
        )
        self._update_lagrangian_constraint(
            env_manager,
            constraint_name="sensor_terminal",
            enabled_key="sensor_terminal_lagrangian_enabled",
            target_key="sensor_terminal_target_exposure",
            lambda_init_key="sensor_terminal_lambda_init",
            lambda_lr_key="sensor_terminal_lambda_lr",
            lambda_max_key="sensor_terminal_lambda_max",
            latest_rate=stats_module.LATEST_SENSOR_TERMINAL_EXPOSURE,
            penalty_param_name="sensor_terminal_penalty_lambda",
            curr_step=curr_step,
        )
        # Re-adds maxstep-rate-driven adaptive tuning, removed previously when it targeted
        # reward_maxstep_penalty (a one-time end-of-episode charge) and "never converged to a
        # useful policy" (see carAgent.cs's rwMaxStepPenalty comment) - this version targets
        # reward_time_penalty instead (the flat PER-STEP cost, already accumulating the whole
        # episode), a continuous lever that should react faster/more smoothly to maxstep_rate than
        # a one-shot terminal charge could. Independent constraint/lambda from "crash" despite both
        # ultimately feeding a "reward_*_penalty" parameter - different target metric (maxstep_rate,
        # not crash_rate) and different parameter (reward_time_penalty, not reward_terminal_penalty).
        self._update_lagrangian_constraint(
            env_manager,
            constraint_name="maxstep_timepenalty",
            enabled_key="lagrangian_maxstep_enabled",
            target_key="lagrangian_target_maxstep_rate",
            lambda_init_key="lagrangian_maxstep_lambda_init",
            lambda_lr_key="lagrangian_maxstep_lambda_lr",
            lambda_max_key="lagrangian_maxstep_lambda_max",
            latest_rate=stats_module.LATEST_MAXSTEP_RATE,
            penalty_param_name="reward_time_penalty",
            curr_step=curr_step,
        )

    # Self-paced curriculum on Agent.MaxStep (carAgent.cs reads max_step_budget every episode) -
    # NOT a Lagrangian penalty despite reusing maxstep_target_rate as its target and a
    # similar-looking update rule: this adjusts the episode's actual time budget/task structure,
    # not a reward coefficient. Clamped both directions (adaptive_step_budget_min/max), unlike the
    # penalty lambda which only floors at 0 - a step budget can't be allowed to grow unboundedly
    # either. No-ops entirely if adaptive_step_budget_enabled isn't set.
    def _update_adaptive_step_budget(self, env_manager: EnvManager, curr_step: float) -> None:
        samplers = self.param_manager.get_current_samplers()
        enabled_sampler = samplers.get("adaptive_step_budget_enabled")
        if enabled_sampler is None or getattr(enabled_sampler, "value", 0.0) < 0.5:
            return

        bootstrap_total = self._bootstrap_total_steps(samplers)
        if curr_step < bootstrap_total:
            return
        if not self._step_budget_gate_logged:
            self._step_budget_gate_logged = True
            self.logger.info(
                f"[StepBudget] bootstrap finished at step {bootstrap_total:.0f} - now active "
                f"(curr_step={curr_step:.0f})."
            )

        rate = stats_module.LATEST_MAXSTEP_RATE
        if rate is None or rate == self._step_budget_last_consumed_rate:
            return
        self._step_budget_last_consumed_rate = rate

        target = samplers["maxstep_target_rate"].value
        lr = samplers["adaptive_step_budget_lr"].value
        lo = samplers["adaptive_step_budget_min"].value
        hi = samplers["adaptive_step_budget_max"].value
        if self._step_budget is None:
            self._step_budget = samplers["max_step_budget"].value

        self._step_budget = min(hi, max(lo, self._step_budget + lr * (rate - target)))

        self.logger.info(
            f"[StepBudget] maxstep_rate={rate:.3f} target={target:.3f} "
            f"max_step_budget={self._step_budget:.0f}"
        )
        # See the matching comment in _update_lagrangian_constraint - same set_stat pattern,
        # graphable in TensorBoard under "Environment/Adaptive Step Budget".
        for trainer in self.trainers.values():
            trainer.stats_reporter.set_stat("Environment/Adaptive Step Budget", self._step_budget)
        env_manager.set_env_parameters(
            {"max_step_budget": ConstantSettings(value=self._step_budget)}
        )

    def reset_env_if_ready(self, env: EnvManager) -> None:
        # Get the sizes of the reward buffers.
        reward_buff = {k: list(t.reward_buffer) for (k, t) in self.trainers.items()}
        curr_step = {k: int(t.get_step) for (k, t) in self.trainers.items()}
        max_step = {k: int(t.get_max_steps) for (k, t) in self.trainers.items()}
        # Attempt to increment the lessons of the brains who
        # were ready.
        updated, param_must_reset = self.param_manager.update_lessons(
            curr_step, max_step, reward_buff
        )
        if updated:
            for trainer in self.trainers.values():
                trainer.reward_buffer.clear()
        # If ghost trainer swapped teams
        ghost_controller_reset = self.ghost_controller.should_reset()
        if param_must_reset or ghost_controller_reset:
            self._reset_env(env)  # This reset also sends the new config to env
            self.end_trainer_episodes()
        elif updated:
            env.set_env_parameters(self.param_manager.get_current_samplers())

        # Single global step (decisions summed across all workers, matching the trainer's own
        # "Step:" counter) - used only to gate Lagrangian/step-budget until bootstrap finishes (see
        # _bootstrap_total_steps). Falls back to 0 if there are no trainers yet.
        current_global_step = max(curr_step.values()) if curr_step else 0
        self._update_lagrangian_penalties(env, current_global_step)
        self._update_adaptive_step_budget(env, current_global_step)

    @timed
    def advance(self, env_manager: EnvManager) -> int:
        # Get steps
        with hierarchical_timer("env_step"):
            new_step_infos = env_manager.get_steps()
            self._register_new_behaviors(env_manager, new_step_infos)
            num_steps = env_manager.process_steps(new_step_infos)

        # Report current lesson for each environment parameter
        for (
            param_name,
            lesson_number,
        ) in self.param_manager.get_current_lesson_number().items():
            for trainer in self.trainers.values():
                trainer.stats_reporter.set_stat(
                    f"Environment/Lesson Number/{param_name}", lesson_number
                )

        for trainer in self.trainers.values():
            if not trainer.threaded:
                with hierarchical_timer("trainer_advance"):
                    trainer.advance()

        return num_steps

    def _register_new_behaviors(
        self, env_manager: EnvManager, step_infos: List[EnvironmentStep]
    ) -> None:
        """
        Handle registration (adding trainers and managers) of new behaviors ids.
        :param env_manager:
        :param step_infos:
        :return:
        """
        step_behavior_ids: Set[str] = set()
        for s in step_infos:
            step_behavior_ids |= set(s.name_behavior_ids)
        new_behavior_ids = step_behavior_ids - self.registered_behavior_ids
        self._create_trainers_and_managers(env_manager, new_behavior_ids)
        self.registered_behavior_ids |= step_behavior_ids

    def join_threads(self, timeout_seconds: float = 1.0) -> None:
        """
        Wait for threads to finish, and merge their timer information into the main thread.
        :param timeout_seconds:
        :return:
        """
        self.kill_trainers = True
        for t in self.trainer_threads:
            try:
                t.join(timeout_seconds)
            except Exception:
                pass

        with hierarchical_timer("trainer_threads") as main_timer_node:
            for trainer_thread in self.trainer_threads:
                thread_timer_stack = get_timer_stack_for_thread(trainer_thread)
                if thread_timer_stack:
                    main_timer_node.merge(
                        thread_timer_stack.root,
                        root_name="thread_root",
                        is_parallel=True,
                    )
                    merge_gauges(thread_timer_stack.gauges)

    def trainer_update_func(self, trainer: Trainer) -> None:
        while not self.kill_trainers:
            with hierarchical_timer("trainer_advance"):
                trainer.advance()
