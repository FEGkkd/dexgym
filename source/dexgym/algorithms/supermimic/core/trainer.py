"""Alternating real-physics/model/BPTT/TD(lambda) trainer with injected adapters."""

import time
from dataclasses import dataclass

import torch

from .imagination import ImaginationEngine, critic_td_lambda_batch
from .optimization import frozen_parameters, optimize_critic, safe_gradient_step, validate_optimizer_ownership


@dataclass(frozen=True)
class TrainerConfig:
    mode: str = "pure_model_based"
    physics_steps_per_iteration: int = 1
    world_window: int = 8
    world_updates_per_iteration: int = 1
    world_batch: int = 64
    imagination_batch: int = 32
    actor_updates_per_iteration: int = 1
    max_imagination_horizon: int = 32
    world_warmup_updates: int = 30
    reference_pd_bootstrap_steps: int = 0
    boundary_probe_steps: int = 0
    termination_pos_weight: float = 1.0
    critic_epochs: int = 2
    critic_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    trust_interval: int = 25
    validation_batch: int = 64
    value_bound: bool = False
    actor_bootstrap_warmup_updates: int = 0

    def __post_init__(self):
        if self.mode not in ("pure_model_based", "teacher_bootstrap_model_based",
                             "supervised_init_model_based"):
            raise ValueError("Unknown SuperMimic training mode")
        if self.mode == "pure_model_based" and self.reference_pd_bootstrap_steps:
            raise ValueError("Strict Pure mode cannot collect teacher dynamics data")
        if self.world_window != 8 or self.max_imagination_horizon not in (1, 8, 32):
            raise ValueError("Current algorithm uses WM8 and a 1/8/32 imagination curriculum")
        for name in ("physics_steps_per_iteration", "world_updates_per_iteration", "world_batch",
                     "imagination_batch", "actor_updates_per_iteration", "critic_epochs",
                     "critic_minibatches", "trust_interval", "validation_batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (self.world_warmup_updates < 0 or self.reference_pd_bootstrap_steps < 0
                or self.boundary_probe_steps < 0 or self.termination_pos_weight <= 0):
            raise ValueError("World warmup, bootstrap or termination weight is invalid")
        if self.actor_bootstrap_warmup_updates < 0:
            raise ValueError("actor_bootstrap_warmup_updates must be nonnegative")


class SuperMimicTrainer:
    def __init__(self, *, collector, replay, robot, task, modules, optimizers,
                 trust_evaluator, config: TrainerConfig,
                 train_env_range, validation_env_range,
                 supervised_initializer=None, observation_noise=None):
        if collector.num_envs != replay.num_envs or robot.spec != replay.robot_spec or task.spec != replay.task_spec:
            raise ValueError("Collector, Replay, Robot and Task contracts differ")
        if set(optimizers) != {"world_model", "actor", "critic"}:
            raise ValueError("Trainer requires world, Actor and Critic optimizers")
        if not (0 <= train_env_range[0] < train_env_range[1] <= validation_env_range[0]
                < validation_env_range[1] <= replay.num_envs):
            raise ValueError("Training and held-out validation environment ranges overlap")
        if config.mode == "supervised_init_model_based" and supervised_initializer is None:
            raise ValueError("Supervised mode needs a task-supplied Actor initializer")
        self.collector, self.replay, self.robot, self.task = collector, replay, robot, task
        self.world_model, self.actor, self.critic = (
            modules.world_model, modules.actor, modules.critic)
        self.termination_head = modules.termination if task.termination_mode == "learned" else None
        validate_optimizer_ownership(modules, optimizers, task.termination_mode)
        existing_owner = getattr(self.world_model, "_supermimic_optimizer_owner", None)
        if existing_owner is not None and existing_owner is not optimizers["world_model"]:
            raise ValueError("Shared World Model parameters require one common World Model optimizer")
        self.world_model._supermimic_optimizer_owner = optimizers["world_model"]
        self.observation_noise = observation_noise
        if observation_noise is not None:
            observation_noise.validate(task.spec)
            if getattr(collector, "observation_noise", None) is not observation_noise:
                raise ValueError("Real Collector and Imagination must share the observation noise contract")
            self.actor.observation_noise_spec = observation_noise.spec
        self.optimizers = optimizers
        self.trust_evaluator, self.config = trust_evaluator, config
        self.train_env_range, self.validation_env_range = train_env_range, validation_env_range
        self.value_upper_bound = None
        if config.value_bound:
            if not hasattr(task, "reward_upper_bound"):
                raise ValueError("Value bound needs a task-proven reward upper bound")
            reward_bound = float(task.reward_upper_bound())
            if not 0 < reward_bound < float("inf"):
                raise ValueError("Task reward upper bound must be finite and positive")
            self.value_upper_bound = reward_bound / (1 - config.gamma)
        self.engine = ImaginationEngine(gamma=config.gamma, lam=config.lam,
                                        value_upper_bound=self.value_upper_bound)
        self.iteration = 0
        self.allowed_horizon = 0
        self.trust_report = {}
        self.counts = {"real_transitions": 0, "actor_transitions": 0,
                       "reference_pd_transitions": 0, "boundary_probe_transitions": 0,
                       "world_updates": 0, "actor_updates": 0, "critic_updates": 0,
                       "nonfinite_world": 0, "nonfinite_actor": 0,
                       "nonfinite_critic": 0, "trust_failures": 0}
        if supervised_initializer is not None and config.mode == "supervised_init_model_based":
            supervised_initializer(self.actor, self.optimizers["actor"])

    def _stamp(self):
        if self.replay.device.type == "cuda":
            torch.cuda.synchronize(self.replay.device)
        return time.perf_counter()

    def _update_normalizers_from_real(self, transition):
        low, high = self.train_env_range
        selection = slice(low, high)
        task_state = transition.task_state.index(selection)
        state, action, next_state = (transition.state[selection], transition.action[selection],
                                     transition.next_state[selection])
        context = transition.dynamics_context[selection]
        self.world_model.update_norms(state, action, context, next_state, self.robot)
        self.actor.update_norms(transition.history[selection], self.task.goal(state, task_state))
        self.critic.update_norms(self.task.critic_features(state, task_state))

    def collect(self):
        reports = []
        for _ in range(self.config.physics_steps_per_iteration):
            policy = ("reference_pd" if self.counts["reference_pd_transitions"] <
                      self.config.reference_pd_bootstrap_steps * self.replay.num_envs else
                      "boundary_probe" if self.counts["boundary_probe_transitions"] <
                      self.config.boundary_probe_steps * self.replay.num_envs else "actor")
            transition = self.collector.step(self.actor, policy=policy)
            self.replay.append(transition)
            self._update_normalizers_from_real(transition)
            self.counts["real_transitions"] += self.replay.num_envs
            key = {"actor": "actor_transitions", "reference_pd": "reference_pd_transitions",
                   "boundary_probe": "boundary_probe_transitions"}[policy]
            self.counts[key] += self.replay.num_envs
            reports.append({"policy": policy, "reward_mean": (
                float(transition.reward.mean()) if transition.reward is not None else None),
                "terminated": int(transition.terminated.sum()),
                "truncated": int(transition.truncated.sum())})
        return reports

    def update_world_model(self):
        try:
            window = self.replay.sample_windows(self.config.world_batch,
                                                self.config.world_window,
                                                env_range=self.train_env_range)
        except ValueError as error:
            return {"updated": False, "reason": str(error)}
        loss, errors = self.world_model.autoregressive_loss(
            window, self.robot, termination_head=self.termination_head,
            task=self.task if self.termination_head is not None else None,
            termination_pos_weight=self.config.termination_pos_weight)
        parameters = list(self.world_model.parameters())
        if self.termination_head is not None:
            parameters.extend(self.termination_head.parameters())
        result = safe_gradient_step(loss, parameters, self.optimizers["world_model"])
        self.counts["world_updates" if result["updated"] else "nonfinite_world"] += 1
        return {**result, "loss": float(loss.detach()),
                "valid_transition_count": int(window.mask.sum()),
                "prediction_error_mean": float(errors[window.mask.bool()].detach().mean()),
                "positive_termination_count": int((window.terminated & window.mask.bool()).sum())}

    def validate_world_model(self):
        self.trust_report = self.trust_evaluator.evaluate_available(
            self.world_model, self.replay, self.robot,
            batch=self.config.validation_batch, env_range=self.validation_env_range,
            source_id=0)
        self.allowed_horizon = min(self.config.max_imagination_horizon,
                                   self.trust_evaluator.trusted_horizon(self.trust_report))
        if not self.allowed_horizon:
            self.counts["trust_failures"] += 1
        return self.trust_report

    def _actor_bootstrap_weight(self) -> float:
        # Preserve the original 1/8/32 curriculum behavior by default. A new
        # task can ramp in the terminal Critic as it learns from imagined
        # returns; this avoids the random Critic directing the first BPTT step.
        if self.allowed_horizon == 1:
            return 0.0
        warmup = self.config.actor_bootstrap_warmup_updates
        return min(1.0, self.counts["actor_updates"] / warmup) if warmup else 1.0

    def update_actor_critic(self):
        if self.counts["world_updates"] < self.config.world_warmup_updates or not self.allowed_horizon:
            return {"updated": False, "reason": "world_warmup_or_trust"}
        try:
            start = self.replay.sample_starts(
                self.config.imagination_batch, horizon=self.allowed_horizon,
                valid_start=self.task.valid_imagination_start,
                env_range=self.train_env_range, source_id=0)
        except ValueError as error:
            return {"updated": False, "reason": str(error)}
        actor_began = self._stamp()
        bootstrap_weight = self._actor_bootstrap_weight()
        with frozen_parameters(self.world_model, self.critic, self.termination_head):
            rollout = self.engine.rollout(
                start, self.actor, self.world_model, self.critic, self.robot, self.task,
                horizon=self.allowed_horizon,
                bootstrap_weight=bootstrap_weight,
                termination_head=self.termination_head,
                observation_noise=self.observation_noise)
            actor_result = safe_gradient_step(rollout.actor_loss, self.actor.parameters(),
                                              self.optimizers["actor"])
        actor_finished = self._stamp()
        if not actor_result["updated"]:
            self.counts["nonfinite_actor"] += 1
            return {"updated": False, "reason": actor_result["reason"]}
        self.counts["actor_updates"] += 1
        features, targets = critic_td_lambda_batch(rollout, gamma=self.config.gamma,
                                                    lam=self.config.lam,
                                                    value_upper_bound=self.value_upper_bound)
        critic_result = optimize_critic(self.critic, self.optimizers["critic"],
                                        features, targets,
                                        epochs=self.config.critic_epochs,
                                        minibatches=self.config.critic_minibatches)
        critic_finished = self._stamp()
        self.counts["critic_updates"] += critic_result["updates"]
        self.counts["nonfinite_critic"] += critic_result["skipped_nonfinite"]
        return {"updated": True, "horizon": self.allowed_horizon,
                "actor_bootstrap_weight": bootstrap_weight,
                "actor_loss": float(rollout.actor_loss.detach()),
                "actor_grad_norm": actor_result["grad_norm"],
                "actor_bptt_seconds": actor_finished - actor_began,
                "critic_optimization_seconds": critic_finished - actor_finished,
                "critic": critic_result}

    def run_iteration(self):
        began = self._stamp()
        collected = self.collect()
        t_collect = self._stamp()
        world = [self.update_world_model() for _ in range(self.config.world_updates_per_iteration)]
        t_world = self._stamp()
        validated = False
        if (self.counts["world_updates"] >= self.config.world_warmup_updates
                and self.iteration % self.config.trust_interval == 0):
            self.validate_world_model()
            validated = True
        t_trust = self._stamp()
        actor = [self.update_actor_critic() for _ in range(self.config.actor_updates_per_iteration)]
        t_actor = self._stamp()
        self.iteration += 1
        return {"iteration": self.iteration, "collection": collected, "world": world,
                "actor": actor, "trust": self.trust_report if validated else None,
                "allowed_horizon": self.allowed_horizon, "counts": self.counts.copy(),
                "phase_seconds": {"collection": t_collect - began,
                                  "world": t_world - t_collect,
                                  "trust": t_trust - t_world,
                                  "actor_critic": t_actor - t_trust}}
