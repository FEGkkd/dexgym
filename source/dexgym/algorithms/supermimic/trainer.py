"""Task-independent Newton collection and learned-model SuperMimic updates."""

from dataclasses import dataclass
import time

import torch

from .critic_optimizer import optimize_critic
from .optimization import safe_gradient_step
from .rewards import REWARD_PROFILES, reward_upper_bound
from .trust_evaluator import trusted_imagination_horizon


@dataclass(frozen=True)
class TrainerConfig:
    mode: str = "pure_model_based"
    physics_steps_per_iteration: int = 1
    world_updates_per_iteration: int = 1
    world_batch: int = 128
    imagination_batch: int = 64
    actor_updates_per_iteration: int = 1
    max_imagination_horizon: int = 32
    world_warmup_updates: int = 100
    reference_pd_bootstrap_steps: int = 0
    boundary_probe_steps: int = 0
    termination_pos_weight: float = 1.0
    critic_epochs: int = 2
    critic_minibatches: int = 4
    trust_interval: int = 25
    validation_batch: int = 64

    def __post_init__(self):
        if self.mode not in ("pure_model_based", "teacher_bootstrap_model_based", "supervised_init_model_based"):
            raise ValueError("Unknown SuperMimic startup mode")
        if self.reference_pd_bootstrap_steps and self.mode == "pure_model_based":
            raise ValueError("Strict pure mode cannot collect reference PD bootstrap data")
        if self.boundary_probe_steps < 0 or self.termination_pos_weight <= 0:
            raise ValueError("Boundary collection and positive class weight must be valid")
        for name in ("physics_steps_per_iteration", "world_updates_per_iteration", "world_batch",
                     "imagination_batch", "actor_updates_per_iteration", "max_imagination_horizon",
                     "critic_epochs", "critic_minibatches", "trust_interval", "validation_batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


class SuperMimicTrainer:
    """A single alternating learner used by every robot/task adapter."""

    def __init__(self, *, collector, replay, robot, task, world_model, actor, critic,
                 world_optimizer, actor_optimizer, critic_optimizer, trust_evaluator,
                 config: TrainerConfig, train_env_range=None, validation_env_range=None):
        self.collector, self.replay, self.robot, self.task = collector, replay, robot, task
        self.world_model, self.actor, self.critic = world_model, actor, critic
        self.world_optimizer, self.actor_optimizer, self.critic_optimizer = (
            world_optimizer, actor_optimizer, critic_optimizer)
        self.trust_evaluator, self.config = trust_evaluator, config
        self.train_env_range = train_env_range
        self.validation_env_range = validation_env_range
        self.iteration = 0
        self.counts = {"real_transitions": 0, "actor_transitions": 0, "reference_pd_transitions": 0,
                       "boundary_probe_transitions": 0,
                       "world_updates": 0, "actor_updates": 0, "critic_updates": 0,
                       "nonfinite_world": 0, "nonfinite_actor": 0, "nonfinite_critic": 0,
                       "trust_failures": 0}
        self.trust_report = {}
        self.boundary_termination_report = {}
        self.allowed_horizon = 0
        self.resume_recollection_steps_required = 0
        self.resume_boundary_probe_steps_remaining = 0

    def _update_normalizers_from_real(self):
        slot = (self.replay.steps - 1) % self.replay.capacity
        state, action, next_state = (self.replay.state[slot], self.replay.action[slot],
                                      self.replay.next_state[slot])
        frame = self.replay.frame[slot]
        goal = self.task.future_goal(frame)
        self.world_model.update_norms(state, action, next_state)
        self.actor.update_norms(self.replay.history[slot], goal)
        self.critic.update_norms(state, self.task.reference.data["q_arm"][frame], goal[:, 0])

    def collect(self):
        logs = []
        for _ in range(self.config.physics_steps_per_iteration):
            policy = ("boundary_probe" if self.resume_boundary_probe_steps_remaining > 0 else
                      "reference_pd" if self.counts["reference_pd_transitions"] <
                      self.config.reference_pd_bootstrap_steps * self.replay.num_envs else
                      "boundary_probe" if self.counts["boundary_probe_transitions"] <
                      self.config.boundary_probe_steps * self.replay.num_envs else "actor")
            result = self.collector.step(policy=policy)
            if self.resume_boundary_probe_steps_remaining > 0:
                self.resume_boundary_probe_steps_remaining -= 1
            self._update_normalizers_from_real()
            count = self.replay.num_envs
            self.counts["real_transitions"] += count
            self.counts[{"actor": "actor_transitions", "reference_pd": "reference_pd_transitions",
                         "boundary_probe": "boundary_probe_transitions"}[policy]] += count
            logs.append(result)
        return logs

    def update_world_model(self):
        try:
            window = self.replay.sample_windows(self.config.world_batch, 8,
                                                env_range=self.train_env_range)
        except ValueError:
            return {"updated": False, "reason": "empty_replay"}
        valid = window["mask"].bool()
        if self.task.termination_mode == "learned":
            frame = torch.where(valid, window["frame"], torch.zeros_like(window["frame"]))
            window["termination_goal"] = self.task.arrival_goal(frame.reshape(-1)).reshape(
                self.config.world_batch, 8, -1)
        else:
            window.pop("terminated")
        loss, errors = self.world_model.autoregressive_loss(
            window, self.robot.action_map, update_normalizers=False,
            termination_pos_weight=self.config.termination_pos_weight,
            termination_module=getattr(self.task, "termination_module", None))
        parameters = list(self.world_model.parameters())
        if getattr(self.task, "termination_module", None) is not None:
            parameters.extend(self.task.termination_module.parameters())
        result = safe_gradient_step(loss, parameters, self.world_optimizer)
        if result["updated"]:
            self.counts["world_updates"] += 1
        else:
            self.counts["nonfinite_world"] += 1
        return {**result, "loss": float(loss.detach()),
                "valid_length_histogram": self.replay.valid_length_histogram(window),
                "prediction_error_mean": float(errors[valid].detach().mean()),
                "terminated_positive_count": int((window.get("terminated", torch.zeros_like(valid)) & valid).sum()),
                "terminated_negative_count": int(valid.sum() -
                                                   (window.get("terminated", torch.zeros_like(valid)) & valid).sum())}

    def validate_model(self):
        report = self.trust_evaluator.evaluate_available(
            self.world_model, self.replay, self.robot.action_map,
            batch=self.config.validation_batch, env_range=self.validation_env_range,
            source_id=0, termination_mode=self.task.termination_mode,
            termination_thresholds=self.task.thresholds,
            termination_module=getattr(self.task, "termination_module", None))
        self.trust_report = report
        if self.task.termination_mode == "learned" and self.counts["boundary_probe_transitions"]:
            try:
                self.boundary_termination_report = self.trust_evaluator.evaluate_horizon(
                    self.world_model, self.replay, self.robot.action_map, horizon=1,
                    batch=max(1024, self.config.validation_batch), env_range=self.validation_env_range,
                    source_id=2, termination_module=getattr(self.task, "termination_module", None))
            except ValueError as error:
                self.boundary_termination_report = {"available": False, "reason": str(error)}
        self.allowed_horizon = min(self.config.max_imagination_horizon,
                                   trusted_imagination_horizon(report))
        if not self.allowed_horizon:
            self.counts["trust_failures"] += 1
        return report

    def update_actor_critic(self):
        horizon = self.allowed_horizon
        if self.replay.steps < self.resume_recollection_steps_required:
            return {"updated": False, "reason": "resume_replay_recollection"}
        if not horizon or self.counts["world_updates"] < self.config.world_warmup_updates:
            return {"updated": False, "reason": "warmup_or_trust"}
        try:
            start = self.replay.sample_starts(
                self.config.imagination_batch, env_range=self.train_env_range,
                horizon=horizon, clip_length=self.task.reference.length)
        except ValueError as error:
            return {"updated": False, "reason": str(error)}
        frozen = tuple(self.world_model.parameters()) + tuple(self.critic.parameters())
        if getattr(self.task, "termination_module", None) is not None:
            frozen += tuple(self.task.termination_module.parameters())
        requires = [parameter.requires_grad for parameter in frozen]
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            rollout = self.task.imagine(self.actor, self.world_model, self.critic,
                                        start, horizon=horizon)
            actor_result = safe_gradient_step(rollout["actor_loss"], self.actor.parameters(),
                                              self.actor_optimizer)
        finally:
            for parameter, old in zip(frozen, requires):
                parameter.requires_grad_(old)
        if not actor_result["updated"]:
            self.counts["nonfinite_actor"] += 1
            return {"updated": False, "reason": actor_result["reason"]}
        self.counts["actor_updates"] += 1
        features, targets, bootstrap = self.task.critic_batch(self.critic, rollout)
        critic_result = optimize_critic(
            self.critic, self.critic_optimizer, features, targets,
            epochs=self.config.critic_epochs, minibatches=self.config.critic_minibatches)
        if getattr(self.task, "reward_profile", None) in REWARD_PROFILES:
            value_bound = reward_upper_bound(REWARD_PROFILES[self.task.reward_profile]) / (1 - 0.99)
            critic_result["derived_positive_value_upper_bound"] = value_bound
            critic_result["overestimation_alarm"] = (
                critic_result["predicted_value_mean"] is not None and
                critic_result["predicted_value_mean"] > value_bound * 1.1)
        self.counts["critic_updates"] += critic_result["updates"]
        self.counts["nonfinite_critic"] += critic_result["skipped_nonfinite"]
        return {"updated": True, "horizon": horizon,
                "actor_loss": float(rollout["actor_loss"].detach()),
                "actor_grad_norm": actor_result["grad_norm"],
                "critic": critic_result, "bootstrap_value_mean": float(bootstrap.mean())}

    def run_iteration(self):
        def stamp():
            if self.replay.device.type == "cuda":
                torch.cuda.synchronize(self.replay.device)
            return time.perf_counter()

        began = stamp()
        collection = self.collect()
        collected = stamp()
        world = [self.update_world_model() for _ in range(self.config.world_updates_per_iteration)]
        modeled = stamp()
        validated = False
        if (self.counts["world_updates"] >= self.config.world_warmup_updates
                and self.iteration % self.config.trust_interval == 0):
            self.validate_model()
            validated = True
        trusted = stamp()
        actor = [self.update_actor_critic() for _ in range(self.config.actor_updates_per_iteration)]
        optimized = stamp()
        self.iteration += 1
        return {"iteration": self.iteration, "collection": collection, "world": world,
                "actor": actor, "counts": self.counts.copy(),
                "allowed_horizon": self.allowed_horizon,
                "trust": self.trust_report if validated else None,
                "boundary_termination": self.boundary_termination_report if validated else None,
                "phase_seconds": {"collection": collected - began, "world": modeled - collected,
                                  "trust": trusted - modeled, "actor_critic": optimized - trusted}}
