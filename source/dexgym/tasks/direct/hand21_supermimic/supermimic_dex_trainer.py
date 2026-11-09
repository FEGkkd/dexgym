"""v4 real-data model norms and imagined-data Critic normalization."""

import torch

from dexgym.algorithms.supermimic.core.imagination import critic_td_lambda_batch
from dexgym.algorithms.supermimic.core.optimization import frozen_parameters, optimize_critic, safe_gradient_step

from .supermimic_schedule import HandSuperMimicTrainer


class HandDexTrainer(HandSuperMimicTrainer):
    @torch.no_grad()
    def _update_normalizers_from_real(self, transition):
        low, high = self.train_env_range
        selection = slice(low, high)
        state, next_state = transition.state[selection], transition.next_state[selection]
        self.world_model.update_norms(
            state,
            transition.executed_command[selection],
            transition.dynamics_context[selection],
            next_state,
            self.robot,
        )
        # Actor statistics are updated before acting by the training collector.
        # Critic statistics are updated only after imagination, below.
        if self.termination_head is not None:
            arrival = transition.next_task_state.index(selection)
            self.termination_head.update_norms(next_state, self.task.termination_features(next_state, arrival))

    @torch.no_grad()
    def refresh_critic_values(self, rollout):
        features = torch.cat((rollout.critic_features, rollout.terminal_features[None]), 0).detach()
        self.critic.update_norms(features)
        rollout.values = self.critic(features)

    def update_actor_critic(self):
        if self.iteration < self.config.wm_only_iters:
            return {"updated": False, "reason": "fixed_wm_only_iterations"}
        horizon = self.config.max_imagination_horizon
        try:
            start = self.replay.sample_starts(
                self.config.imagination_batch,
                horizon=horizon,
                valid_start=self.task.valid_imagination_start,
                env_range=self.train_env_range,
                source_id=0,
            )
        except ValueError as error:
            return {"updated": False, "reason": str(error)}
        actor_began = self._stamp()
        bootstrap_weight = self._actor_bootstrap_weight()
        with frozen_parameters(self.world_model, self.critic, self.termination_head):
            rollout = self.engine.rollout(
                start,
                self.actor,
                self.world_model,
                self.critic,
                self.robot,
                self.task,
                horizon=horizon,
                bootstrap_weight=bootstrap_weight,
                termination_head=self.termination_head,
                observation_noise=self.observation_noise,
            )
            actor_result = safe_gradient_step(
                rollout.actor_loss,
                self.actor.parameters(),
                self.optimizers["actor"],
                grad_value_clip=self.config.actor_grad_value_clip,
            )
        actor_finished = self._stamp()
        if not actor_result["updated"]:
            self.counts["nonfinite_actor"] += 1
            return {"updated": False, "reason": actor_result["reason"]}
        self.counts["actor_updates"] += 1
        # Dex fits Critic statistics on all H+1 detached imagined states,
        # then recomputes values before constructing fixed TD(lambda) targets.
        self.refresh_critic_values(rollout)
        features, targets = critic_td_lambda_batch(
            rollout, gamma=self.config.gamma, lam=self.config.lam, value_upper_bound=self.value_upper_bound
        )
        critic_result = optimize_critic(
            self.critic,
            self.optimizers["critic"],
            features,
            targets,
            epochs=self.config.critic_epochs,
            minibatches=self.config.critic_minibatches,
        )
        critic_finished = self._stamp()
        self.counts["critic_updates"] += critic_result["updates"]
        self.counts["nonfinite_critic"] += critic_result["skipped_nonfinite"]
        return {
            "updated": True,
            "horizon": horizon,
            "actor_bootstrap_weight": bootstrap_weight,
            "actor_loss": float(rollout.actor_loss.detach()),
            "actor_grad_norm": actor_result["grad_norm"],
            "actor_bptt_seconds": actor_finished - actor_began,
            "critic_optimization_seconds": critic_finished - actor_finished,
            "critic": critic_result,
        }
