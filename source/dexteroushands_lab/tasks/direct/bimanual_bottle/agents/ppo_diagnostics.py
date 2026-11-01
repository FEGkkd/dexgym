"""PPO health metrics for the installed RSL-RL rollout format.

The policy update remains the upstream RSL-RL implementation. Diagnostics
read the completed rollout after the update and do not alter optimization.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


def clip_fraction(log_ratio: torch.Tensor, clip_param: float) -> torch.Tensor:
    """Fraction of rollout actions whose updated probability ratio is clipped."""
    return ((log_ratio.exp() - 1.0).abs() > clip_param).float().mean()


def explained_variance(returns: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Variance explained by the critic, with a finite constant-target case."""
    target_variance = returns.var(unbiased=False)
    return torch.where(target_variance > 1e-8,
                       1.0 - (returns - prediction).var(unbiased=False) /
                       target_variance.clamp_min(1e-8), torch.zeros_like(target_variance))


class BimanualDiagnosticPPO(PPO):
    """Add KL, clip fraction, value fit and grouped action statistics to PPO logs."""

    def update(self) -> dict[str, float]:
        losses = super().update()
        # RolloutStorage.clear() only resets its write cursor. Existing data
        # remain valid until the next collection, allowing a post-update check.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]
                                   if torch.device(self.device).type == "cuda" else []):
            batch = next(self.storage.mini_batch_generator(1, 1))
        actor_was_training = self.actor.training
        critic_was_training = self.critic.training
        self.actor.eval()
        self.critic.eval()
        try:
            with torch.no_grad():
                self.actor(batch.observations, masks=batch.masks,
                           hidden_state=batch.hidden_states[0], stochastic_output=True)
                new_log_prob = self.actor.get_output_log_prob(batch.actions)
                log_ratio = new_log_prob - batch.old_actions_log_prob.squeeze(-1)
                current_params = self.actor.output_distribution_params
                kl = self.actor.get_kl_divergence(batch.old_distribution_params,
                                                   current_params).mean()
                values = self.critic(batch.observations, masks=batch.masks,
                                     hidden_state=batch.hidden_states[1])
                std = self.actor.output_std.detach().reshape(-1)
                mean_action = self.actor.output_mean
                latent_mean = current_params[0]
                diagnostics = {
                    "approx_kl": kl,
                    "ppo_clip_fraction": clip_fraction(log_ratio, self.clip_param),
                    "explained_variance": explained_variance(batch.returns, values),
                    "raw_action_abs_mean": batch.actions.abs().mean(),
                    "raw_action_element_outside_fraction": (batch.actions.abs() > 1).float().mean(),
                    "policy_action_abs_mean": mean_action.abs().mean(),
                    "policy_mean_element_outside_fraction": (mean_action.abs() > 1).float().mean(),
                    "latent_policy_mean_abs_mean": latent_mean.abs().mean(),
                    "latent_policy_mean_outside_fraction": (latent_mean.abs() > 1).float().mean(),
                    "policy_entropy": self.actor.output_entropy.mean(),
                    "left_wrist_translation_std": std[0:3].mean(),
                    "left_wrist_rotation_std": std[3:6].mean(),
                    "left_finger_std": std[6:26].mean(),
                    "right_wrist_translation_std": std[26:29].mean(),
                    "right_wrist_rotation_std": std[29:32].mean(),
                    "right_finger_std": std[32:52].mean(),
                }
                losses.update({name: float(value) for name, value in diagnostics.items()})
        finally:
            self.actor.train(actor_was_training)
            self.critic.train(critic_was_training)
        return losses
