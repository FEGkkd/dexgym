"""Count latent Gaussian limit violations and executed tanh action saturation."""

import math

import torch


class ActionAudit:
    """Accumulate rollout counts by exact active target span; span zero is a cross-trajectory goal."""

    FIELDS = (
        "env_steps",
        "latent_mean_over_limit_components",
        "latent_mean_over_limit_steps",
        "gaussian_sample_over_limit_components",
        "gaussian_sample_over_limit_steps",
        "executed_saturated_components",
        "executed_saturated_steps",
    )

    def __init__(self, *, action_dim: int, max_span: int, device: str | torch.device) -> None:
        self.action_dim = action_dim
        self.max_span = max_span
        self.counts = torch.zeros(max_span + 1, len(self.FIELDS), device=device, dtype=torch.float64)

    def record(self, latent_mean: torch.Tensor, actions: torch.Tensor, goal_span: torch.Tensor) -> None:
        """Record one policy step before the environment advances its active goal."""
        if latent_mean.shape != actions.shape or actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError("Expected matching latent means and actions[N, action_dim]")
        if goal_span.shape != (len(actions),) or bool(((goal_span < 0) | (goal_span > self.max_span)).any()):
            raise ValueError("Expected goal span in [0, max_span]")
        mean_over = latent_mean.abs() > 1.0
        # tanh is monotone: |sample_latent| > 1 iff |executed_action| > tanh(1).
        sample_over = actions.abs() > math.tanh(1.0)
        saturated = actions.abs() >= 0.95
        step_counts = torch.stack(
            (
                torch.ones(len(actions), device=actions.device),
                mean_over.sum(dim=-1),
                mean_over.any(dim=-1),
                sample_over.sum(dim=-1),
                sample_over.any(dim=-1),
                saturated.sum(dim=-1),
                saturated.any(dim=-1),
            ),
            dim=-1,
        ).to(self.counts.dtype)
        self.counts.index_add_(0, goal_span.long(), step_counts)

    def drain(self, *, iteration: int) -> list[dict[str, int]]:
        """Return per-span counts for one PPO rollout and begin the next one."""
        cpu = self.counts.cpu().to(torch.int64).tolist()
        self.counts.zero_()
        return [
            {"iteration": iteration, "goal_span": span, **dict(zip(self.FIELDS, values, strict=True))}
            for span, values in enumerate(cpu)
        ]
