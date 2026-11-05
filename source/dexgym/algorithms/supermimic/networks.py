"""World model, goal-conditioned deterministic actor, and privileged critic."""

import torch
from torch import nn
from torch.nn import functional as F

from .action import ArmActionMap


class RunningNorm(nn.Module):
    def __init__(self, dimension: int, min_std: float | torch.Tensor = 1e-2, max_count: int = 200_000_000):
        super().__init__()
        if max_count < 1:
            raise ValueError("Normalizer maximum count must be positive")
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("variance", torch.ones(dimension))
        self.register_buffer("count", torch.tensor(0.0))
        floor = torch.as_tensor(min_std, dtype=torch.float32)
        if floor.ndim > 1 or (floor.ndim == 1 and floor.shape != (dimension,)) or bool((floor <= 0).any()):
            raise ValueError("Normalizer minimum std must be positive scalar or per-feature vector")
        self.register_buffer("min_std", floor, persistent=False)
        self.max_count = max_count

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        values = values.detach().reshape(-1, self.mean.numel()).float()
        if len(values) == 0:
            return
        remaining = self.max_count - int(self.count.item())
        if remaining <= 0:
            return
        values = values[:remaining]
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Normalizer data must be finite")
        batch_mean = values.mean(0)
        batch_var = values.var(0, unbiased=False)
        amount = float(len(values))
        total = self.count + amount
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * amount / total
        moment = self.variance * self.count + batch_var * amount + delta.square() * self.count * amount / total
        self.mean.copy_(new_mean)
        self.variance.copy_(moment / total)
        self.count.copy_(total)

    def std(self):
        return self.variance.sqrt().clamp_min(self.min_std)

    def forward(self, values: torch.Tensor):
        return (values - self.mean) / self.std()


def mlp(widths, *, output_zero=False):
    layers = []
    for i, (before, after) in enumerate(zip(widths[:-1], widths[1:])):
        linear = nn.Linear(before, after)
        if output_zero and i == len(widths) - 2:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i != len(widths) - 2:
            layers.extend((nn.LayerNorm(after), nn.SiLU()))
    return nn.Sequential(*layers)


class DeltaWorldModel(nn.Module):
    """Predict q/qd deltas; command state follows the exact analytic servo map."""

    def __init__(self, hidden=2048):
        super().__init__()
        self.input_norm = RunningNorm(56)
        self.delta_norm = RunningNorm(28, min_std=0.01)
        self.network = mlp((56, hidden, hidden, hidden, 28), output_zero=True)
        self.termination_head = mlp((42 + 90, 128, 128, 1))
        nn.init.constant_(self.termination_head[-1].bias, -4.0)

    @torch.no_grad()
    def update_norms(self, state, action, next_state):
        self.input_norm.update(torch.cat((state, action), dim=-1))
        self.delta_norm.update(next_state[..., :28] - state[..., :28])

    def forward(self, state: torch.Tensor, action: torch.Tensor, mapping: ArmActionMap):
        x = self.input_norm(torch.cat((state, action), dim=-1))
        delta = self.network(x) * self.delta_norm.std() + self.delta_norm.mean
        command = mapping(action, state[..., 28:42])
        return torch.cat((state[..., :28] + delta, command), dim=-1)

    def termination_probability(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != 42 or goal.shape[-1] != 90 or state.shape[:-1] != goal.shape[:-1]:
            raise ValueError("Termination head expects matching [...,42] state and [...,90] goal")
        return torch.sigmoid(self.termination_head(torch.cat((state, goal), dim=-1)).squeeze(-1))

    def autoregressive_loss(self, windows: dict[str, torch.Tensor], mapping: ArmActionMap, *,
                            update_normalizers: bool = True, termination_pos_weight: float = 1.0):
        state, action, truth = windows["state"], windows["action"], windows["next_state"]
        if action.shape[1] != 8:
            raise ValueError("World model requires eight slots, with invalid transitions masked")
        mask = windows.get("mask", torch.ones_like(action[..., 0])).bool()
        if mask.shape != action.shape[:2] or not bool(mask[:, 0].all()):
            raise ValueError("World model window mask must include every first transition")
        if update_normalizers:
            self.update_norms(state[mask], action[mask], truth[mask])
        prediction = state[:, 0]
        has_term = "terminated" in windows or "termination_goal" in windows
        if has_term and ("terminated" not in windows or "termination_goal" not in windows):
            raise ValueError("Termination labels require aligned next-state goals")
        if has_term and (
            windows["terminated"].shape != action.shape[:2]
            or windows["termination_goal"].shape != (*action.shape[:2], 90)
        ):
            raise ValueError("Termination labels and goals must align with eight real transitions")
        losses, errors, term_losses = [], [], []
        for step in range(8):
            prediction = self(prediction, action[:, step], mapping)
            delta = (prediction[..., :28] - truth[:, step, :28]) / self.delta_norm.std()
            losses.append(delta[mask[:, step]].square().sum())
            error = (prediction[..., :28] - truth[:, step, :28]).square().mean(-1).sqrt()
            errors.append(torch.where(mask[:, step], error, torch.zeros_like(error)))
            if has_term and bool(mask[:, step].any()):
                logits = self.termination_head(torch.cat((
                    prediction[mask[:, step]], windows["termination_goal"][:, step][mask[:, step]]
                ), dim=-1)).squeeze(-1)
                labels = windows["terminated"][:, step][mask[:, step]].float()
                term_losses.append(F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum",
                    pos_weight=torch.as_tensor(termination_pos_weight, device=logits.device)))
        objective = torch.stack(losses).sum() / (mask.sum() * 28)
        if term_losses:
            objective = objective + torch.stack(term_losses).sum() / mask.sum()
        return objective, torch.stack(errors, dim=1)


class GoalActor(nn.Module):
    def __init__(self, goal_latent=32, *, goal_latent_tanh: bool = False,
                 normalization_mode: str = "legacy"):
        super().__init__()
        if normalization_mode not in ("legacy", "running"):
            raise ValueError("Unknown actor normalization mode")
        self.normalization_mode = normalization_mode
        self.goal_latent_tanh = goal_latent_tanh
        self.observation_norm = RunningNorm(56) if normalization_mode == "running" else nn.Identity()
        self.goal_norm = RunningNorm(90) if normalization_mode == "running" else nn.Identity()
        self.goal_encoder = mlp((900, 2048, 1024, 512, 256, goal_latent))
        self.decoder = mlp((560 + goal_latent, 2048, 1024, 512, 256, 14), output_zero=True)

    @torch.no_grad()
    def update_norms(self, history: torch.Tensor, goal: torch.Tensor):
        if self.normalization_mode == "running":
            self.observation_norm.update(history)
            self.goal_norm.update(goal)

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        if goal.shape[-2:] != (10, 90):
            raise ValueError("Goal encoder expects 10x90 future reference")
        latent = self.goal_encoder(self.goal_norm(goal).flatten(-2))
        return latent.tanh() if self.goal_latent_tanh else latent

    def forward(self, history: torch.Tensor, goal: torch.Tensor):
        if history.shape[-2:] != (10, 56) or goal.shape[-2:] != (10, 90):
            raise ValueError("Actor expects 10x56 history and 10x90 goal")
        latent = self.encode_goal(goal)
        return self.decoder(torch.cat((self.observation_norm(history).flatten(-2), latent), dim=-1))


class PrivilegedCritic(nn.Module):
    def __init__(self, *, normalization_mode: str = "legacy"):
        super().__init__()
        if normalization_mode not in ("legacy", "running"):
            raise ValueError("Unknown critic normalization mode")
        self.normalization_mode = normalization_mode
        self.input_norm = RunningNorm(42 + 14 + 90) if normalization_mode == "running" else nn.Identity()
        self.network = mlp((42 + 14 + 90, 256, 128, 1))

    @torch.no_grad()
    def update_norms(self, state: torch.Tensor, reference_q: torch.Tensor, next_link_goal: torch.Tensor):
        if self.normalization_mode == "running":
            self.input_norm.update(torch.cat((state, reference_q, next_link_goal), dim=-1))

    def forward(self, state: torch.Tensor, reference_q: torch.Tensor, next_link_goal: torch.Tensor):
        return self.forward_features(torch.cat((state, reference_q, next_link_goal), dim=-1))

    def forward_features(self, features: torch.Tensor):
        if features.shape[-1] != 42 + 14 + 90:
            raise ValueError("Critic expects 146D state/reference features")
        return self.network(self.input_norm(features)).squeeze(-1)
