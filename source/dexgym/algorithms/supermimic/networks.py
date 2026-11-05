"""World model, goal-conditioned deterministic actor, and privileged critic."""

import torch
from torch import nn

from .action import ArmActionMap


class RunningNorm(nn.Module):
    def __init__(self, dimension: int, min_std: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("variance", torch.ones(dimension))
        self.register_buffer("count", torch.tensor(0.0))
        self.min_std = min_std

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        values = values.detach().reshape(-1, self.mean.numel()).float()
        if len(values) == 0:
            return
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

    @torch.no_grad()
    def update_norms(self, state, action, next_state):
        self.input_norm.update(torch.cat((state, action), dim=-1))
        self.delta_norm.update(next_state[..., :28] - state[..., :28])

    def forward(self, state: torch.Tensor, action: torch.Tensor, mapping: ArmActionMap):
        x = self.input_norm(torch.cat((state, action), dim=-1))
        delta = self.network(x) * self.delta_norm.std() + self.delta_norm.mean
        command = mapping(action, state[..., 28:42])
        return torch.cat((state[..., :28] + delta, command), dim=-1)

    def autoregressive_loss(self, windows: dict[str, torch.Tensor], mapping: ArmActionMap):
        state, action, truth = windows["state"], windows["action"], windows["next_state"]
        if action.shape[1] != 8:
            raise ValueError("World model requires eight real continuous transitions")
        self.update_norms(state, action, truth)
        prediction = state[:, 0]
        losses, errors = [], []
        for step in range(8):
            prediction = self(prediction, action[:, step], mapping)
            delta = (prediction[..., :28] - truth[:, step, :28]) / self.delta_norm.std()
            losses.append(delta.square().mean())
            errors.append((prediction[..., :28] - truth[:, step, :28]).square().mean(-1).sqrt())
        return torch.stack(losses).mean(), torch.stack(errors, dim=1)


class GoalActor(nn.Module):
    def __init__(self, goal_latent=32):
        super().__init__()
        self.goal_encoder = mlp((900, 2048, 1024, 512, 256, goal_latent))
        self.decoder = mlp((560 + goal_latent, 2048, 1024, 512, 256, 14), output_zero=True)

    def forward(self, history: torch.Tensor, goal: torch.Tensor):
        if history.shape[-2:] != (10, 56) or goal.shape[-2:] != (10, 90):
            raise ValueError("Actor expects 10x56 history and 10x90 goal")
        latent = self.goal_encoder(goal.flatten(-2))
        return self.decoder(torch.cat((history.flatten(-2), latent), dim=-1))


class PrivilegedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = mlp((42 + 14 + 90, 256, 128, 1))

    def forward(self, state: torch.Tensor, reference_q: torch.Tensor, next_link_goal: torch.Tensor):
        return self.network(torch.cat((state, reference_q, next_link_goal), dim=-1)).squeeze(-1)
