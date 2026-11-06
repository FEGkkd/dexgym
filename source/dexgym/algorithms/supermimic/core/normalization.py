"""FP32 Welford-style running normalization updated from real training data only."""

import torch
from torch import nn


class RunningNorm(nn.Module):
    def __init__(self, dimension: int, *, min_std: float = 1e-2, max_count: int = 200_000_000):
        super().__init__()
        if dimension < 1 or min_std <= 0 or max_count < 1:
            raise ValueError("RunningNorm dimension, floor and count limit must be positive")
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float32))
        self.register_buffer("variance", torch.ones(dimension, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float32))
        self.min_std = min_std
        self.max_count = max_count

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        if values.shape[-1] != self.mean.numel():
            raise ValueError("Normalizer feature dimension differs")
        values = values.detach().reshape(-1, self.mean.numel()).float()
        remaining = self.max_count - int(self.count.item())
        if remaining <= 0 or len(values) == 0:
            return
        values = values[:remaining]
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Normalizer input is nonfinite")
        amount = len(values)
        total = self.count + amount
        batch_mean = values.mean(0)
        batch_var = values.var(0, unbiased=False)
        difference = batch_mean - self.mean
        next_mean = self.mean + difference * amount / total
        moment = (self.variance * self.count + batch_var * amount
                  + difference.square() * self.count * amount / total)
        self.mean.copy_(next_mean)
        self.variance.copy_(moment / total)
        self.count.copy_(total)

    def std(self):
        return self.variance.sqrt().clamp_min(self.min_std)

    def forward(self, values: torch.Tensor):
        if values.shape[-1] != self.mean.numel():
            raise ValueError("Normalizer feature dimension differs")
        return (values - self.mean) / self.std()

