"""The same bounded position-servo mapping for Newton and imagination."""

import torch


class ArmActionMap:
    def __init__(
        self, origin: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, scale: torch.Tensor, max_delta=0.08
    ):
        self.origin, self.lower, self.upper, self.scale = origin, lower, upper, scale
        self.max_delta = float(max_delta)
        if (
            origin.shape != lower.shape
            or origin.shape != upper.shape
            or origin.shape != scale.shape
            or origin.numel() != 14
        ):
            raise ValueError("Expected four 14D arm tensors")

    def __call__(self, raw_action: torch.Tensor, previous_command: torch.Tensor) -> torch.Tensor:
        bounded = raw_action.tanh()
        desired = torch.clamp(self.origin + self.scale * bounded, min=self.lower, max=self.upper)
        return torch.clamp(desired, min=previous_command - self.max_delta, max=previous_command + self.max_delta)
