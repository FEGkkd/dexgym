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


class ArmJointDeltaActionMap(ArmActionMap):
    """Versioned relative servo command: one bounded joint increment per policy step."""

    def __init__(self, origin: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
                 *, delta_q_max: float = 0.01):
        if not 0 < delta_q_max <= 0.04:
            raise ValueError("Joint delta limit must be positive and within the servo safety envelope")
        observation_scale = torch.minimum((upper - lower) / 2,
                                          torch.full_like(lower, 0.25))
        super().__init__(origin, lower, upper, observation_scale, max_delta=delta_q_max)
        self.delta_q_max = float(delta_q_max)

    def __call__(self, raw_action: torch.Tensor,
                 previous_command: torch.Tensor) -> torch.Tensor:
        if raw_action.shape != previous_command.shape or raw_action.shape[-1] != 14:
            raise ValueError("Joint delta action and previous command must be 14D and aligned")
        return torch.clamp(previous_command + self.delta_q_max * raw_action.tanh(),
                           min=self.lower, max=self.upper)
