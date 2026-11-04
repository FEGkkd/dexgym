"""Hard position-command limits for the simulated Marvin arms."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointCommandLimiter:
    dt: float
    velocity_limit: float
    acceleration_limit: float
    tracking_lag_limit: float

    def apply(
        self,
        action: torch.Tensor,
        actual: torch.Tensor,
        previous: torch.Tensor,
        previous_velocity: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        desired_velocity = action * self.velocity_limit
        velocity = previous_velocity + (desired_velocity - previous_velocity).clamp(
            -self.acceleration_limit * self.dt, self.acceleration_limit * self.dt
        )
        velocity = velocity.clamp(-self.velocity_limit, self.velocity_limit)
        target = previous + velocity * self.dt
        target = torch.maximum(
            torch.minimum(target, actual + self.tracking_lag_limit), actual - self.tracking_lag_limit
        )
        target = torch.maximum(torch.minimum(target, upper), lower)
        velocity = (target - previous) / self.dt
        return target, velocity
