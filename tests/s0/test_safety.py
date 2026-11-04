"""Joint commands must respect velocity, acceleration, limits and tracking lag."""

import torch
from dexgym.tasks.marvin_s0.safety import JointCommandLimiter


def test_command_limiter_bounds_velocity_acceleration_lag_and_limits():
    limiter = JointCommandLimiter(dt=0.01, velocity_limit=0.8, acceleration_limit=4.0, tracking_lag_limit=0.05)
    actual = torch.tensor([[0.49, 0.0]])
    previous = torch.tensor([[0.49, 0.04]])
    previous_velocity = torch.tensor([[0.0, 0.0]])
    lower = torch.tensor([[-0.5, -0.5]])
    upper = torch.tensor([[0.5, 0.5]])
    target, velocity = limiter.apply(
        torch.tensor([[9.0, float("nan")]]), actual, previous, previous_velocity, lower, upper
    )
    assert torch.isfinite(target).all()
    assert torch.all(target <= upper)
    assert torch.all(target >= lower)
    assert torch.all(torch.abs(velocity) <= 0.04 + 1e-5)
    assert torch.all(torch.abs(target - actual) <= 0.05 + 1e-7)
