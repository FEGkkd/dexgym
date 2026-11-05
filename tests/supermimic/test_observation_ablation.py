"""Goal-latent and sensor-noise ablations preserve the arm-only contract."""

import torch
from dexgym.algorithms.supermimic.networks import GoalActor
from dexgym.algorithms.supermimic.observation import noisy_proprio_frame


def test_goal_latent_tanh_is_configurable_and_bounds_encoder_output():
    goal = torch.zeros(2, 10, 90)
    bounded = GoalActor(goal_latent_tanh=True)
    unbounded = GoalActor(goal_latent_tanh=False)
    for actor in (bounded, unbounded):
        with torch.no_grad():
            actor.goal_encoder[-1].weight.zero_()
            actor.goal_encoder[-1].bias.fill_(5.0)
    assert bounded.encode_goal(goal).abs().max() < 1.0
    assert unbounded.encode_goal(goal).min() == 5.0


def test_sensor_noise_only_changes_q_and_qd_and_keeps_actor_gradient():
    torch.manual_seed(7)
    frame = torch.zeros(32, 56, requires_grad=True)
    scale = torch.full((14,), 2.0)
    noisy = noisy_proprio_frame(frame, scale, position_rad=0.01, velocity_rad_s=0.5)
    assert noisy.shape == frame.shape
    assert noisy[:, :14].abs().max() <= 0.005
    assert noisy[:, 14:28].abs().max() <= 0.25
    assert noisy[:, :28].abs().sum() > 0
    torch.testing.assert_close(noisy[:, 28:], frame[:, 28:])
    noisy.sum().backward()
    torch.testing.assert_close(frame.grad, torch.ones_like(frame))


def test_zero_sensor_noise_is_deterministic():
    frame = torch.randn(3, 56)
    torch.testing.assert_close(noisy_proprio_frame(frame, torch.ones(14), 0.0, 0.0), frame)
