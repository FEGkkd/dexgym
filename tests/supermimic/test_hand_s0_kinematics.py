"""Hand FK oracle and gradients, plus parity with the actual PPO command map."""
import mujoco
import numpy as np
import torch

from dexgym.tasks.direct.hand21_supermimic.hand_kinematics import Hand21Kinematics
from dexgym.tasks.direct.hand21_supermimic.supermimic_robot import Hand21DynamicsAdapter
from dexgym.tasks.direct.hand21_supermimic.logic import residual_joint_target


def test_standalone_fk_matches_mujoco_and_has_correct_directional_derivative():
    torch.manual_seed(21)
    fk = Hand21Kinematics(dtype=torch.float64)
    q = fk.limits[:, 0] + torch.rand(32, 20, dtype=torch.float64) * (fk.limits[:, 1]-fk.limits[:, 0])
    actual = fk.landmarks(q).detach().numpy()
    data = mujoco.MjData(fk.model)
    expected = []
    for pose in q.numpy():
        data.qpos[fk.qpos_indices] = pose
        mujoco.mj_forward(fk.model, data)
        points = np.stack((data.site_xpos[fk.tip_ids], data.xpos[fk.level1], data.xpos[fk.level2]))
        rotation = data.xmat[fk.wrist_id].reshape(3, 3)
        expected.append((points-data.xpos[fk.wrist_id]) @ rotation)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
    q = q[:2].requires_grad_()
    direction = torch.randn_like(q)
    _, derivative = torch.autograd.functional.jvp(fk.landmarks, q, direction)
    eps = 1e-6
    difference = (fk.landmarks(q + eps*direction) - fk.landmarks(q-eps*direction))/(2*eps)
    torch.testing.assert_close(derivative, difference, atol=1e-8, rtol=1e-6)
    assert derivative.abs().max() > .001


def test_hand_model_integrates_exact_command_and_only_learns_physical_delta():
    robot = Hand21DynamicsAdapter()
    state = torch.zeros(3, 60)
    action = torch.stack((torch.zeros(20), torch.full((20,), .2), torch.full((20,), 20.)))
    expected = residual_joint_target(state[:, 40:], action.tanh(), robot.fk.limits,
                                     scale=.1, deadzone=.1)
    delta = torch.full((3, 40), .02)
    arrival = robot.integrate(state, delta, action, torch.empty(3, 0))
    torch.testing.assert_close(arrival[:, 40:], expected)
    torch.testing.assert_close(robot.physical_delta(state, arrival), delta)
    assert robot.spec.state_dim == 60 and robot.spec.action_dim == 20
    # One maximum bounded action changes a command by at most .09 rad.
    assert arrival[:, 40:].abs().max() <= .090001
