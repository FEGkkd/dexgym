"""FK witness sampling creates legal, diverse and non-leaking SE(3) goals."""

from types import SimpleNamespace

import torch

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot
from dexgym.tasks.marvin_supermimic.se3_goal_sampler import SE3GoalSampler
from dexgym.tasks.marvin_supermimic.se3_reach_hold import unpack_dual_tcp_pose


def _robot():
    return MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                           action_schema="arm_joint_delta_v1")


def test_near_sampling_randomizes_both_arms_and_discards_joint_witness():
    robot = _robot()
    samples = SE3GoalSampler(robot, train_seed=7, evaluation_seed=9001).sample(16, "near")
    assert samples.start_q.shape == (16, 14)
    assert samples.goal_pose.shape == (16, 18)
    assert samples.position_difficulty_m.shape == (16, 2)
    assert samples.rotation_difficulty_rad.shape == (16, 2)
    assert not hasattr(samples, "q_goal")
    assert torch.all(samples.start_q >= robot.action_map.lower + 0.02)
    assert torch.all(samples.start_q <= robot.action_map.upper - 0.02)
    assert (samples.start_q[:, :7].std(0) > 0.01).all()
    assert (samples.start_q[:, 7:].std(0) > 0.01).all()
    assert (samples.position_difficulty_m > 0.005).all()
    assert (samples.rotation_difficulty_rad > 0.005).all()
    position, rotation = unpack_dual_tcp_pose(samples.goal_pose)
    assert position.shape == (16, 2, 3)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                               torch.eye(3).expand(16, 2, 3, 3), atol=1e-5, rtol=1e-5)


def test_evaluation_goal_set_is_seeded_and_independent_from_training_rng():
    robot = _robot()
    first = SE3GoalSampler(robot, train_seed=1, evaluation_seed=91)
    second = SE3GoalSampler(robot, train_seed=800, evaluation_seed=91)
    first.sample(8, "near")
    a = first.sample_evaluation(4, "near")
    b = second.sample_evaluation(4, "near")
    assert torch.equal(a.start_q, b.start_q)
    assert torch.equal(a.goal_pose, b.goal_pose)
    assert not torch.equal(a.start_q[0], a.start_q[1])


def test_medium_and_wider_curricula_have_larger_bimanual_difficulty():
    robot = _robot()
    sampler = SE3GoalSampler(robot, train_seed=12, evaluation_seed=13)
    near = sampler.sample(24, "near")
    medium = sampler.sample(24, "medium")
    wider = sampler.sample(24, "wider_feasible")
    assert float(near.position_difficulty_m.mean()) < float(medium.position_difficulty_m.mean())
    assert float(medium.position_difficulty_m.mean()) < float(wider.position_difficulty_m.mean())
