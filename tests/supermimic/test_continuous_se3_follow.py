"""Contract tests for a moving TCP target, distinct from fixed-goal Reach & Hold."""

import math
from types import SimpleNamespace

import pytest
import torch
from dexgym.assets.supermimic import PROCEDURAL_REFERENCE
from dexgym.tasks.marvin_supermimic.se3_reach_hold import rotation_6d_from_matrix


class LinearTCPRobot:
    class Spec:
        control_dt = 0.02

    spec = Spec()

    def __init__(self):
        self.action_map = type("Map", (), {
            "origin": torch.zeros(14), "scale": torch.ones(14),
            "lower": torch.full((14,), -1.0), "upper": torch.full((14,), 1.0)})()

    def fk(self, q):
        pos = q.new_zeros((*q.shape[:-1], 2, 3, 3))
        pos[..., 0, 2, 0] = q[..., 0]
        pos[..., 1, 2, 0] = q[..., 7]
        rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(
            *q.shape[:-1], 2, 3, 3, 3).clone()
        return pos, rotation


class MovingPoseReference:
    names = ("forward",)
    lengths = torch.tensor([201])
    hashes = {"forward": "test"}

    def pose_at(self, motion_id, frame):
        target = torch.zeros((len(frame), 2, 9), dtype=torch.float32)
        target[:, 0, 0] = frame * 0.001
        target[:, 1, 0] = -frame * 0.001
        target[:, :, 3:] = rotation_6d_from_matrix(torch.eye(3))
        return target.flatten(1)

    def future_goal_at(self, motion_id, frame, count):
        offsets = torch.arange(1, count + 1)
        indices = (frame[:, None] + offsets).clamp(max=200)
        return self.pose_at(motion_id[:, None].expand_as(indices).reshape(-1),
                            indices.reshape(-1)).reshape(len(frame), count, 18)


def test_reference_provider_is_time_indexed_and_reverses_pose_not_rotation_encoding():
    from dexgym.tasks.marvin_supermimic.se3_trajectory import SE3TCPReferenceProvider

    provider = SE3TCPReferenceProvider({"procedural_bimanual_reach_v1": PROCEDURAL_REFERENCE}, "cpu",
                                       include_reverse=True)
    assert provider.names == ("procedural_bimanual_reach_v1",
                             "procedural_bimanual_reach_v1_reverse")
    assert provider.lengths.tolist() == [201, 201]
    ids = torch.tensor([0, 1])
    frame = torch.tensor([17, 17])
    torch.testing.assert_close(provider.pose_at(ids, frame)[0],
                               provider.pose_at(torch.tensor([1]), torch.tensor([183]))[0])
    future = provider.future_goal_at(ids, frame, 10)
    torch.testing.assert_close(future[:, 0], provider.pose_at(ids, frame + 1))
    torch.testing.assert_close(provider.future_goal_at(ids, torch.tensor([199, 199]), 10)[:, 1],
                               provider.pose_at(ids, torch.tensor([200, 200])))


def test_task_has_10x18_moving_goal_and_scores_arrival_frame():
    from dexgym.tasks.marvin_supermimic.se3_trajectory import MarvinContinuousSE3Follow

    task = MarvinContinuousSE3Follow(MovingPoseReference(), LinearTCPRobot())
    assert (task.spec.observation_dim, task.spec.goal_dim, task.spec.goal_horizon,
            task.spec.critic_dim, task.spec.task_data_dim) == (68, 18, 10, 222, 0)
    state = torch.zeros(1, 42)
    state[:, 0] = 0.001
    state[:, 7] = -0.001
    task_state = task.task_state(torch.tensor([0]), torch.tensor([0]))
    goal = task.goal(state, task_state)
    assert goal.shape == (1, 10, 18)
    torch.testing.assert_close(goal[0, 0, 0], torch.tensor(0.001))
    assert task.observation(state, torch.zeros(1, 14), task_state).shape == (1, 68)
    assert task.critic_features(state, task_state).shape == (1, 222)
    next_state = state.clone()
    next_state[:, 0] = 0.002
    next_state[:, 7] = -0.002
    arrival = task.advance(task_state)
    reward_correct = task.reward(state, next_state, torch.zeros(1, 14), arrival)
    reward_wrong = task.reward(state, next_state, torch.zeros(1, 14), task_state)
    assert reward_correct.item() > reward_wrong.item()
    assert task.valid_imagination_start(task_state, 32).item()
    assert not task.valid_imagination_start(task.task_state(torch.tensor([0]), torch.tensor([169])),
                                            32).item()


def test_moving_tcp_reward_preserves_fk_gradient_and_large_error_is_not_failure():
    from dexgym.tasks.marvin_supermimic.se3_trajectory import MarvinContinuousSE3Follow

    task = MarvinContinuousSE3Follow(MovingPoseReference(), LinearTCPRobot())
    state = torch.zeros(1, 42)
    next_state = state.clone().requires_grad_()
    arrival = task.task_state(torch.tensor([0]), torch.tensor([20]))
    reward = task.reward(state, next_state, torch.zeros(1, 14), arrival)
    reward.backward()
    assert torch.isfinite(next_state.grad).all()
    assert next_state.grad[0, 0].abs() > 1e-5
    assert next_state.grad[0, 7].abs() > 1e-5
    assert not task.hard_termination(next_state.detach(), arrival,
                                     torch.zeros(1, dtype=torch.int32)).item()
    assert math.isfinite(task.reward_upper_bound())


def test_moving_orientation_reward_has_nonzero_so3_gradient():
    from dexgym.tasks.marvin_supermimic.se3_trajectory import ContinuousSE3RewardConfig, MarvinContinuousSE3Follow

    class RotatingRobot(LinearTCPRobot):
        def fk(self, q):
            pos, rotation = super().fk(q)
            angle = q[:, [0, 7]]
            cosine, sine = angle.cos(), angle.sin()
            rotation[:, :, 2, 0, 0] = cosine
            rotation[:, :, 2, 1, 0] = sine
            rotation[:, :, 2, 0, 1] = -sine
            rotation[:, :, 2, 1, 1] = cosine
            return pos, rotation

    class RotatingTarget(MovingPoseReference):
        def pose_at(self, motion_id, frame):
            target = super().pose_at(motion_id, frame).reshape(-1, 2, 9)
            angle = frame.float() * 0.005
            cosine, sine = angle.cos(), angle.sin()
            target[:, :, 3] = cosine[:, None]
            target[:, :, 4] = sine[:, None]
            target[:, :, 5] = 0
            target[:, :, 6] = -sine[:, None]
            target[:, :, 7] = cosine[:, None]
            target[:, :, 8] = 0
            return target.flatten(1)

    task = MarvinContinuousSE3Follow(
        RotatingTarget(), RotatingRobot(),
        reward_config=ContinuousSE3RewardConfig(
            position_weight=0, rotation_weight=1, progress_weight=0,
            action_weight=0, action_rate_weight=0, command_jump_weight=0,
            velocity_weight=0, soft_joint_limit_weight=0))
    state = torch.zeros(1, 42)
    next_state = state.clone().requires_grad_()
    task.reward(state, next_state, torch.zeros(1, 14),
                task.task_state(torch.tensor([0]), torch.tensor([20]))).backward()
    assert torch.isfinite(next_state.grad).all()
    assert next_state.grad[0, 0].abs() > 1e-5
    assert next_state.grad[0, 7].abs() > 1e-5


def test_task_rejects_full_54dof_state_and_keeps_only_arm_state_in_mdp():
    from dexgym.tasks.marvin_supermimic.se3_trajectory import MarvinContinuousSE3Follow

    task = MarvinContinuousSE3Follow(MovingPoseReference(), LinearTCPRobot())
    current = task.task_state(torch.tensor([0]), torch.tensor([0]))
    with pytest.raises(ValueError, match="42D arm-only"):
        task.observation(torch.zeros(1, 54), torch.zeros(1, 14), current)
    with pytest.raises(ValueError, match="42D arm-only"):
        task.critic_features(torch.zeros(1, 54), current)
    with pytest.raises(ValueError, match="42D arm-only"):
        task.reward(torch.zeros(1, 42), torch.zeros(1, 54),
                    torch.zeros(1, 14), task.advance(current))


def test_complete_episode_success_requires_both_tcps_to_follow_all_frames():
    from dexgym.tasks.marvin_supermimic.se3_trajectory import MarvinContinuousSE3Follow

    task = MarvinContinuousSE3Follow(MovingPoseReference(), LinearTCPRobot())
    trace = []
    for frame in range(1, 201):
        state = torch.zeros(1, 42)
        state[0, 0], state[0, 7] = frame * 0.001, -frame * 0.001
        trace.append(SimpleNamespace(
            next_state=state,
            next_task_state=task.task_state(torch.tensor([0]), torch.tensor([frame])),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.tensor([frame == 200])))
    assert task.episode_success(trace, complete=True)
    for item in trace[100:120]:
        item.next_state[0, 7] += 0.05
    assert not task.episode_success(trace, complete=True)


