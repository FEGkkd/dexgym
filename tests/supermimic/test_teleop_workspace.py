"""Causal dual TCP target stream contracts for the teleoperation task."""

import torch

from dexgym.tasks.marvin_supermimic.se3_reach_hold import rotation_6d_from_matrix
from tests.supermimic.test_continuous_se3_follow import LinearTCPRobot


class RampTarget:
    names = ("ramp",)
    lengths = torch.tensor([201])
    hashes = {"ramp": "synthetic"}

    def pose_at(self, motion_id, frame):
        value = torch.zeros(len(frame), 2, 9)
        value[:, 0, 0] = frame.float() * 0.01
        value[:, 1, 0] = -frame.float() * 0.01
        value[:, :, 3:] = rotation_6d_from_matrix(torch.eye(3))
        return value.flatten(1)


def test_actor_sees_only_current_target_and_arrival_scores_commanded_target():
    from dexgym.tasks.marvin_supermimic.teleop_trajectory import MarvinTeleopWorkspaceFollow

    task = MarvinTeleopWorkspaceFollow(RampTarget(), LinearTCPRobot())
    old = task.task_state(torch.tensor([0]), torch.tensor([5]))
    arrival = task.advance(old)
    state = torch.zeros(1, 42)
    next_state = state.clone()
    next_state[0, 0], next_state[0, 7] = 0.05, -0.05
    assert task.spec.goal_horizon == 1
    assert task.spec.goal_dim == 18
    assert task.spec.critic_dim == 60
    torch.testing.assert_close(task.goal(state, old)[0, 0], RampTarget().pose_at(old.motion_id, old.frame)[0])
    assert task.reward(state, next_state, torch.zeros(1, 14), arrival) > task.reward(
        state, next_state, torch.zeros(1, 14), task.task_state(torch.tensor([0]), torch.tensor([7])))
    assert task.observation(state, torch.zeros(1, 14), old).shape == (1, 68)
    assert task.critic_features(state, old).shape == (1, 60)


def test_teleop_reward_keeps_bptt_gradient_and_no_joint_witness():
    from dexgym.tasks.marvin_supermimic.teleop_trajectory import MarvinTeleopWorkspaceFollow

    task = MarvinTeleopWorkspaceFollow(RampTarget(), LinearTCPRobot())
    start = torch.zeros(1, 42)
    next_state = start.clone().requires_grad_()
    arrival = task.task_state(torch.tensor([0]), torch.tensor([6]))
    task.reward(start, next_state, torch.zeros(1, 14), arrival).backward()
    assert torch.isfinite(next_state.grad).all()
    assert next_state.grad[0, [0, 7]].abs().min() > 0
    assert "q_goal" not in task.checkpoint_contract()


def test_fk_bank_has_disjoint_splits_and_uses_only_current_pose(tmp_path):
    from dexgym.motion.teleop_workspace import (TeleopWorkspaceReferenceBank,
                                                generate_teleop_workspace_bank)

    path = tmp_path / "targets.npz"
    report = generate_teleop_workspace_bank(path, seed=17, split_counts=(3, 2, 1))
    bank = TeleopWorkspaceReferenceBank(path, "cpu")
    assert report["clips"] == 6
    assert bank.lengths.tolist() == [201] * 6
    assert bank.pose_at(torch.tensor([0, 5]), torch.tensor([10, 20])).shape == (2, 18)
    assert bank.reset_q_at(torch.tensor([0]), torch.tensor([10])).shape == (1, 14)
    assert set(bank.sample_ids(100, split="train").tolist()) <= {0, 1, 2}
    assert set(bank.sample_ids(100, split="validation").tolist()) <= {3, 4}
    assert set(bank.sample_ids(100, split="test").tolist()) == {5}


def test_live_target_stream_is_causal_and_enforces_fixed_base_50hz_contract():
    import pytest
    from dexgym.tasks.marvin_supermimic.teleop_stream import StatefulTeleopPolicy

    class StreamRobot(LinearTCPRobot):
        class Spec:
            control_dt = 0.02
            state_dim = 42
            action_dim = 14
        spec = Spec()

        @staticmethod
        def executed_command(state, action):
            return state[:, 28:42] + 0.01 * action.tanh()

    class ZeroActor(torch.nn.Module):
        def forward(self, history, goal):
            assert history.shape == (1, 10, 68)
            assert goal.shape == (1, 1, 18)
            return history.new_zeros((1, 14))

    policy = StatefulTeleopPolicy(ZeroActor(), StreamRobot(), device="cpu")
    pos = torch.zeros(2, 3)
    rotation = torch.eye(3).expand(2, 3, 3).clone()
    state = torch.zeros(1, 42)
    action, command = policy.step(state, 0.0, pos, rotation)
    assert action.shape == command.shape == (1, 14)
    with pytest.raises(ValueError, match="fixed-base"):
        policy.step(state, 0.02, pos, rotation, frame_id="world")
    with pytest.raises(ValueError, match="FP32"):
        policy.step(state, 0.02, pos.double(), rotation)
    with pytest.raises(ValueError, match="50 Hz"):
        policy.step(state, 0.1, pos, rotation)
    policy.step(state, 0.02, pos, rotation)
    with pytest.raises(ValueError, match="current received"):
        policy.reference.pose_at(torch.tensor([0]), torch.tensor([0]))


def test_reverse_evaluation_bank_changes_time_direction_without_new_joint_targets(tmp_path):
    import numpy as np
    from dexgym.motion.teleop_workspace import generate_teleop_workspace_bank
    from scripts.supermimic.reverse_teleop_workspace_bank import reverse_test_bank

    source, output = tmp_path / "source.npz", tmp_path / "reversed.npz"
    generate_teleop_workspace_bank(source, seed=8, split_counts=(1, 1, 2))
    report = reverse_test_bank(source, output)
    assert report["clips"] == 2
    with np.load(source, allow_pickle=False) as old, np.load(output, allow_pickle=False) as new:
        assert new["split"].tolist() == [2, 2]
        np.testing.assert_array_equal(new["tcp_pos"][0, 0], old["tcp_pos"][2, -1])
        np.testing.assert_array_equal(new["tcp_pos"][1, -1], old["tcp_pos"][3, 0])
