"""Batch controller must match the tested scalar SE(3) command semantics."""

import math

import pytest
import torch

from dexteroushands_lab.assets.wrist_pose_control import WristPoseController
from dexteroushands_lab.assets.batched_wrist_pose import BatchedWristPoseController


def test_batch_matches_scalar_and_resets_only_selected_rows():
    pos = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]], dtype=torch.float64)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float64)
    batch = BatchedWristPoseController(2, device="cpu", dtype=torch.float64)
    batch.reset(torch.arange(2), pos, quat)
    scalar = WristPoseController(tuple(pos[0].tolist()), (0, 0, 0, 1))
    delta = torch.zeros(2, 6, dtype=torch.float64)
    delta[0] = torch.tensor([0.002, -0.001, 0.003, 0.01, 0.0, 0.0])
    delta[1, 0] = 0.002
    batch.command(delta)
    scalar.command(tuple(delta[0, :3].tolist()), tuple(delta[0, 3:].tolist()))
    for _ in range(10):
        pose, velocity = batch.step(0.002)
        scalar_pose, scalar_velocity = scalar.step(0.002)
        assert pose[0].tolist() == pytest.approx(scalar_pose, abs=1e-11)
        assert velocity[0].tolist() == pytest.approx(scalar_velocity, abs=1e-11)
    before = batch.current_pose[1].clone()
    batch.reset(torch.tensor([0]), pos[:1], quat[:1])
    assert torch.equal(batch.current_pose[1], before)
    assert torch.equal(batch.current_velocity[0], torch.zeros(6, dtype=torch.float64))


def test_batch_clips_workspace_without_crashing_and_zero_action_holds_target():
    batch = BatchedWristPoseController(1, device="cpu")
    batch.reset(torch.tensor([0]), torch.tensor([[0.0, 0.0, 0.5]]),
                torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    batch.command(torch.tensor([[10.0, 0.0, -10.0, 0.0, 0.0, 0.0]]))
    target = batch.target_pose.clone()
    assert target[0, 2] >= batch.limits.min_height
    assert torch.linalg.vector_norm(target[0, :3] - batch.initial_pos[0]) <= 0.600001
    assert batch.workspace_clip_count[0] > 0
    batch.command(torch.zeros(1, 6))
    assert torch.equal(batch.target_pose, target)
    pose, velocity = batch.step(0.002)
    assert torch.linalg.vector_norm(velocity[0, :3]) <= 0.150001
    assert torch.linalg.vector_norm(velocity[0, 3:]) <= 1.000001
    assert math.isfinite(float(pose.sum()))
