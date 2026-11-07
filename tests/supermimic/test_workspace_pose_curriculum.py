"""A reachable-workspace curriculum is defined by TCP pose error, not a name."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK
from dexgym.algorithms.supermimic.rewards import so3_geodesic
from dexgym.motion.teleop_workspace import DEFAULT_MJCF


def test_fk_curriculum_has_measured_dual_arm_pose_bins_and_no_goal_joint_label(tmp_path):
    from dexgym.motion.teleop_workspace import (
        TeleopWorkspaceReferenceBank, generate_teleop_reach_curriculum_bank)

    target = tmp_path / "measured_pose_curriculum.npz"
    generate_teleop_reach_curriculum_bank(
        target, seed=20261113, split_counts=(3, 3, 3), frames=601)
    with np.load(target, allow_pickle=False) as bank:
        assert str(bank["version"]) == "marvin_teleop_workspace_bank_v4"
        assert int(bank["fps"]) == 50
        assert not any("goal_q" in key or "q_goal" in key for key in bank.files)
        assert sorted(set(bank["difficulty"].tolist())) == [0, 1, 2]
        start = torch.tensor(bank["q_reset_witness"][:, 0], dtype=torch.float32)
        target_position = torch.tensor(bank["tcp_pos"][:, 0], dtype=torch.float32)
        target_rotation = torch.tensor(Rotation.from_quat(
            bank["tcp_quat_xyzw"][:, 0].reshape(-1, 4)).as_matrix().reshape(-1, 2, 3, 3),
            dtype=torch.float32)
        difficulty = bank["difficulty"].copy()
    fk = DifferentiableMarvinFK(DEFAULT_MJCF, "cpu")
    position, rotation = fk(start)
    position_error = torch.linalg.vector_norm(position[:, :, 2] - target_position, dim=-1)
    rotation_error = so3_geodesic(rotation[:, :, 2], target_rotation)
    limits = ((0.015, 0.12, 0.035, 0.45),
              (0.08, 0.35, 0.15, 1.1),
              (0.20, 1.1, 0.4, 3.0))
    for level, (p_min, p_max, r_min, r_max) in enumerate(limits):
        selected = torch.tensor(difficulty == level)
        assert bool(((position_error[selected] >= p_min - 0.003)
                     & (position_error[selected] <= p_max + 0.003)).all())
        assert bool(((rotation_error[selected] >= r_min - 0.003)
                     & (rotation_error[selected] <= r_max + 0.003)).all())
    reference = TeleopWorkspaceReferenceBank(target, "cpu")
    assert reference.version == "marvin_teleop_workspace_bank_v4"
