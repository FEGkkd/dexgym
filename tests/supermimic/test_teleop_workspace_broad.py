"""Regression tests for expanding causal TCP targets beyond the Home neighborhood."""

import mujoco
import numpy as np
import torch


def test_broad_bank_has_long_safe_targets_and_nonlocal_resets(tmp_path):
    from dexgym.motion.teleop_workspace import (
        DEFAULT_MJCF, TeleopWorkspaceReferenceBank, generate_teleop_workspace_bank)

    path = tmp_path / "broad.npz"
    report = generate_teleop_workspace_bank(
        path, seed=28, split_counts=(3, 1, 1), frames=601, profile="broad_v2")
    bank = TeleopWorkspaceReferenceBank(path, "cpu")
    with np.load(path, allow_pickle=False) as data:
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
        home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")]
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                  for name in data["joint_names_arm"]]
        addresses = model.jnt_qposadr[joints]
        assert str(data["version"]) == "marvin_teleop_workspace_bank_v2"
        assert data["tcp_pos"].shape == (5, 601, 2, 3)
        assert float(np.abs(data["q_reset_witness"][:, 0] - home[addresses]).max()) > 0.5
        assert float(np.abs(np.diff(data["q_reset_witness"], axis=1)).max()) <= 0.009
        assert float(np.ptp(data["tcp_pos"][:, 0, 0, 0])) > 0.25
    assert report["frames"] == 601
    assert bank.lengths.tolist() == [601] * 5
    assert set(bank.hashes) == {"marvin_teleop_workspace_bank_v2"}
    assert bank.pose_at(torch.tensor([0, 4]), torch.tensor([0, 600])).shape == (2, 18)


def test_broad_bank_rejects_too_short_or_misaligned_episode(tmp_path):
    import pytest
    from dexgym.motion.teleop_workspace import generate_teleop_workspace_bank

    for frames in (201, 602):
        with pytest.raises(ValueError, match="broad"):
            generate_teleop_workspace_bank(
                tmp_path / "invalid.npz", seed=1, split_counts=(1, 1, 1),
                frames=frames, profile="broad_v2")


def test_reference_episode_duration_controls_full_physical_evaluation():
    import pytest
    from dexgym.motion.teleop_workspace import reference_episode_control_steps

    assert reference_episode_control_steps(torch.tensor([201]), 4.0) == 200
    assert reference_episode_control_steps(torch.tensor([601, 401]), 12.0) == 600
    with pytest.raises(ValueError, match="reference duration"):
        reference_episode_control_steps(torch.tensor([601]), 4.0)


def test_far_goal_probe_separates_safe_reset_from_hidden_fk_goal(tmp_path):
    from dexgym.motion.teleop_workspace import (
        DEFAULT_MJCF, TeleopWorkspaceReferenceBank, generate_teleop_reach_probe_bank)

    path = tmp_path / "far_goals.npz"
    generate_teleop_reach_probe_bank(path, seed=19, split_counts=(1, 1, 3), frames=601)
    bank = TeleopWorkspaceReferenceBank(path, "cpu")
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
    data = mujoco.MjData(model)
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")]
    with np.load(path, allow_pickle=False) as raw:
        assert not any("goal_q" in field for field in raw.files)
        np.testing.assert_array_equal(raw["tcp_pos"][:, 0], raw["tcp_pos"][:, -1])
        np.testing.assert_array_equal(raw["q_reset_witness"][:, 0],
                                      raw["q_reset_witness"][:, -1])
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                  for name in raw["joint_names_arm"]]
        addresses = model.jnt_qposadr[joints]
        for index in range(5):
            data.qpos[:] = home
            data.qpos[addresses] = raw["q_reset_witness"][index, 0]
            mujoco.mj_forward(model, data)
            assert data.ncon == 0
            actual = np.stack([data.site_xpos[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{side}")] for side in "LR"])
            assert np.linalg.norm(actual - raw["tcp_pos"][index, 0], axis=-1).min() > 0.05
    assert bank.pose_at(torch.tensor([2]), torch.tensor([400])).shape == (1, 18)
