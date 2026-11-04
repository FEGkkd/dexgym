"""The sampled target FK must agree with the pinned MuJoCo model."""

import os

import mujoco
import numpy as np
import pytest
import torch
from dexgym.tasks.marvin_s0.kinematics import MarvinArmFK, orientation_error


@pytest.fixture(scope="module")
def model():
    source = os.environ.get("DEXGYM_MARVIN_MJCF")
    if not source:
        pytest.skip("DEXGYM_MARVIN_MJCF is required for the external asset")
    return mujoco.MjModel.from_xml_path(source)


def test_batch_fk_matches_mujoco_for_legal_random_poses(model):
    fk = MarvinArmFK(model, device="cpu")
    rng = np.random.default_rng(91)
    limits = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in fk.names])
    q = limits[:, 0] + rng.uniform(0.1, 0.9, (128, 14)) * (limits[:, 1] - limits[:, 0])
    poses = fk(torch.as_tensor(q, dtype=torch.float64))
    data = mujoco.MjData(model)
    for i in range(len(q)):
        for j, name in enumerate(fk.names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = q[i, j]
        mujoco.mj_forward(model, data)
        for side, suffix in enumerate(("L", "R")):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{suffix}")
            assert np.linalg.norm(poses[i, side, :3].numpy() - data.xpos[bid]) < 1e-8
            expected = data.xquat[bid, [1, 2, 3, 0]]
            assert abs(np.dot(poses[i, side, 3:].numpy(), expected)) > 1 - 1e-10


def test_orientation_error_is_sign_invariant():
    current = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0, 0.0, -1.0]], dtype=torch.float64)
    assert torch.max(torch.abs(orientation_error(target, current))) < 1e-12
