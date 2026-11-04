"""Small, source-backed checks for the external Marvin/Wuji model contract."""

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from dexgym.assets.marvin_wuji.frames import angular_distance_deg, wxyz_to_xyzw
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract, qpos_to_newton
from dexgym.assets.marvin_wuji.robot_cfg import (
    JOINT_ORDER,
    PACKAGED_KEYFRAMES,
    PACKAGED_USD,
    SOURCE_SHA256,
    make_robot_cfg,
)


def test_quaternion_convention_and_sign():
    np.testing.assert_array_equal(wxyz_to_xyzw(np.array([1.0, 0.0, 0.0, 0.0])), [0.0, 0.0, 0.0, 1.0])
    assert angular_distance_deg([0, 0, 0, 1], [0, 0, 0, -1]) == 0.0
    assert angular_distance_deg([0, 0, 0, 1], [0, 0, 1, 0]) == 180.0


def test_packaged_usd_is_validated_asset():
    import hashlib

    assert PACKAGED_USD.is_file()
    assert hashlib.sha256(PACKAGED_USD.read_bytes()).hexdigest() == (
        "4643b25b3ddc1412f83ef2f089bfc4fe6dd90ad2f36ef76efd9d48112ecd1ceb"
    )
    cfg = make_robot_cfg()
    assert cfg.spawn.usd_path == str(PACKAGED_USD.resolve())
    assert tuple(cfg.joint_ordering) == JOINT_ORDER
    keyframes = json.loads(PACKAGED_KEYFRAMES.read_text())
    assert keyframes["source_sha256"] == SOURCE_SHA256
    assert tuple(keyframes["joint_order"]) == JOINT_ORDER
    assert set(keyframes["keyframes"]) == {"hardware_zero", "home", "measured_j5_zero", "forward_pose"}
    assert all(len(q) == 54 for q in keyframes["keyframes"].values())


def test_external_source_joint_contract():
    source = os.environ.get("DEXGYM_MARVIN_MJCF")
    if not source:
        pytest.skip("Set DEXGYM_MARVIN_MJCF for external source contract check")
    import hashlib

    path = Path(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256
    model = mujoco.MjModel.from_xml_path(str(path))
    contract = inspect_joint_contract(model)
    assert contract.names == JOINT_ORDER
    assert tuple(map(len, contract.groups.values())) == (7, 7, 20, 20)
    assert [contract.ctrl_by_name[f"Joint{i}_L"] for i in range(1, 8)] == list(range(7))
    assert [contract.ctrl_by_name[f"Joint{i}_R"] for i in range(1, 8)] == list(range(7, 14))
    assert any(contract.qpos_by_name[name] != contract.ctrl_by_name[name] for name in contract.names)
    reversed_names = list(reversed(contract.names))
    assert np.array_equal(
        qpos_to_newton(np.arange(54), contract, reversed_names),
        np.array([contract.qpos_by_name[name] for name in reversed_names]),
    )
