"""The experimental Newton profile must stay tied to the pinned source drives."""

import os

import mujoco
import pytest
from dexgym.assets.marvin_wuji.controller import (
    ARM_JOINT_NAMES,
    ARM_KD_SOURCE,
    ARM_KP_SOURCE,
    USD_ANGULAR_GAIN_TO_SI,
    match_mujoco_arm_gains_in_memory,
)
from dexgym.assets.marvin_wuji.robot_cfg import resolve_source_mjcf


def test_arm_tracking_profile_matches_pinned_xml():
    if not os.environ.get("DEXGYM_MARVIN_MJCF"):
        pytest.skip("Set DEXGYM_MARVIN_MJCF for external source controller contract")
    model = mujoco.MjModel.from_xml_path(str(resolve_source_mjcf()))
    assert len(ARM_JOINT_NAMES) == 14
    for index, name in enumerate(ARM_JOINT_NAMES):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = next(aid for aid in range(model.nu) if model.actuator_trnid[aid, 0] == joint)
        assert model.actuator_gainprm[actuator, 0] == ARM_KP_SOURCE[index % 7]
        assert -model.actuator_biasprm[actuator, 2] == ARM_KD_SOURCE[index % 7]
        assert model.actuator_forcerange[actuator, 1] > 0


def test_matched_mujoco_gains_leave_effort_limits_and_xml_untouched():
    if not os.environ.get("DEXGYM_MARVIN_MJCF"):
        pytest.skip("Set DEXGYM_MARVIN_MJCF for external source controller contract")
    import hashlib

    source = resolve_source_mjcf()
    digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(source))
    limits_before = model.actuator_forcerange.copy()
    result = match_mujoco_arm_gains_in_memory(model, damping_scale_source=0.5)
    assert result["source_xml_unchanged"]
    assert (model.actuator_forcerange == limits_before).all()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest_before
    for index, name in enumerate(ARM_JOINT_NAMES):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = next(aid for aid in range(model.nu) if model.actuator_trnid[aid, 0] == joint)
        assert model.actuator_gainprm[actuator, 0] == pytest.approx(ARM_KP_SOURCE[index % 7] * USD_ANGULAR_GAIN_TO_SI)
        assert -model.actuator_biasprm[actuator, 2] == pytest.approx(ARM_KD_SOURCE[index % 7] * 0.5)
