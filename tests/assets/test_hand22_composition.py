"""Protect Cube hand equivalence, original arm FK and explicit SI drives."""

import hashlib
import json

from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_KEYFRAMES, PACKAGED_MJCF, make_robot_cfg

from scripts.assets.validate_hand22_composition import validate


def test_composition_against_independent_sources():
    assert validate()["pass"]


def test_explicit_drives_and_generated_source_hash():
    meta = json.loads(PACKAGED_KEYFRAMES.read_text())
    assert hashlib.sha256(PACKAGED_MJCF.read_bytes()).hexdigest() == meta["source_sha256"]
    cfg = make_robot_cfg()
    actuator = cfg.actuators["all"]
    assert len(actuator.stiffness) == len(actuator.damping) == len(actuator.joint_effort_limit) == 54
    for name, values in meta["drives"].items():
        assert actuator.stiffness[name] == values["kp"]
        assert actuator.damping[name] == values["kd"]
        assert actuator.joint_effort_limit[name] == values["effort"]
