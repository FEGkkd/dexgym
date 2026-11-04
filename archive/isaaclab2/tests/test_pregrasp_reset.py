"""CPU contracts for geometry-only bimanual pregrasp Reset states."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math

import pytest

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.pregrasp_reset import (
    PregraspResetBank, PregraspResetGenerator, PregraspTemplate,
    TEMPLATE_IDS, validate_pregrasp_state,
)


def _spec(side):
    template = PregraspTemplate.load(TEMPLATE_IDS[0 if side == "left" else 1])
    names = tuple(template.joint_center_rad)
    return SimpleNamespace(
        side=side, variant="beta1", source_commit="pinned-test-commit",
        usd_sha256=f"{side}-usd", asset_tree_sha256=f"{side}-tree",
        joint_names=names,
        joints=tuple(SimpleNamespace(name=name, lower=-2.0, upper=3.0)
                     for name in names),
    )


def test_two_named_templates_draw_reproducible_distinct_joint_preshapes():
    bottle = BottleConfig()
    specs = (_spec("left"), _spec("right"))
    generator = PregraspResetGenerator(bottle, *specs)
    a = generator.draw(seed=42)
    b = generator.draw(seed=42)
    c = generator.draw(seed=43)
    assert a == b and a != c
    assert a.mode == "geometry_pregrasp"
    assert a.metadata["pregrasp_template_ids"] == list(TEMPLATE_IDS)
    assert a.metadata["geometry_qualified"] is False
    assert a.metadata["hold_validated"] is False
    assert not a.validated and a.validation is None
    assert a.left.joint_position != c.left.joint_position
    assert a.right.joint_position != c.right.joint_position
    validate_pregrasp_state(a, bottle, *specs)


def test_pregrasp_state_rejects_joint_limit_target_and_cap_mismatch():
    bottle = BottleConfig()
    specs = (_spec("left"), _spec("right"))
    sample = PregraspResetGenerator(bottle, *specs).draw(seed=4)
    with pytest.raises(ValueError, match="physical and controller"):
        validate_pregrasp_state(replace(
            sample, right=replace(sample.right, joint_target=(0.0,) * 20)),
            bottle, *specs)
    with pytest.raises(ValueError, match="Cap pose"):
        validate_pregrasp_state(replace(
            sample, cap=replace(sample.cap,
                                root_pose=(0, 0, .7, 0, 0, 0, 1))),
            bottle, *specs)
    bad = list(sample.left.joint_position)
    bad[0] = 10
    with pytest.raises(ValueError, match="official limit"):
        validate_pregrasp_state(replace(
            sample, left=replace(sample.left, joint_position=tuple(bad),
                                 joint_target=tuple(bad))), bottle, *specs)


def test_separate_geometry_bank_rejects_unverified_and_preserves_hold_flag(tmp_path):
    bottle = BottleConfig()
    specs = (_spec("left"), _spec("right"))
    candidate = PregraspResetGenerator(bottle, *specs).draw(seed=17)
    bank = PregraspResetBank(tmp_path / "pregrasp.json")
    with pytest.raises(ValueError, match="geometry-qualified"):
        bank.add(candidate)
    evidence = {"official_mjcf": {
                    "qualified": True,
                    "backend": "official_beta1_MJCF_mesh_and_engineering_cylinder_proxy"},
                "native_physx": {
                    "qualified": True,
                    "backend": "official_native_USD_PhysX_one_step_contact"}}
    certified = replace(candidate, metadata={**candidate.metadata,
        "geometry_qualified": True, "geometry_evidence": evidence})
    bank.add(certified)
    assert bank.load() == (certified,)
    for seed in (0, 1, 2):
        sample = bank.draw_runtime(seed=seed)
        validate_pregrasp_state(sample, bottle, *specs)
        assert sample.metadata["geometry_qualified"] is True
        assert sample.metadata["rigid_transform_preserves_collision_clearance"] is True
        assert sample.validated is False
        assert math.dist(sample.bottle.root_pose[:3], certified.bottle.root_pose[:3]) < .04
        assert sample.left.joint_position == certified.left.joint_position
    assert bank.draw_runtime(seed=2) == bank.draw_runtime(seed=2)
    with pytest.raises(ValueError, match="above the floor"):
        bank.select(seed=0, bottle_position_xyz=(0, 0, .05))


def test_template_rejects_wrong_bottle_profile():
    specs = (_spec("left"), _spec("right"))
    with pytest.raises(ValueError, match="70 mm"):
        PregraspResetGenerator(BottleConfig(bottle_radius=.03), *specs)


def test_checked_in_bank_uses_actual_right_tripod_fingertip_sites():
    bank_path = (Path(__file__).resolve().parents[1] / "docs/assets" /
                 "phase0_pregrasp_reset_bank.json")
    samples = PregraspResetBank(bank_path).load()
    assert len(samples) >= 3
    for sample in samples:
        geometry = sample.metadata["geometry_evidence"]["official_mjcf"]
        sites = geometry["right_fingertip_sites"]
        for digit in ("thumb", "index_finger", "middle_finger"):
            assert .001 <= sites[digit]["radial_gap_to_cap_sidewall_m"] <= .012
            assert abs(sites[digit]["axial_rel_cap_m"]) <= .0115
        for digit in ("ring_finger", "pinky"):
            assert sites[digit]["radial_gap_to_cap_sidewall_m"] >= .020
        assert geometry["right_thumb_tripod_site_opposition_rad"] >= math.radians(125)
