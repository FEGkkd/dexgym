"""Offline contracts for reproducible bottle-relative initial states."""

import math
from types import SimpleNamespace

import pytest

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.grasp_init import BimanualGraspInitializer, GraspInitBank


def _spec(side):
    return SimpleNamespace(
        side=side, variant="beta1", source_commit="official-test-commit",
        usd_sha256=f"{side}-usd", asset_tree_sha256=f"{side}-tree",
        joint_names=tuple(f"{side}_{i}" for i in range(20)),
        joints=tuple(SimpleNamespace(name=f"{side}_{i}", lower=-0.5, upper=1.0)
                     for i in range(20)),
    )


def test_modes_keep_full_state_and_do_not_claim_unvalidated_grasp():
    init = BimanualGraspInitializer(BottleConfig(), _spec("left"), _spec("right"))
    for mode in ("stable_pregrasp", "weak_contact", "near_contact"):
        sample = init.generate(mode=mode, seed=13, bottle_pose=(0, 0, 0.5, 0, 0, 0, 1))
        assert sample.mode == mode
        assert not sample.validated
        assert len(sample.left.joint_position) == len(sample.right.joint_position) == 20
        assert sample.left.wrist_target_pose == sample.left.root_pose
        assert sample.left.joint_target == sample.left.joint_position
        assert sample.cap.root_pose[2] > sample.bottle.root_pose[2]
        assert sample.metadata["asset_variant"] == "beta1"
        assert init.generate(mode=mode, seed=13, bottle_pose=(0, 0, 0.5, 0, 0, 0, 1)) == sample


def test_bottle_rotation_rotates_all_bottle_relative_positions():
    init = BimanualGraspInitializer(BottleConfig(), _spec("left"), _spec("right"))
    a = init.generate(mode="near_contact", seed=2, bottle_pose=(0, 0, 0.5, 0, 0, 0, 1))
    qz = math.sqrt(0.5)
    b = init.generate(mode="near_contact", seed=2, bottle_pose=(0, 0, 0.5, 0, 0, qz, qz))
    assert b.left.root_pose[0] == pytest.approx(-a.left.root_pose[1], abs=1e-6)
    assert b.left.root_pose[1] == pytest.approx(a.left.root_pose[0], abs=1e-6)
    assert b.cap.root_pose[2] == pytest.approx(a.cap.root_pose[2], abs=1e-6)


def test_bank_rejects_candidate_without_gpu_hold_evidence(tmp_path):
    init = BimanualGraspInitializer(BottleConfig(), _spec("left"), _spec("right"))
    candidate = init.generate(mode="stable_pregrasp", seed=7)
    bank = GraspInitBank(tmp_path / "bank.json")
    with pytest.raises(ValueError, match="validated"):
        bank.add(candidate)
    with pytest.raises(ValueError, match="metric"):
        candidate.with_validation({"fixture": "gravity_on_no_support", "passed": True})
    assert not bank.path.exists()


def test_bank_round_trip_preserves_full_validated_state(tmp_path):
    init = BimanualGraspInitializer(BottleConfig(), _spec("left"), _spec("right"))
    sample = init.generate(mode="stable_pregrasp", seed=17).with_validation({
        "fixture": "gravity_on_no_support", "passed": True,
        "duration_s": 1.0, "hold_displacement_m": 0.01,
        "hold_rotation_rad": 0.05, "left_only_displacement_m": 0.01,
        "left_contact_ratio": 0.9, "right_contact_ratio": 0.9,
        "left_only_contact_ratio": 0.9, "ground_contact_ratio": 0.0,
        "max_contact_peak_n": 5.0, "max_peak_force_guard_n": 100.0,
    })
    bank = GraspInitBank(tmp_path / "bank.json")
    bank.add(sample)
    assert bank.load() == (sample,)
