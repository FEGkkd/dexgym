"""Offline contracts for reproducible bottle-relative initial states."""

import math
from types import SimpleNamespace

import pytest

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.grasp_init import BimanualGraspInitializer, GraspInitBank
from dexteroushands_lab.scenes.grasp_synthesis import GraspSynthesisFixture


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
        "initial_min_separation_m": -0.001,
        "max_initial_penetration_guard_m": 0.002,
        "left_thumb_contact_time_s": 0.4,
        "left_other_finger_count": 2,
        "opposing_contact_coverage_ratio": 0.9,
        "support_released": True, "reset_replay_passed": True,
    })
    bank = GraspInitBank(tmp_path / "bank.json")
    bank.add(sample)
    assert bank.load() == (sample,)


@pytest.mark.parametrize("overrides", [
    {"initial_min_separation_m": -0.04},
    {"initial_min_separation_m": -0.0021},
    {"max_initial_penetration_guard_m": 0.005},
    {"max_contact_peak_n": 312.0},
    {"max_peak_force_guard_n": 500.0},
    {"opposing_contact_coverage_ratio": 0.1},
    {"support_released": False},
    {"reset_replay_passed": False},
])
def test_bank_rejects_unphysical_or_unreplayed_grasp(overrides):
    init = BimanualGraspInitializer(BottleConfig(), _spec("left"), _spec("right"))
    sample = init.generate(mode="stable_pregrasp", seed=17)
    metrics = {
        "fixture": "gravity_on_no_support", "passed": True,
        "duration_s": 1.0, "hold_displacement_m": 0.01,
        "hold_rotation_rad": 0.05, "left_only_displacement_m": 0.01,
        "left_contact_ratio": 0.9, "right_contact_ratio": 0.9,
        "left_only_contact_ratio": 0.9, "ground_contact_ratio": 0.0,
        "max_contact_peak_n": 5.0, "max_peak_force_guard_n": 100.0,
        "initial_min_separation_m": -0.001,
        "max_initial_penetration_guard_m": 0.002,
        "left_thumb_contact_time_s": 0.4,
        "left_other_finger_count": 2,
        "opposing_contact_coverage_ratio": 0.9,
        "support_released": True, "reset_replay_passed": True,
    }
    metrics.update(overrides)
    with pytest.raises(ValueError):
        sample.with_validation(metrics)


def test_replay_opposition_requires_thumb_and_two_other_fingers():
    contacts = {"bodies": {
        "Bottle|l_thumb_distal": {"samples": 30},
        "l_index_finger_distal|Bottle": {"samples": 26},
        "Bottle|l_middle_finger_distal": {"samples": 25},
        "Bottle|l_ring_finger_distal": {"samples": 1},
    }}
    times = GraspSynthesisFixture._left_finger_contact_times(contacts, 0.002)
    assert GraspSynthesisFixture._opposition_contact_passed(times)
    assert not GraspSynthesisFixture._opposition_contact_passed(
        {"thumb": 0.06, "index": 0.052, "middle": 0.049})
