import json
from pathlib import Path

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.bottle.mimic_contract import mimic_reset_geometry
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.reset_bank import (
    make_candidate, make_hybrid_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_changes_local_hand_shape_but_preserves_screw_geometry():
    stable = grasp_sample_from_dict(json.loads(
        (ROOT / "docs/assets/phase1_mimic_pgs_grasp_init_bank.json").read_text())[0])
    natural = grasp_sample_from_dict(json.loads(
        (ROOT / "docs/assets/phase0_pregrasp_reset_bank.json").read_text())[9])
    a = make_candidate(stable, natural, level="A", seed=123)
    b = make_candidate(stable, natural, level="A", seed=124)
    assert a.mode == "geometry_pregrasp" and not a.validated
    assert not a.metadata["geometry_qualified"]
    assert a.left.root_pose != natural.left.root_pose
    assert a.left.joint_position != natural.left.joint_position
    assert a.left.joint_position != b.left.joint_position
    assert a.metadata["phase1b_reset_level"] == "A"
    check = mimic_reset_geometry(a.bottle.root_pose, a.cap.root_pose,
                                 a.thread_initial_angle_rad, BottleConfig())
    assert abs(check["pitch_error_m"]) < 5e-5


def test_level_a_closes_more_than_b_for_same_seed():
    stable = grasp_sample_from_dict(json.loads(
        (ROOT / "docs/assets/phase1_mimic_pgs_grasp_init_bank.json").read_text())[0])
    natural = grasp_sample_from_dict(json.loads(
        (ROOT / "docs/assets/phase0_pregrasp_reset_bank.json").read_text())[9])
    a = make_candidate(stable, natural, level="A", seed=123)
    b = make_candidate(stable, natural, level="B", seed=123)
    assert a.metadata["left_blend"] > b.metadata["left_blend"]


def test_hybrid_keeps_mimic_pose_and_local_natural_right_tripod():
    stable = grasp_sample_from_dict(json.loads(
        (ROOT / "docs/assets/phase1_mimic_pgs_grasp_init_bank.json").read_text())[0])
    natural = grasp_sample_from_dict(json.loads(
        (ROOT / "docs/assets/phase0_pregrasp_reset_bank.json").read_text())[15])
    candidate = make_hybrid_candidate(stable, natural, seed=45001,
                                       left_open_blend=.05,
                                       left_outward_m=.0002,
                                       right_inward_m=.001)
    assert not candidate.validated and not candidate.metadata["geometry_qualified"]
    assert candidate.left.joint_position != stable.left.joint_position
    assert candidate.right.joint_position == natural.right.joint_position
    assert candidate.right.root_pose != stable.right.root_pose
    check = mimic_reset_geometry(candidate.bottle.root_pose, candidate.cap.root_pose,
                                 candidate.thread_initial_angle_rad, BottleConfig())
    assert abs(check["pitch_error_m"]) < 5e-5
