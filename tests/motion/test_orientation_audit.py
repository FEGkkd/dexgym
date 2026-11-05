from pathlib import Path

from dexgym.motion.bvh import human_fk, parse_bvh
from dexgym.motion.orientation_audit import audit_hand_orientation
from test_bvh import BVH


def test_one_axis_wrist_is_reconstructed_but_not_independent_full_6d(tmp_path: Path):
    source = tmp_path / "one_axis.bvh"
    source.write_text(BVH.replace("Left", "Right").replace("0 0 0 90 0 0 0 0 0 0 0 90", "0 0 0 90 0 0 0 0 0 0 0 10"))
    motion = parse_bvh(source)
    fk = human_fk(motion, scale_m_per_unit=0.01, remove_t_pose=True)
    audit = audit_hand_orientation(motion, fk, hand_names=("RightHand",))
    hand = audit["hands"]["right"]
    assert hand["orientation_available"]
    assert hand["orientation_reconstruction_valid"]
    assert hand["orientation_source"] == "bvh_reconstructed"
    assert hand["active_local_rotation_axes"] == ["X"] or hand["active_local_rotation_axes"] == ["Z"]
    assert hand["orientation_confidence"] == "low"
    assert hand["rotation_quality"] == "implausible_angular_speed"
    assert not audit["full_6d_accepted"]


def test_static_wrist_is_not_intrinsically_invalid_when_source_is_validated(tmp_path: Path):
    source = tmp_path / "static.bvh"
    source.write_text(BVH.replace("Left", "Right").replace("0 0 0 90 0 0 0 0 0 0 0 90", "0 0 0 90 0 0 0 0 0 0 0 0"))
    motion = parse_bvh(source)
    fk = human_fk(motion, scale_m_per_unit=0.01, remove_t_pose=True)
    audit = audit_hand_orientation(motion, fk, hand_names=("RightHand",), source_full_orientation_validated=True)
    hand = audit["hands"]["right"]
    assert hand["active_local_rotation_axes"] == []
    assert hand["full_6d_accepted"]
    assert hand["orientation_confidence"] == "high"
