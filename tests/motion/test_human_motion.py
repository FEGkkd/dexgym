from pathlib import Path

import numpy as np
import pytest
from dexgym.motion.bvh import human_fk, parse_bvh, resample_motion
from dexgym.motion.human_motion import build_upper_body
from test_bvh import BVH


def test_schema_rejects_missing_bilateral_nodes(tmp_path: Path):
    source = tmp_path / "one_hand.bvh"
    source.write_text(BVH)
    fk = resample_motion(human_fk(parse_bvh(source), scale_m_per_unit=0.01, remove_t_pose=True))
    with pytest.raises(ValueError, match="LeftShoulder"):
        build_upper_body(fk, {"hands": {}})


def test_real_cmu_m1_schema_if_materialized():
    path = Path("/home/ubuntu/dexgym_data/cmu/cmu-mocap/data/141/141_16.bvh")
    if not path.exists():
        pytest.skip("External CMU data not materialized")
    from dexgym.motion.orientation_audit import audit_hand_orientation

    motion = parse_bvh(path)
    fk = resample_motion(human_fk(motion, scale_m_per_unit=0.07, remove_t_pose=True))
    audit = audit_hand_orientation(motion, human_fk(motion, scale_m_per_unit=0.07, remove_t_pose=True))
    data = build_upper_body(fk, audit)
    assert data["hand_pose"].shape == (len(fk.time), 2, 7)
    assert data["hand_angular_velocity"].shape == (len(fk.time), 2, 3)
    assert not data["hand_orientation_valid"].any()
    assert np.allclose(np.linalg.norm(data["hand_pose"][..., 3:], axis=-1), 1, atol=1e-5)
