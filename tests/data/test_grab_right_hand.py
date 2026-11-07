"""Contract checks for GRAB -> Wuji right-hand fingertip targets."""

from pathlib import Path

import numpy as np
import pytest
from dexgym.data.grab_mano import load_mano_model, load_subject_template, mano_keypoints
from dexgym.data.wuji_retarget import RightHandFK, WujiRetargetSession, retarget_sequence

ROOT = Path(__file__).resolve().parents[2]
MANO = ROOT / "data/models/mano_v1_2/MANO_RIGHT.pkl"
TEMPLATE = ROOT / "data/tools/subject_meshes/male/s1_rhand.ply"
RIGHT_XML = Path(
    "/home/ubuntu/teleop_ws/src/whole_body_mujoco_driver/model/source/wujihand2_beta2/mjcf/right.xml"
)
SDK = Path("/home/ubuntu/teleop_ws/third_party/wuji_c_api/lib/libwuji_sdk_c.so")


@pytest.mark.skipif(not MANO.is_file() or not TEMPLATE.is_file(), reason="local GRAB/MANO data unavailable")
def test_mano_landmarks_are_wrist_relative_and_finger_ordered():
    model = load_mano_model(MANO)
    template = load_subject_template(TEMPLATE)
    pose = np.zeros((2, 45), dtype=np.float32)
    pose[1, 1] = 0.4  # rotate index MCP around an axis transverse to the finger
    points = mano_keypoints(pose, template, model)

    assert points.shape == (2, 21, 3)
    np.testing.assert_allclose(points[:, 0], 0.0, atol=1e-6)
    assert np.linalg.norm(points[0, 8] - points[0, 7]) < 0.05  # index tip follows index DIP
    assert np.linalg.norm(points[0, 4] - points[0, 3]) < 0.05  # thumb tip follows thumb IP
    assert np.linalg.norm(points[0, 20] - points[0, 19]) < 0.05  # pinky tip follows pinky DIP
    assert np.linalg.norm(points[1, 8] - points[0, 8]) > 0.005
    np.testing.assert_allclose(points[1, 0], points[0, 0], atol=1e-6)


@pytest.mark.skipif(not SDK.is_file(), reason="local Wuji SDK unavailable")
def test_sdk_accepts_right_hand_landmarks_and_rejects_wrong_shape():
    with WujiRetargetSession(SDK) as session:
        with pytest.raises(ValueError, match="21"):
            session.step(np.zeros((20, 3), dtype=np.float32))
        points = np.zeros((21, 3), dtype=np.float32)
        points[:, 0] = np.arange(21) * 0.005
        q = session.step(points)
    assert q.shape == (20,)
    assert np.isfinite(q).all()


@pytest.mark.skipif(not RIGHT_XML.is_file(), reason="local Wuji hand model unavailable")
def test_fk_returns_reachable_fingertips_in_wrist_frame():
    fk = RightHandFK(RIGHT_XML)
    q, clipped = fk.clamp(np.full(20, 100.0, dtype=np.float32))
    assert clipped == 20
    assert np.isfinite(q).all()
    assert np.all(q <= fk.upper + 1e-6)
    points = fk.tips(q)
    assert points.shape == (5, 3)
    assert np.isfinite(points).all()
    assert np.max(np.linalg.norm(points, axis=-1)) < 0.35


@pytest.mark.skipif(not all(p.is_file() for p in (MANO, TEMPLATE, RIGHT_XML, SDK)), reason="local inputs absent")
def test_retargeted_sequence_targets_are_exact_robot_fk():
    model = load_mano_model(MANO)
    template = load_subject_template(TEMPLATE)
    pose = np.zeros((3, 45), dtype=np.float32)
    pose[:, 0] = [0.0, 0.25, 0.5]
    points = mano_keypoints(pose, template, model)
    fk = RightHandFK(RIGHT_XML)
    with WujiRetargetSession(SDK) as session:
        q, tips, metrics = retarget_sequence(points, session, fk)
    assert q.shape == (3, 20)
    assert tips.shape == (3, 5, 3)
    assert metrics["frames"] == 3
    assert np.isfinite(q).all() and np.isfinite(tips).all()
    np.testing.assert_allclose(tips, np.stack([fk.tips(frame) for frame in q]), atol=1e-6)
