from pathlib import Path

import numpy as np
from dexgym.motion.bvh import human_fk, parse_bvh, resample_motion
from scipy.spatial.transform import Rotation

BVH = """HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT LeftForeArm
 {
  OFFSET 0 10 0
  CHANNELS 3 Yrotation Zrotation Xrotation
  JOINT LeftHand
  {
   OFFSET 0 10 0
   CHANNELS 3 Xrotation Yrotation Zrotation
   End Site
   {
    OFFSET 0 2 0
   }
  }
 }
}
MOTION
Frames: 3
Frame Time: 0.008333333333
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 90 0 0 0 0 0 0 0 0
0 0 0 90 0 0 0 0 0 0 0 90
"""


def test_hierarchy_channels_and_declared_euler_order(tmp_path: Path):
    path = tmp_path / "sample.bvh"
    path.write_text(BVH)
    motion = parse_bvh(path)
    assert motion.frames.shape == (3, 12)
    assert [joint.name for joint in motion.joints[:3]] == ["Hips", "LeftForeArm", "LeftHand"]
    assert motion.joints[2].parent == 1
    assert motion.joints[2].channels == ("Xrotation", "Yrotation", "Zrotation")
    fk = human_fk(motion, scale_m_per_unit=0.01, remove_t_pose=True)
    assert fk.time.shape == (2,)
    np.testing.assert_allclose(fk.time, [0, motion.frame_time], atol=1e-10)
    np.testing.assert_allclose(fk.position[0, 2], [-0.2, 0, 0], atol=1e-8)
    expected = Rotation.from_euler("ZXY", [90, 0, 0], degrees=True).as_matrix()
    np.testing.assert_allclose(fk.rotation[0, 2], expected, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(fk.quat_xyzw, axis=-1), 1, atol=1e-8)


def test_hand_local_rotation_and_resample_slerp(tmp_path: Path):
    path = tmp_path / "sample.bvh"
    path.write_text(BVH)
    fk = human_fk(parse_bvh(path), scale_m_per_unit=0.01, remove_t_pose=True)
    np.testing.assert_allclose(
        fk.local_rotation[1, 2], Rotation.from_euler("XYZ", [0, 0, 90], degrees=True).as_matrix()
    )
    sampled = resample_motion(fk, sample_rate_hz=50)
    assert sampled.time[0] == 0
    np.testing.assert_allclose(sampled.time[-1], fk.time[-1])
    assert np.all(np.isfinite(sampled.quat_xyzw))
    assert np.all(np.sum(sampled.quat_xyzw[1:] * sampled.quat_xyzw[:-1], axis=-1) >= 0)


def test_120_to_50_hz_preserves_duration_and_mid_rotation(tmp_path: Path):
    hierarchy = BVH.split("MOTION\n")[0]
    frames = ["0 0 0 0 0 0 0 0 0 0 0 0"]
    for index in range(121):
        frames.append(f"0 0 0 0 0 0 0 0 0 0 0 {index * 0.75}")
    source = tmp_path / "one_second.bvh"
    source.write_text(hierarchy + "MOTION\nFrames: 122\nFrame Time: 0.008333333333\n" + "\n".join(frames) + "\n")
    fk = resample_motion(human_fk(parse_bvh(source), scale_m_per_unit=0.01, remove_t_pose=True))
    assert len(fk.time) == 51
    np.testing.assert_allclose(fk.time[-1], 1, atol=1e-8)
    np.testing.assert_allclose(
        fk.local_rotation[25, 2], Rotation.from_euler("Z", 45, degrees=True).as_matrix(), atol=1e-8
    )
    np.testing.assert_allclose(np.linalg.norm(fk.quat_xyzw, axis=-1), 1, atol=1e-8)


def test_resampling_preserves_nonroot_translation_channels(tmp_path: Path):
    source = tmp_path / "translated_child.bvh"
    source.write_text("""HIERARCHY
ROOT Root
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Child
 {
  OFFSET 0 1 0
  CHANNELS 4 Xposition Zrotation Yrotation Xrotation
  End Site
  {
   OFFSET 0 1 0
  }
 }
}
MOTION
Frames: 3
Frame Time: 0.5
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 10 0 0 0
""")
    fk = human_fk(parse_bvh(source), scale_m_per_unit=0.01, remove_t_pose=True)
    sampled = resample_motion(fk, sample_rate_hz=4)
    np.testing.assert_allclose(sampled.position[1, 1], [0.05, 0.01, 0], atol=1e-8)


def test_real_cmu_tpose_side_and_hand_channel_audit():
    path = Path("/home/ubuntu/dexgym_data/cmu/cmu-mocap/data/141/141_16.bvh")
    if not path.exists():
        import pytest

        pytest.skip("External CMU data not materialized")
    motion = parse_bvh(path)
    fk = human_fk(motion, scale_m_per_unit=1, remove_t_pose=False)
    names = {joint.name: i for i, joint in enumerate(motion.joints)}
    assert fk.position[0, names["LeftHand"], 0] > fk.position[0, names["RightHand"], 0]
    for name in ("LeftHand", "RightHand"):
        joint = motion.joints[names[name]]
        assert joint.channels == ("Zrotation", "Yrotation", "Xrotation")
        channels = motion.frames[1:, joint.channel_start : joint.channel_start + 3]
        assert np.ptp(channels[:, 0]) == 0
        assert np.ptp(channels[:, 1]) == 0
        assert np.ptp(channels[:, 2]) > 1
