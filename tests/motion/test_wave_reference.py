import numpy as np
from dexgym.motion.reference_library import MarvinReference, bvh_to_marvin_rotation, smooth_so3
from scipy.spatial.transform import Rotation


def test_mujoco_replay_steps_full_model(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "motion"))
    from replay_marvin_reference_mujoco import replay

    source = Path("/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml")
    reference = Path(
        "/home/ubuntu/dexgym_data/marvin_motion_library_v1/preview_motions/cmu_141_16_wave_right_segments_v2.npz"
    )
    if not reference.exists():
        import pytest

        pytest.skip("Run wave retarget first")
    report = replay(reference, source, tmp_path, max_frames=101)
    assert report["physics_steps"] >= 1000
    assert report["max_actual_joint_speed_rad_s"] > 0
    assert (tmp_path / "mujoco_pd_states.npz").exists()


def test_runtime_arm_gain_profile_preserves_source_asset():
    import hashlib
    import sys
    from pathlib import Path

    import mujoco

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "motion"))
    from replay_marvin_reference_mujoco import apply_tracking_profile

    source = Path("/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(source))
    report = apply_tracking_profile(model)
    assert report["source_xml_unchanged"]
    assert report["arm_actuator_count"] == 14
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_generated_wave_has_real_wrist_motion_and_exact_reference_clock():
    from pathlib import Path

    import pytest

    path = Path(
        "/home/ubuntu/dexgym_data/marvin_motion_library_v1/preview_motions/cmu_141_16_wave_right_segments_v2.npz"
    )
    if not path.exists():
        pytest.skip("Run retarget script first")
    ref = np.load(path, allow_pickle=False)
    np.testing.assert_allclose(ref["time"], np.arange(len(ref["time"])) * 0.02, atol=1e-12)
    assert ref["q_arm"].shape == (len(ref["time"]), 14)
    assert ref["q_full"].shape == (len(ref["time"]), 54)
    assert ref["tcp_quat_xyzw"].shape == (len(ref["time"]), 2, 4)
    assert ref["link_angular_vel"].shape == (len(ref["time"]), 2, 3, 3)
    assert np.all(np.ptp(ref["q_arm"][:, 11:14], axis=0) > 0.5)
    assert np.max(abs(ref["q_arm"][:, :7] - ref["q_arm"][0, :7])) == 0
    assert np.ptp(ref["human_hand_filtered_pos"][:, 1, 1]) > 0.5
    assert np.ptp(ref["mapped_target_pos"][:, 1, 2]) > 0.4
    raw = Rotation.from_quat(ref["mapped_target_raw_quat_xyzw"])
    filtered = Rotation.from_quat(ref["mapped_target_quat_xyzw"][:, 1])
    assert np.rad2deg((raw * filtered.inv()).magnitude()).max() > 20


def test_mujoco_body_to_tcp_site_offset_on_nonzero_wrist_poses():
    from pathlib import Path

    import mujoco
    import pytest

    source = Path("/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml")
    ref_path = Path(
        "/home/ubuntu/dexgym_data/marvin_motion_library_v1/preview_motions/cmu_141_16_wave_right_segments_v2.npz"
    )
    if not ref_path.exists():
        pytest.skip("Run retarget script first")
    ref = np.load(ref_path, allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    for frame in (80, 250, 500, 800, 1100):
        data.qpos[:] = ref["q_full"][frame]
        mujoco.mj_forward(model, data)
        for side, name in enumerate(("L", "R")):
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{name}")
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{name}")
            body_r = Rotation.from_quat(data.xquat[body][[1, 2, 3, 0]])
            local_r = Rotation.from_quat(model.site_quat[site][[1, 2, 3, 0]])
            expected = (body_r * local_r).as_quat()
            actual = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_quat()
            assert np.rad2deg((Rotation.from_quat(expected) * Rotation.from_quat(actual).inv()).magnitude()) < 1e-4
            assert np.linalg.norm(data.site_xpos[site] - ref["tcp_pos"][frame, side]) < 1e-5


def test_newton_full_pd_state_is_physical_and_finite():
    import json
    from pathlib import Path

    import pytest

    root = Path("/home/ubuntu/dexgym_data/marvin_motion_library_v1/physics_replays/newton_1env")
    if not (root / "newton_pd_states.npz").exists():
        pytest.skip("Run Newton replay first")
    states = np.load(root / "newton_pd_states.npz", allow_pickle=False)
    report = json.loads((root / "newton_pd_report.json").read_text())
    assert report["complete"] and report["physics_steps"] == 10 * (report["frames"] - 1)
    assert report["state_finite"] and report["newton_contact_total"] == 0
    assert report["illegal_joint_state_count"] == 0
    assert np.max(abs(np.diff(states["q"][:, 0, 27]))) > 0.001
    assert np.max(states["tcp_position_error_mm"][:, 0, 1]) < 5


def test_bvh_axis_map_is_proper_and_anatomical_right_maps_outward():
    matrix = bvh_to_marvin_rotation()
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == 1
    np.testing.assert_allclose(matrix @ [1, 0, 0], [-1, 0, 0])
    np.testing.assert_allclose(matrix @ [0, 1, 0], [0, 0, 1])


def test_so3_filter_preserves_real_rotation_and_reduces_spike():
    angle = np.radians([0, 5, 10, 80, 15, 20, 25])
    quat = Rotation.from_rotvec(np.column_stack([angle, np.zeros((7, 2))])).as_quat()
    filtered = smooth_so3(quat, sigma_frames=1.0)
    assert np.linalg.norm(filtered - quat, axis=1).max() > 0
    assert np.linalg.norm(Rotation.from_quat(filtered[-1]).as_rotvec()) > 0.1
    raw_step = (Rotation.from_quat(quat[1:]) * Rotation.from_quat(quat[:-1]).inv()).magnitude()
    smoothed_step = (Rotation.from_quat(filtered[1:]) * Rotation.from_quat(filtered[:-1]).inv()).magnitude()
    assert smoothed_step.max() < raw_step.max()


def test_loader_clamps_future_horizon_and_interpolates(tmp_path):
    time = np.arange(3) * 0.02
    q = np.tile(np.arange(14), (3, 1)).astype(float)
    q[:, 0] = [0, 1, 2]
    link_pos = np.zeros((3, 2, 3, 3))
    link_pos[:, 1, 2, 0] = [0, 1, 2]
    link_quat = np.zeros((3, 2, 3, 4))
    link_quat[..., 3] = 1
    path = tmp_path / "motion.npz"
    np.savez(path, time=time, fps=np.array(50), q_arm=q, link_pos=link_pos, link_quat_xyzw=link_quat)
    ref = MarvinReference(path)
    assert ref.at(0.01)["q_arm"][0] == 0.5
    goal = ref.future_links(0.04, count=10)
    assert goal["pos"].shape == (10, 2, 3, 3)
    np.testing.assert_array_equal(goal["pos"][:, 1, 2, 0], np.full(10, 2.0))
