import numpy as np
import torch

from dexgym.motion.library import MotionLibrary


def _make_motion(path, name, length, offset):
    q = np.full((length, 14), offset, dtype=np.float32)
    quat = np.zeros((length, 2, 3, 4), dtype=np.float32)
    quat[..., 3] = 1
    np.savez(path, motion_id=name, fps=50, time=np.arange(length) / 50,
             joint_names_arm=np.array([f"Joint{j}_{side}" for side in ("L", "R") for j in range(1, 8)]),
             link_names=np.array([["Link3_L", "Link4_L", "tcp_L"],
                                  ["Link3_R", "Link4_R", "tcp_R"]]),
             q_arm=q, qd_arm=np.zeros_like(q), q_full=np.zeros((length, 54), np.float32),
             link_pos=np.full((length, 2, 3, 3), offset, np.float32),
             link_quat_xyzw=quat, link_linear_vel=np.zeros((length, 2, 3, 3), np.float32),
             link_angular_vel=np.zeros((length, 2, 3, 3), np.float32),
             tcp_pos=np.full((length, 2, 3), offset, np.float32),
             tcp_quat_xyzw=quat[:, :, 2], tcp_linear_vel=np.zeros((length, 2, 3), np.float32),
             tcp_angular_vel=np.zeros((length, 2, 3), np.float32))


def test_multi_motion_current_future_and_horizon_sampling_do_not_cross_clips(tmp_path):
    short, long = tmp_path / "short.npz", tmp_path / "long.npz"
    _make_motion(short, "short", 5, 1.0)
    _make_motion(long, "long", 9, 7.0)
    library = MotionLibrary({"short": short, "long": long}, "cpu")
    assert library.lengths.tolist() == [5, 9]
    assert library.fps.tolist() == [50, 50]
    ids = torch.tensor([0, 1])
    frames = torch.tensor([1, 3])
    assert library.reference_at(ids, frames)["link_pos"][:, 0, 0, 0].tolist() == [1.0, 7.0]
    assert library.future_goal_at(ids, frames, 2)[:, 0, 0].tolist() == [1.0, 7.0]
    sampled_ids, sampled_frames = library.sample_reset(64, horizon=4)
    assert ((sampled_frames + 4) < library.lengths[sampled_ids]).all()
    assert all(len(value) == 64 for value in library.reference_at(sampled_ids, sampled_frames).values())
    assert set(library.hashes) == {"short", "long"}
