"""Add an independently solved left 6D arm to the pinned right-arm wave reference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from dexgym.motion.reference_library import bvh_to_marvin_rotation, so3_distance_deg
from retarget_marvin_wave import BVH, LINKS, MJCF, OUT, MarvinFK, read_human

BASE = OUT / "validated_motions/cmu_141_16_wave_right.npz"
OUTPUT = OUT / "preview_motions/cmu_141_16_wave_bimanual.npz"


def angular_velocity(quat):
    rot = Rotation.from_quat(quat)
    out = np.zeros((len(quat), 3))
    out[1:] = (rot[1:] * rot[:-1].inv()).as_rotvec() / 0.02
    out[0] = out[1]
    return out


def run(base_path=BASE, output=OUTPUT, offset=np.array([0.15, -0.1, 0.05]), scale=0.85, anchor_s=0.7):
    with np.load(base_path, allow_pickle=False) as base:
        arrays = {k: base[k].copy() for k in base.files}
    robot = MarvinFK(MJCF)
    human = read_human(BVH)
    align = Rotation.from_matrix(bvh_to_marvin_rotation())
    n = len(human["time"])
    arm = human["arm_0"][0]
    elbow = human["forearm_0"][0]
    hand = human["hand_0"][0]
    left_qadr = robot.arm_qadr[:7]
    left_dadr = np.array([robot.dadr[name] for name in robot.arm_names[:7]])
    left_limits = np.array(
        [
            robot.model.jnt_range[mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in robot.arm_names[:7]
        ]
    )
    lower, upper = left_limits[:, 0] + 0.005, left_limits[:, 1] - 0.005
    shoulder_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "Link1_L")
    elbow_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "Link4_L")
    shoulder_robot = robot.data.xpos[shoulder_id].copy()
    upper_len = np.linalg.norm(robot.data.xpos[elbow_id] - shoulder_robot)
    fore_len = np.linalg.norm(robot.tcp(0)[0] - robot.data.xpos[elbow_id])
    upper_vec = align.apply(elbow - arm)
    upper_vec /= np.linalg.norm(upper_vec, axis=1)[:, None]
    fore = align.apply(hand - elbow)
    fore /= np.linalg.norm(fore, axis=1)[:, None]
    elbow_target = shoulder_robot + upper_len * upper_vec
    target_pos = shoulder_robot + offset + scale * (upper_len * upper_vec + fore_len * fore)
    raw_target_pos = shoulder_robot + upper_len * upper_vec + fore_len * fore
    sample_mask = arrays["phase"] == "human"
    sample_time = arrays["original_bvh_time"][sample_mask]
    right_q = np.column_stack(
        [np.interp(human["time"], sample_time, arrays["q_arm"][sample_mask, j]) for j in range(7, 14)]
    )

    def set_pose(left, frame):
        robot.data.qpos[left_qadr] = left
        robot.data.qpos[robot.right_qadr] = right_q[frame]
        mujoco.mj_forward(robot.model, robot.data)

    anchor = int(round(anchor_s / 0.02))

    def position_residual(q):
        set_pose(q, anchor)
        return np.r_[(robot.tcp(0)[0] - target_pos[anchor]) / 0.01, (q - robot.home[left_qadr]) / 1.0]

    seed = least_squares(
        position_residual, np.clip(robot.home[left_qadr], lower, upper), bounds=(lower, upper), max_nfev=100
    ).x
    set_pose(seed, anchor)
    hand_to_tcp = (align * Rotation.from_quat(human["hand_0"][1][anchor])).inv() * Rotation.from_quat(robot.tcp(0)[1])
    target_quat = (align * Rotation.from_quat(human["hand_0"][1]) * hand_to_tcp).as_quat()
    raw_target_quat = target_quat.copy()
    center = left_limits.mean(axis=1)
    span = np.diff(left_limits, axis=1).ravel()
    q_human = np.zeros((n, 7))
    previous = seed.copy()
    for frame in range(n):
        target_r = Rotation.from_quat(target_quat[frame])

        def residual(q):
            set_pose(q, frame)
            p, quat = robot.tcp(0)
            rot_error = (target_r * Rotation.from_quat(quat).inv()).as_rotvec()
            return np.r_[
                (p - target_pos[frame]) / 0.012,
                rot_error / 0.12,
                (robot.data.xpos[elbow_id] - elbow_target[frame]) * 0.8 / 0.05,
                (q - previous) / 0.35,
                (q - center) / span * 0.07,
            ]

        def jacobian(q):
            set_pose(q, frame)
            jp = np.zeros((3, robot.model.nv))
            jr = np.zeros_like(jp)
            ep = np.zeros_like(jp)
            er = np.zeros_like(jp)
            mujoco.mj_jacSite(robot.model, robot.data, jp, jr, robot.site_ids[0])
            mujoco.mj_jacBody(robot.model, robot.data, ep, er, elbow_id)
            return np.vstack(
                (
                    jp[:, left_dadr] / 0.012,
                    -jr[:, left_dadr] / 0.12,
                    ep[:, left_dadr] * 0.8 / 0.05,
                    np.eye(7) / 0.35,
                    np.diag(0.07 / span),
                )
            )

        guesses = (previous, seed, np.clip(robot.home[left_qadr], lower, upper))
        solutions = [
            least_squares(residual, guess, jac=jacobian, bounds=(lower, upper), max_nfev=70) for guess in guesses[:2]
        ]
        if min(np.linalg.norm(residual(x.x)) for x in solutions) > 2:
            solutions.append(least_squares(residual, guesses[2], jac=jacobian, bounds=(lower, upper), max_nfev=70))
        previous = min(solutions, key=lambda x: np.linalg.norm(residual(x.x))).x.copy()
        q_human[frame] = previous
        if frame % 25 == 0:
            print(f"left IK {frame}/{n}", flush=True)

    # Reuse the already validated exact 50 Hz clock and the same explicit transitions.
    source_t = arrays["original_bvh_time"]
    q_core = PchipInterpolator(human["time"], q_human, axis=0)(source_t[sample_mask])
    trans_n = int(np.count_nonzero(arrays["phase"] == "transition_in")) + 1
    u = np.linspace(0, 1, trans_n)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    home_left = robot.home[left_qadr]
    q_in = home_left + s[:-1, None] * (q_core[0] - home_left)
    q_out = q_core[-1] + s[1:, None] * (home_left - q_core[-1])
    q_left = np.concatenate((q_in, q_core, q_out))
    if len(q_left) != len(arrays["time"]):
        raise ValueError("Bimanual retime clock mismatch")
    q_left = gaussian_filter1d(q_left, sigma=5.0, axis=0, mode="nearest")
    q_left[0] = home_left
    q_left[-1] = home_left
    arrays["q_arm"][:, :7] = q_left
    arrays["q_full"][:, left_qadr] = q_left
    arrays["qd_arm"] = np.gradient(arrays["q_arm"], 0.02, axis=0)
    arrays["qdd_arm"] = np.gradient(arrays["qd_arm"], 0.02, axis=0)
    pose_p = arrays["tcp_pos"]
    pose_q = arrays["tcp_quat_xyzw"]
    link_p = arrays["link_pos"]
    link_q = arrays["link_quat_xyzw"]
    contacts = arrays["reference_contact_count"]
    for i, q in enumerate(arrays["q_full"]):
        robot.set_full(q)
        contacts[i] = robot.data.ncon
        for side in range(2):
            pose_p[i, side], pose_q[i, side] = robot.tcp(side)
            for j, name in enumerate(LINKS[side]):
                if name.startswith("tcp_"):
                    link_p[i, side, j], link_q[i, side, j] = pose_p[i, side], pose_q[i, side]
                else:
                    bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    link_p[i, side, j] = robot.data.xpos[bid]
                    link_q[i, side, j] = Rotation.from_matrix(robot.data.xmat[bid].reshape(3, 3)).as_quat()
    mapped_p = np.column_stack([np.interp(source_t, human["time"], target_pos[:, j]) for j in range(3)])
    mapped_q = Slerp(human["time"], Rotation.from_quat(target_quat))(source_t).as_quat()
    arrays["mapped_target_pos"][:, 0] = mapped_p
    arrays["mapped_target_quat_xyzw"][:, 0] = mapped_q
    raw_p = np.column_stack([np.interp(source_t, human["time"], raw_target_pos[:, j]) for j in range(3)])
    raw_q = Slerp(human["time"], Rotation.from_quat(raw_target_quat))(source_t).as_quat()
    arrays["mapped_target_raw_pos"] = np.stack((raw_p, arrays["mapped_target_raw_pos"]), axis=1)
    arrays["mapped_target_raw_quat_xyzw"] = np.stack((raw_q, arrays["mapped_target_raw_quat_xyzw"]), axis=1)
    arrays["mapped_left_elbow_target_pos"] = np.column_stack(
        [np.interp(source_t, human["time"], elbow_target[:, j]) for j in range(3)]
    )
    arrays["left_elbow_target_error_mm"] = (
        np.linalg.norm(link_p[:, 0, 1] - arrays["mapped_left_elbow_target_pos"], axis=1) * 1000
    )
    arrays["tcp_pos_error_mm"] = np.linalg.norm(pose_p - arrays["mapped_target_pos"], axis=2) * 1000
    arrays["tcp_rot_error_deg"] = so3_distance_deg(
        pose_q.reshape(-1, 4), arrays["mapped_target_quat_xyzw"].reshape(-1, 4)
    ).reshape(-1, 2)
    arrays["tcp_linear_vel"] = np.gradient(pose_p, 0.02, axis=0)
    arrays["link_linear_vel"] = np.gradient(link_p, 0.02, axis=0)
    arrays["tcp_angular_vel"] = np.stack([angular_velocity(pose_q[:, s]) for s in range(2)], axis=1)
    arrays["link_angular_vel"] = np.stack(
        [np.stack([angular_velocity(link_q[:, s, j]) for j in range(3)], axis=1) for s in range(2)], axis=1
    )
    forearm_quat = arrays["human_forearm_pose"][:, 0, 3:7]
    left_hand_quat = arrays["human_hand_raw_quat_xyzw"][:, 0]
    left_relative = (Rotation.from_quat(forearm_quat).inv() * Rotation.from_quat(left_hand_quat)).as_quat()
    arrays["human_left_hand_relative_forearm_raw_quat_xyzw"] = left_relative
    arrays["human_left_hand_relative_forearm_filtered_quat_xyzw"] = left_relative.copy()
    left_channel_range = np.ptp(arrays["human_left_hand_local_channels_deg"], axis=0)
    arrays["human_left_hand_local_activity_deg"] = left_channel_range
    arrays["motion_id"] = np.array("cmu_141_16_wave_bimanual")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    mask = arrays["phase"] == "human"
    report = {
        "candidate": str(output),
        "source_right_reference": str(base_path),
        "source_right_q_sha256": hashlib.sha256(np.load(base_path)["q_full"].tobytes()).hexdigest(),
        "left_hand_to_tcp_quat_xyzw": hand_to_tcp.as_quat().tolist(),
        "left_mapping": {
            "offset_m": offset.tolist(),
            "scale": scale,
            "anchor_s": anchor_s,
            "alignment_matrix": align.as_matrix().tolist(),
            "upper_len_m": upper_len,
            "fore_len_m": fore_len,
        },
        "left_position_mm": {
            "mean": float(arrays["tcp_pos_error_mm"][mask, 0].mean()),
            "p95": float(np.percentile(arrays["tcp_pos_error_mm"][mask, 0], 95)),
            "max": float(arrays["tcp_pos_error_mm"][mask, 0].max()),
        },
        "left_orientation_deg": {
            "mean": float(arrays["tcp_rot_error_deg"][mask, 0].mean()),
            "p95": float(np.percentile(arrays["tcp_rot_error_deg"][mask, 0], 95)),
            "max": float(arrays["tcp_rot_error_deg"][mask, 0].max()),
        },
        "left_elbow_p95_mm": float(np.percentile(arrays["left_elbow_target_error_mm"][mask], 95)),
        "left_joint_ranges_rad": np.ptp(q_left, axis=0).tolist(),
        "left_local_hand_channel_ranges_deg": left_channel_range.tolist(),
        "contact_frames": int(np.count_nonzero(contacts)),
        "max_speed_rad_s": float(np.max(abs(arrays["qd_arm"]))),
        "max_accel_rad_s2": float(np.max(abs(arrays["qdd_arm"]))),
    }
    path = OUT / "reports/cmu_141_16_wave_bimanual_retarget.json"
    path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.output), indent=2))
