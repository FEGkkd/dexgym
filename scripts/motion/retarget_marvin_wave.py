"""Retarget actual CMU 141_16 to a 54 DoF Marvin preview, with explicit 6D targets."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from dexgym.motion.bvh import human_fk, parse_bvh
from dexgym.motion.reference_library import bvh_to_marvin_rotation, smooth_so3, so3_distance_deg

BVH = Path("/home/ubuntu/dexgym_data/cmu/cmu-mocap/data/141/141_16.bvh")
MJCF = Path("/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml")
OUT = Path("/home/ubuntu/dexgym_data/marvin_motion_library_v1")
SIDES = ("L", "R")
LINKS = (("Link3_L", "Link4_L", "tcp_L"), ("Link3_R", "Link4_R", "tcp_R"))


def _pose_at(rot: np.ndarray, pos: np.ndarray, old_time: np.ndarray, new_time: np.ndarray):
    pos_new = np.column_stack([np.interp(new_time, old_time, pos[:, i]) for i in range(3)])
    quat_new = Slerp(old_time, Rotation.from_matrix(rot))(new_time).as_quat()
    return pos_new, quat_new


def read_human(bvh: Path, sigma_frames: float = 5.0):
    source = parse_bvh(bvh)
    meta = json.loads((OUT.parent / "human_motion_v1/metadata/141_16.json").read_text())
    fk = human_fk(source, scale_m_per_unit=meta["length_scale_m_per_source_unit"], remove_t_pose=True)
    lookup = {j.name: i for i, j in enumerate(fk.joints)}
    sample_time = np.arange(int(np.floor(fk.time[-1] / 0.02)) + 1) * 0.02
    raw_local = fk.local_rotation[:, lookup["RightHand"]]
    smooth_local_q = smooth_so3(Rotation.from_matrix(raw_local).as_quat(), sigma_frames=sigma_frames)
    parent_world = fk.rotation[:, lookup["RightForeArm"]]
    smooth_world = parent_world @ Rotation.from_quat(smooth_local_q).as_matrix()
    arrays = {"time": sample_time, "source_time": fk.time, "meta": meta}
    torso_index = lookup["Spine1"]
    arrays["torso"] = _pose_at(fk.rotation[:, torso_index], fk.position[:, torso_index], fk.time, sample_time)
    for side in ("Left", "Right"):
        i = 0 if side == "Left" else 1
        for label in ("Shoulder", "Arm", "ForeArm", "Hand"):
            index = lookup[side + label]
            arrays[f"{label.lower()}_{i}"] = _pose_at(
                fk.rotation[:, index], fk.position[:, index], fk.time, sample_time
            )
    hand_index = lookup["RightHand"]
    arrays["hand_raw_1"] = _pose_at(fk.rotation[:, hand_index], fk.position[:, hand_index], fk.time, sample_time)
    arrays["hand_1"] = _pose_at(smooth_world, fk.position[:, hand_index], fk.time, sample_time)
    arrays["local_right_raw_quat"] = Rotation.from_matrix(raw_local).as_quat()
    arrays["local_right_filtered_quat"] = smooth_local_q
    return arrays


def _mean_p95_max(x):
    x = np.asarray(x)
    return {"mean": float(np.mean(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


class MarvinFK:
    def __init__(self, xml: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (54, 54, 54):
            raise ValueError("Expected full 54 DoF Marvin")
        self.joint_names = tuple(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(54))
        self.qadr = {n: int(self.model.jnt_qposadr[i]) for i, n in enumerate(self.joint_names)}
        self.dadr = {n: int(self.model.jnt_dofadr[i]) for i, n in enumerate(self.joint_names)}
        self.ctrl = {}
        for aid in range(self.model.nu):
            jid = int(self.model.actuator_trnid[aid, 0])
            name = self.joint_names[jid]
            if name in self.ctrl:
                raise ValueError(f"Duplicate control {name}")
            self.ctrl[name] = aid
        if len(self.ctrl) != 54 or len(set(self.qadr.values())) != 54:
            raise ValueError("Invalid 54 DoF name mapping")
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.home = self.model.key_qpos[home_id].copy()
        self.arm_names = tuple(f"Joint{i}_{side}" for side in SIDES for i in range(1, 8))
        self.arm_qadr = np.array([self.qadr[n] for n in self.arm_names])
        self.right_names = self.arm_names[7:]
        self.right_qadr = self.arm_qadr[7:]
        self.right_dadr = np.array([self.dadr[n] for n in self.right_names])
        self.right_limits = np.array(
            [
                self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in self.right_names
            ]
        )
        self.site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{s}") for s in SIDES]
        self.elbow_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link4_R")
        self.shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link1_R")
        self.set_full(self.home)
        self.shoulder_origin = self.data.xpos[self.shoulder_id].copy()

    def set_full(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def set_right(self, q):
        self.data.qpos[self.right_qadr] = q
        mujoco.mj_forward(self.model, self.data)

    def tcp(self, side=1):
        site = self.site_ids[side]
        return self.data.site_xpos[site].copy(), Rotation.from_matrix(self.data.site_xmat[site].reshape(3, 3)).as_quat()

    def solve(self, pos, quat, elbow, previous, *, max_nfev=55, elbow_weight=0.08):
        target_r = Rotation.from_quat(quat)
        center = np.mean(self.right_limits, axis=1)
        span = np.diff(self.right_limits, axis=1).ravel()
        lower, upper = self.right_limits[:, 0] + 0.005, self.right_limits[:, 1] - 0.005
        previous = np.clip(previous, lower, upper)

        def residual(q):
            self.set_right(q)
            current_p, current_q = self.tcp()
            rotation_error = (target_r * Rotation.from_quat(current_q).inv()).as_rotvec()
            return np.concatenate(
                (
                    (current_p - pos) / 0.012,
                    rotation_error / 0.12,
                    (self.data.xpos[self.elbow_id] - elbow) * elbow_weight / 0.05,
                    (q - previous) / 0.35,
                    (q - center) / span * 0.07,
                )
            )

        def jacobian(q):
            self.set_right(q)
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site_ids[1])
            ep = np.zeros_like(jp)
            er = np.zeros_like(jp)
            mujoco.mj_jacBody(self.model, self.data, ep, er, self.elbow_id)
            return np.vstack(
                (
                    jp[:, self.right_dadr] / 0.012,
                    -jr[:, self.right_dadr] / 0.12,
                    ep[:, self.right_dadr] * elbow_weight / 0.05,
                    np.eye(7) / 0.35,
                    np.diag(0.07 / span),
                )
            )

        seeds = (previous, np.clip(self.home[self.right_qadr], lower, upper))
        best = None
        for seed in seeds:
            result = least_squares(residual, seed, jac=jacobian, bounds=(lower, upper), max_nfev=max_nfev, ftol=2e-5)
            cost = np.linalg.norm(residual(result.x))
            if best is None or cost < best[0]:
                best = cost, result.x.copy()
            if cost < 2:
                break
        self.set_right(best[1])
        return best[1]

    def position_seed(self, pos):
        lower, upper = self.right_limits[:, 0] + 0.005, self.right_limits[:, 1] - 0.005

        def fun(q):
            self.set_right(q)
            return np.r_[((self.tcp()[0] - pos) / 0.01), (q - self.home[self.right_qadr]) / 1.0]

        result = least_squares(
            fun, np.clip(self.home[self.right_qadr], lower, upper), bounds=(lower, upper), max_nfev=100
        )
        self.set_right(result.x)
        return result.x.copy(), self.tcp()[1]


def _mapped(human, robot, scale, offset, mapping_mode="legacy"):
    align = bvh_to_marvin_rotation()
    h_shoulder = human["shoulder_1"][0] if mapping_mode == "legacy" else human["arm_1"][0]
    h_elbow = human["forearm_1"][0]
    h_hand = human["hand_1"][0]
    base = robot.shoulder_origin + offset
    if mapping_mode == "legacy":
        pos = base + scale * (h_hand - h_shoulder) @ align.T
        elbow = base + scale * (h_elbow - h_shoulder) @ align.T
    else:
        upper = (h_elbow - h_shoulder) @ align.T
        upper /= np.linalg.norm(upper, axis=1, keepdims=True)
        upper_length = 0.2875638399617261
        elbow = robot.shoulder_origin + upper_length * upper
        if mapping_mode == "physical_shoulder":
            pos = base + scale * (h_hand - h_shoulder) @ align.T
        elif mapping_mode == "segments":
            forearm = (h_hand - h_elbow) @ align.T
            forearm /= np.linalg.norm(forearm, axis=1, keepdims=True)
            forearm_length = 0.40939591175141044
            pos = base + scale * (upper_length * upper + forearm_length * forearm)
        else:
            raise ValueError(f"Unknown mapping mode {mapping_mode}")
    return pos, elbow


def make_candidate(human, robot, *, scale, offset, anchor, name, output, mapping_mode="legacy", elbow_weight=0.08):
    align = bvh_to_marvin_rotation()
    target_p, target_e = _mapped(human, robot, scale, offset, mapping_mode)
    original_p, original_e = _mapped(human, robot, 1.0, np.zeros(3), mapping_mode)
    hand_q = human["hand_1"][1]
    anchor_index = int(round(anchor / 0.02))
    _, anchor_robot_q = robot.position_seed(target_p[anchor_index])
    hand_to_tcp = (Rotation.from_matrix(align) * Rotation.from_quat(hand_q[anchor_index])).inv() * Rotation.from_quat(
        anchor_robot_q
    )
    target_q = (Rotation.from_matrix(align) * Rotation.from_quat(hand_q) * hand_to_tcp).as_quat()
    original_q = (Rotation.from_matrix(align) * Rotation.from_quat(human["hand_raw_1"][1]) * hand_to_tcp).as_quat()
    previous = robot.home[robot.right_qadr].copy()
    q_human = np.zeros((len(target_p), 7))
    for i in range(len(target_p)):
        previous = robot.solve(target_p[i], target_q[i], target_e[i], previous, elbow_weight=elbow_weight)
        q_human[i] = previous
        if i % 25 == 0:
            print(f"{name}: IK {i}/{len(target_p)}", flush=True)

    # Retiming uses exact 50 Hz samples. Quintic endpoints make marked transitions C2.
    base_duration = human["time"][-1]
    speed_scale = 2.0
    max_speed = 2.0
    max_accel = 12.0
    applied_speed_scale = speed_scale
    for _ in range(5):
        applied_speed_scale = speed_scale
        core_duration = base_duration * speed_scale
        core_n = int(np.ceil(core_duration / 0.02)) + 1
        core_t = np.arange(core_n) * 0.02
        source_t = np.minimum(core_t / speed_scale, base_duration)
        q_core = PchipInterpolator(human["time"], q_human, axis=0)(source_t)
        vel = np.gradient(q_core, 0.02, axis=0)
        acc = np.gradient(vel, 0.02, axis=0)
        if np.max(abs(vel)) <= max_speed and np.max(abs(acc)) <= max_accel:
            break
        speed_scale *= 1.5
    trans_n = 51
    u = np.linspace(0, 1, trans_n)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    home_r = robot.home[robot.right_qadr]
    q_in = home_r[None, :] + s[:, None] * (q_core[0] - home_r)[None, :]
    q_out = q_core[-1][None, :] + s[:, None] * (home_r - q_core[-1])[None, :]
    q_right = np.concatenate((q_in[:-1], q_core, q_out[1:]))
    time = np.arange(len(q_right), dtype=np.float64) * 0.02
    phase = np.array(["transition_in"] * (trans_n - 1) + ["human"] * core_n + ["transition_out"] * (trans_n - 1))
    raw_time = np.r_[np.full(trans_n - 1, 0.0), source_t, np.full(trans_n - 1, base_duration)]
    q_arm = np.tile(robot.home[robot.arm_qadr], (len(time), 1))
    q_arm[:, 7:] = q_right
    q_full = np.tile(robot.home, (len(time), 1))
    q_full[:, robot.arm_qadr] = q_arm
    qd = np.gradient(q_arm, 0.02, axis=0)
    qdd = np.gradient(qd, 0.02, axis=0)
    pose_pos = np.zeros((len(time), 2, 3))
    pose_q = np.zeros((len(time), 2, 4))
    link_p = np.zeros((len(time), 2, 3, 3))
    link_q = np.zeros((len(time), 2, 3, 4))
    contacts = np.zeros(len(time), dtype=int)
    for i, q in enumerate(q_full):
        robot.set_full(q)
        contacts[i] = robot.data.ncon
        for side in range(2):
            pose_pos[i, side], pose_q[i, side] = robot.tcp(side)
            for j, link in enumerate(LINKS[side]):
                if link.startswith("tcp_"):
                    link_p[i, side, j], link_q[i, side, j] = pose_pos[i, side], pose_q[i, side]
                else:
                    bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, link)
                    link_p[i, side, j] = robot.data.xpos[bid]
                    link_q[i, side, j] = Rotation.from_matrix(robot.data.xmat[bid].reshape(3, 3)).as_quat()
    # Target pose during transitions is the endpoint FK blend, labelled separately.
    desired_p = np.zeros_like(pose_pos)
    desired_q = np.zeros_like(pose_q)
    desired_p[:, 0], desired_q[:, 0] = pose_pos[:, 0], pose_q[:, 0]
    interp_target_p = np.column_stack([np.interp(raw_time, human["time"], target_p[:, k]) for k in range(3)])
    interp_target_q = Slerp(human["time"], Rotation.from_quat(target_q))(raw_time).as_quat()
    desired_p[:, 1], desired_q[:, 1] = interp_target_p, interp_target_q
    human_mask = phase == "human"
    pos_err = np.linalg.norm(pose_pos[human_mask, 1] - desired_p[human_mask, 1], axis=1) * 1000
    rot_err = so3_distance_deg(pose_q[human_mask, 1], desired_q[human_mask, 1])
    interp_elbow = np.column_stack([np.interp(raw_time, human["time"], target_e[:, k]) for k in range(3)])
    elbow_error = np.linalg.norm(link_p[:, 1, 1] - interp_elbow, axis=1) * 1000
    elbow_direction_actual = link_p[:, 1, 1] - robot.shoulder_origin
    elbow_direction_target = interp_elbow - robot.shoulder_origin
    elbow_cos = np.sum(elbow_direction_actual * elbow_direction_target, axis=1) / (
        np.linalg.norm(elbow_direction_actual, axis=1) * np.linalg.norm(elbow_direction_target, axis=1)
    )
    velocity = np.gradient(pose_pos, 0.02, axis=0)
    link_velocity = np.gradient(link_p, 0.02, axis=0)

    def omega(quat):
        result = np.zeros(quat.shape[:-1] + (3,))
        for k in range(1, len(quat)):
            delta = Rotation.from_quat(quat[k]) * Rotation.from_quat(quat[k - 1]).inv()
            result[k] = delta.as_rotvec() / 0.02
        result[0] = result[1]
        return result

    angular = np.stack([omega(pose_q[:, s]) for s in range(2)], axis=1)
    link_angular = np.stack([np.stack([omega(link_q[:, s, j]) for j in range(3)], axis=1) for s in range(2)], axis=1)

    # Retain human and both raw/adjusted mapping targets on the full reference clock.
    def interp(a):
        return np.column_stack([np.interp(raw_time, human["time"], a[:, k]) for k in range(3)])

    raw_human_pos = np.stack([interp(human[f"hand_{s}"][0]) for s in range(2)], axis=1)
    filtered_human_pos = raw_human_pos.copy()
    raw_human_q = np.stack(
        [
            Slerp(human["time"], Rotation.from_quat(human["hand_raw_1"][1] if s else human["hand_0"][1]))(
                raw_time
            ).as_quat()
            for s in range(2)
        ],
        axis=1,
    )
    filtered_human_q = np.stack(
        [Slerp(human["time"], Rotation.from_quat(human[f"hand_{s}"][1]))(raw_time).as_quat() for s in range(2)], axis=1
    )
    original_map_p = interp(original_p)
    original_map_q = Slerp(human["time"], Rotation.from_quat(original_q))(raw_time).as_quat()
    path = output / "preview_motions" / f"cmu_141_16_wave_right_{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=time,
        fps=np.array(50),
        original_bvh_time=raw_time,
        phase=phase,
        q_arm=q_arm,
        qd_arm=qd,
        qdd_arm=qdd,
        q_full=q_full,
        joint_names_arm=np.array(robot.arm_names),
        joint_names_full=np.array(robot.joint_names),
        tcp_pos=pose_pos,
        tcp_quat_xyzw=pose_q,
        tcp_linear_vel=velocity,
        tcp_angular_vel=angular,
        link_names=np.array(LINKS),
        link_pos=link_p,
        link_quat_xyzw=link_q,
        link_linear_vel=link_velocity,
        link_angular_vel=link_angular,
        human_hand_raw_pos=raw_human_pos,
        human_hand_raw_quat_xyzw=raw_human_q,
        human_hand_filtered_pos=filtered_human_pos,
        human_hand_filtered_quat_xyzw=filtered_human_q,
        mapped_target_raw_pos=original_map_p,
        mapped_target_raw_quat_xyzw=original_map_q,
        mapped_target_pos=desired_p,
        mapped_target_quat_xyzw=desired_q,
        mapped_right_elbow_target_pos=interp_elbow,
        right_elbow_target_error_mm=elbow_error,
        right_elbow_direction_cosine=elbow_cos,
        tcp_pos_error_mm=np.linalg.norm(pose_pos - desired_p, axis=2) * 1000,
        tcp_rot_error_deg=so3_distance_deg(pose_q.reshape(-1, 4), desired_q.reshape(-1, 4)).reshape(-1, 2),
        reference_contact_count=contacts,
    )
    # Dense interpolation contacts: check between every pair of 50 Hz references.
    midpoint_contacts = 0
    for i in range(len(q_full) - 1):
        robot.set_full((q_full[i] + q_full[i + 1]) * 0.5)
        midpoint_contacts += robot.data.ncon
    metrics = {
        "path": str(path),
        "frames": len(time),
        "duration_s": float(time[-1]),
        "human_duration_s": float(base_duration),
        "speed_scale": applied_speed_scale,
        "retime_speed_accel_within_design": bool(np.max(np.abs(qd)) <= max_speed and np.max(np.abs(qdd)) <= max_accel),
        "offline_right_pos_mm": _mean_p95_max(pos_err),
        "offline_right_rot_deg": _mean_p95_max(rot_err),
        "offline_right_elbow_mm": _mean_p95_max(elbow_error[human_mask]),
        "offline_right_elbow_direction_cosine": float(np.mean(elbow_cos[human_mask])),
        "max_joint_step_rad": float(np.max(np.abs(np.diff(q_arm, axis=0)))),
        "max_joint_speed_rad_s": float(np.max(np.abs(qd))),
        "max_joint_accel_rad_s2": float(np.max(np.abs(qdd))),
        "reference_contact_frames": int(np.count_nonzero(contacts)),
        "reference_contact_total": int(contacts.sum()),
        "midpoint_contact_total": int(midpoint_contacts),
        "joint_limit_violations": int(
            np.count_nonzero(
                np.any(
                    (q_right < robot.right_limits[:, 0] - 1e-8) | (q_right > robot.right_limits[:, 1] + 1e-8), axis=1
                )
            )
        ),
        "q_right_range_rad": np.ptp(q_right, axis=0).tolist(),
        "left_max_joint_drift_rad": float(np.max(np.abs(q_arm[:, :7] - q_arm[0, :7]))),
        "hand_to_tcp_quat_xyzw": hand_to_tcp.as_quat().tolist(),
        "alignment_matrix": align.tolist(),
        "scale": scale,
        "offset_m": offset.tolist(),
        "anchor_s": anchor,
        "mapping_mode": mapping_mode,
        "elbow_weight": elbow_weight,
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", type=Path, default=BVH)
    ap.add_argument("--mjcf", type=Path, default=MJCF)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--sigma-frames", type=float, default=5.0)
    ap.add_argument("--candidate-mode", choices=("legacy", "anatomical"), default="legacy")
    args = ap.parse_args()
    if (
        hashlib.sha256(args.bvh.read_bytes()).hexdigest()
        != "d428d2c4fa8873d077537567ad32c95b9687a479a10522d314b203fddea37daf"
    ):
        raise ValueError("CMU 141_16 BVH SHA256 changed")
    if (
        hashlib.sha256(args.mjcf.read_bytes()).hexdigest()
        != "6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce"
    ):
        raise ValueError("Marvin MJCF SHA256 changed")
    human = read_human(args.bvh, args.sigma_frames)
    raw_local = human["local_right_raw_quat"]
    filtered_local = human["local_right_filtered_quat"]
    dt = human["meta"]["original_frame_time_s"]
    speed_raw = (
        np.rad2deg((Rotation.from_quat(raw_local[1:]) * Rotation.from_quat(raw_local[:-1]).inv()).magnitude()) / dt
    )
    speed_filtered = (
        np.rad2deg((Rotation.from_quat(filtered_local[1:]) * Rotation.from_quat(filtered_local[:-1]).inv()).magnitude())
        / dt
    )
    filter_delta = so3_distance_deg(raw_local, filtered_local)
    human_quality = {
        "orientation_source": "bvh_fk_reconstructed",
        "retarget_orientation_usable": True,
        "source_full_orientation_validated": False,
        "right_local_active_axes": ["X"],
        "right_wrist_raw_speed_deg_s": _mean_p95_max(speed_raw),
        "right_wrist_filtered_speed_deg_s": _mean_p95_max(speed_filtered),
        "right_wrist_raw_abnormal_step_count_over_1800_deg_s": int(np.count_nonzero(speed_raw > 1800)),
        "right_wrist_filtered_abnormal_step_count_over_1800_deg_s": int(np.count_nonzero(speed_filtered > 1800)),
        "right_wrist_raw_high_speed_step_count_over_720_deg_s": int(np.count_nonzero(speed_raw > 720)),
        "right_wrist_filtered_high_speed_step_count_over_720_deg_s": int(np.count_nonzero(speed_filtered > 720)),
        "filter_rotation_delta_deg": _mean_p95_max(filter_delta),
        "filter_position_delta_mm": 0.0,
        "filter": {"space": "local_hand_SO3_chordal_S3", "sigma_raw_frames": args.sigma_frames},
        "limitations": human["meta"]["orientation_limitations"],
    }
    robot = MarvinFK(args.mjcf)
    candidates = []
    specs = (
        (
            ("first", 1.0, 0.0, np.array([0.0, -0.03, 0.0]), "legacy", 0.08),
            ("tuned", 0.85, 0.7, np.array([0.0, -0.03, 0.0]), "legacy", 0.08),
        )
        if args.candidate_mode == "legacy"
        else (
            ("physical_shoulder_v1", 0.85, 0.7, np.array([-0.1, -0.05, 0.1]), "physical_shoulder", 0.4),
            ("segments_v1", 0.85, 0.7, np.array([-0.15, -0.1, 0.05]), "segments", 0.4),
            ("segments_v2", 0.85, 0.7, np.array([-0.15, -0.1, 0.05]), "segments", 0.8),
        )
    )
    for name, scale, anchor, offset, mode, elbow_weight in specs:
        robot.set_full(robot.home)
        candidates.append(
            make_candidate(
                human,
                robot,
                scale=scale,
                offset=offset,
                anchor=anchor,
                name=name,
                output=args.output,
                mapping_mode=mode,
                elbow_weight=elbow_weight,
            )
        )

    def rank(c):
        return (
            c["reference_contact_total"] * 1000
            + c["midpoint_contact_total"] * 1000
            + c["offline_right_pos_mm"]["p95"]
            + 2 * c["offline_right_rot_deg"]["p95"]
            + 0.2 * c["offline_right_elbow_mm"]["p95"]
        )

    best = min(candidates, key=rank)
    reports = args.output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "motion_id": "cmu_141_16_wave_right",
        "source_commit": "09a07f54f3bbb58797325f009282d0b2048a2871",
        "bvh_path": str(args.bvh),
        "bvh_sha256": hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
        "mjcf_path": str(args.mjcf),
        "mjcf_sha256": hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
        "human_quality": human_quality,
        "candidates": candidates,
        "best": best,
        "status": "preview_requires_physics",
        "source_frames_including_t_pose": 300,
        "removed_t_pose": True,
        "source_frame_time_s": dt,
        "human_source_frame_range_inclusive": [1, 299],
        "human_source_time_range_s": [0.0, float(human["source_time"][-1])],
    }
    suffix = "" if args.candidate_mode == "legacy" else "_anatomical"
    (reports / f"cmu_141_16_wave_right_retarget{suffix}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
