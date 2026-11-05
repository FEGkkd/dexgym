"""Real mj_step position-actuator replay of a 54 DoF Marvin reference."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _stats(array):
    return {"mean": float(np.mean(array)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def apply_tracking_profile(model: mujoco.MjModel, *, damping_scale: float = 0.5) -> dict:
    """Runtime-only Newton comparable arm gains, matching the existing controller profile."""
    kp = (400.0, 400.0, 440.0, 340.0, 200.0, 120.0, 85.0)
    kd = (45.0, 45.0, 40.0, 34.0, 23.0, 16.0, 12.0)
    if model.nu != 54 or model.njnt != 54 or not 0 < damping_scale <= 2:
        raise ValueError("Unexpected model or damping scale")
    count = 0
    for side in ("L", "R"):
        for i in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"Joint{i + 1}_{side}")
            aids = [a for a in range(model.nu) if model.actuator_trnid[a, 0] == jid]
            if len(aids) != 1:
                raise ValueError("Unexpected actuator mapping")
            aid = aids[0]
            if not (
                math.isclose(model.actuator_gainprm[aid, 0], kp[i])
                and math.isclose(model.actuator_biasprm[aid, 1], -kp[i])
                and math.isclose(model.actuator_biasprm[aid, 2], -kd[i])
            ):
                raise ValueError("Unexpected source actuator gains")
            tuned_kp = kp[i] * 180 / math.pi
            tuned_kd = kd[i] * damping_scale
            model.actuator_gainprm[aid, 0] = tuned_kp
            model.actuator_biasprm[aid, 1] = -tuned_kp
            model.actuator_biasprm[aid, 2] = -tuned_kd
            count += 1
    return {
        "name": "newton_comparable_runtime_arm",
        "arm_actuator_count": count,
        "kp_source_to_runtime_scale": 180 / math.pi,
        "kd_source_to_runtime_scale": damping_scale,
        "source_xml_unchanged": True,
        "force_limits_unchanged": True,
    }


def replay(
    reference: Path, mjcf: Path, output: Path, *, max_frames: int | None = None, profile: str = "source"
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ref = np.load(reference, allow_pickle=False)
    time = ref["time"][:max_frames]
    q_target = ref["q_full"][:max_frames]
    qd_target = ref["qd_arm"][:max_frames]
    arm_names = tuple(ref["joint_names_arm"].tolist())
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    profile_record = {"name": "source_mjcf", "source_xml_unchanged": True}
    if profile == "newton_comparable":
        profile_record = apply_tracking_profile(model)
    elif profile != "source":
        raise ValueError("Unknown profile")
    data = mujoco.MjData(model)
    if (model.nq, model.nv, model.nu) != (54, 54, 54):
        raise ValueError("Expected full 54 DoF physical model")
    if not np.allclose(np.diff(time), 0.02, atol=1e-10):
        raise ValueError("Reference clock is not exact 50 Hz")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    qadr = {n: int(model.jnt_qposadr[i]) for i, n in enumerate(names)}
    dadr = {n: int(model.jnt_dofadr[i]) for i, n in enumerate(names)}
    control = {}
    for aid in range(model.nu):
        name = names[int(model.actuator_trnid[aid, 0])]
        if name in control:
            raise ValueError(f"Duplicate actuator {name}")
        control[name] = aid
    if set(control) != set(names) or tuple(ref["joint_names_full"]) != tuple(names):
        raise ValueError("Reference/model joint name mismatch")
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{s}") for s in ("L", "R")]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("Link4_L", "Link4_R")]
    data.qpos[:] = q_target[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    substeps = int(round(0.02 / model.opt.timestep))
    if not np.isclose(substeps * model.opt.timestep, 0.02, atol=1e-10):
        raise ValueError("MuJoCo timestep must divide 0.02 s")
    positions = np.zeros((len(time), 54))
    velocities = np.zeros_like(positions)
    tcp_pos = np.zeros((len(time), 2, 3))
    tcp_quat = np.zeros((len(time), 2, 4))
    elbow_pos = np.zeros((len(time), 2, 3))
    forces = np.zeros((len(time), 54))
    contacts = np.zeros(len(time), dtype=int)
    forbidden = []
    all_joint_limits = np.array([model.jnt_range[i] for i in range(model.njnt)])
    limit_violations = np.zeros(len(time), dtype=int)
    sat_count = 0
    force_count = 0
    physical_steps = 0
    contact_substeps = 0
    contact_substep_total = 0
    illegal_substep_states = 0

    def command(q):
        for name in names:
            data.ctrl[control[name]] = q[qadr[name]]

    def record(i):
        positions[i] = data.qpos
        velocities[i] = data.qvel
        forces[i] = data.actuator_force
        contacts[i] = data.ncon
        limit_violations[i] = np.count_nonzero(
            (data.qpos < all_joint_limits[:, 0] - 1e-5) | (data.qpos > all_joint_limits[:, 1] + 1e-5)
        )
        for s, sid in enumerate(site_ids):
            tcp_pos[i, s] = data.site_xpos[sid]
            tcp_quat[i, s] = Rotation.from_matrix(data.site_xmat[sid].reshape(3, 3)).as_quat()
            elbow_pos[i, s] = data.xpos[body_ids[s]]
        if data.ncon:
            for j in range(data.ncon):
                c = data.contact[j]
                forbidden.append(
                    [
                        i,
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
                    ]
                )

    command(q_target[0])
    record(0)
    for i in range(1, len(time)):
        for sub in range(substeps):
            alpha = (sub + 1) / substeps
            command((1 - alpha) * q_target[i - 1] + alpha * q_target[i])
            mujoco.mj_step(model, data)
            physical_steps += 1
            if data.ncon:
                contact_substeps += 1
                contact_substep_total += data.ncon
                for j in range(data.ncon):
                    c = data.contact[j]
                    if len(forbidden) < 20:
                        forbidden.append(
                            [
                                i,
                                sub,
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
                            ]
                        )
            illegal_substep_states += int(
                np.count_nonzero(
                    (data.qpos < all_joint_limits[:, 0] - 1e-5) | (data.qpos > all_joint_limits[:, 1] + 1e-5)
                )
            )
            finite = np.isfinite(model.actuator_forcerange).all(axis=1)
            limited = model.actuator_forcelimited.astype(bool) & finite
            if np.any(limited):
                act = data.actuator_force[limited]
                bounds = model.actuator_forcerange[limited]
                sat_count += int(np.count_nonzero((act <= bounds[:, 0] + 1e-3) | (act >= bounds[:, 1] - 1e-3)))
                force_count += len(act)
        record(i)
    actual_arm_q = positions[:, [qadr[n] for n in arm_names]]
    actual_arm_qd = velocities[:, [dadr[n] for n in arm_names]]
    reference_pos = ref["tcp_pos"][: len(time)]
    reference_quat = ref["tcp_quat_xyzw"][: len(time)]
    position_error = np.linalg.norm(tcp_pos - reference_pos, axis=-1) * 1000
    orientation_error = np.rad2deg(
        (
            Rotation.from_quat(tcp_quat.reshape(-1, 4)) * Rotation.from_quat(reference_quat.reshape(-1, 4)).inv()
        ).magnitude()
    ).reshape(len(time), 2)
    q_error = actual_arm_q - ref["q_arm"][: len(time)]
    qd_error = actual_arm_qd - qd_target
    actual_acc = np.gradient(actual_arm_qd, 0.02, axis=0)
    np.savez_compressed(
        output / "mujoco_pd_states.npz",
        time=time,
        q=positions,
        qd=velocities,
        tcp_pos=tcp_pos,
        tcp_quat_xyzw=tcp_quat,
        elbow_pos=elbow_pos,
        actuator_force=forces,
        contact_count=contacts,
        joint_limit_violations=limit_violations,
        tcp_position_error_mm=position_error,
        tcp_orientation_error_deg=orientation_error,
        q_target=q_target,
        qd_arm_target=qd_target,
    )
    report = {
        "backend": "MuJoCo",
        "reference": str(reference),
        "reference_q_full_sha256": hashlib.sha256(q_target.tobytes()).hexdigest(),
        "model": str(mjcf),
        "source_mjcf_sha256": hashlib.sha256(mjcf.read_bytes()).hexdigest(),
        "profile": profile_record,
        "physics_timestep_s": float(model.opt.timestep),
        "control_timestep_s": 0.02,
        "physics_steps": physical_steps,
        "frames": len(time),
        "complete": len(time) == len(ref["time"]),
        "per_side": {},
        "max_actual_joint_speed_rad_s": float(np.max(abs(actual_arm_qd))),
        "max_actual_joint_accel_rad_s2": float(np.max(abs(actual_acc))),
        "torque_saturation_rate": float(sat_count / force_count) if force_count else None,
        "contact_frames": int(np.count_nonzero(contacts)),
        "contact_total": int(contact_substep_total),
        "contact_substep_count": int(contact_substeps),
        "contact_frames_sampled_50hz": int(np.count_nonzero(contacts)),
        "contact_examples": forbidden[:20],
        "illegal_joint_state_count": int(illegal_substep_states),
        "illegal_joint_state_50hz_count": int(limit_violations.sum()),
        "state_finite": bool(np.isfinite(positions).all() and np.isfinite(velocities).all()),
    }
    for s, name in enumerate(("left", "right")):
        sl = slice(s * 7, (s + 1) * 7)
        report["per_side"][name] = {
            "tcp_pos_mm": _stats(position_error[:, s]),
            "tcp_rot_deg": _stats(orientation_error[:, s]),
            "joint_rmse_rad": float(np.sqrt(np.mean(q_error[:, sl] ** 2))),
            "joint_velocity_rmse_rad_s": float(np.sqrt(np.mean(qd_error[:, sl] ** 2))),
            "max_actual_joint_speed_rad_s": float(np.max(abs(actual_arm_qd[:, sl]))),
            "max_actual_joint_accel_rad_s2": float(np.max(abs(actual_acc[:, sl]))),
        }
    (output / "mujoco_pd_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument(
        "--mjcf",
        type=Path,
        default=Path("/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml"),
    )
    ap.add_argument(
        "--output", type=Path, default=Path("/home/ubuntu/dexgym_data/marvin_motion_library_v1/physics_replays")
    )
    ap.add_argument("--profile", choices=("source", "newton_comparable"), default="source")
    args = ap.parse_args()
    print(json.dumps(replay(args.reference, args.mjcf, args.output, profile=args.profile), indent=2))


if __name__ == "__main__":
    main()
