"""Independently validate and register the actual CMU 141_16 Marvin reference."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from dexgym.motion.bvh import parse_bvh
from dexgym.motion.reference_library import MarvinReference, bvh_to_marvin_rotation, so3_distance_deg
from retarget_marvin_wave import BVH, LINKS, MJCF, OUT, read_human


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_video(path, min_duration):
    if not path.is_file() or path.stat().st_size < 5000:
        return False
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and float(result.stdout.strip()) >= min_duration


def _geometry_sets(model):
    body = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) for i in range(model.ngeom)]
    collision = [i for i in range(model.ngeom) if model.geom_contype[i] and model.geom_conaffinity[i]]
    hand_r = [i for i in collision if body[i].startswith("r_")]
    other = [
        i
        for i in collision
        if body[i] in ("pedestal", "center_mount") or body[i].endswith("_L") or body[i].startswith("l_")
    ]
    return hand_r, other


def assess(candidate: Path, root: Path = OUT):
    r = np.load(candidate, allow_pickle=False)
    candidate_q_hash = hashlib.sha256(r["q_full"].tobytes()).hexdigest()
    source_hash = _hash(MJCF)
    usd_path = (
        Path(__file__).resolve().parents[2]
        / "source/dexgym/assets/marvin_wuji/usd/dual_arm_marvin_no_hand_contacts.usda"
    )
    usd_hash = _hash(usd_path)
    time = r["time"]
    n = len(time)
    required = {
        "fps",
        "time",
        "q_arm",
        "qd_arm",
        "qdd_arm",
        "q_full",
        "tcp_pos",
        "tcp_quat_xyzw",
        "tcp_linear_vel",
        "tcp_angular_vel",
        "link_names",
        "link_pos",
        "link_quat_xyzw",
        "link_linear_vel",
        "link_angular_vel",
        "human_hand_raw_pos",
        "human_hand_raw_quat_xyzw",
        "human_hand_filtered_pos",
        "human_hand_filtered_quat_xyzw",
        "mapped_target_raw_pos",
        "mapped_target_raw_quat_xyzw",
        "mapped_target_pos",
        "mapped_target_quat_xyzw",
        "tcp_pos_error_mm",
        "tcp_rot_error_deg",
        "phase",
        "original_bvh_time",
    }
    missing = sorted(required - set(r.files))
    if missing:
        raise ValueError(f"NPZ missing {missing}")
    shape_ok = (
        r["q_arm"].shape == (n, 14)
        and r["q_full"].shape == (n, 54)
        and r["tcp_pos"].shape == (n, 2, 3)
        and r["tcp_quat_xyzw"].shape == (n, 2, 4)
        and r["link_pos"].shape == (n, 2, 3, 3)
        and r["link_quat_xyzw"].shape == (n, 2, 3, 4)
        and r["link_linear_vel"].shape == (n, 2, 3, 3)
        and r["link_angular_vel"].shape == (n, 2, 3, 3)
    )
    exact_clock = bool(int(r["fps"]) == 50 and np.array_equal(time, np.arange(n, dtype=np.float64) * 0.02))
    finite = all(np.isfinite(r[key]).all() for key in required if key in r and np.issubdtype(r[key].dtype, np.number))
    quat_norm = max(
        float(np.max(abs(np.linalg.norm(r[key], axis=-1) - 1)))
        for key in ("tcp_quat_xyzw", "link_quat_xyzw", "mapped_target_quat_xyzw", "human_hand_raw_quat_xyzw")
    )
    names = tuple(r["joint_names_full"].tolist())
    arm = tuple(r["joint_names_arm"].tolist())
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    model_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt))
    name_ok = names == model_names and arm == tuple(f"Joint{i}_{s}" for s in ("L", "R") for i in range(1, 8))
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")]
    hand_indices = [i for i, name in enumerate(names) if name.startswith(("l_", "r_"))]
    hand_home_max = float(np.max(abs(r["q_full"][:, hand_indices] - home[hand_indices])))
    left_static_max = float(np.max(abs(r["q_arm"][:, :7] - r["q_arm"][0, :7])))
    all_limits = np.array([model.jnt_range[j] for j in range(model.njnt)])
    joint_limit_violations = int(
        np.count_nonzero((r["q_full"] < all_limits[:, 0] - 1e-7) | (r["q_full"] > all_limits[:, 1] + 1e-7))
    )
    speed = float(np.max(abs(r["qd_arm"])))
    accel = float(np.max(abs(r["qdd_arm"])))
    max_step = float(np.max(abs(np.diff(r["q_arm"], axis=0))))
    right_wrist_range = np.ptp(r["q_arm"][:, 11:14], axis=0).tolist()
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{s}") for s in ("L", "R")]
    expected_links = tuple(tuple(row) for row in LINKS)
    link_names = tuple(tuple(row) for row in r["link_names"])
    link_names_ok = link_names == expected_links
    maximum_fk_pos_mm = 0.0
    maximum_fk_rot_deg = 0.0
    max_link_pos_mm = 0.0
    max_link_rot_deg = 0.0
    contact_frames = 0
    midpoint_contacts = 0
    hand_r, other = _geometry_sets(model)
    hand_clearance_min_m = float("inf")
    for frame in range(n):
        data.qpos[:] = r["q_full"][frame]
        mujoco.mj_forward(model, data)
        contact_frames += int(data.ncon > 0)
        for side in range(2):
            sid = site_ids[side]
            p = data.site_xpos[sid]
            q = Rotation.from_matrix(data.site_xmat[sid].reshape(3, 3)).as_quat()
            maximum_fk_pos_mm = max(maximum_fk_pos_mm, float(np.linalg.norm(p - r["tcp_pos"][frame, side]) * 1000))
            maximum_fk_rot_deg = max(maximum_fk_rot_deg, float(so3_distance_deg(q, r["tcp_quat_xyzw"][frame, side])))
            for j, name in enumerate(LINKS[side]):
                if name.startswith("tcp_"):
                    lp, lq = p, q
                else:
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    lp = data.xpos[bid]
                    lq = Rotation.from_matrix(data.xmat[bid].reshape(3, 3)).as_quat()
                max_link_pos_mm = max(max_link_pos_mm, float(np.linalg.norm(lp - r["link_pos"][frame, side, j]) * 1000))
                max_link_rot_deg = max(
                    max_link_rot_deg, float(so3_distance_deg(lq, r["link_quat_xyzw"][frame, side, j]))
                )
        if frame % 10 == 0 or frame == n - 1:
            for a in hand_r:
                for b in other:
                    hand_clearance_min_m = min(
                        hand_clearance_min_m, float(mujoco.mj_geomDistance(model, data, a, b, 1.0, None))
                    )
        if frame < n - 1:
            data.qpos[:] = (r["q_full"][frame] + r["q_full"][frame + 1]) * 0.5
            mujoco.mj_forward(model, data)
            midpoint_contacts += int(data.ncon > 0)
    replay_dir = root / "physics_replays"
    reports = {
        "mujoco_source": json.loads((replay_dir / "mujoco_pd_report.json").read_text()),
        "mujoco_profile": json.loads((replay_dir / "mujoco_newton_comparable/mujoco_pd_report.json").read_text()),
        "newton_1env": json.loads((replay_dir / "newton_1env/newton_pd_report.json").read_text()),
        "newton_16env": json.loads((replay_dir / "newton_16env/newton_pd_report.json").read_text()),
    }
    video_paths = {k: root / "previews" / f"cmu_141_16_{k}.mp4" for k in ("human", "human_vs_marvin", "newton_pd")}
    videos_ok = {k: _valid_video(path, 2 if k == "human" else 20) for k, path in video_paths.items()}
    loader = MarvinReference(candidate)
    at = loader.at(0.113)
    future = loader.future_links(float(time[-1]) - 0.02, count=10)
    loader_ok = (
        at["q_arm"].shape == (14,)
        and future["pos"].shape == (10, 2, 3, 3)
        and np.allclose(future["pos"][-1], r["link_pos"][-1])
    )
    mask = r["phase"] == "human"
    right_pos = r["tcp_pos_error_mm"][mask, 1]
    right_rot = r["tcp_rot_error_deg"][mask, 1]
    elbow_available = "right_elbow_target_error_mm" in r.files and "right_elbow_direction_cosine" in r.files
    elbow_p95 = float(np.percentile(r["right_elbow_target_error_mm"][mask], 95)) if elbow_available else float("inf")
    elbow_direction_mean = float(np.mean(r["right_elbow_direction_cosine"][mask])) if elbow_available else float("-inf")
    offline = {
        "position_mm": {
            "mean": float(right_pos.mean()),
            "p95": float(np.percentile(right_pos, 95)),
            "max": float(right_pos.max()),
        },
        "rotation_deg": {
            "mean": float(right_rot.mean()),
            "p95": float(np.percentile(right_rot, 95)),
            "max": float(right_rot.max()),
        },
    }
    offline_gate = (
        offline["position_mm"]["p95"] <= 10
        and offline["rotation_deg"]["p95"] <= 5
        and maximum_fk_pos_mm < 0.001
        and maximum_fk_rot_deg < 0.001
        and max_link_pos_mm < 0.001
        and max_link_rot_deg < 0.001
        and contact_frames == 0
        and midpoint_contacts == 0
        and hand_clearance_min_m > 0
        and joint_limit_violations == 0
        and max_step < 0.1
        and speed <= 2
        and accel <= 12
        and min(right_wrist_range) > 0.25
        and left_static_max == 0
        and hand_home_max == 0
        and elbow_p95 <= 100
        and elbow_direction_mean >= 0.95
    )
    physics_gate = (
        all(
            x["complete"]
            and x["state_finite"]
            and x["illegal_joint_state_count"] == 0
            and Path(x["reference"]).resolve() == candidate.resolve()
            and x.get("reference_q_full_sha256") == candidate_q_hash
            and x.get("source_mjcf_sha256") == source_hash
            for x in reports.values()
        )
        and reports["newton_16env"]["num_envs"] == 16
        and all(reports[key].get("usd_sha256") == usd_hash for key in ("newton_1env", "newton_16env"))
    )
    for key in ("mujoco_profile", "newton_1env", "newton_16env"):
        p = reports[key]
        if (
            p.get("contact_total", p.get("newton_contact_total", 0))
            or p.get("source_geometry_contact_frame_env_count", 0)
            or p.get("source_geometry_contact_physics_step_env_count", 0)
        ):
            physics_gate = False
        if p["torque_saturation_rate"] > 0.01:
            physics_gate = False
        for side in ("left", "right"):
            if p["per_side"][side]["tcp_pos_mm"]["mean"] > 25 or p["per_side"][side]["tcp_rot_deg"]["mean"] > 8:
                physics_gate = False
    manifest_path = root / "previews/render_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    video_provenance_ok = manifest.get("reference_q_full_sha256") == hashlib.sha256(r["q_full"].tobytes()).hexdigest()
    schema_gate = (
        shape_ok
        and exact_clock
        and finite
        and quat_norm < 1e-5
        and name_ok
        and link_names_ok
        and loader_ok
        and all(videos_ok.values())
        and video_provenance_ok
    )
    gate_a = _hash(BVH) == "d428d2c4fa8873d077537567ad32c95b9687a479a10522d314b203fddea37daf"
    accepted = bool(gate_a and offline_gate and physics_gate and schema_gate)
    result = {
        "status": "validated" if accepted else "preview_failed_gates",
        "gate_a": gate_a,
        "gate_b": bool(offline_gate),
        "gate_c": bool(physics_gate),
        "gate_d": bool(schema_gate),
        "candidate_path": str(candidate),
        "reference_q_full_sha256": candidate_q_hash,
        "frames": n,
        "duration_s": float(time[-1]),
        "exact_50hz": exact_clock,
        "shape_ok": shape_ok,
        "finite": finite,
        "quaternion_max_norm_error": quat_norm,
        "joint_names_match": name_ok,
        "link_names_match": link_names_ok,
        "loader_future_10_ok": loader_ok,
        "original_bvh_sha256": _hash(BVH),
        "source_mjcf_sha256": _hash(MJCF),
        "offline_right_tcp": offline,
        "reference_max_joint_step_rad": max_step,
        "right_elbow_error_p95_mm": elbow_p95,
        "right_elbow_direction_cosine_mean": elbow_direction_mean,
        "reference_max_speed_rad_s": speed,
        "reference_max_accel_rad_s2": accel,
        "right_wrist_joint_ranges_rad": right_wrist_range,
        "left_static_max_rad": left_static_max,
        "hands_home_max_rad": hand_home_max,
        "joint_limit_violations": joint_limit_violations,
        "fk_max_tcp_position_error_mm": maximum_fk_pos_mm,
        "fk_max_tcp_rotation_error_deg": maximum_fk_rot_deg,
        "fk_max_link_position_error_mm": max_link_pos_mm,
        "fk_max_link_rotation_error_deg": max_link_rot_deg,
        "contact_frames_reference": contact_frames,
        "midpoint_contacts_reference": midpoint_contacts,
        "right_hand_min_clearance_to_support_or_left_m": hand_clearance_min_m,
        "video_paths": {k: str(v) for k, v in video_paths.items()},
        "videos_ok": videos_ok,
        "video_provenance_ok": video_provenance_ok,
        "physical_reports": {
            k: str(
                replay_dir
                / (
                    "mujoco_pd_report.json"
                    if k == "mujoco_source"
                    else "mujoco_newton_comparable/mujoco_pd_report.json"
                    if k == "mujoco_profile"
                    else "newton_1env/newton_pd_report.json"
                    if k == "newton_1env"
                    else "newton_16env/newton_pd_report.json"
                )
            )
            for k in reports
        },
        "physics_summary": {
            k: {
                "right": v["per_side"]["right"],
                "left": v["per_side"]["left"],
                "contacts": v.get("contact_total", v.get("newton_contact_total")),
                "illegal": v["illegal_joint_state_count"],
                "saturation_rate": v["torque_saturation_rate"],
            }
            for k, v in reports.items()
        },
    }
    return result


def register(candidate: Path, root: Path = OUT, retarget_report: Path | None = None):
    result = assess(candidate, root)
    report_path = root / "reports/cmu_141_16_wave_right_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2))
    if result["status"] != "validated":
        return result
    retarget_report = retarget_report or root / "reports/cmu_141_16_wave_right_retarget.json"
    retarget = json.loads(retarget_report.read_text())
    best = retarget["best"]
    source = parse_bvh(BVH)
    m1_meta = json.loads((root.parent / "human_motion_v1/metadata/141_16.json").read_text())
    joints = {j.name: j for j in source.joints}
    left = joints["LeftHand"]
    right = joints["RightHand"]
    src_time = np.arange(1, len(source.frames)) * source.frame_time
    usd = (
        Path(__file__).resolve().parents[2]
        / "source/dexgym/assets/marvin_wuji/usd/dual_arm_marvin_no_hand_contacts.usda"
    )
    config = {
        "source_motion": "CMU 141_16 Wave Hello",
        "source_commit": retarget["source_commit"],
        "source_bvh_path": str(BVH),
        "source_bvh_sha256": _hash(BVH),
        "source_mjcf_path": str(MJCF),
        "newton_usd_path": str(usd),
        "human_source_frame_range_inclusive": [1, 299],
        "source_frame_time_s": source.frame_time,
        "source_duration_after_t_pose_removal_s": float(src_time[-1] - src_time[0]),
        "all_original_action_used": True,
        "removed_artificial_t_pose": True,
        "orientation_source": "bvh_fk_reconstructed",
        "retarget_orientation_usable": True,
        "orientation_reconstruction_valid": m1_meta["orientation_reconstruction_valid"],
        "human_orientation_confidence": m1_meta["orientation_confidence"],
        "human_m1_orientation_audit": m1_meta["audit"],
        "independent_full_3axis_wrist_measurement": False,
        "filter": retarget["human_quality"]["filter"],
        "filter_stats": retarget["human_quality"],
        "position_mapping": {
            "rotation_matrix": best["alignment_matrix"],
            "scale": best["scale"],
            "offset_m": best["offset_m"],
            "anchor_s": best["anchor_s"],
            "mode": best.get("mapping_mode", "legacy"),
            "human_physical_shoulder_node": "RightArm" if best.get("mapping_mode") != "legacy" else "RightShoulder",
            "marvin_upper_arm_length_m": 0.2875638399617261,
            "marvin_elbow_to_tcp_length_m": 0.40939591175141044,
        },
        "right_hand_to_tcp_quat_xyzw": best["hand_to_tcp_quat_xyzw"],
        "right_tcp_target_formula": "R_alignment @ R_human_hand_filtered @ R_hand_to_tcp",
        "tcp_frame": "MuJoCo tcp_L and tcp_R site; Newton TCP_Link body plus MJCF site fixed transform",
        "objective": {
            "tcp_position_scale_m": 0.012,
            "tcp_rotation_scale_rad": 0.12,
            "elbow_weight": best.get("elbow_weight", 0.08),
            "elbow_scale_m": 0.05,
            "joint_continuity_scale_rad": 0.35,
            "joint_center_weight": 0.07,
            "joint_bounds_margin_rad": 0.005,
        },
        "retime": {
            "speed_scale": best["speed_scale"],
            "fps": 50,
            "transition_in_s": 1.0,
            "transition_out_s": 1.0,
            "interpolation": "PCHIP joint paths; quintic C2 home transitions",
        },
        "source_mjcf_sha256": _hash(MJCF),
        "newton_usd_sha256": _hash(usd),
        "newton_asset_hand_contacts_enabled": False,
        "physical_profile": "source MJCF in-memory arm Kp x 180/pi, Kd x 0.5; Newton imported Kp and Kd x 0.5",
        "candidate_paths": [x["path"] for x in retarget["candidates"]],
        "validation_report": str(root / "reports/cmu_141_16_wave_right_validation.json"),
    }
    config["human_joint_hierarchy"] = m1_meta["joint_hierarchy"]
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata/retarget_config.json").write_text(json.dumps(config, indent=2))
    valid_path = root / "validated_motions/cmu_141_16_wave_right.npz"
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(candidate, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    human = read_human(BVH)
    full_human_time = arrays["original_bvh_time"]

    def human_pose(label):
        position, quat = human[label]
        p = np.column_stack([np.interp(full_human_time, human["time"], position[:, axis]) for axis in range(3)])
        from scipy.spatial.transform import Slerp

        q = Slerp(human["time"], Rotation.from_quat(quat))(full_human_time).as_quat()
        return np.column_stack((p, q))

    shoulder = np.stack([human_pose(f"shoulder_{side}") for side in range(2)], axis=1)
    elbow = np.stack([human_pose(f"forearm_{side}") for side in range(2)], axis=1)
    torso = human_pose("torso")
    source_model = mujoco.MjModel.from_xml_path(str(MJCF))
    origin = mujoco.MjData(source_model)
    origin.qpos[:] = source_model.key_qpos[mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_KEY, "home")]
    mujoco.mj_forward(source_model, origin)
    shoulder_body = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, "Link1_R")
    mapped_elbow = (
        arrays["mapped_right_elbow_target_pos"]
        if "mapped_right_elbow_target_pos" in arrays
        else origin.xpos[shoulder_body]
        + np.asarray(best["offset_m"])
        + best["scale"] * (elbow[:, 1, :3] - shoulder[:, 1, :3]) @ bvh_to_marvin_rotation().T
    )
    elbow_error = np.linalg.norm(arrays["link_pos"][:, 1, 1] - mapped_elbow, axis=1) * 1000
    human_mask = arrays["phase"] == "human"
    result["right_elbow_target_error_mm"] = {
        "mean": float(np.mean(elbow_error[human_mask])),
        "p95": float(np.percentile(elbow_error[human_mask], 95)),
        "max": float(np.max(elbow_error[human_mask])),
    }
    raw_relative = (
        Rotation.from_quat(elbow[:, 1, 3:]).inv() * Rotation.from_quat(arrays["human_hand_raw_quat_xyzw"][:, 1])
    ).as_quat()
    filtered_relative = (
        Rotation.from_quat(elbow[:, 1, 3:]).inv() * Rotation.from_quat(arrays["human_hand_filtered_quat_xyzw"][:, 1])
    ).as_quat()
    raw_local = human["local_right_raw_quat"]
    filtered_local = human["local_right_filtered_quat"]

    def local_speed(q):
        return (
            np.rad2deg((Rotation.from_quat(q[1:]) * Rotation.from_quat(q[:-1]).inv()).magnitude()) / source.frame_time
        )

    arrays.update(
        human_bvh_time=src_time - src_time[0],
        human_torso_pose=torso,
        human_shoulder_pose=shoulder,
        human_elbow_pose=elbow,
        human_upper_arm_pose=np.stack([human_pose(f"arm_{side}") for side in range(2)], axis=1),
        human_forearm_pose=elbow.copy(),
        human_right_hand_relative_forearm_raw_quat_xyzw=raw_relative,
        human_right_hand_relative_forearm_filtered_quat_xyzw=filtered_relative,
        human_right_hand_local_raw_quat_xyzw=raw_local,
        human_right_hand_local_filtered_quat_xyzw=filtered_local,
        human_right_hand_local_raw_speed_deg_s=local_speed(raw_local),
        human_right_hand_local_filtered_speed_deg_s=local_speed(filtered_local),
        human_local_wrist_active_axes=np.array(["X", "X"]),
        human_orientation_reconstruction_valid=np.array(True),
        human_orientation_confidence=np.array("low"),
        mapped_right_elbow_target_pos=mapped_elbow,
        robot_right_elbow_pos=arrays["link_pos"][:, 1, 1],
        right_elbow_target_error_mm=elbow_error,
        human_left_hand_local_channels_deg=source.frames[1:, left.channel_start : left.channel_start + 3],
        human_right_hand_local_channels_deg=source.frames[1:, right.channel_start : right.channel_start + 3],
        human_hand_local_channel_names=np.array(["Zrotation", "Yrotation", "Xrotation"]),
        human_source_high_confidence_orientation_valid=np.array(False),
        retarget_orientation_usable=np.array(True),
        orientation_source=np.array("bvh_fk_reconstructed"),
        metadata_json=np.array(json.dumps(config, sort_keys=True)),
    )
    np.savez_compressed(valid_path, **arrays)
    result["validated_path"] = str(valid_path)
    result["validated_sha256"] = _hash(valid_path)
    report_path.write_text(json.dumps(result, indent=2))
    index = {
        "schema": "MarvinMotionLibrary v1",
        "motions": [
            {
                "id": "cmu_141_16_wave_right",
                "status": "validated",
                "npz": str(valid_path),
                "metadata": str(root / "metadata/retarget_config.json"),
                "validation": str(report_path),
                "frames": result["frames"],
                "fps": 50,
                "duration_s": result["duration_s"],
            }
        ],
    }
    (root / "index.json").write_text(json.dumps(index, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--output-root", type=Path, default=OUT)
    ap.add_argument("--retarget-report", type=Path)
    args = ap.parse_args()
    result = register(args.candidate, args.output_root, args.retarget_report)
    print(json.dumps({k: v for k, v in result.items() if k != "physics_summary"}, indent=2))
    if result["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
