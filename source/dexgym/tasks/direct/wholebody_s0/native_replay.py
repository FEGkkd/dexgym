"""Audit wholebody_s0 reference clips with the native MuJoCo position servos."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import SIDES
from .bag import digest
from .library import MotionLibrary, SPLITS
from .prepare import write_json, write_npz
from .robot import FINGER_SITES, Robot

# Use the existing DexGym native reference audit conventions. The 85 mm
# threshold is the P95 over frames of each frame's mean of ten tip errors.
PHYSICS_DT_S = 0.002
HARD_CONTACT_M = 0.003
AUDIT_CONTACT_M = 0.0005
JOINT_OVERFLOW_RAD = 0.05
MEAN_TIP_P95_MM = 85.0
FINGERS = tuple(name.removesuffix("_tip").removesuffix("_finger") for name in FINGER_SITES)
CRITICAL_WARNINGS = tuple(
    getattr(mujoco.mjtWarning, "mjWARN_" + name)
    for name in ("BADQPOS", "BADQVEL", "BADQACC", "BADCTRL", "CONTACTFULL", "CNSTRFULL")
)


def distribution(values):
    values = np.asarray(values, dtype=float).ravel()
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("Cannot summarize nonfinite replay metrics")
    p50, p95 = np.percentile(values, [50, 95])
    return {
        "samples": len(values), "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))), "p50": float(p50),
        "p95": float(p95), "max": float(values.max()),
    }


def checked_clip(library, entry):
    source = (library.root / entry["path"]).resolve()
    if not source.is_relative_to(library.root) or digest(source) != entry["sha256"]:
        raise ValueError(f"Reference checksum mismatch: {entry['path']}")
    return library.load_clip(entry)


def site_rotation_error_deg(measured_xyzw, target_xyzw):
    measured = Rotation.from_quat(measured_xyzw.reshape(-1, 4))
    target = Rotation.from_quat(target_xyzw.reshape(-1, 4))
    return np.degrees((target.inv() * measured).magnitude()).reshape(measured_xyzw.shape[:-1])


class PositionReplay:
    def __init__(self, model_path, manifest):
        self.robot = robot = Robot(model_path)
        self.model = model = robot.model
        self.data = robot.data
        if robot.names != tuple(manifest["joint_names"]) or (
            robot.qpos != manifest["qpos_indices"] or robot.ctrl != manifest["actuator_indices"]
        ):
            raise ValueError("MJCF joint order, qpos or actuator mapping differs from Motion Library")
        if not np.array_equal(robot.bounds, manifest["joint_limits_rad"]):
            raise ValueError("MJCF joint limits differ from Motion Library")
        self.qpos_ids = np.asarray(robot.qpos)
        self.qvel_ids = np.asarray([int(model.joint(name).dofadr[0]) for name in robot.names])
        self.actuator_ids = ids = np.asarray(robot.ctrl)
        self.control_dt_s = 1 / manifest["rate_hz"]
        self.substeps = round(self.control_dt_s / model.opt.timestep)
        if not np.isclose(model.opt.timestep, PHYSICS_DT_S, rtol=0, atol=1e-12) or not np.isclose(
            self.substeps * model.opt.timestep, self.control_dt_s, rtol=0, atol=1e-12
        ):
            raise ValueError("Native replay requires 2 ms physics and an integer number of steps per reference sample")
        self.kp = model.actuator_gainprm[ids, 0]
        self.kd = -model.actuator_biasprm[ids, 2]
        expected_bias = np.column_stack((np.zeros(len(ids)), -self.kp, -self.kd))
        if not (np.all(model.actuator_gaintype[ids] == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE)
                and np.all(model.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_JOINT)
                and np.all(model.actuator_dyntype[ids] == mujoco.mjtDyn.mjDYN_NONE)
                and np.allclose(model.actuator_gear[ids], [1, 0, 0, 0, 0, 0])
                and np.allclose(model.actuator_biasprm[ids, :3], expected_bias)
                and np.all(self.kp > 0) and np.all(self.kd >= 0)
                and model.actuator_ctrllimited[ids].all() and model.actuator_forcelimited[ids].all()):
            raise ValueError("The MJCF must supply one limited native position servo per joint")
        self.ctrl_bounds = model.actuator_ctrlrange[ids]
        self.site_ids = robot.tcp + robot.tips
        self.geom_names = tuple(model.geom(i).name or f"geom_{i}" for i in range(model.ngeom))
        self.body_names = tuple(model.body(i).name or f"body_{i}" for i in range(model.nbody))
        self.geom_bodies = model.geom_bodyid.copy()
        self.geom_groups = tuple(self.body_group(int(body)) for body in self.geom_bodies)

    def body_group(self, body):
        """Classify the geom by the nearest named arm or hand ancestor."""
        while body:
            name = self.body_names[body]
            if name.startswith(("l_", "r_")):
                return "left_hand" if name[0] == "l" else "right_hand"
            if name.startswith(("Base_", "Link", "TCP_Link_")) and name.endswith(("_L", "_R")):
                return "left_arm" if name.endswith("_L") else "right_arm"
            body = int(self.model.body_parentid[body])
        return "scene"

    def record_state(self):
        d = self.data
        return (
            d.qpos[self.qpos_ids], d.qvel[self.qvel_ids], float(d.time),
            d.site_xpos[self.site_ids], d.site_xmat[self.site_ids].reshape(12, 3, 3),
        )

    def contact_depth(self, pairs):
        contact = self.data.contact
        distances = contact.dist
        if not np.isfinite(distances).all():
            return np.nan, None
        peak = 0.0
        deepest = None
        substep_pairs = {}
        for idx in np.flatnonzero(distances < 0):
            a, b = sorted((int(contact.geom1[idx]), int(contact.geom2[idx])))
            depth = -float(distances[idx])
            substep_pairs[a, b] = max(substep_pairs.get((a, b), 0), depth)
            if depth > peak:
                peak, deepest = depth, (a, b)
        for pair, depth in substep_pairs.items():
            history = pairs.setdefault(pair, [0, 0, 0.0])
            history[0] += 1
            history[1] += depth > AUDIT_CONTACT_M
            history[2] = max(history[2], depth)
        return peak, deepest

    def pair_description(self, pair):
        a, b = pair
        return {
            "geoms": [self.geom_names[a], self.geom_names[b]],
            "bodies": [self.body_names[self.geom_bodies[a]], self.body_names[self.geom_bodies[b]]],
            "groups": sorted((self.geom_groups[a], self.geom_groups[b])),
        }

    def causes(self, depth, expected_time_s):
        d = self.data
        causes = []
        if (not np.isfinite(depth) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                or not np.isfinite(d.qacc).all() or not np.isfinite(d.time) or abs(d.time - expected_time_s) > 1e-8):
            causes.append("invalid_physics")
        warnings = [warning.name for warning in CRITICAL_WARNINGS if d.warning.number[int(warning)]]
        if warnings:
            causes.append("solver_warning")
        if depth > HARD_CONTACT_M:
            causes.append("collision")
        q = d.qpos[self.qpos_ids]
        if np.any(q < self.robot.bounds[:, 0] - JOINT_OVERFLOW_RAD) or np.any(
            q > self.robot.bounds[:, 1] + JOINT_OVERFLOW_RAD
        ):
            causes.append("joint_limit")
        return causes, warnings

    def run(self, clip):
        m, d = self.model, self.data
        q, qd = clip["q_ref"], clip["qd_ref"]
        low, high = self.ctrl_bounds.T
        if np.any(q < low - 1e-6) or np.any(q > high + 1e-6):
            raise ValueError("Reference contains a position outside the native actuator ctrlrange")
        controls = np.clip(q, low, high)
        mujoco.mj_resetData(m, d)
        d.qpos[self.qpos_ids], d.qvel[self.qvel_ids] = q[0], qd[0]
        d.ctrl[self.actuator_ids] = controls[0]
        mujoco.mj_forward(m, d)
        reference_pose = np.concatenate((clip["robot_wrist_pose"], clip["robot_fingertip_pose"].reshape(len(q), 10, 7)), axis=1)
        measured_initial = d.site_xpos[self.site_ids]
        if not np.allclose(measured_initial, reference_pose[0, :, :3], rtol=0, atol=1e-6):
            raise ValueError("Native FK at the reset does not match the stored reference site positions")
        initial_quat = Rotation.from_matrix(d.site_xmat[self.site_ids].reshape(12, 3, 3)).as_quat()
        if site_rotation_error_deg(initial_quat, reference_pose[0, :, 3:]).max() > 0.001:
            raise ValueError("Native FK at the reset does not match the stored reference site rotations")

        # Use a separate pair map for the reset so contact substep denominators
        # count integrated states only. The reset is still checked for failure.
        reset_depth, reset_pair = self.contact_depth({})
        causes, warnings = self.causes(reset_depth, 0)
        first_event = self.event(0, 0, causes, warnings, reset_depth, reset_pair) if causes else None
        states = [self.record_state()]
        depths = []
        pairs = {}
        for target_frame in range(1, len(q)):
            if first_event is not None:
                break
            d.ctrl[self.actuator_ids] = controls[target_frame]
            for substep in range(1, self.substeps + 1):
                mujoco.mj_step(m, d)
                mujoco.mj_fwdPosition(m, d)
                depth, pair = self.contact_depth(pairs)
                depths.append(depth)
                expected = ((target_frame - 1) * self.substeps + substep) * m.opt.timestep
                causes, warnings = self.causes(depth, expected)
                if causes:
                    first_event = self.event(target_frame, substep, causes, warnings, depth, pair)
                    break
            if substep == self.substeps and (first_event is None or not {
                "invalid_physics", "solver_warning"
            }.intersection(first_event["causes"])):
                states.append(self.record_state())

        qpos, qvel, time, positions, matrices = (np.stack(part) for part in zip(*states))
        measured = np.concatenate((positions, Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(-1, 12, 4)), axis=-1)
        target = reference_pose[1:len(states)]
        actual = measured[1:]
        position_error_mm = 1000 * np.linalg.norm(actual[..., :3] - target[..., :3], axis=-1)
        angle_error_deg = site_rotation_error_deg(actual[..., 3:], target[..., 3:]) if len(actual) else np.empty((0, 12))
        relative_wrist_mm = 1000 * np.linalg.norm(
            (actual[:, 1, :3] - actual[:, 0, :3]) - (target[:, 1, :3] - target[:, 0, :3]), axis=-1,
        )
        trace = {
            "time_s": time, "qpos": qpos, "qvel": qvel, "reference_target": controls[:len(states)],
            "source_frame_index": clip["source_frame_index"][:len(states)],
            "wrist_pose": measured[:, :2], "tip_pose": measured[:, 2:].reshape(-1, 2, 5, 7),
            "wrist_position_mm": position_error_mm[:, :2], "wrist_orientation_deg": angle_error_deg[:, :2],
            "tip_position_mm": position_error_mm[:, 2:].reshape(-1, 2, 5),
            "tip_orientation_deg": angle_error_deg[:, 2:].reshape(-1, 2, 5),
            "relative_wrist_mm": relative_wrist_mm, "penetration_substep_m": np.asarray(depths),
        }
        metrics = error_metrics(trace)
        complete = first_event is None and len(states) == len(q) and len(depths) == (len(q) - 1) * self.substeps
        rejected = first_event["causes"] if first_event else []
        if complete and metrics["mean_of_10_tips_by_frame_mm"]["p95"] > MEAN_TIP_P95_MM:
            rejected = ["fingertip_p95"]
        pair_rows = [
            {**self.pair_description(pair), "penetrating_substeps": n, "substeps_over_0_5_mm": audit,
             "max_depth_mm": 1000 * depth}
            for pair, (n, audit, depth) in sorted(pairs.items(), key=lambda pair: (-pair[1][2], pair[0]))
        ]
        finite_depths = trace["penetration_substep_m"][np.isfinite(trace["penetration_substep_m"])]
        return {
            "complete": complete, "accepted": complete and not rejected, "rejection_reasons": rejected,
            "first_event": first_event, "planned_control_intervals": len(q) - 1,
            "measured_control_intervals": len(actual), "executed_physics_substeps": len(depths),
            "planned_physics_substeps": (len(q) - 1) * self.substeps, "metrics": metrics,
            "contact": {
                "reset_depth_mm": 1000 * reset_depth if np.isfinite(reset_depth) else None,
                "peak_depth_mm": 1000 * max(float(finite_depths.max(initial=0)), reset_depth if np.isfinite(reset_depth) else 0),
                "penetrating_substeps": int(np.count_nonzero(finite_depths > 0)),
                "substeps_over_0_5_mm": int(np.count_nonzero(finite_depths > AUDIT_CONTACT_M)),
                "substeps_over_3_mm": int(np.count_nonzero(finite_depths > HARD_CONTACT_M)),
                "geom_pairs": pair_rows,
            },
        }, trace

    def event(self, frame, substep, causes, warnings, depth, pair):
        time = float(self.data.time)
        return {
            "target_frame": frame, "substep": substep, "time_s": time if np.isfinite(time) else None,
            "causes": causes, "warnings": warnings, "depth_mm": 1000 * depth if np.isfinite(depth) else None,
            "deepest_pair": self.pair_description(pair) if pair is not None else None,
        }

    def controller(self):
        m = self.model
        return {
            "type": "native_mjcf_joint_position_servo",
            "initial_state": "first q_ref and qd_ref",
            "control": "hold q_ref[i] during interval ending at timestamp i",
            "physics_dt_s": float(m.opt.timestep), "control_dt_s": self.control_dt_s,
            "substeps_per_control_interval": self.substeps, "gravity_m_s2": m.opt.gravity.tolist(),
            "integrator": mujoco.mjtIntegrator(m.opt.integrator).name, "solver": mujoco.mjtSolver(m.opt.solver).name,
            "cone": mujoco.mjtCone(m.opt.cone).name, "kp": self.kp.tolist(), "kd": self.kd.tolist(),
            "force_range": m.actuator_forcerange[self.actuator_ids].tolist(),
            "position_target_range_rad": self.ctrl_bounds.tolist(),
        }


def error_metrics(trace):
    wrist_position = trace["wrist_position_mm"]
    wrist_rotation = trace["wrist_orientation_deg"]
    tip_position = trace["tip_position_mm"]
    tip_rotation = trace["tip_orientation_deg"]
    return {
        "measured_control_intervals": len(wrist_position),
        "wrist_position_mm": {side: distribution(wrist_position[:, i]) for i, side in enumerate(SIDES)},
        "wrist_orientation_deg": {side: distribution(wrist_rotation[:, i]) for i, side in enumerate(SIDES)},
        "tip_position_mm": {"all": distribution(tip_position), **{
            side: distribution(tip_position[:, i]) for i, side in enumerate(SIDES)
        }},
        "tip_orientation_deg": {"all": distribution(tip_rotation), **{
            side: distribution(tip_rotation[:, i]) for i, side in enumerate(SIDES)
        }},
        "tip_position_by_finger_mm": {
            side: {finger: distribution(tip_position[:, i, j]) for j, finger in enumerate(FINGERS)}
            for i, side in enumerate(SIDES)
        },
        "mean_of_10_tips_by_frame_mm": distribution(tip_position.mean(axis=(1, 2))),
        "relative_wrist_vector_mm": distribution(trace["relative_wrist_mm"]),
    }


ERROR_FIELDS = ("wrist_position_mm", "wrist_orientation_deg", "tip_position_mm", "tip_orientation_deg", "relative_wrist_mm")


@dataclass
class Worker:
    library: MotionLibrary
    engine: PositionReplay
    output: Path


def init_worker(source, model_path, output):
    global worker
    library = MotionLibrary.open(source)
    worker = Worker(library, PositionReplay(model_path, library.manifest), output)


def replay_entry(entry):
    clip = checked_clip(worker.library, entry)
    report, trace = worker.engine.run(clip)
    trace_path = f"traces/{entry['id']}.npz"
    write_npz(worker.output / trace_path, trace)
    return {
        "id": entry["id"], "source_episode": entry["source_episode"], "split": entry["split"],
        "task": entry["task"], "trial": entry["trial"], "frames": entry["frames"],
        "trace_path": trace_path, **report,
    }


def pooled_metrics(rows, output):
    arrays = defaultdict(list)
    for row in rows:
        with np.load(output / row["trace_path"], allow_pickle=False) as trace:
            for key in ERROR_FIELDS:
                arrays[key].append(trace[key])
    return error_metrics({key: np.concatenate(arrays[key]) for key in ERROR_FIELDS}) if rows else None


def contact_summary(rows):
    integrated = sum(row["executed_physics_substeps"] for row in rows)
    contact = [row["contact"] for row in rows]
    depth = max((item["peak_depth_mm"] for item in contact), default=None)
    audited = sum(item["substeps_over_0_5_mm"] for item in contact)
    first_hard_groups = Counter()
    for row in rows:
        event = row["first_event"]
        if event and "collision" in event["causes"] and event["deepest_pair"]:
            first_hard_groups[" + ".join(event["deepest_pair"]["groups"])] += 1
    return {
        "attempted_clips": len(rows), "executed_physics_substeps": integrated, "peak_depth_mm": depth,
        "clips_with_penetration": sum(item["peak_depth_mm"] > 0 for item in contact),
        "clips_over_0_5_mm": sum(item["peak_depth_mm"] > 1000 * AUDIT_CONTACT_M for item in contact),
        "clips_over_3_mm": sum(item["peak_depth_mm"] > 1000 * HARD_CONTACT_M for item in contact),
        "penetrating_substeps": sum(item["penetrating_substeps"] for item in contact),
        "substeps_over_0_5_mm": audited, "over_0_5_mm_rate": audited / integrated if integrated else None,
        "substeps_over_3_mm": sum(item["substeps_over_3_mm"] for item in contact),
        "first_hard_collision_by_group_pair": dict(first_hard_groups),
    }


def coverage(rows, rate_hz):
    complete = sum(row["complete"] for row in rows)
    passed = [row for row in rows if row["accepted"]]
    return {
        "attempted": len(rows), "complete": complete, "complete_rate": complete / len(rows) if rows else None,
        "accepted": len(passed), "accepted_rate": len(passed) / len(rows) if rows else None,
        "reference_frames": sum(row["frames"] for row in rows),
        "accepted_frames": sum(row["frames"] for row in passed),
        "accepted_duration_s": sum(row["frames"] for row in passed) / rate_hz,
        "accepted_playback_time_s": sum(row["planned_control_intervals"] for row in passed) / rate_hz,
    }


def export_filtered(library, rows, output, native_provenance):
    root = output / "motion_library"
    (root / "clips").mkdir(parents=True)
    by_id = {row["id"]: row for row in rows}
    selected = []
    for entry in library.manifest["clips"]:
        if not by_id[entry["id"]]["accepted"]:
            continue
        path = root / entry["path"]
        shutil.copyfile(library.root / entry["path"], path)
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Filtered library copy does not match the input: {path}")
        selected.append(entry)
    manifest = {**library.manifest, "clips": selected, "splits": {}, "native_replay": native_provenance}
    seen = {}
    for split in SPLITS:
        subset = [entry for entry in selected if entry["split"] == split]
        episodes = {entry["source_episode"] for entry in subset}
        for episode in episodes:
            if seen.setdefault(episode, split) != split:
                raise ValueError(f"Source episode is present in more than one split: {episode}")
        manifest["splits"][split] = {
            "path": f"{split}.json", "episodes": len(episodes), "clips": len(subset),
            "frames": sum(entry["frames"] for entry in subset),
        }
        write_json(root / f"{split}.json", [entry["path"] for entry in subset])
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    filtered = MotionLibrary.open(manifest_path)
    for entry in filtered.manifest["clips"]:
        filtered.load_clip(entry)
    return manifest["splits"], digest(manifest_path)


def write_clip_table(path, rows):
    fields = (
        "id", "source_episode", "split", "task", "trial", "frames", "complete", "accepted",
        "planned_control_intervals", "measured_control_intervals", "first_event_time_s", "rejection_reasons",
        "left_wrist_p95_mm", "right_wrist_p95_mm", "left_wrist_p95_deg", "right_wrist_p95_deg",
        "all_tips_p95_mm", "mean_of_10_tips_by_frame_p95_mm", "peak_penetration_mm",
        "substeps_over_0_5_mm", "substeps_over_3_mm",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            values = {field: row[field] for field in fields if field in row}
            for side in SIDES:
                for key, unit in (("wrist_position_mm", "mm"), ("wrist_orientation_deg", "deg")):
                    values[f"{side}_wrist_p95_{unit}"] = (metrics[key][side] or {}).get("p95")
            values.update(
                first_event_time_s=(row["first_event"] or {}).get("time_s"),
                rejection_reasons="|".join(row["rejection_reasons"]),
                all_tips_p95_mm=(metrics["tip_position_mm"]["all"] or {}).get("p95"),
                mean_of_10_tips_by_frame_p95_mm=(metrics["mean_of_10_tips_by_frame_mm"] or {}).get("p95"),
                peak_penetration_mm=row["contact"]["peak_depth_mm"],
                substeps_over_0_5_mm=row["contact"]["substeps_over_0_5_mm"],
                substeps_over_3_mm=row["contact"]["substeps_over_3_mm"],
            )
            writer.writerow(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Same MJCF at a different local path.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    source = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    if args.workers < 1 or output.exists():
        parser.error("--workers must be positive and --output must name a new directory")
    library = MotionLibrary.open(source)
    entries = library.manifest["clips"]
    if not entries or len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Expected a nonempty Motion Library with distinct clip IDs")
    for entry in entries:
        name = entry["id"] + ".npz"
        if Path(name).name != name or Path(entry["path"]) != Path("clips") / name:
            raise ValueError(f"Invalid clip ID or path: {entry['path']}")
    source_sha256 = digest(source)
    model_info = library.manifest["provenance"]["robot_model"]
    model_path = (args.model or Path(model_info["path"])).resolve(strict=True)
    if digest(model_path) != model_info["sha256"]:
        raise ValueError("The MJCF checksum differs from the Motion Library provenance")
    engine = PositionReplay(model_path, library.manifest)
    if engine.control_dt_s != 0.02:
        raise ValueError("wholebody_s0 replay expects 50 Hz references")
    provenance = {
        "source_manifest": str(source), "source_manifest_sha256": source_sha256,
        "robot_model": {"path": str(model_path), "sha256": model_info["sha256"]},
        "mujoco_version": mujoco.__version__, "controller": engine.controller(),
        "criteria": {
            "complete_required": True, "max_penetration_mm": 1000 * HARD_CONTACT_M,
            "audit_penetration_mm": 1000 * AUDIT_CONTACT_M, "max_joint_overflow_rad": JOINT_OVERFLOW_RAD,
            "max_per_clip_frame_mean_tip_p95_mm": MEAN_TIP_P95_MM,
            "reference_geometry": "robot_wrist_pose and robot_fingertip_pose from the source clip",
        },
        "report": "../replay_report.json",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wholebody-replay-", dir=output.parent) as temp:
        staging = Path(temp)
        (staging / "traces").mkdir()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(source, model_path, staging)) as pool:
            rows = []
            for i, row in enumerate(pool.map(replay_entry, entries), start=1):
                rows.append(row)
                if i % 20 == 0 or i == len(entries):
                    print(f"Replayed {i}/{len(entries)} clips", flush=True)
        if digest(source) != source_sha256:
            raise ValueError("Source manifest changed during replay")
        accepted = [row for row in rows if row["accepted"]]
        complete = [row for row in rows if row["complete"]]
        failures = Counter(reason for row in rows for reason in row["rejection_reasons"])
        split_sizes, manifest_hash = export_filtered(library, rows, staging, provenance)
        summary = {
            "task": "wholebody_s0", "provenance": provenance,
            "overall": coverage(rows, library.manifest["rate_hz"]),
            "splits": {split: coverage([row for row in rows if row["split"] == split], library.manifest["rate_hz"]) for split in SPLITS},
            "by_motion_type": {
                task: coverage([row for row in rows if row["task"] == task], library.manifest["rate_hz"])
                for task in sorted({row["task"] for row in rows})
            },
            "rejections_by_reason": dict(failures),
            "errors_all_measured_intervals": pooled_metrics(rows, staging),
            "errors_completed_clips": pooled_metrics(complete, staging),
            "errors_accepted_clips": pooled_metrics(accepted, staging),
            "contact_all_attempts": contact_summary(rows), "contact_accepted_clips": contact_summary(accepted),
            "filtered_library": {"path": "motion_library/manifest.json", "sha256": manifest_hash, "splits": split_sizes},
        }
        write_json(staging / "summary.json", summary)
        write_json(staging / "replay_report.json", {"summary": summary, "clips": rows})
        write_clip_table(staging / "clips.csv", rows)
        staging.rename(output)
    print(f"Completed {len(complete)}/{len(entries)}, accepted {len(accepted)}/{len(entries)}. Results: {output}")


if __name__ == "__main__":
    main()
