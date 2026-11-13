"""Replay the teleoperation stack's offline Wuji and Marvin solvers."""

from pathlib import Path
import subprocess
import tempfile

import numpy as np

from dexgym.data.wuji_retarget import WujiRetargetSession

from . import SIDES
from .bag import EPOCH_NS, manus_keypoint_order
from .sync import RATE_HZ, interpolate

NATIVE_COLUMNS = 26


def arm_options(parameters, robot):
    p = parameters
    identity = np.eye(3).ravel().tolist()
    low = p["arm.joint_lower_rad"]
    high = p["arm.joint_upper_rad"]
    backoff = p["arm.backoff_scales"]
    control_rate = p.get("arm.control_rate_hz", 250)
    ticks = round(control_rate / RATE_HZ)
    if control_rate != ticks * RATE_HZ:
        raise ValueError(f"Control rate {control_rate} is not a multiple of {RATE_HZ}")
    opts = [
        *robot.home[:14],
        *p["arm.tracker_world_to_left_base_rotation"], *p["arm.tracker_world_to_right_base_rotation"],
    ]
    for side, rotation, translation in zip(SIDES, robot.base_rotation, robot.base_position):
        configured_rotation = p.get(f"arm.world_from_{side}_base_rotation", rotation.ravel())
        configured_translation = p.get(f"arm.world_from_{side}_base_translation", translation)
        if not np.allclose(configured_rotation, rotation.ravel(), atol=1e-6) or not np.allclose(
            configured_translation, translation, atol=1e-6
        ):
            raise ValueError(f"{side} arm base calibration disagrees with the 54-DoF model")
        opts.extend((*configured_rotation, *configured_translation))
    opts.extend((
        *p["arm.translation_scale"], p["arm.rotation_scale"], p.get("arm.absolute_orientation", False),
        *p.get("arm.left_hand_to_tool_rotation", identity), *p.get("arm.right_hand_to_tool_rotation", identity),
        *low, *high, *low, *high,
        np.deg2rad(p["arm.joint_limit_margin_deg"]),
        p["arm.fk_position_tolerance_m"], p["arm.fk_orientation_tolerance_rad"],
        p["arm.max_raw_joint_step_rad"], p["arm.max_command_rate_rad_sec"], p["arm.max_command_dt_sec"],
        p["arm.max_translation_rate_m_sec"], p["arm.max_rotation_rate_rad_sec"],
        p["arm.max_tracking_error_rad"], p["arm.hard_tracking_error_rad"], p["arm.tracker_timeout_sec"],
        p.get("arm.analytic_max_step_dt_sec", 0.016), p.get("arm.analytic_source_nullspace_enabled", False),
        np.deg2rad(p["arm.nullspace_activation_deg"]), np.deg2rad(p["arm.nullspace_rate_deg_sec"]),
        np.deg2rad(p["arm.nullspace_probe_deg"]), np.deg2rad(p["arm.arm_angle_limit_deg"]),
        p["arm.arm_angle_centering_weight"], p["arm.axis_projection_enabled"],
        p["arm.axis_projection_min_retained_ratio"], p["arm.workspace_boundary_enabled"],
        p["arm.workspace_boundary_reserve_m"], p["arm.workspace_boundary_transition_m"],
        p["arm.left_tracker_position_lowpass_hz"], p["arm.right_tracker_position_lowpass_hz"],
        p["arm.tracker_hold_timeout_sec"], p["arm.tracker_position_adaptive_filter"],
        ticks, p.get("arm.bimanual_registration_enabled", False),
        p.get("arm.bimanual_position_tolerance_m", 0.01), len(backoff), *backoff,
    ))
    return opts, int(control_rate)


def run_arm_tool(aligned, config, robot, tool, kine_config):
    parameters = config["teleop"]
    opts, control_rate = arm_options(parameters, robot)
    # Keep a strictly positive and episode-relative clock in the C++ core.
    time = aligned["time_s"] + 1
    columns = [time]
    for side in SIDES:
        key = "tracker_" + side
        valid = aligned[key + "_valid"] & aligned["tracker_safety_valid"]
        pose = np.where(valid[:, None], aligned[key + "_pose"], [0, 0, 0, 0, 0, 0, 1])
        columns.extend((valid, *pose.T))
    with tempfile.TemporaryDirectory(prefix="wholebody-s0-ik-") as folder:
        root = Path(folder)
        settings, wrists = root / "settings.txt", root / "wrists.txt"
        result, anchor = root / "result.txt", root / "anchor.txt"
        settings.write_text(" ".join(format(float(x), ".17g") for x in opts) + "\n", encoding="utf-8")
        np.savetxt(wrists, np.column_stack(columns), fmt="%.17g")
        call = subprocess.run(
            [tool, kine_config, settings, wrists, result, anchor],
            capture_output=True, text=True, timeout=300,
        )
        if call.returncode:
            raise RuntimeError(f"Marvin offline IK failed ({call.returncode}): {call.stderr[-3000:]}")
        raw = np.loadtxt(result, ndmin=2)
        if raw.shape != (len(time), 1 + 2 * NATIVE_COLUMNS) or not np.allclose(raw[:, 0], time, rtol=0, atol=1e-9):
            raise RuntimeError("Marvin replay did not preserve every aligned frame")
        frame, paired, error = np.loadtxt(anchor)
    arm = raw[:, 1:].reshape(len(time), 2, NATIVE_COLUMNS)
    return {
        "arm_accepted": arm[..., 0].astype(bool), "arm_safety_code": arm[..., 1].astype(np.int16),
        "arm_rate_limited": arm[..., 2].astype(bool),
        "arm_position_error_m": arm[..., 3].astype(np.float32),
        "arm_orientation_error_rad": arm[..., 4].astype(np.float32),
        "arm_q": arm[..., 5:12].reshape(len(time), 14).astype(np.float32),
        "target_tcp_base": arm[..., 12:19].astype(np.float32),
        "native_tcp_base": arm[..., 19:26].astype(np.float32),
    }, {
        "frame": int(frame), "method": "paired_wrist_midpoint_to_home" if paired else "independent_model_home",
        "relative_wrist_vector_error_m": float(error),
        "profile": parameters.get("arm.control_pipeline", "manimux_analytic_ik_position"),
        "offline_control_hz": control_rate,
    }


def run_hand_sdk(parsed, aligned, streams, config, library_path):
    """Evaluate the retargeter at recorded message cadence and sample at 50 Hz."""
    positions = []
    valid_sides = []
    reports = {}
    for side in SIDES:
        key = "manus_" + side
        prefix = f"{key}__"
        topology = streams[key]["topology"]
        points = parsed[prefix + "pose"][:, manus_keypoint_order(topology), :3].astype(np.float32)
        parameters = config["teleop"]
        turn = float(parameters.get(f"hand.{side}.manus_rotation_z_deg", 15 if side == "left" else -15))
        angle = np.deg2rad(turn)
        c, s = np.float32(np.cos(angle)), np.float32(np.sin(angle))
        x, y = points[..., 0].copy(), points[..., 1].copy()
        points[..., 0], points[..., 1] = c * x - s * y, s * x + c * y
        alpha = float(parameters.get(f"hand.{side}.output_lp_alpha", 1.0))
        timeout = int(parameters.get(f"hand.{side}.max_input_age_ms", 100)) * 1_000_000
        if not 0 < alpha <= 1 or timeout <= 0:
            raise ValueError(f"Invalid {side} hand filter parameters")
        times = parsed[prefix + "header_ns"]
        arrival = parsed[prefix + "monotonic_ns"]
        valid = parsed[prefix + "valid"].copy()
        latency = parsed[prefix + "receive_ros_ns"] - times
        valid &= (latency >= 0) & (latency <= timeout)
        configured_id = int(parameters.get(f"hand.{side}.glove_id", 0))
        if configured_id:
            valid &= parsed[prefix + "glove_id"].astype(np.uint32) == np.uint32(configured_id)
        raw_q = np.full((len(valid), 20), np.nan, dtype=np.float32)
        errors = 0
        last_ns = None
        with WujiRetargetSession(library_path, side=side) as sdk:
            for index in np.flatnonzero(valid):
                try:
                    target = sdk.step(points[index])
                except RuntimeError:
                    errors += 1
                    continue
                # Production CommandSmoother uses float64 state, casts only its output.
                now = int(arrival[index])
                if last_ns is None or now <= last_ns or now - last_ns >= timeout or alpha == 1.0:
                    filtered = target.astype(float)
                else:
                    filtered += alpha * (target - filtered)
                raw_q[index] = filtered
                last_ns = now
        indexes = aligned[key + "_brackets"].copy()
        good = indexes[:, 0] >= 0
        # Interpolate the native outputs at the same source bracket as the human skeleton.
        good[good] = np.isfinite(raw_q[indexes[good]]).all(axis=(1, 2))
        indexes[~good] = -1
        positions.append(interpolate(raw_q, times, aligned["time_ns"], indexes))
        valid_sides.append(good)
        reports[side] = {
            "rotation_z_deg": turn, "output_lp_alpha": alpha, "raw_messages": len(raw_q),
            "sdk_step_errors": errors, "native_valid_samples": int(np.isfinite(raw_q).all(axis=1).sum()),
            "aligned_valid_frames": int(good.sum()),
        }
    return np.column_stack(positions), np.column_stack(valid_sides), reports
