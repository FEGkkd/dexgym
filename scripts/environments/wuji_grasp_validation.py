"""Deterministic fixed-base contact, grasp, and finger-driven lift checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(
    description="Validate fixed-base WUJI hand contact and grasp physics."
)
parser.add_argument(
    "--task", default="DexterousHands-Wuji-Hand-Reorientation-Direct-v0"
)
parser.add_argument(
    "--scenario", choices=("single", "dual", "multi", "lift"), required=True
)
parser.add_argument(
    "--finger",
    choices=("thumb", "index", "middle", "ring", "pinky"),
    default="index",
    help="Finger selected by the single-finger press scenario.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--contact_threshold", type=float, default=0.05)
parser.add_argument("--gravity", type=float, default=9.81)
parser.add_argument("--lift_threshold", type=float, default=0.005)
parser.add_argument(
    "--placement_penetration",
    type=float,
    default=0.001,
    help="Desired fingertip travel beyond the analytically estimated cube surface.",
)
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--no_fail_exit",
    action="store_true",
    help="Write measurements but return zero when acceptance criteria fail.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dexteroushands_lab.tasks as dexteroushands_tasks  # noqa: E402
from dexteroushands_lab.assets.wuji_hand import (  # noqa: E402
    WUJI_HAND_CONTACT_FRICTION,
    WUJI_HAND_USD_RELATIVE_PATH,
    WUJI_OBJECT_MASS_KG,
    WUJI_OBJECT_SIZE_M,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402


dexteroushands_tasks.register_tasks()


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_TIP_BODY_NAMES = (
    "right_finger1_tip_link",
    "right_finger2_tip_link",
    "right_finger3_tip_link",
    "right_finger4_tip_link",
    "right_finger5_tip_link",
)
# Official collision mesh centers are offset from each tip_link rigid-body origin.
# These offsets were measured in the native USD collision prototypes.
FINGER_TIP_LOCAL_OFFSETS = (
    (0.0, 0.0, -0.00608249),
    (0.0, 0.0, -0.00685046),
    (0.0, 0.0, -0.00685047),
    (0.0, 0.0, -0.00685042),
    (0.0, 0.0, -0.00685047),
)
FINGER_JOINT_SLICES = {
    name: slice(index * 4, (index + 1) * 4)
    for index, name in enumerate(FINGER_NAMES)
}
FLEX_LOCAL_INDICES = (0, 2, 3)
VALIDATION_OPEN_DELTA_RAD = 0.65
VALIDATION_PRESS_OVERTRAVEL_RAD = 0.25
SINGLE_FINGER_PROBE_SIZE_M = 0.020
ACTIVE_FINGERS = {
    "single": (args.finger,),
    "dual": ("thumb", "index"),
    "multi": FINGER_NAMES,
    "lift": FINGER_NAMES,
}

# Only the official fingertip rigid bodies are desired contact bodies for
# these scripted probes. All palm and intermediate links are diagnostic.
UNDESIRED_CONTACT_BODY_NAMES = set(
    ("right_palm_link",)
    + tuple(
        f"right_finger{finger}_link{link}"
        for finger in range(1, 6)
        for link in range(1, 5)
    )
)


def _to_list(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().tolist()


def _phase_lengths(total_steps: int) -> tuple[int, int, int]:
    if total_steps < 120:
        raise ValueError("--steps must be at least 120 for settle, close, and hold phases.")
    settle = max(20, int(total_steps * 0.10))
    close = max(60, int(total_steps * 0.45))
    hold = total_steps - settle - close
    return settle, close, hold


def _output_paths() -> tuple[Path, Path]:
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = REPO_ROOT / "artifacts" / "wuji_grasp_validation" / f"{stamp}_{args.scenario}"
        return directory / "summary.json", directory / "timeseries.csv"
    if args.output.suffix.lower() == ".json":
        return args.output, args.output.with_suffix(".csv")
    return args.output / "summary.json", args.output / "timeseries.csv"


def _set_gravity_equivalent_force(env, enabled: bool) -> None:
    object_asset = env.object
    forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
    torques = torch.zeros_like(forces)
    if enabled:
        forces[:, 0, 2] = -WUJI_OBJECT_MASS_KG * args.gravity
    object_asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=torch.tensor([0], device=env.device, dtype=torch.long),
        env_ids=torch.arange(env.num_envs, device=env.device, dtype=torch.long),
        is_global=True,
    )


def _set_object_kinematic(env, enabled: bool) -> None:
    """Toggle the validation object after contact calibration is complete."""
    for env_id in range(env.num_envs):
        prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Object")
        attribute = prim.GetAttribute("physics:kinematicEnabled")
        if not attribute.IsValid():
            raise RuntimeError(f"Object kinematic attribute is missing at {prim.GetPath()}")
        attribute.Set(bool(enabled))


def _step_sim(env, count: int = 1) -> None:
    for _ in range(count):
        env.scene.write_data_to_sim()
        env.sim.step(render=not args.headless)
        env.scene.update(dt=env.physics_dt)


def _write_robot_state(env, joint_pos_1d: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    robot = env.robot

    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += env.scene.env_origins
    root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    joint_pos = joint_pos_1d.expand(env.num_envs, -1).clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.set_joint_position_target_index(
        target=joint_pos, joint_ids=env.actuated_joint_ids, env_ids=env_ids
    )
    robot.write_joint_position_to_sim_index(
        position=joint_pos, joint_ids=env.actuated_joint_ids, env_ids=env_ids
    )
    robot.write_joint_velocity_to_sim_index(
        velocity=joint_vel, joint_ids=env.actuated_joint_ids, env_ids=env_ids
    )


def _write_object_state(env, object_position: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    object_asset = env.object
    object_pose = object_asset.data.default_root_pose.torch.clone()
    if object_position.ndim == 1:
        object_position = object_position.expand(env.num_envs, -1)
    object_pose[:, :3] = object_position
    object_pose[:, :3] += env.scene.env_origins
    object_pose[:, 3:7] = torch.tensor(
        (0.0, 0.0, 0.0, 1.0), device=env.device, dtype=object_pose.dtype
    )
    object_velocity = torch.zeros_like(object_asset.data.default_root_vel.torch)
    object_asset.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
    object_asset.write_root_velocity_to_sim_index(
        root_velocity=object_velocity, env_ids=env_ids
    )
    _set_gravity_equivalent_force(env, enabled=False)


def _write_initial_state(
    env, open_joint_pos: torch.Tensor, object_position: torch.Tensor
) -> None:
    _write_robot_state(env, open_joint_pos)
    _write_object_state(env, object_position)


def _tip_site_positions(env) -> torch.Tensor:
    actual_tip_names = tuple(env._contact_body_names[index] for index in env._tip_body_ids)
    if actual_tip_names != FINGER_TIP_BODY_NAMES:
        raise RuntimeError(
            "Unexpected WUJI distal-body order for tip calibration: "
            f"{actual_tip_names} != {FINGER_TIP_BODY_NAMES}"
        )
    distal_pose = env.robot.data.body_link_pose_w.torch[:, env._tip_body_ids]
    local_offsets = torch.tensor(
        FINGER_TIP_LOCAL_OFFSETS,
        device=env.device,
        dtype=distal_pose.dtype,
    ).expand(env.num_envs, -1, -1)
    world_offsets = quat_apply(
        distal_pose[:, :, 3:7].reshape(-1, 4),
        local_offsets.reshape(-1, 3),
    ).reshape(env.num_envs, len(FINGER_NAMES), 3)
    return distal_pose[:, :, :3] + world_offsets - env.scene.env_origins[:, None, :]


def _calibrate_object_position(
    env, open_pose: torch.Tensor, close_pose: torch.Tensor
) -> tuple[torch.Tensor, dict[str, object]]:
    parked_position = torch.tensor(
        (0.0, 0.0, 1.5), device=env.device, dtype=open_pose.dtype
    )
    _write_initial_state(env, open_pose, parked_position)
    _step_sim(env, count=4)
    open_tips = _tip_site_positions(env).clone()

    _write_robot_state(env, close_pose)
    _step_sim(env, count=4)
    closed_tips = _tip_site_positions(env).clone()

    active_indices = [FINGER_NAMES.index(name) for name in ACTIVE_FINGERS[args.scenario]]
    motion = closed_tips[:, active_indices] - open_tips[:, active_indices]
    motion_norm = torch.linalg.norm(motion, dim=-1, keepdim=True)
    if torch.any(motion_norm < 1.0e-4):
        raise RuntimeError(
            "Tip calibration found less than 0.1 mm open-to-closed travel for an active finger."
        )
    direction = motion / motion_norm

    # For an axis-aligned cube the center-to-surface distance along unit ray
    # u is half_extent / max(abs(u)). Each active fingertip predicts a cube
    # center; their mean is the least-squares center for a multi-finger close.
    object_size = (
        SINGLE_FINGER_PROBE_SIZE_M
        if args.scenario == "single"
        else WUJI_OBJECT_SIZE_M
    )
    half_extent = 0.5 * object_size
    ray_extent = half_extent / torch.amax(torch.abs(direction), dim=-1, keepdim=True)
    center_predictions = closed_tips[:, active_indices] + direction * (
        ray_extent - args.placement_penetration
    )
    predicted_object_position = center_predictions.mean(dim=1)
    if args.scenario == "single":
        object_position = predicted_object_position
        placement_source = "active fingertip trajectory"
    elif args.scenario == "dual":
        object_position = closed_tips[:, active_indices].mean(dim=1)
        placement_source = "closed active-tip midpoint"
    else:
        active_closed_tips = closed_tips[:, active_indices]
        object_position = 0.5 * (
            active_closed_tips.amin(dim=1) + active_closed_tips.amax(dim=1)
        )
        placement_source = "closed active-tip bounding-box center"

    _write_robot_state(env, open_pose)
    _step_sim(env, count=4)
    calibration = {
        "tip_frame_source": "official WUJI USD tip_link rigid bodys",
        "open_tip_positions_m": _to_list(open_tips),
        "closed_tip_positions_m": _to_list(closed_tips),
        "active_tip_travel_m": _to_list(motion_norm.squeeze(-1)),
        "center_predictions_m": _to_list(center_predictions),
        "mean_predicted_object_position_m": _to_list(predicted_object_position),
        "selected_object_position_m": _to_list(object_position),
        "selected_object_position_source": placement_source,
        "placement_penetration_m": args.placement_penetration,
    }
    return object_position, calibration


def _scenario_targets(env) -> tuple[torch.Tensor, torch.Tensor]:
    closed = env.default_joint_pos[0].clone()
    open_pose = closed.clone()
    for finger in FINGER_NAMES:
        finger_start = FINGER_JOINT_SLICES[finger].start
        for local_index in FLEX_LOCAL_INDICES:
            open_pose[finger_start + local_index] -= VALIDATION_OPEN_DELTA_RAD
    open_pose = torch.maximum(
        torch.minimum(open_pose, env.joint_upper_limits[0]), env.joint_lower_limits[0]
    )

    target = open_pose.clone()
    for finger in ACTIVE_FINGERS[args.scenario]:
        target[FINGER_JOINT_SLICES[finger]] = closed[FINGER_JOINT_SLICES[finger]]
    if args.scenario in ("dual", "multi"):
        for finger in ACTIVE_FINGERS[args.scenario]:
            finger_start = FINGER_JOINT_SLICES[finger].start
            for local_index in FLEX_LOCAL_INDICES:
                target[finger_start + local_index] += VALIDATION_PRESS_OVERTRAVEL_RAD
        target = torch.maximum(
            torch.minimum(target, env.joint_upper_limits[0]), env.joint_lower_limits[0]
        )
    if args.scenario == "lift":
        for finger in FINGER_NAMES:
            finger_start = FINGER_JOINT_SLICES[finger].start
            for local_index in FLEX_LOCAL_INDICES:
                target[finger_start + local_index] += 0.12
        target = torch.maximum(
            torch.minimum(target, env.joint_upper_limits[0]), env.joint_lower_limits[0]
        )
    return open_pose, target


def _record_step(
    env,
    target: torch.Tensor,
    phase: str,
    step_index: int,
    baseline_object_pos: torch.Tensor,
    baseline_object_quat: torch.Tensor,
    active_tip_local_indices: list[int],
    undesired_ids: list[int],
) -> list[dict[str, object]]:
    normal = env._contact_forces_with_object()
    friction = env._contact_friction_forces_with_object()
    normal_magnitude = torch.linalg.norm(normal, dim=-1)
    friction_magnitude = torch.linalg.norm(friction, dim=-1)
    normal_force_sum = normal_magnitude.sum(dim=-1)
    tangential_force_sum = friction_magnitude.sum(dim=-1)
    friction_ratio = torch.where(
        normal_force_sum > args.contact_threshold,
        tangential_force_sum / normal_force_sum.clamp_min(1.0e-8),
        torch.zeros_like(normal_force_sum),
    )
    contact = normal_magnitude > args.contact_threshold
    tip_contact = contact[:, env._tip_body_ids]

    object_pose = env.object.data.root_pose_w.torch
    object_pos = object_pose[:, :3] - env.scene.env_origins
    object_quat = object_pose[:, 3:7]
    object_lin_vel = env.object.data.root_lin_vel_w.torch
    object_ang_vel = env.object.data.root_ang_vel_w.torch
    displacement = object_pos - baseline_object_pos
    quat_dot = torch.sum(object_quat * baseline_object_quat, dim=-1).abs().clamp(max=1.0)
    orientation_drift = 2.0 * torch.acos(quat_dot)

    joint_pos = env.robot.data.joint_pos.torch[:, env.actuated_joint_ids]
    joint_error = torch.abs(joint_pos - target)
    applied_torque = torch.abs(
        env.robot.data.applied_torque.torch[:, env.actuated_joint_ids]
    )
    torque_ratio = applied_torque / env.effort_limits.clamp_min(1.0e-6)

    rows: list[dict[str, object]] = []
    for env_index in range(env.num_envs):
        contact_ids = contact[env_index].nonzero(as_tuple=False).flatten().tolist()
        contact_names = [env._contact_body_names[index] for index in contact_ids]
        per_tip = {
            FINGER_NAMES[index]: bool(tip_contact[env_index, index].item())
            for index in range(len(FINGER_NAMES))
        }
        per_tip_force = {
            FINGER_NAMES[index]: float(
                normal_magnitude[env_index, env._tip_body_ids[index]].item()
            )
            for index in range(len(FINGER_NAMES))
        }
        active_tip_count = sum(
            int(tip_contact[env_index, index].item()) for index in active_tip_local_indices
        )
        rows.append(
            {
                "step": step_index,
                "time_s": step_index * env.physics_dt,
                "phase": phase,
                "env_id": env_index,
                "contact_count": int(contact[env_index].sum().item()),
                "tip_contact_count": int(tip_contact[env_index].sum().item()),
                "active_tip_contact_count": active_tip_count,
                "undesired_contact_count": int(contact[env_index, undesired_ids].sum().item()),
                "contact_body_names": "|".join(contact_names),
                "tip_contact_json": json.dumps(per_tip, sort_keys=True),
                "tip_normal_force_json": json.dumps(per_tip_force, sort_keys=True),
                "normal_force_sum_n": float(normal_force_sum[env_index].item()),
                "normal_force_max_n": float(normal_magnitude[env_index].max().item()),
                "tangential_force_sum_n": float(tangential_force_sum[env_index].item()),
                "friction_to_normal_ratio": float(friction_ratio[env_index].item()),
                "object_x_m": float(object_pos[env_index, 0].item()),
                "object_y_m": float(object_pos[env_index, 1].item()),
                "object_z_m": float(object_pos[env_index, 2].item()),
                "object_speed_mps": float(torch.linalg.norm(object_lin_vel[env_index]).item()),
                "object_angular_speed_rps": float(torch.linalg.norm(object_ang_vel[env_index]).item()),
                "object_displacement_m": float(torch.linalg.norm(displacement[env_index]).item()),
                "slip_xy_m": float(torch.linalg.norm(displacement[env_index, :2]).item()),
                "lift_height_m": float(displacement[env_index, 2].item()),
                "orientation_drift_rad": float(orientation_drift[env_index].item()),
                "joint_target_error_mean_rad": float(joint_error[env_index].mean().item()),
                "joint_target_error_max_rad": float(joint_error[env_index].max().item()),
                "torque_saturation_fraction": float((torque_ratio[env_index] >= 0.98).float().mean().item()),
                "max_torque_ratio": float(torque_ratio[env_index].max().item()),
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]], env_index: int, hold_steps: int) -> dict[str, object]:
    env_rows = [row for row in rows if row["env_id"] == env_index]
    preclose_rows = [row for row in env_rows if row["phase"] in ("initial", "settle")]
    hold_rows = [row for row in env_rows if row["phase"] == "hold"]
    if len(hold_rows) != hold_steps:
        raise RuntimeError(f"Expected {hold_steps} hold samples, got {len(hold_rows)}")

    active_count_required = {"single": 1, "dual": 2, "multi": 3, "lift": 3}[args.scenario]
    sustained_ratio = sum(
        int(row["active_tip_contact_count"] >= active_count_required) for row in hold_rows
    ) / len(hold_rows)
    tip_contact_ratio = sum(
        int(row["tip_contact_count"] >= active_count_required) for row in hold_rows
    ) / len(hold_rows)
    maximum_force = max(float(row["normal_force_max_n"]) for row in env_rows)
    maximum_speed = max(float(row["object_speed_mps"]) for row in env_rows)
    maximum_joint_error = max(float(row["joint_target_error_max_rad"]) for row in hold_rows)
    maximum_saturation = max(float(row["torque_saturation_fraction"]) for row in hold_rows)
    maximum_lift = max(float(row["lift_height_m"]) for row in env_rows)
    close_and_hold_rows = [
        row for row in env_rows if row["phase"] in ("close", "probe", "hold")
    ]
    active_contact_steps = sum(
        int(row["active_tip_contact_count"] >= active_count_required)
        for row in close_and_hold_rows
    )
    contacted_active_tip_names = sorted(
        name
        for name in ACTIVE_FINGERS[args.scenario]
        if any(json.loads(str(row["tip_contact_json"]))[name] for row in close_and_hold_rows)
    )
    contact_coverage_pass = len(contacted_active_tip_names) >= active_count_required
    max_friction_ratio = max(
        float(row["friction_to_normal_ratio"])
        for row in env_rows
        if float(row["normal_force_sum_n"]) > args.contact_threshold
    ) if maximum_force > args.contact_threshold else 0.0
    preclose_max_force = max(float(row["normal_force_max_n"]) for row in preclose_rows)
    preclose_max_displacement = max(
        float(row["object_displacement_m"]) for row in preclose_rows
    )
    initial_condition_pass = (
        preclose_max_force <= args.contact_threshold
        and preclose_max_displacement < 0.002
    )
    final = hold_rows[-1]
    drop_time = next(
        (
            float(row["time_s"])
            for row in hold_rows
            if float(row["lift_height_m"]) < -0.04
            or int(row["active_tip_contact_count"]) == 0
        ),
        None,
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in env_rows
        for key in (
            "normal_force_max_n",
            "tangential_force_sum_n",
            "object_speed_mps",
            "joint_target_error_max_rad",
        )
    )

    contact_physics_pass = (
        finite
        and initial_condition_pass
        and maximum_force > args.contact_threshold
        and maximum_force < 500.0
        and maximum_speed < 5.0
        and max_friction_ratio <= 0.72
        and maximum_joint_error < 1.2
        and maximum_saturation < 0.80
    )
    # Coverage is diagnostic. A dual or multi-finger free-body grasp needs a
    # separate controller/pose test; this harness intentionally gates only
    # the contact and friction measurement chain.
    if args.scenario == "single":
        scenario_pass = (
            active_contact_steps >= 1
            and maximum_force > args.contact_threshold
            and float(final["object_displacement_m"]) < 0.10
        )
    elif args.scenario == "dual":
        scenario_pass = (
            sustained_ratio >= 0.40
            and float(final["slip_xy_m"]) < 0.04
            and float(final["lift_height_m"]) > -0.04
        )
    elif args.scenario == "multi":
        scenario_pass = (
            tip_contact_ratio >= 0.60
            and float(final["slip_xy_m"]) < 0.03
            and float(final["lift_height_m"]) > -0.04
        )
    else:
        scenario_pass = (
            tip_contact_ratio >= 0.60
            and maximum_lift >= args.lift_threshold
            and float(final["lift_height_m"]) >= 0.003
        )

    return {
        "env_id": env_index,
        "pass": bool(contact_physics_pass),
        "contact_physics_pass": bool(contact_physics_pass),
        "common_physics_pass": bool(contact_physics_pass),
        "grasp_outcome_pass": bool(scenario_pass),
        "scenario_pass": bool(scenario_pass),
        "contact_coverage_pass": bool(contact_coverage_pass),
        "contacted_active_tip_names": contacted_active_tip_names,
        "initial_condition_pass": bool(initial_condition_pass),
        "preclose_max_normal_force_n": preclose_max_force,
        "preclose_max_object_displacement_m": preclose_max_displacement,
        "sustained_active_tip_contact_ratio": sustained_ratio,
        "sustained_required_tip_contact_ratio": tip_contact_ratio,
        "required_active_tip_count": active_count_required,
        "active_contact_steps": active_contact_steps,
        "max_normal_force_n": maximum_force,
        "max_tangential_force_n": max(float(row["tangential_force_sum_n"]) for row in env_rows),
        "max_friction_to_normal_ratio": max_friction_ratio,
        "expected_effective_friction": WUJI_HAND_CONTACT_FRICTION,
        "max_object_speed_mps": maximum_speed,
        "max_lift_height_m": maximum_lift,
        "final_lift_height_m": float(final["lift_height_m"]),
        "final_slip_xy_m": float(final["slip_xy_m"]),
        "final_object_displacement_m": float(final["object_displacement_m"]),
        "final_orientation_drift_rad": float(final["orientation_drift_rad"]),
        "max_joint_target_error_rad": maximum_joint_error,
        "max_torque_saturation_fraction": maximum_saturation,
        "drop_time_s": drop_time,
        "final_contact_body_names": str(final["contact_body_names"]).split("|") if final["contact_body_names"] else [],
        "final_undesired_contact_count": int(final["undesired_contact_count"]),
    }


def run() -> tuple[dict[str, object], list[dict[str, object]]]:
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs
    )
    env_cfg.seed = args.seed
    env_cfg.observation_noise_enabled = False
    # Gravity is introduced as a deterministic post-closure force.  This
    # prevents pre-grasp free fall from confounding the contact experiment.
    env_cfg.object.spawn.rigid_props.disable_gravity = True
    # Press tests constrain the cube so every requested finger can load the
    # same contact without the first touch ejecting a free body. Lift remains
    # dynamic and is reported as a separate, non-gating outcome.
    env_cfg.object.spawn.rigid_props.kinematic_enabled = True
    if args.scenario == "single":
        env_cfg.object.spawn.size = (SINGLE_FINGER_PROBE_SIZE_M,) * 3
    env = gym.make(args.task, cfg=env_cfg)
    rows: list[dict[str, object]] = []
    try:
        env.reset(seed=args.seed)
        direct_env = env.unwrapped
        open_pose, close_pose = _scenario_targets(direct_env)
        object_position, calibration = _calibrate_object_position(
            direct_env, open_pose, close_pose
        )
        _write_initial_state(direct_env, open_pose, object_position)
        _step_sim(direct_env)

        baseline_object_pos = object_position.clone()
        baseline_object_quat = torch.zeros(
            (direct_env.num_envs, 4), device=direct_env.device
        )
        baseline_object_quat[:, 3] = 1.0
        settle_steps, close_steps, hold_steps = _phase_lengths(args.steps)
        active_tip_local_indices = [
            FINGER_NAMES.index(name) for name in ACTIVE_FINGERS[args.scenario]
        ]
        undesired_ids = [
            direct_env._contact_body_names.index(name)
            for name in sorted(UNDESIRED_CONTACT_BODY_NAMES)
        ]

        step_index = 0
        open_target = open_pose.expand(direct_env.num_envs, -1)
        rows.extend(
            _record_step(
                direct_env,
                open_target,
                "initial",
                step_index,
                baseline_object_pos,
                baseline_object_quat,
                active_tip_local_indices,
                undesired_ids,
            )
        )
        for _ in range(settle_steps):
            step_index += 1
            target = open_pose.expand(direct_env.num_envs, -1)
            direct_env.robot.set_joint_position_target_index(
                target=target, joint_ids=direct_env.actuated_joint_ids
            )
            direct_env.scene.write_data_to_sim()
            direct_env.sim.step(render=not args.headless)
            direct_env.scene.update(dt=direct_env.physics_dt)
            rows.extend(
                _record_step(
                    direct_env,
                    target,
                    "settle",
                    step_index,
                    baseline_object_pos,
                    baseline_object_quat,
                    active_tip_local_indices,
                    undesired_ids,
                )
            )

        for close_index in range(close_steps):
            step_index += 1
            alpha = (close_index + 1) / close_steps
            interpolated = open_pose + alpha * (close_pose - open_pose)
            target = interpolated.expand(direct_env.num_envs, -1)
            direct_env.robot.set_joint_position_target_index(
                target=target, joint_ids=direct_env.actuated_joint_ids
            )
            direct_env.scene.write_data_to_sim()
            direct_env.sim.step(render=not args.headless)
            direct_env.scene.update(dt=direct_env.physics_dt)
            rows.extend(
                _record_step(
                    direct_env,
                    target,
                    "close",
                    step_index,
                    baseline_object_pos,
                    baseline_object_quat,
                    active_tip_local_indices,
                    undesired_ids,
                )
            )

        if args.scenario != "lift":
            # Isolate collision/filter/material validation from implicit-drive
            # tracking. The calibrated target carries only 1 mm nominal
            # penetration at the selected fingertip surface.
            _write_robot_state(direct_env, close_pose)
            _step_sim(direct_env)
            step_index += 1
            probe_target = close_pose.expand(direct_env.num_envs, -1)
            rows.extend(
                _record_step(
                    direct_env,
                    probe_target,
                    "probe",
                    step_index,
                    baseline_object_pos,
                    baseline_object_quat,
                    active_tip_local_indices,
                    undesired_ids,
                )
            )

        _set_gravity_equivalent_force(
            direct_env, enabled=args.scenario in ("dual", "multi", "lift")
        )
        if args.scenario == "lift":
            # Keep the fixture fixed while closing, then let PhysX integrate
            # the object under the post-closure gravity-equivalent load.
            _set_object_kinematic(direct_env, enabled=False)
        for _ in range(hold_steps):
            step_index += 1
            target = close_pose.expand(direct_env.num_envs, -1)
            direct_env.robot.set_joint_position_target_index(
                target=target, joint_ids=direct_env.actuated_joint_ids
            )
            direct_env.scene.write_data_to_sim()
            direct_env.sim.step(render=not args.headless)
            direct_env.scene.update(dt=direct_env.physics_dt)
            rows.extend(
                _record_step(
                    direct_env,
                    target,
                    "hold",
                    step_index,
                    baseline_object_pos,
                    baseline_object_quat,
                    active_tip_local_indices,
                    undesired_ids,
                )
            )

        per_env = [_summarize(rows, index, hold_steps) for index in range(args.num_envs)]
        hand_mass = float(direct_env.robot.data.body_mass.torch[0].sum().item())
        root_pose_before = direct_env.robot.data.default_root_pose.torch[:, :7]
        root_pose_after = direct_env.robot.data.root_pose_w.torch.clone()
        root_pose_after[:, :3] -= direct_env.scene.env_origins
        root_position_error = torch.linalg.norm(
            root_pose_after[:, :3] - root_pose_before[:, :3], dim=-1
        )
        root_orientation_error = torch.minimum(
            torch.linalg.norm(root_pose_after[:, 3:7] - root_pose_before[:, 3:7], dim=-1),
            torch.linalg.norm(root_pose_after[:, 3:7] + root_pose_before[:, 3:7], dim=-1),
        )
        result = {
            "schema_version": 2,
            "scenario": args.scenario,
            "definition": (
                "finger-driven object elevation followed by gravity-equivalent hold"
                if args.scenario == "lift"
                else (
                    "single-index-finger bounded contact response"
                    if args.scenario == "single"
                    else "scripted finger closure followed by gravity-equivalent hold"
                )
            ),
            "acceptance_mode": "contact physics; grasp outcome is reported but not gating",
            "pass": (
                all(item["contact_physics_pass"] for item in per_env)
                and bool(torch.all(root_position_error < 1.0e-5).item())
                and bool(torch.all(root_orientation_error < 1.0e-5).item())
            ),
            "task": args.task,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "physics_dt": direct_env.physics_dt,
            "contact_threshold_n": args.contact_threshold,
            "active_fingers": list(ACTIVE_FINGERS[args.scenario]),
            "baseline_object_position_m": _to_list(baseline_object_pos),
            "placement_calibration": calibration,
            "validation_open_delta_rad": VALIDATION_OPEN_DELTA_RAD,
            "validation_press_overtravel_rad": VALIDATION_PRESS_OVERTRAVEL_RAD,
            "test_object_size_m": list(env_cfg.object.spawn.size),
            "fixed_base_position_error_m": _to_list(root_position_error),
            "fixed_base_orientation_error": _to_list(root_orientation_error),
            "asset": {
                "usd_path": f"$WUJI_DESCRIPTION_ROOT/{WUJI_HAND_USD_RELATIVE_PATH.as_posix()}",
                "isaac_hand_mass_kg": hand_mass,
                "isaac_collision_body_count": len(direct_env.robot.body_names),
            },
            "per_env": per_env,
        }
        return result, rows
    finally:
        env.close()


def main() -> int:
    summary_path, csv_path = _output_paths()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result, rows = run()
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "WUJI_GRASP_VALIDATION "
        f"scenario={args.scenario} pass={result['pass']} "
        f"summary={summary_path} timeseries={csv_path} "
        f"per_env={result['per_env']}",
        flush=True,
    )
    return 0 if result["pass"] or args.no_fail_exit else 1


try:
    exit_code = main()
except BaseException:
    # Let Python propagate validation errors; Isaac Sim's close path exits the
    # process with status 0 and would otherwise hide the original failure.
    raise

if exit_code != 0:
    # Isaac Sim registers an exit hook that overwrites normal Python exits.
    # The summary is already durable; preserve the hard-gate status explicitly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

# A passing run can use the normal Isaac Sim shutdown path.
simulation_app.close()
raise SystemExit(0)
