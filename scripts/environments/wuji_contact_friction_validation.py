"""Validate WUJI fingertip contact and Coulomb friction without requiring a grasp."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(
    description="Validate the WUJI hand/object contact material with a loaded fingertip probe."
)
parser.add_argument(
    "--task", default="DexterousHands-Wuji-Hand-Reorientation-Direct-v0"
)
parser.add_argument(
    "--finger",
    choices=("thumb", "index", "middle", "ring", "pinky"),
    default="index",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--preload_steps", type=int, default=80)
parser.add_argument("--static_steps", type=int, default=50)
parser.add_argument("--slip_steps", type=int, default=20)
parser.add_argument("--static_load_ratio", type=float, default=0.20)
parser.add_argument("--slip_load_ratio", type=float, default=0.90)
parser.add_argument("--contact_threshold", type=float, default=0.05)
parser.add_argument("--placement_penetration", type=float, default=0.001)
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
from dexteroushands_lab.assets.wuji_mjcf_reference import (  # noqa: E402
    WUJI_OFFICIAL_HAND_CONTACT,
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
# Native USD collision mesh centers relative to each tip_link body frame.
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
OPEN_DELTA_RAD = 0.65
PROBE_SIZE_M = 0.020


def _to_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _output_paths() -> tuple[Path, Path]:
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = (
            REPO_ROOT
            / "artifacts"
            / "wuji_contact_friction_validation"
            / f"{stamp}_{args.finger}"
        )
        return directory / "summary.json", directory / "timeseries.csv"
    if args.output.suffix.lower() == ".json":
        return args.output, args.output.with_suffix(".csv")
    return args.output / "summary.json", args.output / "timeseries.csv"


def _step(env) -> None:
    env.scene.write_data_to_sim()
    env.sim.step(render=not args.headless)
    env.scene.update(dt=env.physics_dt)


def _write_robot_state(env, joint_pos_1d: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    root_pose = env.robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += env.scene.env_origins
    root_velocity = torch.zeros_like(env.robot.data.default_root_vel.torch)
    env.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim_index(
        root_velocity=root_velocity, env_ids=env_ids
    )
    joint_pos = joint_pos_1d.expand(env.num_envs, -1).clone()
    joint_vel = torch.zeros_like(joint_pos)
    env.robot.set_joint_position_target_index(
        target=joint_pos, joint_ids=env.actuated_joint_ids, env_ids=env_ids
    )
    env.robot.write_joint_position_to_sim_index(
        position=joint_pos, joint_ids=env.actuated_joint_ids, env_ids=env_ids
    )
    env.robot.write_joint_velocity_to_sim_index(
        velocity=joint_vel, joint_ids=env.actuated_joint_ids, env_ids=env_ids
    )


def _write_object_state(env, position: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    root_pose = env.object.data.default_root_pose.torch.clone()
    if position.ndim == 1:
        position = position.expand(env.num_envs, -1)
    root_pose[:, :3] = position + env.scene.env_origins
    root_pose[:, 3:7] = torch.tensor(
        (0.0, 0.0, 0.0, 1.0), device=env.device, dtype=root_pose.dtype
    )
    root_velocity = torch.zeros_like(env.object.data.default_root_vel.torch)
    env.object.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim_index(
        root_velocity=root_velocity, env_ids=env_ids
    )


def _tip_site_positions(env) -> torch.Tensor:
    actual_tip_names = tuple(env._contact_body_names[index] for index in env._tip_body_ids)
    if actual_tip_names != FINGER_TIP_BODY_NAMES:
        raise RuntimeError(
            f"Unexpected WUJI distal-body order: {actual_tip_names} != {FINGER_TIP_BODY_NAMES}"
        )
    distal_pose = env.robot.data.body_link_pose_w.torch[:, env._tip_body_ids]
    local_offsets = torch.tensor(
        FINGER_TIP_LOCAL_OFFSETS, device=env.device, dtype=distal_pose.dtype
    ).expand(env.num_envs, -1, -1)
    world_offsets = quat_apply(
        distal_pose[:, :, 3:7].reshape(-1, 4), local_offsets.reshape(-1, 3)
    ).reshape(env.num_envs, len(FINGER_NAMES), 3)
    return distal_pose[:, :, :3] + world_offsets - env.scene.env_origins[:, None, :]


def _poses(env) -> tuple[torch.Tensor, torch.Tensor]:
    closed = env.default_joint_pos[0].clone()
    open_pose = closed.clone()
    finger_slice = FINGER_JOINT_SLICES[args.finger]
    for local_index in FLEX_LOCAL_INDICES:
        open_pose[finger_slice.start + local_index] -= OPEN_DELTA_RAD
    open_pose = torch.maximum(
        torch.minimum(open_pose, env.joint_upper_limits[0]), env.joint_lower_limits[0]
    )
    return open_pose, closed


def _calibrate_fixture(
    env, open_pose: torch.Tensor, closed_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    parked = torch.tensor((0.0, 0.0, 1.5), device=env.device, dtype=open_pose.dtype)
    _write_object_state(env, parked)
    _write_robot_state(env, open_pose)
    for _ in range(4):
        _step(env)
    open_tips = _tip_site_positions(env).clone()

    _write_robot_state(env, closed_pose)
    for _ in range(4):
        _step(env)
    closed_tips = _tip_site_positions(env).clone()

    finger_index = FINGER_NAMES.index(args.finger)
    motion = closed_tips[:, finger_index] - open_tips[:, finger_index]
    travel = torch.linalg.norm(motion, dim=-1, keepdim=True)
    if torch.any(travel < 1.0e-4):
        raise RuntimeError("Selected fingertip moved less than 0.1 mm during calibration.")
    normal_axis = motion / travel

    half_extent = 0.5 * PROBE_SIZE_M
    ray_extent = half_extent / torch.amax(torch.abs(normal_axis), dim=-1, keepdim=True)
    object_position = closed_tips[:, finger_index] + normal_axis * (
        ray_extent - args.placement_penetration
    )

    world_z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand_as(normal_axis)
    tangent_axis = torch.linalg.cross(normal_axis, world_z, dim=-1)
    use_world_y = torch.linalg.norm(tangent_axis, dim=-1) < 0.2
    world_y = torch.tensor((0.0, 1.0, 0.0), device=env.device).expand_as(normal_axis)
    tangent_axis = torch.where(
        use_world_y[:, None],
        torch.linalg.cross(normal_axis, world_y, dim=-1),
        tangent_axis,
    )
    tangent_axis = tangent_axis / torch.linalg.norm(tangent_axis, dim=-1, keepdim=True)

    calibration = {
        "tip_frame_source": "official WUJI USD tip_link rigid body",
        "open_tip_position_m": _to_list(open_tips[:, finger_index]),
        "closed_tip_position_m": _to_list(closed_tips[:, finger_index]),
        "tip_travel_m": _to_list(travel.squeeze(-1)),
        "normal_axis_world": _to_list(normal_axis),
        "tangent_axis_world": _to_list(tangent_axis),
        "object_position_m": _to_list(object_position),
        "placement_penetration_m": args.placement_penetration,
    }
    return object_position, normal_axis, tangent_axis, calibration


def _set_object_force(
    env,
    tangent_axis: torch.Tensor,
    tangent_load_n: float,
) -> None:
    force = tangent_load_n * tangent_axis
    force = force[:, None, :]
    torque = torch.zeros_like(force)
    env.object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=force,
        torques=torque,
        body_ids=torch.tensor([0], device=env.device, dtype=torch.long),
        env_ids=torch.arange(env.num_envs, device=env.device, dtype=torch.long),
        is_global=True,
    )


def _constrain_fixture_to_tangent_axis(
    env,
    baseline_position: torch.Tensor,
    tangent_axis: torch.Tensor,
    allow_tangent_motion: bool = True,
) -> None:
    """Model a fixed preload fixture or a rail with one translation DOF."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    pose = env.object.data.root_pose_w.torch.clone()
    local_position = pose[:, :3] - env.scene.env_origins
    tangent_displacement = torch.sum(
        (local_position - baseline_position) * tangent_axis, dim=-1, keepdim=True
    )
    if not allow_tangent_motion:
        tangent_displacement.zero_()
    pose[:, :3] = (
        baseline_position
        + tangent_displacement * tangent_axis
        + env.scene.env_origins
    )
    pose[:, 3:7] = torch.tensor(
        (0.0, 0.0, 0.0, 1.0), device=env.device, dtype=pose.dtype
    )
    velocity = env.object.data.root_vel_w.torch.clone()
    tangent_speed = torch.sum(
        velocity[:, :3] * tangent_axis, dim=-1, keepdim=True
    )
    if not allow_tangent_motion:
        tangent_speed.zero_()
    velocity[:, :3] = tangent_speed * tangent_axis
    velocity[:, 3:] = 0.0
    env.object.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim_index(
        root_velocity=velocity, env_ids=env_ids
    )


def _sample(
    env,
    phase: str,
    step_index: int,
    tangent_load_n: float,
    baseline_position: torch.Tensor,
    normal_axis: torch.Tensor,
    tangent_axis: torch.Tensor,
    active_body_id: int,
) -> list[dict[str, object]]:
    normal_forces = env._contact_forces_with_object()
    friction_forces = env._contact_friction_forces_with_object()
    normal_vector = normal_forces[:, active_body_id]
    friction_vector = friction_forces[:, active_body_id]
    normal_magnitude = torch.linalg.norm(normal_vector, dim=-1)
    friction_magnitude = torch.linalg.norm(friction_vector, dim=-1)
    ratio = torch.where(
        normal_magnitude > args.contact_threshold,
        friction_magnitude / normal_magnitude.clamp_min(1.0e-8),
        torch.zeros_like(normal_magnitude),
    )

    position = env.object.data.root_pos_w.torch - env.scene.env_origins
    velocity = env.object.data.root_lin_vel_w.torch
    displacement = position - baseline_position
    tangent_displacement = torch.sum(displacement * tangent_axis, dim=-1)
    normal_displacement = torch.sum(displacement * normal_axis, dim=-1)
    tangent_speed = torch.sum(velocity * tangent_axis, dim=-1)
    friction_tangent = torch.sum(friction_vector * tangent_axis, dim=-1)
    normal_reaction = torch.sum(normal_vector * normal_axis, dim=-1)

    all_normal_magnitude = torch.linalg.norm(normal_forces, dim=-1)
    contact_mask = all_normal_magnitude > args.contact_threshold
    rows = []
    for env_index in range(env.num_envs):
        contact_ids = contact_mask[env_index].nonzero(as_tuple=False).flatten().tolist()
        rows.append(
            {
                "step": step_index,
                "time_s": step_index * env.physics_dt,
                "phase": phase,
                "env_id": env_index,
                "tangent_load_n": tangent_load_n,
                "normal_force_n": float(normal_magnitude[env_index].item()),
                "normal_reaction_projection_n": float(normal_reaction[env_index].item()),
                "friction_force_n": float(friction_magnitude[env_index].item()),
                "friction_tangent_projection_n": float(friction_tangent[env_index].item()),
                "friction_to_normal_ratio": float(ratio[env_index].item()),
                "normal_force_x_n": float(normal_vector[env_index, 0].item()),
                "normal_force_y_n": float(normal_vector[env_index, 1].item()),
                "normal_force_z_n": float(normal_vector[env_index, 2].item()),
                "friction_force_x_n": float(friction_vector[env_index, 0].item()),
                "friction_force_y_n": float(friction_vector[env_index, 1].item()),
                "friction_force_z_n": float(friction_vector[env_index, 2].item()),
                "tangent_displacement_m": float(tangent_displacement[env_index].item()),
                "normal_displacement_m": float(normal_displacement[env_index].item()),
                "tangent_speed_mps": float(tangent_speed[env_index].item()),
                "object_speed_mps": float(torch.linalg.norm(velocity[env_index]).item()),
                "contact_body_names": "|".join(
                    env._contact_body_names[index] for index in contact_ids
                ),
            }
        )
    return rows


def _phase_stats(rows: list[dict[str, object]], phase: str) -> dict[str, float]:
    phase_rows = [row for row in rows if row["phase"] == phase]
    contact_rows = [
        row for row in phase_rows if float(row["normal_force_n"]) > args.contact_threshold
    ]
    if not phase_rows:
        raise RuntimeError(f"No samples recorded for {phase} phase.")
    count = len(phase_rows)
    contact_count = len(contact_rows)
    return {
        "sample_count": count,
        "contact_ratio": contact_count / count,
        "mean_normal_force_n": sum(float(row["normal_force_n"]) for row in phase_rows) / count,
        "max_normal_force_n": max(float(row["normal_force_n"]) for row in phase_rows),
        "mean_friction_force_n": sum(float(row["friction_force_n"]) for row in phase_rows) / count,
        "max_friction_ratio": max(float(row["friction_to_normal_ratio"]) for row in phase_rows),
        "opposing_friction_ratio": (
            sum(float(row["friction_tangent_projection_n"]) < -1.0e-4 for row in contact_rows)
            / max(contact_count, 1)
        ),
        "max_abs_tangent_displacement_m": max(
            abs(float(row["tangent_displacement_m"])) for row in phase_rows
        ),
        "max_abs_tangent_speed_mps": max(
            abs(float(row["tangent_speed_mps"])) for row in phase_rows
        ),
        "max_object_speed_mps": max(float(row["object_speed_mps"]) for row in phase_rows),
    }


def run() -> tuple[dict[str, object], list[dict[str, object]]]:
    if args.num_envs != 1:
        raise ValueError("This calibrated fixture currently requires --num_envs 1.")
    if not 0.0 < args.static_load_ratio < float(WUJI_OFFICIAL_HAND_CONTACT["friction"]):
        raise ValueError("--static_load_ratio must be between zero and the configured friction.")
    if args.slip_load_ratio <= float(WUJI_OFFICIAL_HAND_CONTACT["friction"]):
        raise ValueError("--slip_load_ratio must exceed the configured friction.")
    if min(args.preload_steps, args.static_steps, args.slip_steps) < 5:
        raise ValueError("Every phase must contain at least five simulation steps.")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observation_noise_enabled = False
    env_cfg.object.spawn.size = (PROBE_SIZE_M,) * 3
    env_cfg.object.spawn.rigid_props.disable_gravity = True
    env_cfg.object.spawn.rigid_props.kinematic_enabled = False
    env = gym.make(args.task, cfg=env_cfg)
    rows: list[dict[str, object]] = []
    try:
        env.reset(seed=args.seed)
        direct_env = env.unwrapped
        open_pose, closed_pose = _poses(direct_env)
        object_position, normal_axis, tangent_axis, calibration = _calibrate_fixture(
            direct_env, open_pose, closed_pose
        )

        _write_robot_state(direct_env, open_pose)
        _write_object_state(direct_env, object_position)
        _step(direct_env)
        active_body_name = FINGER_TIP_BODY_NAMES[FINGER_NAMES.index(args.finger)]
        active_body_id = direct_env._contact_body_names.index(active_body_name)
        baseline = object_position.clone()
        rows.extend(
            _sample(
                direct_env,
                "initial",
                0,
                0.0,
                baseline,
                normal_axis,
                tangent_axis,
                active_body_id,
            )
        )

        _write_robot_state(direct_env, closed_pose)
        _write_object_state(direct_env, object_position)
        step_index = 0
        phase_baselines: dict[str, list] = {}

        phase_baselines["preload"] = _to_list(baseline)
        for _ in range(args.preload_steps):
            step_index += 1
            _constrain_fixture_to_tangent_axis(
                direct_env,
                baseline,
                tangent_axis,
            )
            direct_env.robot.set_joint_position_target_index(
                target=closed_pose.expand(direct_env.num_envs, -1),
                joint_ids=direct_env.actuated_joint_ids,
            )
            _set_object_force(direct_env, tangent_axis, 0.0)
            _step(direct_env)
            rows.extend(
                _sample(
                    direct_env,
                    "preload",
                    step_index,
                    0.0,
                    baseline,
                    normal_axis,
                    tangent_axis,
                    active_body_id,
                )
            )

        preload_contact_rows = [
            row
            for row in rows[-args.preload_steps * direct_env.num_envs :]
            if float(row["normal_force_n"]) > args.contact_threshold
        ]
        if not preload_contact_rows:
            normal_reference_n = 0.0
        else:
            steady_count = max(5, len(preload_contact_rows) // 4)
            steady_rows = preload_contact_rows[-steady_count:]
            normal_reference_n = sum(
                float(row["normal_force_n"])
                for row in steady_rows
            ) / steady_count
            measured_normal_axis = torch.tensor(
                [
                    [
                        sum(float(row[f"normal_force_{axis}_n"]) for row in steady_rows)
                        / steady_count
                        for axis in ("x", "y", "z")
                    ]
                ],
                device=direct_env.device,
                dtype=tangent_axis.dtype,
            ).expand(direct_env.num_envs, -1)
            measured_normal_axis = measured_normal_axis / torch.linalg.norm(
                measured_normal_axis, dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            calibration["measured_contact_normal_axis_world"] = _to_list(
                measured_normal_axis
            )

        phase_loads = (
            ("static", args.static_steps, normal_reference_n * args.static_load_ratio),
            ("slip", args.slip_steps, normal_reference_n * args.slip_load_ratio),
        )
        for phase, count, tangent_load in phase_loads:
            phase_baselines[phase] = _to_list(
                direct_env.object.data.root_pos_w.torch - direct_env.scene.env_origins
            )
            for _ in range(count):
                step_index += 1
                _constrain_fixture_to_tangent_axis(
                    direct_env, baseline, tangent_axis
                )
                direct_env.robot.set_joint_position_target_index(
                    target=closed_pose.expand(direct_env.num_envs, -1),
                    joint_ids=direct_env.actuated_joint_ids,
                )
                _set_object_force(direct_env, tangent_axis, tangent_load)
                _step(direct_env)
                rows.extend(
                    _sample(
                        direct_env,
                        phase,
                        step_index,
                        tangent_load,
                        baseline,
                        normal_axis,
                        tangent_axis,
                        active_body_id,
                    )
                )

        initial_rows = [row for row in rows if row["phase"] == "initial"]
        initial_condition_pass = all(
            float(row["normal_force_n"]) <= args.contact_threshold
            for row in initial_rows
        )
        preload = _phase_stats(rows, "preload")
        static = _phase_stats(rows, "static")
        slip = _phase_stats(rows, "slip")
        mu = float(WUJI_OFFICIAL_HAND_CONTACT["friction"])
        finite = all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "normal_force_n",
                "friction_force_n",
                "friction_to_normal_ratio",
                "tangent_displacement_m",
                "object_speed_mps",
            )
        )
        static_start = next(row for row in rows if row["phase"] == "static")
        static_end = [row for row in rows if row["phase"] == "static"][-1]
        slip_start = next(row for row in rows if row["phase"] == "slip")
        slip_end = [row for row in rows if row["phase"] == "slip"][-1]
        static_increment = abs(
            float(static_end["tangent_displacement_m"])
            - float(static_start["tangent_displacement_m"])
        )
        slip_increment = abs(
            float(slip_end["tangent_displacement_m"])
            - float(slip_start["tangent_displacement_m"])
        )

        checks = {
            "finite": finite,
            "no_initial_contact": initial_condition_pass,
            "preload_contact": preload["contact_ratio"] >= 0.80,
            "static_contact": static["contact_ratio"] >= 0.80,
            "static_friction_reported": static["mean_friction_force_n"] > 0.005,
            "static_friction_inside_cone": static["max_friction_ratio"] <= mu + 0.02,
            "static_translation_bounded": static_increment < 0.003,
            "slip_contact_observed": slip["contact_ratio"] >= 0.25,
            "slip_friction_opposes_load": slip["opposing_friction_ratio"] >= 0.60,
            "slip_friction_inside_cone": slip["max_friction_ratio"] <= mu + 0.02,
            "supercritical_load_causes_more_motion": slip_increment
            > max(0.0005, 2.0 * static_increment),
            "numerically_bounded": max(
                preload["max_object_speed_mps"],
                static["max_object_speed_mps"],
                slip["max_object_speed_mps"],
            ) < 5.0,
        }
        result = {
            "schema_version": 1,
            "test": "wuji_loaded_fingertip_contact_friction",
            "acceptance_scope": (
                "contact force reporting, friction-cone behavior, and dynamic stability; "
                "complete grasp and lift are not required"
            ),
            "pass": all(checks.values()),
            "checks": checks,
            "task": args.task,
            "finger": args.finger,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "physics_dt": direct_env.physics_dt,
            "fixture": {
                "probe_size_m": PROBE_SIZE_M,
                "probe_mass_kg": float(direct_env.object.data.body_mass.torch[0, 0].item()),
                "rotation_constraint": (
                    "one-dimensional rail: only tangent-axis translation remains dynamic"
                ),
                "measured_steady_normal_force_n": normal_reference_n,
                "static_tangent_load_n": normal_reference_n * args.static_load_ratio,
                "slip_tangent_load_n": normal_reference_n * args.slip_load_ratio,
                "phase_start_positions_m": phase_baselines,
            },
            "material": {
                "effective_friction": mu,
                "combine_mode": "max",
                "source": "WUJI official MJCF hand contact",
            },
            "calibration": calibration,
            "metrics": {
                "initial_max_normal_force_n": max(
                    float(row["normal_force_n"]) for row in initial_rows
                ),
                "preload": preload,
                "static": static,
                "slip": slip,
                "static_tangent_displacement_increment_m": static_increment,
                "slip_tangent_displacement_increment_m": slip_increment,
            },
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
        "WUJI_CONTACT_FRICTION_VALIDATION "
        f"finger={args.finger} pass={result['pass']} "
        f"summary={summary_path} timeseries={csv_path} checks={result['checks']}",
        flush=True,
    )
    return 0 if result["pass"] or args.no_fail_exit else 1


try:
    exit_code = main()
finally:
    simulation_app.close()

raise SystemExit(exit_code)
