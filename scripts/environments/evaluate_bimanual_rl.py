"""Deterministic 200+ episode policy and counterfactual evaluation."""

import argparse
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("grasp_hold", "cap_turn_30"), default="grasp_hold")
parser.add_argument("--policy", choices=("learned", "zero", "random", "left_only", "right_only", "reverse"),
                    default="zero")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--held-out", action="store_true")
parser.add_argument("--stable-eval-bank", type=Path,
                    help="Separate physically validated stable bank for evaluation only")
parser.add_argument("--source", choices=("mixed", "stable", "pregrasp"), default="mixed")
parser.add_argument("--thread-backend", choices=("legacy_d6", "mimic"), default="legacy_d6")
parser.add_argument("--no-right-cap-contact", action="store_true")
parser.add_argument("--thread-axial-stiffness", type=float,
                    help="Controlled thread stiffness ablation in N/m; default uses task config")
parser.add_argument("--finger-increment-scale", type=float,
                    help="Controlled finger target-increment ablation in rad/action")
parser.add_argument("--thread-trace-output", type=Path,
                    help="Optional one-environment 500 Hz thread load trace JSON")
parser.add_argument("--thread-trace-substeps", type=int, default=500)
parser.add_argument("--solver-position-iterations", type=int,
                    help="Controlled PhysX position solver iteration ablation")
parser.add_argument("--solve-articulation-contact-last", action="store_true",
                    help="Controlled PhysX articulation/contact solver-order ablation")
parser.add_argument("--output", type=Path)
parser.add_argument("--episode-rows-output", type=Path,
                    help="Optional per-episode physical outcomes for independent review")
parser.add_argument("--contact-point-output", type=Path,
                    help="Optional one-environment cap/right-hand PhysX contact point trace")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.policy == "learned" and (args.checkpoint is None or not args.checkpoint.is_file()):
    parser.error("--policy learned requires an existing --checkpoint")
if args.stable_eval_bank and (not args.held_out or args.source != "stable"):
    parser.error("--stable-eval-bank requires --held-out --source stable")
if args.thread_trace_output and (args.num_envs != 1 or args.thread_trace_substeps <= 0):
    parser.error("--thread-trace-output requires --num-envs 1 and positive --thread-trace-substeps")
if args.contact_point_output and args.num_envs != 1:
    parser.error("--contact-point-output requires --num-envs 1")
if args.thread_backend == "mimic" and (args.source != "pregrasp" or args.stable_eval_bank):
    parser.error("mimic evaluation currently requires --source pregrasp and no legacy stable bank")
if args.thread_backend == "mimic" and args.thread_trace_output:
    parser.error("legacy explicit-wrench thread trace is unavailable for the mimic backend")
app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from importlib import metadata
from dexteroushands_lab.tasks.direct.bimanual_bottle.agents.rsl_rl_ppo_cfg import BimanualBottlePPORunnerCfg
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.assets.bottle.batched_thread import _rotate
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import (
    GraspHoldDirectEnvCfg, CapTurn30DirectEnvCfg,
)

cfg = GraspHoldDirectEnvCfg() if args.task == "grasp_hold" else CapTurn30DirectEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.thread_backend = args.thread_backend
cfg.evaluation_mode = args.held_out
cfg.stable_eval_bank_path = str(args.stable_eval_bank) if args.stable_eval_bank else None
cfg.no_right_cap_contact_counterfactual = args.no_right_cap_contact
if args.thread_axial_stiffness is not None:
    cfg.thread_axial_stiffness = args.thread_axial_stiffness
if args.finger_increment_scale is not None:
    if args.finger_increment_scale <= 0:
        parser.error("--finger-increment-scale must be positive")
    cfg.finger_increment_scale = args.finger_increment_scale
if args.solver_position_iterations is not None:
    if not 1 <= args.solver_position_iterations <= 255:
        parser.error("--solver-position-iterations must be in [1, 255]")
    cfg.sim.physics.min_position_iteration_count = args.solver_position_iterations
    cfg.sim.physics.max_position_iteration_count = args.solver_position_iterations
if args.solve_articulation_contact_last:
    cfg.sim.physics.solve_articulation_contact_last = True
if args.source == "stable":
    cfg.stable_reset_probability = 1.0
elif args.source == "pregrasp":
    cfg.stable_reset_probability = 0.0
raw_env = BimanualBottleDirectEnv(cfg)
env = RslRlVecEnvWrapper(raw_env)
torch.manual_seed(args.seed)
print("EVAL_STAGE environment_ready", flush=True)

thread_trace_fields = (
    "theta_rad", "axial_m", "desired_axial_m", "constraint_error_m",
    "lateral_error_m", "axial_velocity_m_s", "spin_rad_s",
    "requested_axial_force_n", "requested_total_force_n",
    "applied_axial_force_n", "applied_total_force_n",
)
thread_trace_rows = []
thread_trace_contacts = []
contact_point_rows = []
contact_point_view = None
if args.contact_point_output:
    from isaaclab_physx.physics import PhysxManager

    filters = [f"/World/envs/env_0/RightHand/{body}"
               for body in raw_env.contacts.body_names["right"]]
    contact_point_view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
        "/World/envs/env_0/Cap", filter_patterns=filters, max_contact_data_count=128)
    if (contact_point_view is None
            or list(contact_point_view.sensor_paths) != ["/World/envs/env_0/Cap"]
            or list(contact_point_view.filter_paths[0]) != filters):
        raise RuntimeError("Cap/right PhysX contact point filter order changed")
if args.thread_trace_output:
    original_apply_action = raw_env._apply_action

    def traced_apply_action():
        original_apply_action()
        if len(thread_trace_rows) >= args.thread_trace_substeps:
            return
        result = raw_env._latest_thread
        bottle_pose = raw_env.bottle.data.root_pose_w.torch[0]
        cap_pose = raw_env.cap.data.root_pose_w.torch[0]
        bottle_velocity = raw_env.bottle.data.root_vel_w.torch[0]
        cap_velocity = raw_env.cap.data.root_vel_w.torch[0]
        axis = result.axis_world[0]
        displacement = cap_pose[:3] - bottle_pose[:3]
        relative_velocity = (cap_velocity[:3] - bottle_velocity[:3]
                             - torch.cross(bottle_velocity[3:], displacement, dim=-1))
        axial_velocity = torch.dot(relative_velocity, axis)
        lateral_velocity = relative_velocity - axis * axial_velocity
        lateral = displacement - axis * result.axial[0]
        config = raw_env.thread.config
        requested_axial = (-config.axial_stiffness * result.constraint_error[0]
                           - config.axial_damping * axial_velocity)
        requested = (-config.lateral_stiffness * lateral
                     - config.lateral_damping * lateral_velocity
                     + axis * requested_axial)
        thread_trace_rows.append(torch.stack((
            result.theta_unwrapped[0], result.axial[0], result.desired_axial[0],
            result.constraint_error[0], result.lateral_error[0], axial_velocity,
            torch.dot(cap_velocity[3:] - bottle_velocity[3:], axis),
            requested_axial, torch.linalg.vector_norm(requested),
            torch.dot(result.cap_force[0], axis),
            torch.linalg.vector_norm(result.cap_force[0]),
        )).detach().clone())

    raw_env._apply_action = traced_apply_action

if args.checkpoint is not None and args.policy in ("learned", "left_only", "right_only"):
    agent_cfg = handle_deprecated_rsl_rl_cfg(BimanualBottlePPORunnerCfg(),
                                               metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=args.device)
    print("EVAL_STAGE checkpoint_loaded", flush=True)
else:
    policy = None

results = {"stable": [], "pregrasp": []}
total = 0
max_opening = float("-inf")
sum_action_abs = 0.0
sum_action_saturation = 0.0
action_samples = 0
counterfactual_right_contact_steps = 0
step_metric_sums = {}
start = time.monotonic()
obs = env.get_observations()
print("EVAL_STAGE observations_ready", flush=True)
steps = 0
try:
    while total < args.episodes:
        steps += 1
        if steps % 25 == 0:
            print("EVAL_STAGE progress", steps, total, flush=True)
        if policy is not None:
            with torch.inference_mode():
                action = policy(obs)
            action = action.clone()
            if args.policy == "left_only":
                action[:, 26:] = 0
            elif args.policy == "right_only":
                action[:, :26] = 0
        elif args.policy == "zero":
            action = torch.zeros(args.num_envs, 52, device=args.device)
        else:
            action = torch.randn(args.num_envs, 52, device=args.device) * 0.25
            if args.policy == "left_only":
                action[:, 26:] = 0
            elif args.policy == "right_only":
                action[:, :26] = 0
            elif args.policy == "reverse":
                action[:] = 0
                bottle_q = raw_env.bottle.data.root_pose_w.torch[:, 3:]
                wrist_q = raw_env.right.data.root_pose_w.torch[:, 3:]
                world_axis = _rotate(bottle_q, torch.tensor(
                    [0., 0., 1.], device=args.device).expand(args.num_envs, 3))
                wrist_inverse = torch.cat((-wrist_q[:, :3], wrist_q[:, 3:]), dim=-1)
                local_reverse = _rotate(wrist_inverse, -world_axis)
                action[:, 29:32] = 0.5 * local_reverse
        sum_action_abs += float(action.abs().mean())
        sum_action_saturation += float((action.abs() > 1).float().mean())
        action_samples += 1
        obs, reward, dones, info = env.step(action)
        # DirectRLEnv auto-resets terminated rows before step() returns. Their
        # post-reset cap pose cannot reframe the previous episode's manifold.
        if args.contact_point_output and not bool(dones[0]):
            import warp as wp

            normal_forces, points, normals, distances, counts, starts = contact_point_view.get_contact_data(
                dt=cfg.sim.dt)
            normal_forces, points, normals, distances, counts, starts = map(
                wp.to_torch, (normal_forces, points, normals, distances, counts, starts))
            counts, starts = counts.reshape(-1), starts.reshape(-1)
            cap_pose = raw_env.cap.data.root_pose_w.torch[0]
            inverse_q = torch.cat((-cap_pose[3:6], cap_pose[6:7]))
            for body_index, body_name in enumerate(raw_env.contacts.body_names["right"]):
                begin = int(starts[body_index])
                end = begin + int(counts[body_index])
                for point_index in range(begin, end):
                    force = float(torch.linalg.vector_norm(normal_forces[point_index]))
                    if force < raw_env.contacts.force_floor_n:
                        continue
                    point_local = _rotate(inverse_q.unsqueeze(0),
                                          (points[point_index] - cap_pose[:3]).unsqueeze(0))[0]
                    contact_point_rows.append({
                        "episode_index": total, "policy_step": steps,
                        "body": body_name, "point_cap_local_m": point_local.detach().cpu().tolist(),
                        "normal_w": normals[point_index].detach().cpu().tolist(),
                        "normal_force_n": force,
                        "separation_m": float(distances.reshape(-1)[point_index]),
                        "cap_physics_valid_at_step": bool(raw_env._metrics["cap_ok"][0]),
                    })
        if args.thread_trace_output and len(thread_trace_contacts) * cfg.decimation < args.thread_trace_substeps:
            thread_trace_contacts.append(torch.stack((
                raw_env.contacts.force_magnitudes("Bottle", "left")[0].sum(),
                raw_env.contacts.force_magnitudes("Cap", "right")[0].sum(),
            )).detach().clone())
        if args.no_right_cap_contact:
            touching = (raw_env.contacts.force_magnitudes("Cap", "right")
                        >= raw_env.contacts.force_floor_n).any(dim=-1)
            counterfactual_right_contact_steps += int(touching.sum())
        for name, value in info.get("metrics", {}).items():
            step_metric_sums[name] = step_metric_sums.get(name, 0.0) + float(value)
        episode = info["phase1_episode"]
        max_opening = max(max_opening, float(episode["opening_rad"].max()))
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids):
            done_rows = {name: values[done_ids].detach().cpu().tolist()
                         for name, values in episode.items()}
            for i in range(len(done_ids)):
                group = "stable" if done_rows["source_stable"][i] else "pregrasp"
                results[group].append({name: values[i] for name, values in done_rows.items()})
                total += 1
                if total >= args.episodes:
                    break
    elapsed = time.monotonic() - start
    summary = {"task": args.task, "policy": args.policy,
               "thread_backend": args.thread_backend,
               "checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "held_out": args.held_out, "seed": args.seed,
               "source": args.source,
               "stable_eval_bank_path": str(args.stable_eval_bank) if args.stable_eval_bank else None,
               "thread_axial_stiffness_n_per_m": cfg.thread_axial_stiffness,
               "finger_increment_scale_rad": cfg.finger_increment_scale,
               "solver_position_iterations": cfg.sim.physics.max_position_iteration_count,
               "solve_articulation_contact_last": cfg.sim.physics.solve_articulation_contact_last,
               "no_right_cap_contact_counterfactual": args.no_right_cap_contact,
               "counterfactual_right_contact_steps": counterfactual_right_contact_steps,
               "num_envs": args.num_envs, "episodes": total,
               "elapsed_s": elapsed, "maximum_opening_angle_rad": max_opening,
               "mean_abs_raw_action": sum_action_abs / action_samples,
               "raw_action_saturation_fraction": sum_action_saturation / action_samples,
               "mean_step_metrics": {name: value / action_samples
                                     for name, value in step_metric_sums.items()},
               "same_family_stable_evaluation": (
                   None if args.stable_eval_bank or args.thread_backend == "mimic" else True),
               "groups": {}}
    for name, rows in results.items():
        summary["groups"][name] = {
            "episodes": len(rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
            "failure_rate": sum(row["failure"] for row in rows) / len(rows) if rows else None,
            "mean_final_opening_rad": sum(row["opening_rad"] for row in rows) / len(rows) if rows else None,
            "mean_final_left_contact": sum(row["left_contact"] for row in rows) / len(rows) if rows else None,
            "mean_final_right_contact": sum(row["right_contact"] for row in rows) / len(rows) if rows else None,
            "breakaway_occurrence_rate": sum(row["breakaway_seen"] for row in rows) / len(rows) if rows else None,
            "mean_maximum_opening_rad": sum(row["maximum_opening_rad"] for row in rows) / len(rows) if rows else None,
            "mean_final_axial_travel_m": sum(row["axial_travel_m"] for row in rows) / len(rows) if rows else None,
            "positive_axial_travel_rate": (sum(row["axial_travel_m"] > 0 for row in rows)
                                           / len(rows) if rows else None),
            "mean_abs_progress_pitch_error_m": (
                sum(abs(row["axial_travel_m"] - raw_env.bottle_cfg.thread_pitch
                        * row["opening_rad"] / (2 * torch.pi)) for row in rows)
                / len(rows) if rows else None),
            "mean_abs_constraint_error_m": sum(abs(row["constraint_error_m"]) for row in rows) / len(rows) if rows else None,
            "ground_contact_rate": sum(row["ground_contact"] for row in rows) / len(rows) if rows else None,
            "constraint_failure_rate": sum(row["constraint_failure"] for row in rows) / len(rows) if rows else None,
        }
    if args.no_right_cap_contact and counterfactual_right_contact_steps:
        raise AssertionError("No-right-cap-contact ablation unexpectedly touched the cap")
    print("PHASE1_EVAL", json.dumps(summary, sort_keys=True), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.episode_rows_output:
        args.episode_rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.episode_rows_output.write_text(json.dumps({
            "checkpoint": summary["checkpoint"], "seed": args.seed,
            "thread_backend": args.thread_backend,
            "thread_pitch_m": raw_env.bottle_cfg.thread_pitch,
            "thread_sign": raw_env.bottle_cfg.thread_sign,
            "rows_by_source": results,
        }, indent=2, sort_keys=True) + "\n")
    if args.contact_point_output:
        args.contact_point_output.parent.mkdir(parents=True, exist_ok=True)
        region_summary = {}
        for frame_kind in ("all", "cap_physics_valid_at_step"):
            selected = [row for row in contact_point_rows
                        if frame_kind == "all" or row["cap_physics_valid_at_step"]]
            normal_total = sum(row["normal_force_n"] for row in selected)
            sidewall_load = sum(row["normal_force_n"] for row in selected
                                if abs(row["point_cap_local_m"][2]) < raw_env.bottle_cfg.cap_height / 2 - 0.002
                                and raw_env.bottle_cfg.cap_outer_radius - 0.003
                                <= math.hypot(*row["point_cap_local_m"][:2])
                                <= raw_env.bottle_cfg.cap_outer_radius + 0.003)
            top_load = sum(row["normal_force_n"] for row in selected
                           if row["point_cap_local_m"][2] >= raw_env.bottle_cfg.cap_height / 2 - 0.002)
            region_summary[frame_kind] = {
                "points": len(selected), "normal_force_sum_n": normal_total,
                "sidewall_normal_load_fraction": sidewall_load / normal_total if normal_total else None,
                "top_normal_load_fraction": top_load / normal_total if normal_total else None,
                "minimum_separation_m": min((row["separation_m"] for row in selected), default=None),
                "maximum_point_normal_force_n": max((row["normal_force_n"] for row in selected), default=None),
            }
        args.contact_point_output.write_text(json.dumps({
            "checkpoint": summary["checkpoint"], "seed": args.seed,
            "cap_outer_radius_m": raw_env.bottle_cfg.cap_outer_radius,
            "cap_height_m": raw_env.bottle_cfg.cap_height,
            "coordinate_frame": "cap local XYZ in metres, quaternion xyzw",
            "region_summary": region_summary,
            "rows": contact_point_rows,
        }, indent=2) + "\n")
    if args.thread_trace_output:
        trace = torch.stack(thread_trace_rows).cpu().tolist()
        trace_contacts = torch.stack(thread_trace_contacts).cpu().tolist()
        trace_summary = {
            "policy": args.policy, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "physics_dt_s": cfg.sim.dt, "substeps": len(trace),
            "solver_position_iterations": cfg.sim.physics.max_position_iteration_count,
            "solve_articulation_contact_last": cfg.sim.physics.solve_articulation_contact_last,
            "thread_axial_stiffness_n_per_m": raw_env.thread.config.axial_stiffness,
            "thread_force_limit_n": raw_env.thread.config.max_constraint_force,
            "max_abs_constraint_error_m": max(abs(row[3]) for row in trace),
            "force_clipped_fraction": sum(row[8] > raw_env.thread.config.max_constraint_force + 1e-5
                                          for row in trace) / len(trace),
            "substep_fields": thread_trace_fields,
            "substeps_data": trace,
            "policy_contact_fields": ("left_bottle_force_sum_n", "right_cap_force_sum_n"),
            "policy_contacts_data": trace_contacts,
        }
        args.thread_trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.thread_trace_output.write_text(json.dumps(trace_summary, indent=2) + "\n")
except BaseException as exc:
    print("EVAL_STAGE exception", type(exc).__name__, repr(exc), "steps", steps, flush=True)
    raise
finally:
    print("EVAL_STAGE finally", steps, total, flush=True)
    env.close()
    app.close()
