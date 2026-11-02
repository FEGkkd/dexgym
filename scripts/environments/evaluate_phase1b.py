"""Independent held-out evaluation of unified pregrasp-to-CapTurn30 PPO."""

import argparse
import json
import time
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--hand-variant", choices=("beta1", "beta2"), default="beta2")
parser.add_argument("--policy", choices=("learned", "zero", "random", "no_right", "no_left", "no_hands", "freeze_left", "freeze_right", "reverse"), default="zero")
parser.add_argument("--level", choices=("A", "B", "D"), default="A")
parser.add_argument("--reward-profile", choices=("baseline", "contact_coverage"), default="baseline")
parser.add_argument("--action-distribution", choices=("gaussian", "tanh"), default="gaussian")
parser.add_argument("--pgs-position-iterations", type=int, choices=(8, 16, 32), default=8,
                    help="Controlled PGS solver ablation; the trained task uses eight iterations")
parser.add_argument("--solve-articulation-contact-last", action="store_true",
                    help="Controlled PGS solver-order probe; the trained task uses the default ordering")
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=9117)
parser.add_argument("--output", type=Path)
parser.add_argument("--rows-output", type=Path)
parser.add_argument("--physics-trace-output", type=Path,
                    help="One-env 500 Hz screw/contact trace for a complete replay episode")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.policy in ("learned", "no_right", "no_left", "freeze_left", "freeze_right") and (not args.checkpoint or not args.checkpoint.is_file()):
    parser.error("Learned and ablation policies require --checkpoint")
if args.physics_trace_output and args.num_envs != 1:
    parser.error("--physics-trace-output requires --num-envs 1")
if args.level in ("B", "D") and args.hand_variant == "beta2":
    parser.error("Beta 2 weak-contact and original-distance pregrasps have not been physically qualified")
app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.env import PregraspCapTurn30DirectEnv
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.env_cfg import PregraspCapTurn30DirectEnvCfg
from dexteroushands_lab.tasks.direct.bimanual_bottle.variant_config import configure_bimanual_hand_variant
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.agents.rsl_rl_ppo_cfg import UnifiedPPORunnerCfg
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.action_math import base_evaluation_action
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.pose_spin import relative_pose_yaw, wrapped_delta
from dexteroushands_lab.assets.bottle.batched_thread import _rotate


def main():
    cfg = PregraspCapTurn30DirectEnvCfg()
    configure_bimanual_hand_variant(cfg, args.hand_variant)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.sim.physics.min_position_iteration_count = args.pgs_position_iterations
    cfg.sim.physics.max_position_iteration_count = args.pgs_position_iterations
    cfg.sim.physics.solve_articulation_contact_last = args.solve_articulation_contact_last
    cfg.seed = args.seed
    cfg.unified_reward_profile = args.reward_profile
    if args.level == "D":
        cfg.pregrasp_bank_path = "docs/assets/phase0_pregrasp_reset_bank.json"
        cfg.pregrasp_reset_levels = ()
        cfg.pregrasp_split_key = None
    else:
        cfg.pregrasp_reset_levels = (args.level,)
    cfg.evaluation_mode = True
    cfg.no_right_cap_contact_counterfactual = args.policy in ("no_right", "no_hands")
    cfg.no_left_bottle_contact_counterfactual = args.policy in ("no_left", "no_hands")
    raw = PregraspCapTurn30DirectEnv(cfg)
    env = RslRlVecEnvWrapper(raw)
    torch.manual_seed(args.seed)
    policy = None
    gaussian_std_groups = None
    if args.checkpoint:
        agent_cfg = handle_deprecated_rsl_rl_cfg(UnifiedPPORunnerCfg(), metadata.version("rsl-rl-lib"))
        if args.action_distribution == "tanh":
            agent_cfg.actor.distribution_cfg.class_name = (
                "dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.agents."
                "tanh_distribution:TanhGaussianDistribution")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        std = runner.alg.actor.distribution.std_param.detach().flatten().cpu()
        gaussian_std_groups = {
            name: float(std[start:end].mean()) for name, start, end in (
                ("left_wrist_translation", 0, 3), ("left_wrist_rotation", 3, 6),
                ("left_fingers", 6, 26), ("right_wrist_translation", 26, 29),
                ("right_wrist_rotation", 29, 32), ("right_fingers", 32, 52))}
    obs = env.get_observations()
    physics_trace = []
    if args.physics_trace_output:
        import warp as wp
        original_apply_action = raw._apply_action
        previous_theta = None
        previous_pose_wrapped = None
        pose_opening = 0.0

        def trace_apply_action():
            nonlocal previous_theta, previous_pose_wrapped, pose_opening
            original_apply_action()
            theta = float(raw.thread.theta_unwrapped[0])
            joint_spin = (None if previous_theta is None else
                          (theta - previous_theta) / cfg.sim.dt)
            previous_theta = theta
            bottle_pose = raw.bottle.data.root_pose_w.torch[0]
            cap_pose = raw.cap.data.root_pose_w.torch[0]
            pose_wrapped = relative_pose_yaw(
                bottle_pose[3:].unsqueeze(0), cap_pose[3:].unsqueeze(0))[0]
            pose_spin = None
            if previous_pose_wrapped is not None:
                delta = float(wrapped_delta(pose_wrapped, previous_pose_wrapped))
                pose_opening += delta
                pose_spin = delta / cfg.sim.dt
            previous_pose_wrapped = pose_wrapped
            bottle_angular_velocity_w = raw.bottle.data.root_ang_vel_w.torch[0]
            cap_angular_velocity_w = raw.cap.data.root_ang_vel_w.torch[0]
            bottle_axis_w = _rotate(
                bottle_pose[3:].unsqueeze(0),
                torch.tensor([[0.0, 0.0, 1.0]], device=args.device))[0]
            relative_body_spin = torch.dot(
                cap_angular_velocity_w - bottle_angular_velocity_w, bottle_axis_w)
            drive_state = raw.thread.drive.last_state
            if drive_state is None:
                raise RuntimeError("Mimic resistance drive state missing from substep trace")
            view = raw.contacts.cap_right_manifold_view
            if view is None:
                raise RuntimeError("PGS cap/right manifold contact view is required for trace")
            force, point, normal, separation, counts, starts = map(
                wp.to_torch, view.get_contact_data(dt=cfg.sim.dt))
            force = force.reshape(force.shape[0], -1)
            point = point.reshape(-1, 3)
            normal = normal.reshape(-1, 3)
            counts, starts = counts.reshape(-1), starts.reshape(-1)
            points = []
            contact_overflow = False
            for body_index, body_name in enumerate(raw.contacts.cap_right_manifold_bodies):
                begin = int(starts[body_index])
                end = begin + int(counts[body_index])
                if end > len(force) or end > len(point) or end > separation.numel():
                    contact_overflow = True
                end = min(end, len(force), len(point), separation.numel())
                for j in range(begin, end):
                    magnitude = float(torch.linalg.vector_norm(force[j]))
                    if magnitude < raw.contacts.force_floor_n:
                        continue
                    points.append({"body": body_name,
                                   "point_world_m": point[j].detach().cpu().tolist(),
                                   "normal_world": normal[j].detach().cpu().tolist(),
                                   "normal_force_n": magnitude,
                                   "signed_separation_m": float(separation.reshape(-1)[j])})
            physics_trace.append({
                "substep": len(physics_trace), "time_s": len(physics_trace) * cfg.sim.dt,
                "reset_sample_index": int(raw._reset_sample_index[0]),
                "theta_unwrapped_rad": theta,
                "bottle_pose_world_xyzw": bottle_pose.detach().cpu().tolist(),
                "cap_pose_world_xyzw": cap_pose.detach().cpu().tolist(),
                "left_wrist_pose_world_xyzw": raw.wrists[0].current_pose[0].detach().cpu().tolist(),
                "right_wrist_pose_world_xyzw": raw.wrists[1].current_pose[0].detach().cpu().tolist(),
                "left_actor_action_26d": raw._last_action[0, :26].detach().cpu().tolist(),
                "right_actor_action_26d": raw._last_action[0, 26:].detach().cpu().tolist(),
                "joint_differenced_spin_rad_s": joint_spin,
                "pose_relative_opening_rad": pose_opening,
                "pose_differenced_spin_rad_s": pose_spin,
                "physx_screw_spin_rad_s": float(raw.thread.state.spin_velocity_rad_s[0]),
                "bottle_angular_velocity_world_rad_s": bottle_angular_velocity_w.detach().cpu().tolist(),
                "cap_angular_velocity_world_rad_s": cap_angular_velocity_w.detach().cpu().tolist(),
                "relative_body_spin_projected_rad_s": float(relative_body_spin),
                "drive_sliding": bool(drive_state.sliding[0]),
                "drive_breakaway_seen": bool(drive_state.breakaway_seen[0]),
                "drive_active_force_limit_nm": float(drive_state.active_force_limit_nm[0]),
                "axial_m": float(raw._latest_thread.axial[0]),
                "constraint_error_m": float(raw._latest_thread.constraint_error[0]),
                "left_bottle_normal_force_n": float(raw.contacts.force_magnitudes(
                    "Bottle", "left")[0].sum()),
                "right_cap_normal_force_n": float(raw.contacts.force_magnitudes(
                    "Cap", "right")[0].sum()),
                "contact_buffer_overflow": contact_overflow,
                "right_cap_contact_points": points,
            })

        raw._apply_action = trace_apply_action
    rows = []
    metric_sums = {}
    raw_outside = 0.0
    group_action_sums = {name: {"outside": 0.0, "abs_mean": 0.0}
                         for name in ("left_wrist_translation", "left_wrist_rotation",
                                      "left_fingers", "right_wrist_translation",
                                      "right_wrist_rotation", "right_fingers")}
    start = time.monotonic()
    steps = 0
    try:
        while len(rows) < args.episodes:
            steps += 1
            action = base_evaluation_action(args.policy, policy, obs)
            if args.policy == "reverse":
                bottle_q = raw.bottle.data.root_pose_w.torch[:, 3:]
                wrist_q = raw.right.data.root_pose_w.torch[:, 3:]
                world_axis = _rotate(bottle_q, torch.tensor(
                    [0., 0., 1.], device=args.device).expand(args.num_envs, 3))
                wrist_inverse = torch.cat((-wrist_q[:, :3], wrist_q[:, 3:]), dim=-1)
                action[:, 29:32] = .5 * _rotate(wrist_inverse, -world_axis)
            else:
                if args.policy in ("no_right", "freeze_right"):
                    action[:, 26:] = 0
                elif args.policy in ("no_left", "freeze_left"):
                    action[:, :26] = 0
            if args.policy in ("no_right", "no_hands"):
                action[:, 26:] = 0
            if args.policy in ("no_left", "no_hands"):
                action[:, :26] = 0
            raw_outside += float((action.abs() > 1).float().mean())
            for name, start_index, end_index in (
                    ("left_wrist_translation", 0, 3), ("left_wrist_rotation", 3, 6),
                    ("left_fingers", 6, 26), ("right_wrist_translation", 26, 29),
                    ("right_wrist_rotation", 29, 32), ("right_fingers", 32, 52)):
                group = action[:, start_index:end_index]
                group_action_sums[name]["outside"] += float((group.abs() > 1).float().mean())
                group_action_sums[name]["abs_mean"] += float(group.abs().mean())
            obs, reward, dones, info = env.step(action)
            for name, value in info.get("metrics", {}).items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids):
                ep = info["phase1_episode"]
                batch = {name: tensor[done_ids].detach().cpu().tolist() for name, tensor in ep.items()}
                for i in range(len(done_ids)):
                    rows.append({name: values[i] for name, values in batch.items()})
                    if len(rows) == args.episodes:
                        break
            if steps % 100 == 0:
                print("PHASE1B_EVAL_PROGRESS", steps, len(rows), flush=True)
        summary = {
            "task": "DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0",
            "hand_variant": args.hand_variant,
            "level": args.level, "held_out": True, "policy": args.policy,
            "reward_profile": args.reward_profile,
            "action_distribution": args.action_distribution,
            "pgs_position_iterations": args.pgs_position_iterations,
            "solve_articulation_contact_last": args.solve_articulation_contact_last,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "seed": args.seed, "episodes": len(rows), "num_envs": args.num_envs,
            "steps": steps, "elapsed_s": time.monotonic() - start,
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "failure_rate": sum(r["failure"] for r in rows) / len(rows),
            "ground_contact_rate": sum(r["ground_contact"] for r in rows) / len(rows),
            "constraint_failure_rate": sum(r["constraint_failure"] for r in rows) / len(rows),
            "mean_maximum_opening_rad": sum(r["maximum_opening_rad"] for r in rows) / len(rows),
            "maximum_opening_rad": max(r["maximum_opening_rad"] for r in rows),
            "mean_final_pitch_error_m": sum(abs(r["axial_travel_m"] -
                raw.bottle_cfg.thread_pitch * r["opening_rad"] / (2 * torch.pi)) for r in rows) / len(rows),
            "raw_action_element_outside_fraction": raw_outside / steps,
            "group_raw_action": {name: {key: value / steps for key, value in sums.items()}
                                 for name, sums in group_action_sums.items()},
            "gaussian_std_groups": gaussian_std_groups,
            "mean_step_metrics": {name: total / steps for name, total in metric_sums.items()},
            "distinct_local_reset_rows": len({r["reset_sample_index"] for r in rows}),
            "same_relative_template_family": args.level != "D",
        }
        print("PHASE1B_EVAL", json.dumps(summary, sort_keys=True), flush=True)
        for path, data in ((args.output, summary), (args.rows_output, rows)):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        if args.physics_trace_output:
            args.physics_trace_output.parent.mkdir(parents=True, exist_ok=True)
            args.physics_trace_output.write_text(json.dumps({
                "policy": args.policy, "checkpoint": summary["checkpoint"],
                "episode_success": rows[0]["success"],
                "policy_hz": 50, "physx_hz": 500,
                "pgs_position_iterations": args.pgs_position_iterations,
                "solve_articulation_contact_last": args.solve_articulation_contact_last,
                "cap_outer_radius_m": raw.bottle_cfg.cap_outer_radius,
                "cap_height_m": raw.bottle_cfg.cap_height,
                "rows": physics_trace,
            }, indent=2) + "\n")
    finally:
        env.close()


try:
    main()
except BaseException as exc:
    print("PHASE1B_EVAL_ERROR", type(exc).__name__, repr(exc), flush=True)
    raise
finally:
    app.close()
