"""Deterministic 200+ episode policy and counterfactual evaluation."""

import argparse
import json
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
parser.add_argument("--no-right-cap-contact", action="store_true")
parser.add_argument("--thread-axial-stiffness", type=float,
                    help="Controlled thread stiffness ablation in N/m; default uses task config")
parser.add_argument("--finger-increment-scale", type=float,
                    help="Controlled finger target-increment ablation in rad/action")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.policy == "learned" and (args.checkpoint is None or not args.checkpoint.is_file()):
    parser.error("--policy learned requires an existing --checkpoint")
if args.stable_eval_bank and (not args.held_out or args.source != "stable"):
    parser.error("--stable-eval-bank requires --held-out --source stable")
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
cfg.evaluation_mode = args.held_out
cfg.stable_eval_bank_path = str(args.stable_eval_bank) if args.stable_eval_bank else None
cfg.no_right_cap_contact_counterfactual = args.no_right_cap_contact
if args.thread_axial_stiffness is not None:
    cfg.thread_axial_stiffness = args.thread_axial_stiffness
if args.finger_increment_scale is not None:
    if args.finger_increment_scale <= 0:
        parser.error("--finger-increment-scale must be positive")
    cfg.finger_increment_scale = args.finger_increment_scale
if args.source == "stable":
    cfg.stable_reset_probability = 1.0
elif args.source == "pregrasp":
    cfg.stable_reset_probability = 0.0
raw_env = BimanualBottleDirectEnv(cfg)
env = RslRlVecEnvWrapper(raw_env)
torch.manual_seed(args.seed)
print("EVAL_STAGE environment_ready", flush=True)

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
               "checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "held_out": args.held_out, "seed": args.seed,
               "source": args.source,
               "stable_eval_bank_path": str(args.stable_eval_bank) if args.stable_eval_bank else None,
               "thread_axial_stiffness_n_per_m": cfg.thread_axial_stiffness,
               "finger_increment_scale_rad": cfg.finger_increment_scale,
               "no_right_cap_contact_counterfactual": args.no_right_cap_contact,
               "counterfactual_right_contact_steps": counterfactual_right_contact_steps,
               "num_envs": args.num_envs, "episodes": total,
               "elapsed_s": elapsed, "maximum_opening_angle_rad": max_opening,
               "mean_abs_raw_action": sum_action_abs / action_samples,
               "raw_action_saturation_fraction": sum_action_saturation / action_samples,
               "mean_step_metrics": {name: value / action_samples
                                     for name, value in step_metric_sums.items()},
               "same_family_stable_evaluation": None if args.stable_eval_bank else True,
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
except BaseException as exc:
    print("EVAL_STAGE exception", type(exc).__name__, repr(exc), "steps", steps, flush=True)
    raise
finally:
    print("EVAL_STAGE finally", steps, total, flush=True)
    env.close()
    app.close()
