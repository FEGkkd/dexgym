"""Bounded held-out evaluation for right-hand PPO, zero-action, and joint-replay control."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from dexgym.tasks import RIGHT_HAND_CLEAN_ID, RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.clean_env_cfg import RightHandCleanTrackingCfg
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from dexgym.tasks.ppo_cfg import RightHandCleanTrackingPPOCfg, RightHandTrackingPPOCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from tensordict import TensorDict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("policy", "zero", "joint_replay"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--clean", action="store_true", help="Use the independent 480 Hz clean tracking task")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "policy" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("--mode policy requires an existing --checkpoint")
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")

    load_external_tasks()
    cfg = RightHandCleanTrackingCfg() if args.clean else RightHandTrackingDirectCfg()
    task_id = RIGHT_HAND_CLEAN_ID if args.clean else RIGHT_HAND_ID
    cfg.scene.num_envs = args.num_envs
    cfg.split = args.split
    cfg.seed = args.seed
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    agent_cfg = RightHandCleanTrackingPPOCfg() if args.clean else RightHandTrackingPPOCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = "cuda:0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())

    with launch_simulation(cfg):
        raw_env = gym.make(task_id, cfg=cfg)
        try:
            task = raw_env.unwrapped
            policy = None
            if args.mode == "policy":
                wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
                runner = create_rsl_rl_runner(wrapped, agent_cfg)
                runner.load(str(args.checkpoint.resolve()))
                policy = runner.get_inference_policy(device=task.device)
            torch.manual_seed(args.seed)
            obs_dict, _ = raw_env.reset()
            obs = TensorDict(obs_dict, batch_size=[args.num_envs])
            reward_sum = 0.0
            tip_error_sum = 0.0
            action_sq_sum = 0.0
            reached_count = 0
            completed_count = 0
            terminated_count = 0
            truncated_count = 0
            episode_length_sum = 0
            seen_motions = torch.zeros(task.bank.motion_count, dtype=torch.bool, device=task.device)
            duration_bins = cfg.max_goal_steps + 1
            goal_duration_by_motion = torch.zeros(
                (task.bank.motion_count, duration_bins), dtype=torch.long, device=task.device
            )
            goal_timeout_count = 0
            with torch.inference_mode():
                for _ in range(args.steps):
                    seen_motions[task.motion_ids] = True
                    if args.mode == "policy":
                        action = policy(obs).clamp(-1.0, 1.0)
                    elif args.mode == "zero":
                        action = torch.zeros((args.num_envs, 20), device=task.device)
                    else:
                        goal_q = task.bank.joints(task.motion_ids, task.goal_frame)
                        action = ((goal_q - task.target_q) / cfg.action_delta_rad).clamp(-1.0, 1.0)
                    at_last_frame = task.goal_frame >= task.bank.lengths[task.motion_ids] - 1
                    motion_ids = task.motion_ids.clone()
                    goal_duration = task.goal_age.clone() + 1
                    episode_length = task.episode_length_buf.clone() + 1
                    obs_dict, reward, terminated, truncated, extras = raw_env.step(action)
                    done = terminated | truncated
                    # The +10 sparse bonus is separated from all dense rewards (<1).
                    reached = reward > 9.0
                    completed = reached & at_last_frame & terminated
                    if (goal_duration[reached] > cfg.max_goal_steps).any():
                        raise AssertionError("Reached subgoal exceeded the configured step limit")
                    flat_duration = motion_ids[reached] * duration_bins + goal_duration[reached]
                    goal_duration_by_motion += torch.bincount(
                        flat_duration, minlength=task.bank.motion_count * duration_bins
                    ).reshape(task.bank.motion_count, duration_bins)
                    age_limit_termination = terminated & ~completed & (goal_duration >= cfg.max_goal_steps)
                    goal_timeout_count += int(age_limit_termination.sum().item())
                    completed_count += int(completed.sum().item())
                    reached_count += int(reached.sum().item())
                    terminated_count += int(terminated.sum().item())
                    truncated_count += int(truncated.sum().item())
                    episode_length_sum += int(episode_length[done].sum().item())
                    reward_sum += float(reward.sum().item())
                    tip_error_sum += float(extras["log"]["mean_tip_error_m"].item())
                    action_sq_sum += float(action.square().mean().item())
                    if policy is not None:
                        policy.reset(done.long())
                    obs = TensorDict(obs_dict, batch_size=[args.num_envs])

            env_steps = args.num_envs * args.steps
            episodes = terminated_count + truncated_count
            duration_hist = goal_duration_by_motion.sum(dim=0).cpu()
            duration_count = int(duration_hist.sum().item())
            if duration_count != reached_count:
                raise AssertionError("Goal-duration histogram disagrees with the success count")

            def duration_summary(hist: torch.Tensor) -> dict:
                count = int(hist.sum().item())
                if count == 0:
                    return {"count": 0, "mean_steps": None, "p50_steps": None, "p90_steps": None, "p95_steps": None}
                bins = torch.arange(len(hist))
                cumulative = hist.cumsum(0)
                return {
                    "count": count,
                    "mean_steps": float((bins * hist).sum().item() / count),
                    "p50_steps": int(torch.searchsorted(cumulative, (count + 1) // 2).item()),
                    "p90_steps": int(torch.searchsorted(cumulative, (9 * count + 9) // 10).item()),
                    "p95_steps": int(torch.searchsorted(cumulative, (19 * count + 19) // 20).item()),
                }

            result = {
                "task_id": task_id,
                "mode": args.mode,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "split": args.split,
                "seed": args.seed,
                "goal_tolerance_m": cfg.goal_tolerance_m,
                "goal_dwell_steps": cfg.goal_dwell_steps,
                "subgoal_stride": cfg.subgoal_stride,
                "robot_usd": cfg.scene.robot.spawn.usd_path,
                "num_envs": args.num_envs,
                "steps_per_env": args.steps,
                "environment_steps": env_steps,
                "distinct_motions_seen": int(seen_motions.sum().item()),
                "mean_reward_per_step": reward_sum / env_steps,
                "mean_tip_error_m": tip_error_sum / args.steps,
                "action_rms": (action_sq_sum / args.steps) ** 0.5,
                "subgoals_reached": reached_count,
                "subgoals_per_1000_steps": 1000 * reached_count / env_steps,
                "goal_duration_steps": duration_summary(duration_hist),
                "goal_duration_histogram": duration_hist.tolist(),
                "goal_timeout_count": goal_timeout_count,
                "goal_duration_by_motion": [
                    {"motion": record["source"], **duration_summary(hist.cpu())}
                    for record, hist in zip(task.bank.records, goal_duration_by_motion, strict=True)
                ],
                "completed_motions": completed_count,
                "terminated_episodes": terminated_count,
                "truncated_episodes": truncated_count,
                "motion_completion_rate": completed_count / episodes if episodes else 0.0,
                "mean_ended_episode_length": episode_length_sum / episodes if episodes else None,
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
