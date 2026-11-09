"""Compare saved Hand 2.1 PPO policies with privileged q_ref replay under live task rules."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from dexgym.tasks import HAND21_TELEDEXTER_ID
from dexgym.tasks.direct.right_hand_tracking.logic import action_toward_joint_target
from dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg import WujiHand21TeleDexterCfg
from dexgym.tasks.ppo_cfg import WujiHand21TeleDexterPPOCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from tensordict import TensorDict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("privileged", "mean", "sample"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--split", choices=("train", "validation"), default="validation")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.mode != "privileged" and (args.checkpoint is None or not args.checkpoint.is_file()):
        p.error("Policy modes require a checkpoint")
    torch.manual_seed(args.seed)
    load_external_tasks()
    cfg = WujiHand21TeleDexterCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.split = args.split
    cfg.seed = args.seed
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = gym.make(HAND21_TELEDEXTER_ID, cfg=cfg)
        try:
            task = env.unwrapped
            if args.num_envs < task.bank.motion_count:
                raise ValueError("Need at least one environment per motion")
            policy = None
            if args.mode != "privileged":
                acfg = WujiHand21TeleDexterPPOCfg()
                acfg.seed = args.seed
                acfg.device = "cuda:0"
                acfg = handle_deprecated_rsl_rl_cfg(acfg, check_rsl_rl_version())
                wrapped = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
                runner = create_rsl_rl_runner(wrapped, acfg)
                runner.load(str(args.checkpoint.resolve()))
                policy = runner.get_inference_policy(device=task.device)
            task.curriculum_steps[:] = cfg.curriculum_anneal_steps
            sampler = task.bank.sample_reference_starts
            used = False

            def balanced(count, *, first_fraction=0.9):
                nonlocal used
                if used:
                    return sampler(count, first_fraction=first_fraction)
                used = True
                ids = torch.arange(count, device=task.device) % task.bank.motion_count
                choices = torch.minimum(
                    torch.ceil(task.bank.lengths[ids] * first_fraction).long(), task.bank.lengths[ids] - 1
                )
                frames = torch.floor(torch.rand(count, device=task.device) * choices).long()
                return ids, frames

            torch.manual_seed(args.seed)
            task.bank.sample_reference_starts = balanced
            obs_dict, _ = env.reset()
            task.bank.sample_reference_starts = sampler
            obs = TensorDict(obs_dict, batch_size=[args.num_envs])
            original_get_dones = task._get_dones
            decision = {}

            def audited_get_dones():
                old_span = task.goal_span.clone()
                old_budget = task.fail_budget.clone()
                errors = (task._current_tips() - task.goal_tips).norm(dim=-1)
                terminated, truncated = original_get_dones()
                decision.update(span=old_span, budget=old_budget, errors=errors,
                                reached=task.last_reached.clone(), fail_frames=task.fail_frames.clone(),
                                terminated=terminated.clone(), truncated=truncated.clone())
                return terminated, truncated

            task._get_dones = audited_get_dones
            span_bins = cfg.curriculum_stride_max + 1
            counts = torch.zeros((span_bins, 4), dtype=torch.long, device=task.device)
            reward_sum = torch.zeros((), device=task.device)
            mean_error_sum = torch.zeros((), device=task.device)
            max_error_sum = torch.zeros((), device=task.device)
            episode_length_sum = torch.zeros((), device=task.device)
            episode_count = torch.zeros((), device=task.device)
            action_abs_sum = torch.zeros((), device=task.device)
            action_saturated_count = torch.zeros((), device=task.device)
            with torch.inference_mode():
                for step in range(args.steps):
                    if policy is None:
                        desired = task.bank.joints(task.motion_ids, task.goal_frame)
                        action = action_toward_joint_target(desired, task.target_q,
                            scale=cfg.action_delta_rad, deadzone=cfg.action_deadzone)
                    else:
                        action = policy(obs, stochastic_output=args.mode == "sample")
                    action_abs_sum += action.abs().mean()
                    action_saturated_count += (action.abs() >= 0.95).any(dim=-1).sum()
                    old_episode_length = task.episode_length_buf.clone() + 1
                    obs_dict, reward, terminated, truncated, _ = env.step(action)
                    reached = decision["reached"]
                    timeout = decision["terminated"] & ~reached & (decision["fail_frames"] > decision["budget"])
                    other = decision["terminated"] & ~reached & ~timeout
                    cutoff = decision["truncated"] & ~reached & ~decision["terminated"]
                    for i, mask in enumerate((reached, timeout, other, cutoff)):
                        counts[:, i] += torch.bincount(decision["span"][mask], minlength=span_bins)
                    reward_sum += reward.sum()
                    mean_error_sum += decision["errors"].mean()
                    max_error_sum += decision["errors"].max(dim=-1).values.mean()
                    done = terminated | truncated
                    episode_length_sum += old_episode_length[done].sum()
                    episode_count += done.sum()
                    if policy is not None:
                        policy.reset(done.long())
                    obs = TensorDict(obs_dict, batch_size=[args.num_envs])
                    if (step + 1) % 100 == 0:
                        print(f"EVAL_PROGRESS mode={args.mode} step={step+1}/{args.steps}", flush=True)
            cpu = counts.cpu().tolist()
            total = [sum(row[i] for row in cpu) for i in range(4)]
            result = dict(mode=args.mode, checkpoint=str(args.checkpoint) if args.checkpoint else None,
                split=args.split, seed=args.seed, num_envs=args.num_envs, steps=args.steps,
                goal_tolerance_mm=cfg.goal_tolerance_m*1000,
                curriculum_upper=cfg.curriculum_stride_max,
                totals=dict(zip(("reached", "budget_timeout", "other_termination", "episode_truncation"), total)),
                frame_hit_fraction=total[0]/(args.num_envs*args.steps),
                by_span={
                    str(i): dict(zip(("reached", "budget_timeout", "other_termination", "episode_truncation"), row))
                    for i, row in enumerate(cpu)
                },
                mean_reward_per_step=float(reward_sum.item()/(args.num_envs*args.steps)),
                mean_tip_error_mm=float(mean_error_sum.item()*1000/args.steps),
                mean_max_tip_error_mm=float(max_error_sum.item()*1000/args.steps),
                mean_finished_episode_length=float(episode_length_sum.item()/max(1,episode_count.item())),
                finished_episodes=int(episode_count.item()),
                mean_abs_action=float(action_abs_sum.item()/args.steps),
                action_saturated_step_fraction=float(action_saturated_count.item()/(args.num_envs*args.steps)))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+"\n")
            print("EVAL_DONE", args.output, result["totals"], flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
