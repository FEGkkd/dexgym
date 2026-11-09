"""Record Newton physical states from deterministic Hand 2.1 PPO play on one GRAB clip."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from dexgym.tasks import HAND21_TELEDEXTER_ID
from dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg import WujiHand21TeleDexterCfg
from dexgym.tasks.ppo_cfg import WujiHand21TeleDexterPPOCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from tensordict import TensorDict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", default="s8/hammer_lift.npz")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.steps < 1:
        parser.error("Need an existing checkpoint and positive step count")
    torch.manual_seed(args.seed)
    load_external_tasks()
    cfg = WujiHand21TeleDexterCfg()
    cfg.scene.num_envs = 1
    cfg.split = args.split
    cfg.seed = args.seed
    cfg.cross_trajectory_probability = 0.0  # keep the video on the selected clip
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = gym.make(HAND21_TELEDEXTER_ID, cfg=cfg)
        try:
            task = env.unwrapped
            candidates = [i for i, row in enumerate(task.bank.records)
                          if row.get("origin_source", row["source"]) == args.source]
            if len(candidates) != 1:
                raise ValueError(f"Expected one eligible segment for {args.source}; got {len(candidates)}")
            motion_id = candidates[0]
            acfg = handle_deprecated_rsl_rl_cfg(WujiHand21TeleDexterPPOCfg(), check_rsl_rl_version())
            acfg.device = "cuda:0"
            runner = create_rsl_rl_runner(RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions), acfg)
            runner.load(str(args.checkpoint.resolve()))
            policy = runner.get_inference_policy(device=task.device)
            task.curriculum_steps[:] = cfg.curriculum_anneal_steps
            sampler = task.bank.sample_reference_starts
            used = False

            def selected_start(count, *, first_fraction=0.9):
                nonlocal used
                if used:
                    return sampler(count, first_fraction=first_fraction)
                used = True
                return (torch.tensor([motion_id], device=task.device),
                        torch.zeros(1, dtype=torch.long, device=task.device))

            torch.manual_seed(args.seed)
            task.bank.sample_reference_starts = selected_start
            obs_dict, _ = env.reset()
            task.bank.sample_reference_starts = sampler
            obs = TensorDict(obs_dict, batch_size=[1])
            original_get_dones = task._get_dones
            snapshot = {}

            def observed_get_dones():
                old_frame = task.goal_frame.clone()
                old_span = task.goal_span.clone()
                old_motion = task.motion_ids.clone()
                old_goal_q = task.bank.joints(old_motion, old_frame).clone()
                actual_q = task.robot.data.joint_pos.torch[:, task.right_ids].clone()
                tip_error = (task._current_tips() - task.goal_tips).norm(dim=-1)
                terminated, truncated = original_get_dones()
                snapshot.update(q_actual=actual_q, q_goal=old_goal_q, goal_frame=old_frame,
                                goal_span=old_span, tip_error=tip_error,
                                reached=task.last_reached.clone(), crossed=task.last_crossed.clone(),
                                terminated=terminated.clone(), truncated=truncated.clone())
                return terminated, truncated

            task._get_dones = observed_get_dones
            frames = {key: [] for key in ("q_actual", "q_goal", "goal_frame", "goal_span",
                       "tip_error_mm", "reached", "crossed")}
            stop_reason = "step_limit"
            with torch.inference_mode():
                for _ in range(args.steps):
                    action = policy(obs, stochastic_output=False)
                    obs_dict, _, terminated, truncated, _ = env.step(action)
                    frames["q_actual"].append(snapshot["q_actual"][0].cpu().numpy())
                    frames["q_goal"].append(snapshot["q_goal"][0].cpu().numpy())
                    frames["goal_frame"].append(int(snapshot["goal_frame"][0].item()))
                    frames["goal_span"].append(int(snapshot["goal_span"][0].item()))
                    frames["tip_error_mm"].append(snapshot["tip_error"][0].cpu().numpy() * 1000)
                    frames["reached"].append(bool(snapshot["reached"][0].item()))
                    frames["crossed"].append(bool(snapshot["crossed"][0].item()))
                    if snapshot["crossed"][0]:
                        stop_reason = "source_end_cross_switch"
                        break
                    if terminated[0] or truncated[0]:
                        stop_reason = "episode_ended"
                        break
                    obs = TensorDict(obs_dict, batch_size=[1])
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **{key: np.asarray(value) for key, value in frames.items()},
                                source=np.asarray(args.source), fps=np.float32(30))
            errors = np.asarray(frames["tip_error_mm"])
            summary = dict(source=args.source, split=args.split, checkpoint=str(args.checkpoint.resolve()),
                           frames=len(errors), duration_s=len(errors) / 30, stop_reason=stop_reason,
                           last_goal_frame=frames["goal_frame"][-1],
                           goals_reached=sum(frames["reached"]),
                           mean_max_tip_error_mm=float(errors.max(axis=1).mean()),
                           p90_max_tip_error_mm=float(np.percentile(errors.max(axis=1), 90)),
                           frames_all_tips_within_15mm=float((errors.max(axis=1) < 15).mean()))
            output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"POLICY_PLAY_DONE {output} {summary}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
