"""Bounded Newton contract preflight for IK-free dual-arm SE(3) reach and hold."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # noqa: F401
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector
from dexgym.tasks.marvin_supermimic.se3_env import MarvinSE3ReachHoldCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--level", choices=("near", "medium", "wider_feasible"), default="near")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 64 or not 1 <= args.steps <= 200:
        parser.error("Preflight is bounded to at most 64 environments and 200 steps")
    cfg = MarvinSE3ReachHoldCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.goal_train_seed = args.seed + 101
    cfg.goal_eval_seed = args.seed + 100101
    cfg.curriculum_level = args.level
    cfg.audit_contacts_each_physics_step = True
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            collector = CoreNewtonCollector(env, native.core_robot, native.se3_task)
            transition_count = failures = truncations = 0
            command_error = 0.0
            rewards = []

            def actor(history, goal):
                return torch.zeros((len(history), 14), device=history.device)

            for _ in range(args.steps):
                batch = collector.step(actor)
                transition_count += len(batch.state)
                failures += int(batch.terminated.sum())
                truncations += int(batch.truncated.sum())
                command_error = max(command_error, float((batch.executed_command
                    - batch.next_state[:, 28:42]).abs().max()))
                rewards.append(float(batch.reward.mean()))
            report = {"task_id": native.se3_task.spec.task_id,
                      "action_schema": native.core_robot.spec.action_schema,
                      "goal_shape": list(batch.task_state.data.shape),
                      "state_shape": list(batch.state.shape),
                      "random_start_std_rad": native.command.std(0).tolist(),
                      "physics_transitions": transition_count,
                      "hard_terminations": failures, "truncations": truncations,
                      "command_max_abs_error_rad": command_error,
                      "reward_mean": sum(rewards) / len(rewards),
                      "position_difficulty_mean_m":
                          native.reset_position_difficulty_m.mean(0).tolist(),
                      "rotation_difficulty_mean_rad":
                          native.reset_rotation_difficulty_rad.mean(0).tolist()}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
