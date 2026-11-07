"""Short real Newton parity check for the moving dual-TCP SE(3) task."""

import argparse
import json
from pathlib import Path

import dexgym.tasks.marvin_supermimic  # noqa: F401
import gymnasium as gym
import torch
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector
from dexgym.tasks.marvin_supermimic.se3_trajectory_env import MarvinContinuousSE3FollowCfg
from isaaclab.app import launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 64 or not 1 <= args.steps <= 200:
        parser.error("Newton preflight is bounded to 1-64 environments and 1-200 steps")
    cfg = MarvinContinuousSE3FollowCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.evaluation_mode = False
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Continuous-SE3-Follow-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            collector = CoreNewtonCollector(env, native.core_robot, native.follow_task)
            failures = truncations = 0
            command_error = 0.0
            rewards = []
            motion_counts = torch.zeros(len(native.reference.names), dtype=torch.long)
            hand_command_error = 0.0
            hand_position_error = 0.0

            def zero_actor(history, goal):
                assert goal.shape == (len(history), 10, 18)
                return torch.zeros((len(history), 14), device=history.device)

            for _ in range(args.steps):
                batch = collector.step(zero_actor)
                failures += int(batch.terminated.sum())
                truncations += int(batch.truncated.sum())
                command_error = max(command_error, float((batch.executed_command
                    - batch.next_state[:, 28:42]).abs().max()))
                motion_counts += torch.bincount(batch.task_state.motion_id.cpu(),
                                                minlength=len(motion_counts))
                rewards.append(float(batch.reward.mean()))
                hand_command_error = max(hand_command_error, float((
                    native.full_command[:, native.hand_ids]
                    - native.home_full[native.hand_ids]).abs().max()))
                hand_position_error = max(hand_position_error, float((
                    native.robot.data.joint_pos.torch[:, native.hand_ids]
                    - native.home_full[native.hand_ids]).abs().max()))
            report = {"task_id": native.follow_task.spec.task_id,
                      "reference_names": native.reference.names,
                      "motion_transition_counts": motion_counts.tolist(),
                      "action_schema": native.core_robot.spec.action_schema,
                      "state_shape": list(batch.state.shape),
                      "goal_shape": [len(batch.state), 10, 18],
                      "physics_transitions": args.steps * args.num_envs,
                      "hard_terminations": failures, "truncations": truncations,
                      "command_max_abs_error_rad": command_error,
                      "hand_home_command_max_abs_error_rad": hand_command_error,
                      "hand_physical_position_max_abs_error_rad": hand_position_error,
                      "reward_mean": sum(rewards) / len(rewards)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
