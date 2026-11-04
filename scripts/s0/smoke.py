"""Finite Newton smoke of the 54-joint S0 robot and 14-action task."""

import argparse
import json
from pathlib import Path

import dexgym.tasks  # noqa: F401
import gymnasium as gym
import torch
from dexgym.tasks.marvin_s0.env_cfg import MarvinBimanualS0Cfg
from isaaclab.app import launch_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--bimanual-fraction", type=float, default=0.33)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("num-envs and steps must be positive")
    cfg = MarvinBimanualS0Cfg()
    cfg.scene.num_envs = args.num_envs
    cfg.bimanual_fraction = args.bimanual_fraction
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-Bimanual-SE3-Direct-v0", cfg=cfg)
        try:
            observation, _ = env.reset(seed=112358)
            raw = env.unwrapped
            modes_seen = torch.zeros(3, dtype=torch.bool, device=raw.device)
            timeouts = 0
            terminations = 0
            for step in range(args.steps):
                if observation["policy"].shape != (args.num_envs, 73):
                    raise ValueError("Wrong observation shape")
                if not torch.isfinite(observation["policy"]).all():
                    raise ValueError(f"Nonfinite observation at step {step}")
                action = torch.zeros((args.num_envs, 14), device=raw.device)
                observation, reward, terminated, truncated, _ = env.step(action)
                if not torch.isfinite(reward).all() or not torch.isfinite(raw.robot.data.joint_pos.torch).all():
                    raise ValueError(f"Nonfinite reward or joint state at step {step}")
                modes_seen |= raw.mode_one_hot.bool().any(dim=0)
                timeouts += int(truncated.sum())
                terminations += int(terminated.sum())
            if raw.robot.num_joints != 54 or timeouts < 1 or not modes_seen.all():
                raise ValueError("54-DoF, reset or three-mode smoke condition failed")
            result = {
                "task_id": "DexGym-Marvin-Bimanual-SE3-Direct-v0",
                "num_envs": args.num_envs,
                "steps": args.steps,
                "robot_joint_count": raw.robot.num_joints,
                "observation_size": 73,
                "action_size": 14,
                "modes_seen": modes_seen.detach().cpu().tolist(),
                "timeouts": timeouts,
                "terminations": terminations,
                "finite": True,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
