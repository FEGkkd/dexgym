"""Run an actual multi-environment Newton reset/step/timeout smoke test."""

import argparse

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--policy", help="Exported TorchScript policy.pt for finite-step replay")
    args = parser.parse_args()
    load_external_tasks()
    spec = gym.spec(args.task)
    module, name = spec.kwargs["env_cfg_entry_point"].split(":")
    from importlib import import_module

    cfg = getattr(import_module(module), name)()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    policy = torch.jit.load(args.policy, map_location=cfg.sim.device).eval() if args.policy else None
    with launch_simulation(cfg):
        env = gym.make(args.task, cfg=cfg)
        try:
            obs, _ = env.reset()
            assert obs["policy"].shape == (args.num_envs, 4)
            assert env.action_space.shape[-1] == 1
            initial_goals = obs["policy"][:, 2].clone()
            assert initial_goals.std() > 0
            timeouts = 0
            for _ in range(args.steps):
                with torch.no_grad():
                    action = (
                        policy(obs["policy"])
                        if policy is not None
                        else torch.zeros((args.num_envs, 1), device=env.unwrapped.device)
                    )
                assert action.shape == (args.num_envs, 1)
                assert torch.isfinite(action).all()
                obs, reward, _terminated, truncated, _ = env.step(action)
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(reward).all()
                robot = env.unwrapped.scene["robot"]
                assert torch.isfinite(robot.data.joint_pos.torch).all()
                assert torch.isfinite(robot.data.joint_vel.torch).all()
                assert obs["policy"].shape == (args.num_envs, 4)
                timeouts += int(truncated.sum())
            print(
                f"task={args.task} steps={args.steps} envs={args.num_envs} "
                f"observation={tuple(obs['policy'].shape)} action_space={env.action_space.shape} "
                f"timeouts={timeouts} policy={args.policy or 'zero'}"
            )
            assert timeouts > 0
            assert torch.any(obs["policy"][:, 2] != initial_goals)
        finally:
            env.close()


if __name__ == "__main__":
    main()
