"""Run a short reset/step smoke test for Allegro Hand tracking."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the Allegro tracking DirectTask.")
    parser.add_argument(
        "--task", default="DexterousHands-Allegro-Hand-Track-Direct-v0"
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    return parser


def _finite(value) -> bool:
    import torch

    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    return True


def assert_finite_step_tensors(observation, reward, terminated, truncated) -> None:
    """Raise a useful error if a reset/step result contains non-finite values."""

    if not _finite(observation):
        raise RuntimeError("Allegro observation contains non-finite values")
    if not _finite(reward):
        raise RuntimeError("Allegro reward contains non-finite values")
    if not _finite(terminated) or not _finite(truncated):
        raise RuntimeError("Allegro termination tensors contain non-finite values")


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch

        import dexteroushands_lab.tasks as dexteroushands_tasks
        from isaaclab_tasks.utils import parse_env_cfg

        dexteroushands_tasks.register_tasks()
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        env = gym.make(args.task, cfg=env_cfg)
        try:
            observation, _ = env.reset(seed=0)
            assert_finite_step_tensors(
                observation,
                torch.zeros(args.num_envs, device=env.unwrapped.device),
                torch.zeros(args.num_envs, dtype=torch.bool, device=env.unwrapped.device),
                torch.zeros(args.num_envs, dtype=torch.bool, device=env.unwrapped.device),
            )
            if "policy" not in observation or observation["policy"].shape[-1] != 105:
                raise RuntimeError(f"unexpected Allegro policy observation: {observation}")
            first_reference_index = int(env.unwrapped.reference_indices[0].item())
            for _ in range(args.steps):
                actions = torch.zeros(
                    (args.num_envs, env.action_space.shape[-1]),
                    device=env.unwrapped.device,
                )
                observation, reward, terminated, truncated, _ = env.step(actions)
                assert_finite_step_tensors(observation, reward, terminated, truncated)
            last_reference_index = int(env.unwrapped.reference_indices[0].item())
            print(
                "ALLEGRO_SMOKE "
                f"action_dim={env.action_space.shape[-1]} "
                f"observation_dim={observation['policy'].shape[-1]} "
                f"reference_start={first_reference_index} "
                f"reference_end={last_reference_index} "
                f"terminated={terminated.detach().cpu().tolist()} "
                f"truncated={truncated.detach().cpu().tolist()}",
                flush=True,
            )
        finally:
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
