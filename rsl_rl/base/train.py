"""Train a registered DexterousHands task with RSL-RL."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from importlib import metadata
from pathlib import Path

from rsl_rl import cli_args


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train a DexterousHands task with RSL-RL.")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--stable_train_bank", type=Path,
                        help="Validated stable bank for a controlled bimanual training round")
    parser.add_argument("--stable_reset_probability", type=float)
    parser.add_argument("--finger_increment_scale", type=float,
                        help="Controlled bimanual finger target increment in rad/action")
    parser.add_argument("--source_bank_snapshot", type=Path, action="append",
                        help="Additional local source bank to hash and snapshot into ignored logs")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.task is None:
        raise SystemExit("--task is required")
    if (args.stable_train_bank is not None or
            args.stable_reset_probability is not None or
            args.source_bank_snapshot or args.finger_increment_scale is not None) and not args.task.startswith(
                "DexterousHands-WujiHand2-Bimanual-"):
        raise SystemExit("Stable-bank overrides are for the bimanual tasks only")
    if (args.stable_reset_probability is not None and
            not 0 <= args.stable_reset_probability <= 1):
        raise SystemExit("--stable_reset_probability must be in [0, 1]")
    if args.finger_increment_scale is not None and args.finger_increment_scale <= 0:
        raise SystemExit("--finger_increment_scale must be positive")
    if args.video:
        args.enable_cameras = True
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import dexteroushands_lab.tasks as dexteroushands_tasks
        from isaaclab.utils.dict import print_dict
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        from dexteroushands_lab.utils.policy_metadata import write_policy_artifacts

        dexteroushands_tasks.register_tasks()
        env_cfg = parse_env_cfg(
            args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
        )
        cli_args.apply_env_cfg_overrides(env_cfg, args)
        if args.stable_train_bank is not None:
            env_cfg.stable_train_bank_path = str(args.stable_train_bank)
        if args.stable_reset_probability is not None:
            env_cfg.stable_reset_probability = args.stable_reset_probability
        if args.finger_increment_scale is not None:
            env_cfg.finger_increment_scale = args.finger_increment_scale
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env_cfg.seed = agent_cfg.seed
        repo_root = Path(__file__).resolve().parents[2]
        log_root = repo_root / "logs" / "rsl_rl" / agent_cfg.experiment_name
        log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir = Path(f"{log_dir}_{agent_cfg.run_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            video_kwargs = {
                "video_folder": str(log_dir / "videos" / "train"),
                "step_trigger": lambda step: step % args.video_interval == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume:
            from isaaclab_tasks.utils import get_checkpoint_path

            resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
            runner.load(resume_path)
        env.seed(agent_cfg.seed)
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        write_policy_artifacts(log_dir, args.task, env_cfg=env_cfg)
        if args.task.startswith("DexterousHands-WujiHand2-Bimanual-"):
            from dexteroushands_lab.tasks.direct.bimanual_bottle.experiment_manifest import (
                write_bimanual_training_manifest,
            )

            asset_root = os.environ.get("WUJI_DESCRIPTION_ROOT")
            write_bimanual_training_manifest(
                log_dir, repo_root, args.task, Path(asset_root) if asset_root else None,
                extra_inputs=([args.stable_train_bank] if args.stable_train_bank else []) +
                             (args.source_bank_snapshot or []))
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
