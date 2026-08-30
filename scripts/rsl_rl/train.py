"""Train DexterousHands tasks with the Isaac Lab RSL-RL interface."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Train a DexterousHands task with RSL-RL.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import dexteroushands_lab.tasks as dexteroushands_tasks  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from dexteroushands_lab.utils.policy_metadata import write_policy_artifacts  # noqa: E402

dexteroushands_tasks.register_tasks()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed

    log_root = REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir = Path(f"{log_dir}_{agent_cfg.run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(log_dir / "videos" / "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    env.seed(agent_cfg.seed)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    write_policy_artifacts(log_dir, args_cli.task, env_cfg=env_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    checkpoints = sorted(log_dir.glob("model_*.pt"))
    if checkpoints:
        write_policy_artifacts(
            log_dir,
            args_cli.task,
            env_cfg=env_cfg,
            checkpoint=checkpoints[-1].name,
        )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
