"""Common RSL-RL command-line arguments for project entrypoints."""

from __future__ import annotations

import argparse
import random


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    group.add_argument("--experiment_name", type=str, default=None)
    group.add_argument("--run_name", type=str, default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument("--load_run", type=str, default=None)
    group.add_argument("--checkpoint", type=str, default=None)
    group.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    group.add_argument("--log_project_name", type=str, default=None)


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if getattr(args_cli, "seed", None) is not None:
        cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    if getattr(args_cli, "resume", False):
        cfg.resume = True
    if getattr(args_cli, "load_run", None) is not None:
        cfg.load_run = args_cli.load_run
    if getattr(args_cli, "checkpoint", None) is not None:
        cfg.load_checkpoint = args_cli.checkpoint
    if getattr(args_cli, "experiment_name", None) is not None:
        cfg.experiment_name = args_cli.experiment_name
    if getattr(args_cli, "run_name", None) is not None:
        cfg.run_name = args_cli.run_name
    if getattr(args_cli, "logger", None) is not None:
        cfg.logger = args_cli.logger
    if cfg.logger in {"wandb", "neptune"} and getattr(args_cli, "log_project_name", None):
        cfg.wandb_project = args_cli.log_project_name
        cfg.neptune_project = args_cli.log_project_name
    return cfg
