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
    group.add_argument("--learning_rate", type=float, default=None)
    group.add_argument("--entropy_coef", type=float, default=None)
    group.add_argument("--value_loss_coef", type=float, default=None)
    group.add_argument("--clip_param", type=float, default=None)
    group.add_argument("--num_learning_epochs", type=int, default=None)
    group.add_argument("--num_mini_batches", type=int, default=None)
    group.add_argument("--gamma", type=float, default=None)
    group.add_argument("--lam", type=float, default=None)
    group.add_argument("--desired_kl", type=float, default=None)
    group.add_argument("--max_grad_norm", type=float, default=None)
    group.add_argument("--num_steps_per_env", type=int, default=None)
    group.add_argument("--save_interval", type=int, default=None)
    group.add_argument("--actor_hidden_dims", type=int, nargs="+", default=None)
    group.add_argument("--critic_hidden_dims", type=int, nargs="+", default=None)
    group.add_argument("--disable_curriculum", action="store_true", default=False)
    group.add_argument("--disable_domain_randomization", action="store_true", default=False)
    group.add_argument("--disable_disturbance", action="store_true", default=False)


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
    algorithm = getattr(cfg, "algorithm", None)
    for name in (
        "learning_rate", "entropy_coef", "value_loss_coef", "clip_param",
        "num_learning_epochs", "num_mini_batches", "gamma", "lam",
        "desired_kl", "max_grad_norm",
    ):
        value = getattr(args_cli, name, None)
        if value is not None and algorithm is not None:
            setattr(algorithm, name, value)
    for name in ("num_steps_per_env", "save_interval"):
        value = getattr(args_cli, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args_cli, "actor_hidden_dims", None) is not None:
        cfg.actor.hidden_dims = list(args_cli.actor_hidden_dims)
    if getattr(args_cli, "critic_hidden_dims", None) is not None:
        cfg.critic.hidden_dims = list(args_cli.critic_hidden_dims)
    return cfg


def apply_env_cfg_overrides(env_cfg, args_cli: argparse.Namespace):
    """Apply feature gates from CLI while retaining nested parameter values."""
    gates = (
        ("disable_curriculum", "enable_curriculum", "curriculum"),
        ("disable_domain_randomization", "enable_domain_randomization", "domain_randomization"),
        ("disable_disturbance", "enable_disturbance", "disturbance"),
    )
    for flag, top_level, nested in gates:
        if getattr(args_cli, flag, False):
            if hasattr(env_cfg, top_level):
                setattr(env_cfg, top_level, False)
            group = getattr(env_cfg, nested, None)
            if group is not None and hasattr(group, "enabled"):
                group.enabled = False
    return env_cfg
