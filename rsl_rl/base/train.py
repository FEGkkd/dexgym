"""Train a registered DexterousHands task with RSL-RL."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from importlib import metadata
from pathlib import Path

from rsl_rl import cli_args


def reset_loaded_gaussian_std(runner, std: float) -> None:
    """Start a new task at std=1 after transferring learned policy weights.

    RSL-RL loads the source checkpoint's distribution parameter and Adam
    moments. Reset only that parameter and its optimizer moments; the actor
    mean, critic and observation normalizers remain transferred.
    """
    import torch

    distribution = runner.alg.actor.distribution
    if (distribution is None or getattr(distribution, "std_type", None) != "scalar"
            or not hasattr(distribution, "std_param") or not 0.01 <= std <= 2.0):
        raise ValueError("Std transfer requires scalar Gaussian distribution and std in [0.01, 2.0]")
    with torch.no_grad():
        distribution.std_param.fill_(std)
    runner.alg.optimizer.state.pop(distribution.std_param, None)


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train a DexterousHands task with RSL-RL.")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--hand_variant", choices=("beta1", "beta2"),
                        help="Bimanual WujiHand2 asset; beta1 replays historical checkpoints")
    parser.add_argument("--fixed_pregrasp_seed", type=int,
                        help="Beta 2 unified-task diagnostic: use one frozen training pregrasp row")
    parser.add_argument("--stable_train_bank", type=Path,
                        help="Validated stable bank for a controlled bimanual training round")
    parser.add_argument("--stable_reset_probability", type=float)
    parser.add_argument("--thread_backend", choices=("legacy_d6", "mimic"),
                        help="Bimanual bottle thread solver")
    parser.add_argument("--solver_type", choices=("tgs", "pgs"),
                        help="PhysX solver; validated Mimic stable bank currently requires PGS")
    parser.add_argument("--right_cap_contact_gate",
                        choices=("site_sidewall_proxy", "body_only", "manifold_sidewall"),
                        help="Bimanual right-cap reward gate; body_only reproduces older rounds")
    parser.add_argument("--right_contact_balance_weight", type=float,
                        help="Weight on all-three geometric contact term; default 0.8")
    parser.add_argument("--right_cap_proximity_mode",
                        choices=("full_five", "sidewall_tripod"),
                        help="Controlled cap approach reward topology; full_five reproduces older rounds")
    parser.add_argument("--finger_increment_scale", type=float,
                        help="Controlled bimanual finger target increment in rad/action")
    parser.add_argument("--wrist_translation_scale", type=float,
                        help="Controlled bimanual wrist translation increment in m/action")
    parser.add_argument("--init_action_std", type=float,
                        help="Initial normalized Gaussian action std for a controlled PPO ablation")
    parser.add_argument("--action_distribution", choices=("gaussian", "tanh"),
                        help="Controlled Phase 1B actor distribution; tanh uses exact transformed PPO density")
    parser.add_argument("--unified_reward_profile", choices=("baseline", "contact_coverage"),
                        help="Unified Phase 1B reward profile; same task and actor")
    parser.add_argument("--reset_loaded_action_std", action="store_true",
                        help="On task transfer, restore the loaded Gaussian std to --init_action_std")
    parser.add_argument("--init_checkpoint", type=Path,
                        help="Initialize one actor/critic from an explicit checkpoint for a controlled task transfer")
    parser.add_argument("--solve_articulation_contact_last", action="store_true",
                        help="PhysX contact-last articulation solve order for a controlled experiment")
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
    if (args.hand_variant is not None or args.fixed_pregrasp_seed is not None or
            args.stable_train_bank is not None or
            args.stable_reset_probability is not None or
            args.source_bank_snapshot or args.finger_increment_scale is not None or
            args.wrist_translation_scale is not None or
            args.init_action_std is not None or
            args.reset_loaded_action_std or
            args.thread_backend is not None or
            args.solver_type is not None or
            args.right_cap_contact_gate is not None or
            args.right_contact_balance_weight is not None or
            args.right_cap_proximity_mode is not None or
            args.solve_articulation_contact_last) and not args.task.startswith(
                "DexterousHands-WujiHand2-Bimanual-"):
        raise SystemExit("Bimanual overrides are for the bimanual tasks only")
    if (args.stable_reset_probability is not None and
            not 0 <= args.stable_reset_probability <= 1):
        raise SystemExit("--stable_reset_probability must be in [0, 1]")
    if args.finger_increment_scale is not None and args.finger_increment_scale <= 0:
        raise SystemExit("--finger_increment_scale must be positive")
    if args.wrist_translation_scale is not None and args.wrist_translation_scale <= 0:
        raise SystemExit("--wrist_translation_scale must be positive")
    if args.init_action_std is not None and not 0.01 <= args.init_action_std <= 2.0:
        raise SystemExit("--init_action_std must be in [0.01, 2.0]")
    if args.action_distribution is not None and args.task != (
            "DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0"):
        raise SystemExit("--action_distribution is for the unified Phase 1B task only")
    if args.fixed_pregrasp_seed is not None and args.task != (
            "DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0"):
        raise SystemExit("--fixed_pregrasp_seed is for the unified Phase 1B task only")
    if args.init_checkpoint is not None and (args.resume or not args.init_checkpoint.is_file()):
        raise SystemExit("--init_checkpoint must exist and cannot be combined with --resume")
    if args.reset_loaded_action_std and (
            not (args.resume or args.init_checkpoint) or args.init_action_std is None):
        raise SystemExit("--reset_loaded_action_std requires checkpoint loading and --init_action_std")
    if args.right_contact_balance_weight is not None and not 0 <= args.right_contact_balance_weight <= 1:
        raise SystemExit("--right_contact_balance_weight must be in [0, 1]")
    if args.thread_backend == "mimic" and args.stable_reset_probability != 0 and (
            args.solver_type != "pgs" or args.stable_train_bank is None):
        raise SystemExit("Mimic stable resets require --solver_type pgs and a validated --stable_train_bank")
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
        if args.hand_variant is not None:
            from dexteroushands_lab.tasks.direct.bimanual_bottle.variant_config import (
                configure_bimanual_hand_variant,
            )
            configure_bimanual_hand_variant(env_cfg, args.hand_variant)
        if args.fixed_pregrasp_seed is not None:
            env_cfg.fixed_pregrasp_seed = args.fixed_pregrasp_seed
        if args.stable_train_bank is not None:
            env_cfg.stable_train_bank_path = str(args.stable_train_bank)
        if args.stable_reset_probability is not None:
            env_cfg.stable_reset_probability = args.stable_reset_probability
        if args.thread_backend is not None:
            env_cfg.thread_backend = args.thread_backend
        if args.solver_type is not None:
            env_cfg.sim.physics.solver_type = 1 if args.solver_type == "tgs" else 0
        if args.right_cap_contact_gate is not None:
            env_cfg.right_cap_contact_gate = args.right_cap_contact_gate
        if args.right_contact_balance_weight is not None:
            env_cfg.right_contact_balance_weight = args.right_contact_balance_weight
        if args.right_cap_proximity_mode is not None:
            env_cfg.right_cap_proximity_mode = args.right_cap_proximity_mode
        if args.finger_increment_scale is not None:
            env_cfg.finger_increment_scale = args.finger_increment_scale
        if args.wrist_translation_scale is not None:
            env_cfg.wrist_translation_scale = args.wrist_translation_scale
        if args.solve_articulation_contact_last:
            env_cfg.sim.physics.solve_articulation_contact_last = True
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
        if args.init_action_std is not None:
            agent_cfg.actor.distribution_cfg.init_std = args.init_action_std
        if args.action_distribution == "tanh":
            agent_cfg.actor.distribution_cfg.class_name = (
                "dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.agents."
                "tanh_distribution:TanhGaussianDistribution")
        if args.unified_reward_profile is not None:
            if not hasattr(env_cfg, "unified_reward_profile"):
                raise SystemExit("--unified_reward_profile requires the Phase 1B task")
            env_cfg.unified_reward_profile = args.unified_reward_profile
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
        if agent_cfg.resume or args.init_checkpoint is not None:
            if args.init_checkpoint is None:
                from isaaclab_tasks.utils import get_checkpoint_path
                resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
            else:
                resume_path = args.init_checkpoint.resolve()
            runner.load(str(resume_path))
            if args.reset_loaded_action_std:
                loaded_std = float(runner.alg.actor.distribution.std_param.detach().mean())
                reset_loaded_gaussian_std(runner, args.init_action_std)
                (log_dir / "phase1_std_transfer.json").write_text(json.dumps({
                    "source_checkpoint": str(resume_path),
                    "source_mean_std": loaded_std,
                    "reset_mean_std": args.init_action_std,
                    "std_type": "scalar",
                    "optimizer_std_moments_cleared": True,
                }, indent=2) + "\n")
                print("PHASE1_TRANSFER_ACTION_STD", loaded_std,
                      args.init_action_std, flush=True)
            if args.init_checkpoint is not None:
                (log_dir / "phase1b_initial_checkpoint.json").write_text(json.dumps({
                    "source_checkpoint": str(resume_path),
                    "single_actor_transfer": True,
                    "runtime_policy_switching": False,
                    "source_iteration": runner.current_learning_iteration,
                }, indent=2) + "\n")
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
        print("TRAIN_STAGE learn_enter", agent_cfg.max_iterations, env.num_envs, flush=True)
        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        except BaseException as exc:
            print("TRAIN_STAGE learn_exception", type(exc).__name__, repr(exc), flush=True)
            raise
        print("TRAIN_STAGE learn_return", runner.current_learning_iteration, flush=True)
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
