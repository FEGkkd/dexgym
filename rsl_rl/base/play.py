"""Play and export a registered DexterousHands RSL-RL policy."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path

from rsl_rl import cli_args


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play a DexterousHands RSL-RL policy.")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--thread_backend", choices=("legacy_d6", "mimic"),
                        help="Bimanual bottle thread solver used by the checkpoint")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _disable_training_enhancements(env_cfg) -> None:
    """Make evaluation deterministic while retaining all task parameters."""
    if hasattr(env_cfg, "enable_curriculum"):
        env_cfg.enable_curriculum = False
    if hasattr(env_cfg, "enable_domain_randomization"):
        env_cfg.enable_domain_randomization = False
    if hasattr(env_cfg, "enable_disturbance"):
        env_cfg.enable_disturbance = False
    for name in ("curriculum", "domain_randomization", "disturbance"):
        group = getattr(env_cfg, name, None)
        if group is not None and hasattr(group, "enabled"):
            group.enabled = False
    if hasattr(env_cfg, "observation_noise_enabled"):
        env_cfg.observation_noise_enabled = False


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.task is None:
        raise SystemExit("--task is required")
    if args.video:
        args.enable_cameras = True
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import dexteroushands_lab.tasks as dexteroushands_tasks
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from dexteroushands_lab.utils.policy_metadata import write_policy_artifacts

        dexteroushands_tasks.register_tasks()
        env_cfg = parse_env_cfg(
            args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
        )
        if args.thread_backend is not None:
            if not args.task.startswith("DexterousHands-WujiHand2-Bimanual-"):
                raise SystemExit("--thread_backend applies only to bimanual bottle tasks")
            env_cfg.thread_backend = args.thread_backend
            if args.thread_backend == "mimic":
                env_cfg.stable_reset_probability = 0.0
        _disable_training_enhancements(env_cfg)
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env_cfg.seed = agent_cfg.seed
        repo_root = Path(__file__).resolve().parents[2]
        log_root = repo_root / "logs" / "rsl_rl" / agent_cfg.experiment_name
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = Path(resume_path).parent
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        export_dir = log_dir / "exported"
        runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
        runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")
        write_policy_artifacts(
            log_dir, args.task, env_cfg=env_cfg, checkpoint=Path(resume_path).name,
            exported_files=["exported/policy.pt", "exported/policy.onnx"],
        )
        obs = env.get_observations()
        if args.steps > 0:
            for _ in range(args.steps):
                obs, _, _, _ = env.step(policy(obs))
        else:
            while simulation_app.is_running():
                obs, _, _, _ = env.step(policy(obs))
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
