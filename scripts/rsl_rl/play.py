"""Play and export a trained DexterousHands RSL-RL policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from importlib import metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Play a DexterousHands RSL-RL policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--steps", type=int, default=0, help="Number of inference steps; 0 runs until the simulator closes.")
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
    # Evaluation must use the clean policy contract, without training corruption.
    if hasattr(env_cfg, "observation_noise_enabled"):
        env_cfg.observation_noise_enabled = False
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed
    log_root = REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = Path(resume_path).parent

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = log_dir / "exported"
    runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
    runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")
    metadata_paths = write_policy_artifacts(
        log_dir,
        args_cli.task,
        env_cfg=env_cfg,
        checkpoint=Path(resume_path).name,
        exported_files=["exported/policy.pt", "exported/policy.onnx"],
    )
    if metadata_paths:
        print(f"[INFO] Wrote policy contract metadata to {log_dir}")

    obs = env.get_observations()
    if args_cli.steps > 0:
        for _ in range(args_cli.steps):
            obs, _, _, _ = env.step(policy(obs))
    else:
        while simulation_app.is_running():
            obs, _, _, _ = env.step(policy(obs))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
