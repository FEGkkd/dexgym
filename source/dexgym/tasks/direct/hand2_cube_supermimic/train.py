"""Train the registered Wuji Hand 2 cube task with the pinned SuperMimic recipe."""

import argparse
import json
import random
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine
from dexgym.assets.wuji_hand2_cube import REFERENCE_SHA
from dexgym.tasks import HAND2_CUBE_SUPERMIMIC_ID

from .adapter import CubeAdapter
from .env_cfg import Hand2CubeSuperMimicCfg
from .supermimic_cfg import default_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wm-batch", type=int, default=2048)
    parser.add_argument("--pi-batch", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = Hand2CubeSuperMimicCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    recipe = default_experiment()
    recipe.seed = args.seed
    recipe.max_iterations = args.iterations
    recipe.wm_batch = args.wm_batch
    recipe.pi_batch = args.pi_batch
    recipe.log_interval = args.log_interval
    recipe.compile = recipe.compile_policy = not args.no_compile
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[5]

    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(root), *cmd], text=True).strip()

    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "task": HAND2_CUBE_SUPERMIMIC_ID,
                "reference_sha": REFERENCE_SHA,
                "scene_revision": cfg.scene_revision,
                "dexgym_commit": git("rev-parse", "HEAD"),
                "worktree_status": git("status", "--porcelain"),
                "num_envs": args.num_envs,
                "backend": "Isaac Lab 3 / Newton MuJoCo Warp",
            },
            indent=2,
        )
    )
    with launch_simulation(cfg):
        env = gym.make(HAND2_CUBE_SUPERMIMIC_ID, cfg=cfg).unwrapped
        try:
            engine = SuperMimicEngine(CubeAdapter(env), recipe, args.output)
            if args.checkpoint:
                engine.load(args.checkpoint)
            engine.learn()
        finally:
            env.close()


if __name__ == "__main__":
    main()
