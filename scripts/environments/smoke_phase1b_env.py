"""Finite reset/step and throughput probe for the unified 52D PGS task."""

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--level", choices=("A", "B"), default="A")
parser.add_argument("--hand-variant", choices=("beta1", "beta2"), default="beta2")
parser.add_argument("--evaluation", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.hand_variant == "beta2" and args.level == "B":
    parser.error("Beta 2 weak-contact pregrasps have not been physically qualified")
app = AppLauncher(args).app

import torch
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.env import PregraspCapTurn30DirectEnv
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.env_cfg import PregraspCapTurn30DirectEnvCfg
from dexteroushands_lab.tasks.direct.bimanual_bottle.variant_config import configure_bimanual_hand_variant
from dexteroushands_lab.tasks.direct.bimanual_bottle.task_contract import actor_dim, critic_dim


def main():
    cfg = PregraspCapTurn30DirectEnvCfg()
    configure_bimanual_hand_variant(cfg, args.hand_variant)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 1001
    cfg.pregrasp_reset_levels = (args.level,)
    cfg.evaluation_mode = args.evaluation
    env = PregraspCapTurn30DirectEnv(cfg)
    try:
        obs, _ = env.reset()
        if obs["policy"].shape != (args.num_envs, actor_dim()) or obs["critic"].shape != (args.num_envs, critic_dim()):
            raise RuntimeError("Unified actor/critic dimensions differ from schema")
        start = time.monotonic()
        rewards = []
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(
                torch.zeros(args.num_envs, 52, device=env.device))
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(obs["critic"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Nonfinite Phase 1B observation or reward")
            rewards.append(float(reward.mean()))
        elapsed = time.monotonic() - start
        print(json.dumps({"num_envs": args.num_envs, "steps": args.steps,
                          "actor_dim": actor_dim(), "critic_dim": critic_dim(),
                          "reward_mean": sum(rewards)/len(rewards),
                          "transitions_per_second": args.num_envs*args.steps/elapsed,
                          "left_contact_rate": float(env._metrics["left_contact"].float().mean()),
                          "right_contact_rate": float(env._metrics["right_contact"].float().mean()),
                          "opening_max_rad": float(env._metrics["opening"].max()),
                          "constraint_error_max_m": float(env._latest_thread.constraint_error.abs().max())}),
              flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
