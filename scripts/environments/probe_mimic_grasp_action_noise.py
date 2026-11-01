"""Measure whether Gaussian exploration destroys a validated PGS grasp.

This is a physics counterfactual, not policy evaluation. Every run starts
from the same one-family stable bank; only normalized action noise changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--stds", type=float, nargs="+", default=(0.0, 0.05, 0.25))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stable-bank", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 25 or any(not 0 <= x <= .65 for x in args.stds):
    parser.error("Positive environments, at least 25 steps, and stds in [0, 0.65] required")
app = AppLauncher(args).app

import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import GraspHoldDirectEnvCfg


def main():
    cfg = GraspHoldDirectEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.thread_backend = "mimic"
    cfg.sim.physics.solver_type = 0
    cfg.stable_train_bank_path = str(args.stable_bank)
    cfg.stable_reset_probability = 1.0
    cfg.finger_increment_scale = .01
    cfg.wrist_translation_scale = .0005
    env = BimanualBottleDirectEnv(cfg)
    try:
        results = []
        for std in args.stds:
            torch.manual_seed(args.seed)
            env.reset()
            seen_success = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
            seen_failure = torch.zeros_like(seen_success)
            right_contact_sum = left_contact_sum = 0.0
            start = time.monotonic()
            for _ in range(args.steps):
                action = torch.randn(args.num_envs, 52, device=env.device) * std
                _, _, _, _, info = env.step(action)
                episode = info["phase1_episode"]
                seen_success |= episode["success"]
                seen_failure |= episode["failure"]
                right_contact_sum += float(episode["right_contact"].float().mean())
                left_contact_sum += float(episode["left_contact"].float().mean())
            results.append({
                "normalized_action_std": std,
                "success_fraction_any_in_50_steps": float(seen_success.float().mean()),
                "failure_fraction_any_in_50_steps": float(seen_failure.float().mean()),
                "left_contact_step_fraction": left_contact_sum / args.steps,
                "right_tripod_step_fraction": right_contact_sum / args.steps,
                "elapsed_s": time.monotonic() - start,
            })
        output = {"fixture": "pgs_mimic_stable_bank_action_noise_ablation",
                  "num_envs": args.num_envs, "steps": args.steps,
                  "seed": args.seed, "stable_bank": str(args.stable_bank),
                  "action_scales": {"finger_rad": cfg.finger_increment_scale,
                                    "wrist_translation_m": cfg.wrist_translation_scale,
                                    "wrist_rotation_rad": cfg.wrist_rotation_scale},
                  "results": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print("MIMIC_ACTION_NOISE", json.dumps(output), flush=True)
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
