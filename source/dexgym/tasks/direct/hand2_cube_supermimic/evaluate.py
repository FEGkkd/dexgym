"""Evaluate deterministic actions in native physics; no replay or imagination.

Every lane contributes the same number of complete episodes. Report goal
completions, drops, survival and orientation error, including zero-success runs.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.dex_reference.networks import Actor
from dexgym.assets.wuji_hand2_cube import REFERENCE_SHA
from dexgym.tasks import HAND2_CUBE_SUPERMIMIC_ID

from .adapter import CubeAdapter
from .env_cfg import Hand2CubeSuperMimicCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes-per-env", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-push", action="store_true", help="Report a separate evaluation without interval pushes")
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes_per_env < 1:
        parser.error("Environment and episode counts must be positive")
    cfg = Hand2CubeSuperMimicCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.velocity_disturbance = not args.no_push
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg):
        env = gym.make(HAND2_CUBE_SUPERMIMIC_ID, cfg=cfg).unwrapped
        try:
            actor = Actor(207, 20).to(env.device).eval()
            checkpoint_iteration = None
            if args.checkpoint:
                payload = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
                if payload["task_id"] != "Wuji-Hand2-Cube-Reorientation" or payload["format_version"] != 1:
                    raise ValueError("Checkpoint task/schema mismatch")
                actor.load_state_dict(payload["modules"]["actor"])
                checkpoint_iteration = payload["iteration"]
            # Reset RNG after loading/initialization for matching evaluation seeds.
            obs = env.reset(seed=args.seed)[0]["policy"]
            task = CubeAdapter(env)
            completed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            successes = torch.zeros(env.num_envs, device=env.device)
            errors = torch.zeros_like(successes)
            returns = torch.zeros_like(successes)
            records = []
            steps = 0
            with torch.inference_mode():
                while bool((completed < args.episodes_per_env).any()):
                    obs_dict, reward, terminated, truncated, _ = env.step(actor(obs))
                    obs = obs_dict["policy"]
                    cap = env.transition
                    if not bool(torch.isfinite(cap["x1"]).all()):
                        raise RuntimeError("Non-finite physical state during evaluation")
                    successes += (env.memory["count"] - cap["memory"]["count"]).squeeze(-1)
                    errors += task.rotation_error(cap["x1"], cap["memory"])
                    returns += reward
                    done_ids = (terminated | truncated).nonzero().flatten()
                    for idx in done_ids.tolist():
                        if completed[idx] < args.episodes_per_env:
                            length = int(env.episode_length_buf[idx])
                            records.append(
                                dict(
                                    env=idx,
                                    episode=int(completed[idx]),
                                    steps=length,
                                    successes=int(successes[idx]),
                                    drop=bool(cap["drop"][idx]),
                                    timeout=bool(truncated[idx]),
                                    mean_angle_rad=float(errors[idx] / length),
                                    episode_return=float(returns[idx]),
                                )
                            )
                            completed[idx] += 1
                    if len(done_ids):
                        obs = env.reset(env_ids=done_ids)[0]["policy"]
                        successes[done_ids] = 0
                        errors[done_ids] = 0
                        returns[done_ids] = 0
                    steps += 1
                    if steps % 500 == 0:
                        print(json.dumps(dict(control_steps=steps, completed_episodes=len(records))), flush=True)
            n = len(records)
            total_steps = sum(r["steps"] for r in records)
            result = dict(
                reference_sha=REFERENCE_SHA,
                checkpoint=str(args.checkpoint) if args.checkpoint else None,
                checkpoint_iteration=checkpoint_iteration,
                seed=args.seed,
                num_envs=env.num_envs,
                episodes_per_env=args.episodes_per_env,
                observation_noise=True,
                startup_randomization=True,
                reset_randomization=True,
                interval_pushes=cfg.velocity_disturbance,
                disturbance_protocol="fresh reference curriculum for every evaluation",
                episodes=n,
                mean_successes=sum(r["successes"] for r in records) / n,
                fraction_at_least_one=sum(r["successes"] >= 1 for r in records) / n,
                fraction_at_least_three=sum(r["successes"] >= 3 for r in records) / n,
                drop_fraction=sum(r["drop"] for r in records) / n,
                mean_episode_seconds=total_steps * env.step_dt / n,
                mean_angle_rad=sum(r["mean_angle_rad"] * r["steps"] for r in records) / total_steps,
                records=records,
            )
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
