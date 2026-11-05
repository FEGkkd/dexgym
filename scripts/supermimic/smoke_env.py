"""Short Newton DirectRLEnv arm-only collection with finite-state checks."""

import argparse
import json
import time

import gymnasium as gym
import torch
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from isaaclab.app import launch_simulation


def run(num_envs: int, steps: int, *, random_start: bool = True):
    cfg = MarvinWaveDirectCfg()
    cfg.scene.num_envs = num_envs
    cfg.random_reference_start = random_start
    started = time.perf_counter()
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            obs, _ = env.reset()
            if obs["policy"].shape != (num_envs, 1460):
                raise ValueError(f"Unexpected policy observation {obs['policy'].shape}")
            native = env.unwrapped
            q_reset = native.robot.data.joint_pos.torch[:, native.arm_ids]
            reset_error = float((q_reset - native.motion.current("q_arm")).abs().max())
            if reset_error > 0.031:
                raise ValueError(f"Reference reset joint order mismatch: {reset_error}")
            rewards = []
            for i in range(steps):
                action = torch.zeros((num_envs, 14), device=env.unwrapped.device)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                    raise ValueError(f"Nonfinite physical transition at step {i}")
                rewards.append(float(reward.mean()))
            hand_ids = torch.tensor([j for j in range(54) if j not in native.arm_ids.tolist()], device=native.device)
            hand_drift = float(
                (native.robot.data.joint_pos.torch[:, hand_ids] - native.home_full[hand_ids]).abs().max()
            )
            return {
                "environments": num_envs,
                "steps": steps,
                "transitions": num_envs * steps,
                "seconds": time.perf_counter() - started,
                "reward_mean": sum(rewards) / len(rewards),
                "terminated": int(terminated.sum()),
                "truncated": int(truncated.sum()),
                "observation_shape": list(obs["policy"].shape),
                "backend": "Newton MJWarp DirectRLEnv",
                "reset_arm_error_max_rad": reset_error,
                "finger_home_drift_max_rad": hand_drift,
            }
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fixed-start", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.num_envs, args.steps, random_start=not args.fixed_start), indent=2))
