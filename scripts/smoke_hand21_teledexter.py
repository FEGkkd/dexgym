"""Exercise the current Hand 2.1 TELEDEXTER task with privileged joint replay."""

import argparse

import gymnasium as gym
import torch
from dexgym.tasks import HAND21_TELEDEXTER_ID
from dexgym.tasks.direct.right_hand_tracking.logic import action_toward_joint_target
from dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg import WujiHand21TeleDexterCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--cross-probability", type=float, default=0.1)
    parser.add_argument("--library-path")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or not 0 <= args.cross_probability <= 1:
        parser.error("Invalid environment count, step count, or cross probability")
    load_external_tasks()
    cfg = WujiHand21TeleDexterCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.split = args.split
    cfg.cross_trajectory_probability = args.cross_probability
    if args.library_path:
        cfg.library_path = args.library_path
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = gym.make(HAND21_TELEDEXTER_ID, cfg=cfg)
        try:
            obs, _ = env.reset()
            task = env.unwrapped
            assert obs["policy"].shape == (args.num_envs, 125)
            start = task.goal_frame - task.goal_span
            fk_error = torch.linalg.vector_norm(
                task._current_tips() - task.bank.fingertips(task.motion_ids, start), dim=-1
            ).max().item()
            assert fk_error < 0.005, f"Hand 2.1 FK mismatch: {fk_error:.6f} m"
            reached_count = 0
            cross_count = 0
            for _ in range(args.steps):
                before_ids = task.motion_ids.clone()
                before_frame = task.goal_frame.clone()
                desired = task.bank.joints(task.motion_ids, task.goal_frame)
                action = action_toward_joint_target(
                    desired, task.target_q, scale=cfg.action_delta_rad, deadzone=cfg.action_deadzone
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                crossed = task.last_crossed & ~(terminated | truncated)
                if crossed.any():
                    assert torch.all(task.motion_ids[crossed] != before_ids[crossed])
                    assert torch.equal(task.goal_frame[crossed], before_frame[crossed])
                reached_count += int(task.last_reached.sum().item())
                cross_count += int(crossed.sum().item())
            print(
                f"HAND21_TELEDEXTER_SMOKE_OK envs={args.num_envs} steps={args.steps} "
                f"reached={reached_count} cross_switches={cross_count} fk_max_mm={fk_error * 1000:.4f}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
