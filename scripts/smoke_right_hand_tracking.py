"""Check right-hand task FK parity and run a small joint-replay baseline."""

import argparse

import gymnasium as gym
import torch
from dexgym.tasks import RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--tolerance-m", type=float, default=None)
    parser.add_argument("--max-goal-steps", type=int, default=None)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    load_external_tasks()
    cfg = RightHandTrackingDirectCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.split = args.split
    if args.stride is not None:
        cfg.subgoal_stride = args.stride
    if args.tolerance_m is not None:
        cfg.goal_tolerance_m = args.tolerance_m
    if args.max_goal_steps is not None:
        cfg.max_goal_steps = args.max_goal_steps
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = gym.make(RIGHT_HAND_ID, cfg=cfg)
        try:
            obs, _ = env.reset()
            task = env.unwrapped
            assert obs["policy"].shape == (args.num_envs, 105)
            assert env.action_space.shape[-1] == 20
            start = task.goal_frame - task.cfg.subgoal_stride
            reference = task.bank.fingertips(task.motion_ids, start)
            initial_error = torch.linalg.vector_norm(task._current_tips() - reference, dim=-1)
            fk_error_m = initial_error.max().item()
            assert fk_error_m < 0.005, f"Wrist-frame FK mismatch: {fk_error_m:.6f} m"
            reached = 0.0
            resets = 0
            tip_errors = []
            joint_errors = []
            for _ in range(args.steps):
                q_goal = task.bank.joints(task.motion_ids, task.goal_frame)
                action = ((q_goal - task.target_q) / cfg.action_delta_rad).clamp(-1.0, 1.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(reward).all()
                reached += task.extras["log"]["goal_reach_fraction"].item() * args.num_envs
                resets += int((terminated | truncated).sum().item())
                tip_errors.append(task.extras["log"]["mean_tip_error_m"].item())
                joint_errors.append(
                    (task.robot.data.joint_pos.torch[:, task.right_ids] - q_goal).abs().mean().item()
                )
            print(
                f"RIGHT_HAND_SMOKE_OK envs={args.num_envs} steps={args.steps} "
                f"split={cfg.split} stride={cfg.subgoal_stride} tolerance_m={cfg.goal_tolerance_m:.3f} "
                f"fk_max_error_m={fk_error_m:.6f} subgoals_reached={reached:.0f} resets={resets} "
                f"tip_error_mean_m={sum(tip_errors) / len(tip_errors):.5f} "
                f"joint_error_mean_rad={sum(joint_errors) / len(joint_errors):.5f}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
