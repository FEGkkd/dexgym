"""Check that the imported WUJI articulation behaves as a fixed-base hand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="DexterousHands-Wuji-Hand-Reorientation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dexteroushands_lab.tasks as dexteroushands_tasks  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

dexteroushands_tasks.register_tasks()


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    try:
        env.reset(seed=0)
        robot = env.unwrapped.robot
        root_pose_before = robot.data.body_link_pose_w.torch[:, 0].clone()
        root_vel_before = robot.data.body_link_vel_w.torch[:, 0].clone()
        for _ in range(args.steps):
            targets = env.unwrapped.joint_targets
            robot.set_joint_position_target_index(
                target=targets, joint_ids=env.unwrapped.actuated_joint_ids
            )
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
        root_pose_after = robot.data.body_link_pose_w.torch[:, 0]
        root_vel_after = robot.data.body_link_vel_w.torch[:, 0]
        pose_error = torch.linalg.norm(root_pose_after[:, :3] - root_pose_before[:, :3], dim=-1)
        quat_error = torch.minimum(
            torch.linalg.norm(root_pose_after[:, 3:7] - root_pose_before[:, 3:7], dim=-1),
            torch.linalg.norm(root_pose_after[:, 3:7] + root_pose_before[:, 3:7], dim=-1),
        )
        velocity = torch.linalg.norm(root_vel_after, dim=-1)
        print(
            "FIXED_BASE_ROOT "
            f"position_error={pose_error.detach().cpu().tolist()} "
            f"quaternion_error={quat_error.detach().cpu().tolist()} "
            f"root_velocity={velocity.detach().cpu().tolist()} "
            f"root_pose_before={root_pose_before[0].detach().cpu().tolist()} "
            f"root_pose_after={root_pose_after[0].detach().cpu().tolist()}",
            flush=True,
        )
    finally:
        env.close()


try:
    main()
finally:
    app.close()
