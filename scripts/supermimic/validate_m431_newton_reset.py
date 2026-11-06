"""Bounded real Newton check of first reset, partial reset and shared Marvin servo."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # noqa: F401 - Gym registration
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.reach_env import MarvinReachDirectCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tracking", "reach"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = MarvinWaveDirectCfg() if args.task == "tracking" else MarvinReachDirectCfg()
    cfg.scene.num_envs = 16
    cfg.seed = 431
    cfg.audit_contacts_each_physics_step = True
    if args.task == "tracking":
        cfg.initialize_hands_from_pd_replay = False
        cfg.random_reference_start = False
        cfg.reset_joint_perturb_rad = 0.0
        cfg.action_schema = "robot_fixed_v3"
    gym_id = ("DexGym-Marvin-SuperMimic-Wave-Direct-v0" if args.task == "tracking" else
              "DexGym-Marvin-SuperMimic-Reach-Direct-v0")
    with launch_simulation(cfg):
        env = gym.make(gym_id, cfg=cfg)
        try:
            native = env.unwrapped
            env.reset()
            for _ in range(3):
                env.step(torch.zeros(16, 14, device=native.device))
            state_before = native.dynamics_adapter.physical_state(native).clone()
            frame_before = (native.motion.frame if args.task == "tracking" else native.reach_frame).clone()
            ids = torch.tensor([1, 4], device=native.device, dtype=torch.long)
            untouched = torch.tensor([index for index in range(16) if index not in (1, 4)],
                                     device=native.device, dtype=torch.long)
            native._reset_idx(ids)
            state_after = native.dynamics_adapter.physical_state(native)
            frame_after = native.motion.frame if args.task == "tracking" else native.reach_frame
            if not torch.equal(state_before[untouched], state_after[untouched]):
                raise AssertionError("Partial reset changed untouched robot states")
            if not torch.equal(frame_before[untouched], frame_after[untouched]):
                raise AssertionError("Partial reset changed untouched task time")
            if not bool((frame_after[ids] == 0).all()):
                raise AssertionError("Partial reset did not restart selected task time")
            env.step(torch.zeros(16, 14, device=native.device))
            if native.robot.num_joints != 54 or native.arm_ids.numel() != 14 or native.hand_ids.numel() != 40:
                raise AssertionError("Marvin full-physics joint contract differs")
            if not torch.allclose(native.full_command[:, native.hand_ids],
                                  native.home_full[native.hand_ids][None].expand(16, -1)):
                raise AssertionError("Hand commands are not fixed at home")
            if not torch.allclose(native.full_command[:, native.arm_ids], native.command):
                raise AssertionError("Final Newton servo substep differs from the command")
            result = {"task": args.task, "num_envs": 16, "partial_reset_env_ids": [1, 4],
                      "untouched_state_exact": True, "untouched_task_time_exact": True,
                      "selected_task_time_reset": True, "joints": 54, "arm_actions": 14,
                      "hands_home_commands": 40,
                      "physics_dt": cfg.sim.dt, "policy_dt": cfg.sim.dt * cfg.decimation}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
