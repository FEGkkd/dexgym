"""Audit 500 Hz Marvin joint velocity and displacement with zero policy action."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # noqa: F401
from dexgym.tasks.marvin_supermimic.se3_env import MarvinSE3ReachHoldCfg
from dexgym.tasks.marvin_supermimic.velocity_audit import substep_velocity_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--control-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9511)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 16 or not 2 <= args.control_steps <= 100:
        parser.error("Velocity audit is bounded to 1-16 environments and 2-100 control steps")
    if args.output.exists():
        parser.error("Velocity audit output already exists")
    cfg = MarvinSE3ReachHoldCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.goal_train_seed = args.seed + 101
    cfg.goal_eval_seed = args.seed + 100101
    cfg.audit_contacts_each_physics_step = True
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0", cfg=cfg)
        native = env.unwrapped
        original_apply = native._apply_action
        try:
            env.reset()
            initial_command = native.command.detach().clone()
            positions, velocities = [], []

            def record_then_apply():
                positions.append(native.robot.data.joint_pos.torch[:, native.arm_ids]
                                 .detach().clone())
                velocities.append(native.robot.data.joint_vel.torch[:, native.arm_ids]
                                  .detach().clone())
                original_apply()

            native._apply_action = record_then_apply
            contact_steps = 0
            for _ in range(args.control_steps):
                _, _, terminated, truncated, _ = env.step(
                    torch.zeros((args.num_envs, 14), device=native.device))
                if bool((terminated | truncated).any()):
                    raise RuntimeError("Zero-action velocity audit ended an Episode")
                contact_steps += int(native.last_policy_contact_counts.sum())
            q = torch.stack(positions).cpu()
            qd = torch.stack(velocities).cpu()
            if len(q) != args.control_steps * cfg.decimation:
                raise RuntimeError("Newton did not sample every 500 Hz substep")
            report = {"kind": "zero_action_newton_substep_velocity_audit",
                      "seed": args.seed,
                      "physics_dt_s": cfg.sim.dt,
                      "policy_dt_s": cfg.sim.dt * cfg.decimation,
                      "control_steps": args.control_steps,
                      "num_envs": args.num_envs,
                      "contact_event_substeps": contact_steps,
                      "max_command_change_rad": float(
                          (native.command - initial_command).abs().max()),
                      "joint_order": list(native.core_robot.spec.joint_order),
                      "per_environment": [
                          substep_velocity_metrics(q[:, index], qd[:, index],
                                                   dt=cfg.sim.dt)
                          for index in range(args.num_envs)]}
            args.output.mkdir(parents=True)
            torch.save({"position": q, "velocity": qd,
                        "report": report}, args.output / "substep_trace.pt")
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            native._apply_action = original_apply
            env.close()


if __name__ == "__main__":
    main()
