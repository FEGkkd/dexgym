"""Finite, fixed-seed local S0 evaluation against a zero-action baseline."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import dexgym.tasks  # noqa: F401
import gymnasium as gym
import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.robot_cfg import SOURCE_SHA256
from dexgym.tasks.marvin_s0.env_cfg import MarvinBimanualS0Cfg
from dexgym.tasks.marvin_s0.ppo_cfg import MarvinS0PPOCfg, compatible_agent_cfg
from isaaclab.app import launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import create_rsl_rl_runner


def statistics(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0, "mean": None, "p95": None, "p99": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, help="Collision-checked fixed evaluation JSON")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or (not args.dataset and args.steps >= 300):
        parser.error("Positive steps are required; local seeded evaluation uses 1..299")
    dataset = json.loads(args.dataset.read_text()) if args.dataset else None
    if dataset:
        expected = dataset.pop("case_set_sha256")
        if hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest() != expected:
            raise ValueError("Fixed evaluation JSON hash mismatch")
        if dataset["source_mjcf_sha256"] != SOURCE_SHA256:
            raise ValueError("Fixed evaluation model hash mismatch")
        cases = dataset["cases"]
        args.num_envs = len(cases)
        budgets = np.maximum(
            args.steps,
            np.ceil(np.array([case["minimum_time_budget_s_from_command_limit"] for case in cases]) / 0.01).astype(int),
        )
        case_labels = np.array([case["category"] for case in cases])
    else:
        budgets = np.full(args.num_envs, args.steps, dtype=int)
        case_labels = np.array(["local_se3_left_or_right"] * args.num_envs)
    max_steps = int(budgets.max())
    cfg = MarvinBimanualS0Cfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = max(3.0, (max_steps + 2) * 0.01)
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-Bimanual-SE3-Direct-v0", cfg=cfg)
        try:
            policy = None
            if args.checkpoint:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
                runner = create_rsl_rl_runner(wrapped, compatible_agent_cfg(MarvinS0PPOCfg()))
                runner.load(str(args.checkpoint.resolve()))
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                obs = wrapped.get_observations()
            raw = env.unwrapped
            obs, _ = env.reset(seed=args.seed)
            if dataset:
                raw.set_eval_cases(
                    torch.tensor([case["start_arm_q_rad"] for case in cases], device=raw.device),
                    torch.tensor([case["goal_tcp_pose_xyzw"] for case in cases], device=raw.device),
                    torch.tensor([case["active_mask"] for case in cases], device=raw.device),
                )
                obs = raw._get_observations()
                identity = expected
            else:
                identity = hashlib.sha256(
                    raw.robot.data.joint_pos.torch[:, raw.arm_ids].detach().cpu().numpy().tobytes()
                    + raw.goal_pose.detach().cpu().numpy().tobytes()
                ).hexdigest()
            mask = raw.active_mask.clone()
            alive = torch.ones(args.num_envs, dtype=torch.bool, device=raw.device)
            succeeded = torch.zeros_like(alive)
            terminated_total = 0
            term_sums = {}
            max_command_speed = torch.zeros((), device=raw.device)
            max_arm_speed = torch.zeros((), device=raw.device)
            sampled_contact_counts = []
            sampled_non_floor_contacts = []
            sampled_contact_cases = np.zeros(args.num_envs, dtype=bool)
            contact_pair_examples = []
            solver = raw.sim.physics_manager._solver  # MJWarp diagnostic API in pinned Newton 1.6.1
            body_names = [
                mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                for body_id in range(solver.mj_model.nbody)
            ]
            world = np.array([name == "world" for name in body_names])
            pedestal = np.array([name is not None and name.endswith("_Geometry_pedestal") for name in body_names])
            body_env = np.array(
                [int(match.group(1)) if (match := re.search(r"_env_(\d+)_", name or "")) else -1 for name in body_names]
            )
            budget_tensor = torch.tensor(budgets, device=raw.device)
            measured_pos = torch.full((args.num_envs, 2), float("nan"), device=raw.device)
            measured_rot = torch.full_like(measured_pos, float("nan"))
            valid_at_budget = torch.zeros_like(alive)
            for step in range(max_steps):
                with torch.inference_mode():
                    action = policy(obs) if policy else torch.zeros((args.num_envs, 14), device=raw.device)
                    obs, reward, terminated, truncated, _ = env.step(action)
                if not torch.isfinite(reward).all() or not torch.isfinite(raw.robot.data.joint_pos.torch).all():
                    raise ValueError("Nonfinite evaluation physics or reward")
                alive &= ~(terminated | truncated)
                succeeded |= (raw.success_steps >= 20) & alive & (step + 1 <= budget_tensor)
                terminated_total += int((terminated & (step + 1 <= budget_tensor)).sum())
                at_budget = step + 1 == budget_tensor
                if torch.any(at_budget):
                    valid_at_budget = torch.where(at_budget, alive, valid_at_budget)
                    pose_pos, pose_rot = raw._pose_errors()
                    measured_pos = torch.where(
                        at_budget[:, None], torch.linalg.vector_norm(pose_pos, dim=-1) * 1000.0, measured_pos
                    )
                    measured_rot = torch.where(
                        at_budget[:, None], torch.linalg.vector_norm(pose_rot, dim=-1) * (180.0 / np.pi), measured_rot
                    )
                max_command_speed = torch.maximum(max_command_speed, raw.command_velocity.abs().max())
                max_arm_speed = torch.maximum(max_arm_speed, raw.robot.data.joint_vel.torch[:, raw.arm_ids].abs().max())
                if step % 10 == 0:
                    contact_count = int(solver.mjw_data.nacon.numpy()[0])
                    sampled_contact_counts.append(contact_count)
                    if contact_count:
                        geoms = solver.mjw_data.contact.geom.numpy()[:contact_count]
                        bodies = solver.mj_model.geom_bodyid[geoms]
                        floor_pedestal = (world[bodies[:, 0]] & pedestal[bodies[:, 1]]) | (
                            world[bodies[:, 1]] & pedestal[bodies[:, 0]]
                        )
                        sampled_non_floor_contacts.append(int((~floor_pedestal).sum()))
                        contact_env_ids = body_env[bodies[~floor_pedestal]].reshape(-1)
                        contact_env_ids = contact_env_ids[(contact_env_ids >= 0) & (contact_env_ids < args.num_envs)]
                        sampled_contact_cases[contact_env_ids] = True
                        if len(contact_pair_examples) < 20:
                            for body_a, body_b in bodies[~floor_pedestal][: 20 - len(contact_pair_examples)]:
                                contact_pair_examples.append([body_names[body_a], body_names[body_b]])
                    else:
                        sampled_non_floor_contacts.append(0)
                for name, value in raw.last_reward_terms.items():
                    term_sums[name] = term_sums.get(name, torch.zeros((), device=raw.device)) + value.mean()
            pos_mm = measured_pos.detach().cpu().numpy()
            rot_deg = measured_rot.detach().cpu().numpy()
            active = mask.detach().cpu().numpy().astype(bool)
            valid = valid_at_budget.detach().cpu().numpy()
            success = succeeded.detach().cpu().numpy()

            def side_metrics(selected_cases):
                sides = {}
                for side, name in enumerate(("left", "right")):
                    selected = active[:, side] & selected_cases
                    sides[name] = {
                        "active_cases": int(selected.sum()),
                        "success_rate": float(success[selected].mean()) if selected.any() else None,
                        "position_error_mm": statistics(pos_mm[selected & valid, side]),
                        "orientation_error_deg": statistics(rot_deg[selected & valid, side]),
                        "terminated_before_budget": int((selected & ~valid).sum()),
                    }
                return sides

            categories = {}
            for category in sorted(set(case_labels)):
                selected = case_labels == category
                categories[category] = {
                    "cases": int(selected.sum()),
                    "success_rate": float(success[selected].mean()),
                    "left_right": side_metrics(selected),
                }
            result = {
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "case_set_sha256": identity,
                "seed": args.seed,
                "num_envs": args.num_envs,
                "steps": max_steps,
                "minimum_steps_per_case": args.steps,
                "category": "fixed_collision_checked" if dataset else "local_se3_left_or_right",
                "success_definition": "<=5 mm and <=2 deg held for >=20 consecutive 100-Hz steps",
                "success_rate": float(success.mean()),
                "terminated": terminated_total,
                "max_command_velocity_rad_s": float(max_command_speed),
                "max_actual_arm_velocity_rad_s": float(max_arm_speed),
                "newton_active_contacts_sampled_every_10_steps": {
                    "samples": len(sampled_contact_counts),
                    "max_total_contacts_across_envs": max(sampled_contact_counts),
                    "samples_with_contact": sum(count > 0 for count in sampled_contact_counts),
                    "max_non_floor_pedestal_contacts": max(sampled_non_floor_contacts),
                    "samples_with_non_floor_pedestal_contacts": sum(count > 0 for count in sampled_non_floor_contacts),
                    "cases_with_sampled_non_floor_pedestal_contact": int(sampled_contact_cases.sum()),
                    "non_floor_pedestal_pair_examples": contact_pair_examples,
                },
                "mean_reward_terms": {name: float(total / max_steps) for name, total in term_sums.items()},
                "left_right": side_metrics(np.ones(args.num_envs, dtype=bool)),
                "categories": categories,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
