"""Compare learned one-step SE(3) reward gradients with real Newton differences.

This is a physical diagnostic, not a controller or a training data source. The
sampler's joint witness is never used as a target action or reward label.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # noqa: F401
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.tasks.marvin_supermimic.se3_env import MarvinSE3ReachHoldCfg


def model_reward_gradient(record, robot, task, state, task_state,
                          base_action, previous_action, epsilon):
    if record["robot_spec"] != asdict(robot.spec) or record["task_spec"] != asdict(task.spec):
        raise ValueError("Diagnostic checkpoint differs from the active Robot or Task schema")
    network = NetworkSpec(**record["contract_manifest"]["networks"]["world_model"]["network_spec"])
    modules = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot)
    model = modules.world_model.to(state.device)
    model.load_state_dict(record["modules"]["world_model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    action = base_action.detach().clone().requires_grad_(True)
    next_state = model(state, action, state.new_zeros((1, robot.spec.context_dim)), robot)
    predicted_reward = task.reward(state, next_state, action, task.advance(task_state),
                                   previous_action)
    gradient, = torch.autograd.grad(predicted_reward.sum(), action)
    with torch.no_grad():
        perturbation = epsilon * torch.eye(robot.spec.action_dim, device=state.device)
        paired_actions = torch.cat((base_action + perturbation,
                                    base_action - perturbation))
        count = len(paired_actions)
        paired_state = state.expand(count, -1)
        paired_task_state = TaskState(
            *(getattr(task_state, name).expand(count, *getattr(task_state, name).shape[1:])
              for name in ("task_id", "motion_id", "frame", "data")))
        paired_next = model(paired_state, paired_actions,
                            state.new_zeros((count, robot.spec.context_dim)), robot)
        paired_reward = task.reward(paired_state, paired_next, paired_actions,
                                    task.advance(paired_task_state),
                                    previous_action.expand(count, -1))
        model_difference = ((paired_reward[:robot.spec.action_dim]
                             - paired_reward[robot.spec.action_dim:]) / (2 * epsilon))
        model_q_response = ((paired_next[:robot.spec.action_dim, :14]
                             - paired_next[robot.spec.action_dim:, :14]) / (2 * epsilon))
        model_qd_response = ((paired_next[:robot.spec.action_dim, 14:28]
                              - paired_next[robot.spec.action_dim:, 14:28]) / (2 * epsilon))
    return (gradient.detach()[0], float(predicted_reward.detach()[0]),
            model_difference, model_q_response, model_qd_response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--world-model-override", type=Path,
                        help="Evaluate an offline WM-only diagnostic beside its source checkpoint")
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--action-epsilon", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-actor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not 1 <= args.trials <= 8 or not 0.05 <= args.action_epsilon <= 1.0
            or not 0 <= args.warmup_steps <= 100
            or (args.warmup_steps and args.warmup_actor is None)):
        parser.error("Gradient diagnosis is bounded to 1-8 goals and epsilon 0.05-1.0")
    records = [(str(path), torch.load(path, map_location="cpu", weights_only=False))
               for path in args.checkpoint]
    if args.world_model_override is not None:
        diagnostic = torch.load(args.world_model_override, map_location="cpu", weights_only=False)
        if not diagnostic.get("diagnostic_only") or "world_model" not in diagnostic:
            parser.error("World Model override must be a diagnostic-only model record")
        sources = [record for name, record in records
                   if name == diagnostic.get("source_checkpoint")]
        if len(sources) != 1:
            parser.error("World Model override source checkpoint must be listed exactly once")
        override = {**sources[0], "modules": {**sources[0]["modules"],
                                              "world_model": diagnostic["world_model"]}}
        records.append((str(args.world_model_override), override))
    cfg = MarvinSE3ReachHoldCfg()
    cfg.scene.num_envs = 30  # two baselines plus positive/negative action for each of 14 joints
    cfg.seed = args.seed
    cfg.goal_train_seed = args.seed + 101
    cfg.goal_eval_seed = args.seed + 100101
    cfg.evaluation_mode = True
    cfg.audit_contacts_each_physics_step = True
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            robot, task = native.core_robot, native.se3_task
            warmup_actor = None
            if args.warmup_actor is not None:
                actor_record = torch.load(args.warmup_actor, map_location="cpu", weights_only=False)
                if (actor_record["robot_spec"] != asdict(robot.spec)
                        or actor_record["task_spec"] != asdict(task.spec)):
                    raise ValueError("Warmup Actor differs from active Robot or Task schema")
                actor_spec = NetworkSpec(**actor_record["contract_manifest"]["networks"]["actor"]["network_spec"])
                warmup_actor = NetworkFactory.build(
                    robot.spec, task.spec, actor_spec, robot_adapter=robot).actor.to(native.device)
                warmup_actor.load_state_dict(actor_record["modules"]["actor"], strict=True)
                warmup_actor.eval()
            env.reset()
            results = []
            for trial in range(args.trials):
                sample = native.goal_sampler.sample_evaluation(1, "near")
                arm_q = sample.start_q.expand(30, -1)
                full_q = native.home_full[None].expand(30, -1).clone()
                full_qd = torch.zeros_like(full_q)
                env_ids = torch.arange(30, device=native.device)
                native._write_robot_reset(env_ids, full_q, full_qd, arm_q,
                                          torch.zeros_like(arm_q))
                native.goal_pose[:] = sample.goal_pose.expand(30, -1)
                native.goal_frame.zero_()
                native.episode_length_buf.zero_()
                native._contact_accum.zero_()
                first_observation = task.observation(
                    robot.physical_state(native), native.last_action, native.task_state())
                native.se3_history[:] = first_observation[:, None].expand(-1, 10, -1)
                for _ in range(args.warmup_steps):
                    with torch.no_grad():
                        warmup_action = warmup_actor(
                            native.se3_history,
                            task.goal(robot.physical_state(native), native.task_state()))
                    _, _, failed, timeout, _ = env.step(warmup_action)
                    if bool((failed | timeout).any()):
                        raise RuntimeError("Warmup Actor failed before gradient comparison")
                if args.warmup_steps:
                    # Reclone the *full* 54DoF physical state. Tiny GPU solver
                    # differences can otherwise compound through the Actor
                    # during warmup and invalidate paired finite differences.
                    full_q = native.robot.data.joint_pos.torch[:1].detach().clone().expand(30, -1).clone()
                    full_qd = native.robot.data.joint_vel.torch[:1].detach().clone().expand(30, -1).clone()
                    command = native.command[:1].detach().clone()
                    previous_action = native.last_action[:1].detach().clone()
                    frame = native.goal_frame[0].clone()
                    elapsed = native.episode_length_buf[0].clone()
                    history = native.se3_history[:1].detach().clone()
                    native._write_robot_reset(env_ids, full_q, full_qd,
                                              full_q[:, native.arm_ids], full_qd[:, native.arm_ids])
                    native.command[:] = command.expand(30, -1)
                    native.previous_command[:] = native.command
                    native.last_action[:] = previous_action.expand(30, -1)
                    native.goal_frame[:] = frame
                    native.episode_length_buf[:] = elapsed
                    native.se3_history[:] = history.expand(30, -1, -1)
                state = robot.physical_state(native).detach().clone()
                state_spread = (state - state[:1]).abs().max(0).values
                if float(state_spread.max()) > 1e-4:
                    raise RuntimeError(
                        "Paired Newton environments diverged before action perturbation: "
                        f"q={float(state_spread[:14].max()):.6g}, "
                        f"qd={float(state_spread[14:28].max()):.6g}, "
                        f"qcmd={float(state_spread[28:].max()):.6g}")
                task_state = native.task_state().index(slice(0, 1))
                with torch.no_grad():
                    base_action = (warmup_actor(native.se3_history[:1],
                                                task.goal(state[:1], task_state))
                                   if warmup_actor is not None else state.new_zeros((1, 14)))
                previous_action = native.last_action[:1].detach().clone()
                gradients = {}
                for path, record in records:
                    gradients[path] = model_reward_gradient(record, robot, task,
                                                             state[:1], task_state,
                                                             base_action, previous_action,
                                                             args.action_epsilon)
                actions = base_action.expand(30, -1).clone()
                actions[1:15] += args.action_epsilon * torch.eye(14, device=native.device)
                actions[15:29] -= args.action_epsilon * torch.eye(14, device=native.device)
                _, reward, terminated, truncated, _ = env.step(actions)
                if bool((terminated | truncated).any()):
                    raise RuntimeError("Finite-difference Newton step failed the safety audit")
                finite_difference = ((reward[1:15] - reward[15:29])
                                     / (2 * args.action_epsilon)).detach()
                physical_next = native.last_physical_arm_state.detach()
                physical_q_response = ((physical_next[1:15, :14]
                                        - physical_next[15:29, :14])
                                       / (2 * args.action_epsilon))
                physical_qd_response = ((physical_next[1:15, 14:28]
                                         - physical_next[15:29, 14:28])
                                        / (2 * args.action_epsilon))
                if not bool(torch.isfinite(finite_difference).all()):
                    raise RuntimeError("Newton finite-difference reward is nonfinite")
                trial_report = {
                    "trial": trial, "goal_position_difficulty_m": sample.position_difficulty_m[0].tolist(),
                    "goal_rotation_difficulty_rad": sample.rotation_difficulty_rad[0].tolist(),
                    "warmup_steps": args.warmup_steps,
                    "baseline_action_abs_tanh_mean": float(base_action.tanh().abs().mean()),
                    "newton_baseline_reward": float(reward[0]),
                    "paired_baseline_reward_difference": float((reward[0] - reward[29]).abs()),
                    "newton_gradient": finite_difference.tolist(), "models": {},
                    "newton_q_action_response": physical_q_response.tolist(),
                    "newton_qd_action_response": physical_qd_response.tolist(),
                }
                for path, (gradient, predicted_reward, model_difference,
                           model_q_response, model_qd_response) in gradients.items():
                    norm = float(gradient.norm())
                    physical_norm = float(finite_difference.norm())
                    cosine = (float(torch.dot(gradient, finite_difference)
                                    / (gradient.norm() * finite_difference.norm()))
                              if norm > 1e-9 and physical_norm > 1e-9 else None)
                    trial_report["models"][path] = {
                        "predicted_reward": predicted_reward,
                        "gradient": gradient.tolist(), "gradient_norm": norm,
                        "model_finite_difference_gradient": model_difference.tolist(),
                        "model_gradient_vs_finite_difference_cosine": float(
                            torch.dot(gradient, model_difference)
                            / (gradient.norm() * model_difference.norm())),
                        "model_gradient_vs_finite_difference_norm_ratio": float(
                            gradient.norm() / model_difference.norm()),
                        "newton_gradient_norm": physical_norm,
                        "gradient_cosine": cosine,
                        "joint_sign_agreement": float((gradient.sign()
                                                       == finite_difference.sign()).float().mean()),
                        "model_q_action_response": model_q_response.tolist(),
                        "model_qd_action_response": model_qd_response.tolist(),
                        "q_action_response_cosine": float(
                            torch.dot(model_q_response.flatten(), physical_q_response.flatten())
                            / (model_q_response.norm() * physical_q_response.norm())),
                        "q_action_response_norm_ratio": float(
                            model_q_response.norm() / physical_q_response.norm()),
                        "qd_action_response_cosine": float(
                            torch.dot(model_qd_response.flatten(), physical_qd_response.flatten())
                            / (model_qd_response.norm() * physical_qd_response.norm())),
                        "qd_action_response_norm_ratio": float(
                            model_qd_response.norm() / physical_qd_response.norm()),
                    }
                results.append(trial_report)
            output = {"evaluation_kind": "one_step_actor_action_gradient_model_vs_newton",
                      "action_epsilon_raw": args.action_epsilon,
                      "warmup_actor": str(args.warmup_actor) if args.warmup_actor else None,
                      "warmup_steps": args.warmup_steps,
                      "num_envs": 30, "seed": args.seed, "trials": results,
                      "training_data_used": False, "teacher_or_ik_used": False}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2))
            print(json.dumps(output), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
