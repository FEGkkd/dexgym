"""Collect bounded, teacher-free paired Newton action probes for WM diagnosis."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # noqa: F401
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.tasks.marvin_supermimic.physics_contract import marvin_se3_physics_config
from dexgym.tasks.marvin_supermimic.se3_env import MarvinSE3ReachHoldCfg
from dexgym.tasks.marvin_supermimic.se3_probe import (
    make_probe_transition, paired_probe_actions)


def collect_one(env, replay, *, epsilon: float, level: str):
    native = env.unwrapped
    robot, task = native.core_robot, native.se3_task
    actions, groups = paired_probe_actions(native.num_envs, robot.spec.action_dim,
                                            epsilon=epsilon, device=native.device)
    sample = native.goal_sampler.sample(int(groups[-1]) + 1, level)
    start_q = sample.start_q[groups]
    goal_pose = sample.goal_pose[groups]
    env_ids = torch.arange(native.num_envs, device=native.device)
    full_q = native.home_full[None].expand(native.num_envs, -1).clone()
    full_qd = torch.zeros_like(full_q)
    native._write_robot_reset(env_ids, full_q, full_qd, start_q,
                              torch.zeros_like(start_q))
    native.goal_pose[:] = goal_pose
    native.goal_frame.zero_()
    native.episode_length_buf.zero_()
    native._contact_accum.zero_()
    state = robot.physical_state(native).detach().clone()
    task_state = native.task_state()
    previous_action = native.last_action.detach().clone()
    first_observation = task.observation(state, previous_action, task_state)
    native.se3_history[:] = first_observation[:, None].expand(
        -1, task.spec.history_length, -1)
    history = native.se3_history.detach().clone()
    context = robot.dynamics_context(native).detach().clone()
    _, physical_reward, terminated, truncated, _ = env.step(actions.clone())
    if bool(truncated.any()):
        raise RuntimeError("One-step Newton action probe unexpectedly timed out")
    transition = make_probe_transition(
        robot, task, state=state, action=actions,
        previous_action=previous_action,
        next_state=native.last_physical_arm_state.detach().clone(),
        history=history, task_state=task_state,
        dynamics_context=context, terminated=terminated.detach().clone(),
        failure_reason=native.last_failure_reason.detach().clone())
    if not torch.allclose(transition.reward, physical_reward, atol=5e-4, rtol=5e-3):
        raise RuntimeError("Probe Task reward differs from real Newton reward")
    if not torch.allclose(transition.executed_command,
                          transition.next_state[:, 28:42], atol=1e-6, rtol=1e-6):
        raise RuntimeError("Probe action mapping differs from Newton command")
    replay.append(transition)
    group_size = 2 * robot.spec.action_dim + 2
    complete = native.num_envs // group_size
    baselines = torch.arange(complete, device=native.device) * group_size
    baseline_reward_difference = (
        transition.reward[baselines] - transition.reward[baselines + group_size - 1]
    ).abs()
    state_spread = torch.stack([
        (state[groups == index] - state[groups == index][:1]).abs().max()
        for index in range(complete)])
    return {
        "epsilon_raw": epsilon,
        "complete_paired_start_groups": complete,
        "physical_transitions": native.num_envs,
        "hard_terminations": int(transition.terminated.sum()),
        "probe_truncations": int(transition.truncated.sum()),
        "maximum_pre_action_group_state_spread": float(state_spread.max()),
        "maximum_paired_baseline_reward_difference": float(baseline_reward_difference.max()),
        "mean_position_difficulty_m": sample.position_difficulty_m.mean(0).tolist(),
        "mean_rotation_difficulty_rad": sample.rotation_difficulty_rad.mean(0).tolist(),
        "source_id": 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--level", choices=("near",), default="near")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 30 <= args.num_envs <= 4096:
        parser.error("Probe collection requires 30-4096 Newton environments")
    if args.output.exists():
        parser.error("Probe output already exists; preserve prior physical data")
    cfg = MarvinSE3ReachHoldCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.goal_train_seed = args.seed + 101
    cfg.goal_eval_seed = args.seed + 100101
    cfg.audit_contacts_each_physics_step = True
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            robot, task = native.core_robot, native.se3_task
            train = SchemaReplay(2, args.num_envs, robot.spec, task.spec, native.device)
            validation = SchemaReplay(1, args.num_envs, robot.spec, task.spec, native.device)
            report = {"kind": "strict_pure_newton_paired_action_probe",
                      "seed": args.seed, "num_envs": args.num_envs,
                      "level": args.level, "teacher_or_ik_used": False,
                      "goal_joint_witness_stored": False,
                      "robot": asdict(robot.spec), "task": asdict(task.spec),
                      "robot_contract": robot.checkpoint_contract(),
                      "task_contract": task.checkpoint_contract(),
                      "physics_config": marvin_se3_physics_config(cfg, robot),
                      "train": [], "validation": []}
            env.reset()
            for epsilon in (0.1, 0.5):
                report["train"].append(collect_one(env, train, epsilon=epsilon,
                                                    level=args.level))
            report["validation"].append(collect_one(env, validation,
                                                     epsilon=0.1, level=args.level))
            args.output.mkdir(parents=True)
            torch.save({"train": train.state_dict(),
                        "validation": validation.state_dict(),
                        "metadata": report}, args.output / "probes.pt")
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
