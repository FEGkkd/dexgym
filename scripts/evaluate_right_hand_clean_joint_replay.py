"""Run every clean motion through the actual 30 Hz task action loop."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from dexgym.tasks import RIGHT_HAND_CLEAN_ID
from dexgym.tasks.direct.right_hand_tracking.clean_env_cfg import RightHandCleanTrackingCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    torch.manual_seed(args.seed)
    load_external_tasks()
    cfg = RightHandCleanTrackingCfg()
    cfg.split = args.split
    cfg.seed = args.seed
    manifest = json.loads(Path(cfg.library_path).read_text())
    cfg.scene.num_envs = sum(m["split"] == args.split for m in manifest["motions"])
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    if cfg.decimation * cfg.sim.dt != 1 / 30:
        raise AssertionError("Joint Replay requires an exact 30 Hz policy loop")

    with launch_simulation(cfg):
        env = gym.make(RIGHT_HAND_CLEAN_ID, cfg=cfg)
        try:
            obs, _ = env.reset()
            task = env.unwrapped
            if obs["policy"].shape != (task.num_envs, 105) or env.action_space.shape[-1] != 20:
                raise AssertionError("Unexpected observation or action shape")
            ids = torch.arange(task.bank.motion_count, device=task.device)
            if task.num_envs != len(ids) or (task.bank.lengths <= cfg.subgoal_stride).any():
                raise AssertionError("Each clean motion needs an environment and at least one subgoal")

            # Assign one complete motion to each environment. The task's normal
            # random reset is replaced by its first reference frame only once.
            q0 = task.home_q[None].repeat(task.num_envs, 1)
            q0[:, task.right_ids] = task.bank.joints(ids, torch.zeros_like(ids))
            task.robot.write_joint_position_to_sim_index(position=q0)
            task.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
            task.robot.actuators.target_command.set_position_index(value=q0, full_data=True)
            task.scene.write_data_to_sim()
            task.sim.forward()
            task.scene.update(0.0)
            task.full_target[:] = q0
            task.target_q[:] = q0[:, task.right_ids]
            task.motion_ids[:] = ids
            task.goal_frame[:] = cfg.subgoal_stride
            task.goal_tips[:] = task.bank.fingertips(ids, task.goal_frame)
            task.dwell.zero_()
            task.goal_age.zero_()
            task.completed.zero_()
            task.goals_reached.zero_()
            task.previous_action.zero_()
            task.action_delta.zero_()
            task.episode_length_buf.zero_()
            initial_tip_error = (
                task._current_tips() - task.bank.fingertips(ids, torch.zeros_like(ids))
            ).norm(dim=-1).max().item()
            if initial_tip_error >= 0.005:
                raise AssertionError(f"Initial FK mismatch: {initial_tip_error:.6f} m")

            # Capture the task's decision before DirectRLEnv auto-resets any
            # terminal environment. This also audits the final subgoal.
            original_get_dones = task._get_dones
            decision = {}

            def audited_get_dones():
                before_frame = task.goal_frame.clone()
                before_dwell = task.dwell.clone()
                before_tips = task._current_tips().clone()
                before_goal = task.goal_tips.clone()
                terminated, truncated = original_get_dones()
                decision.update(
                    frame=before_frame,
                    dwell=before_dwell,
                    tips=before_tips,
                    goal=before_goal,
                    next_frame=task.goal_frame.clone(),
                    next_dwell=task.dwell.clone(),
                    terminated=terminated.clone(),
                    truncated=truncated.clone(),
                )
                return terminated, truncated

            task._get_dones = audited_get_dones

            done = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
            completed = torch.zeros_like(done)
            timed_out = torch.zeros_like(done)
            truncated_by_episode = torch.zeros_like(done)
            goals = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
            observed_dwell = torch.zeros_like(goals)
            steps = torch.zeros_like(goals)
            active_error_sum = 0.0
            active_error_count = 0
            max_error_m = 0.0
            with torch.inference_mode():
                for _ in range(args.max_steps):
                    active = ~done
                    if not active.any():
                        break
                    previous_frame = task.goal_frame.clone()
                    motion_ids = task.motion_ids.clone()
                    q_goal = task.bank.joints(motion_ids, previous_frame)
                    action = ((q_goal - task.target_q) / cfg.action_delta_rad).clamp(-1.0, 1.0)
                    action[~active] = 0.0
                    _, reward, terminated, truncated, _ = env.step(action)
                    if not torch.isfinite(reward).all():
                        raise AssertionError("Non-finite replay reward")
                    if not torch.equal(terminated, decision["terminated"]) or not torch.equal(
                        truncated, decision["truncated"]
                    ):
                        raise AssertionError("Returned terminal flags differ from the task decision")
                    reached = reward > 9.0  # sparse bonus is 10; dense term is at most 1.
                    if not torch.equal(decision["frame"][active], previous_frame[active]):
                        raise AssertionError("Task evaluated an unexpected subgoal frame")
                    tip_error = (decision["tips"] - decision["goal"]).norm(dim=-1)
                    within = (tip_error < cfg.goal_tolerance_m).all(dim=-1)
                    if not torch.equal(decision["dwell"][active], observed_dwell[active]):
                        raise AssertionError("Task dwell counter differs from independently observed history")
                    observed_dwell = torch.where(within, observed_dwell + 1, torch.zeros_like(observed_dwell))
                    expected_reached = observed_dwell >= cfg.goal_dwell_steps
                    if not torch.equal(reached[active], expected_reached[active]):
                        raise AssertionError("10 mm / two-step reach decision differs from observed fingertips")
                    expected_next_dwell = torch.where(reached, torch.zeros_like(observed_dwell), observed_dwell)
                    if not torch.equal(decision["next_dwell"][active], expected_next_dwell[active]):
                        raise AssertionError("Task dwell did not reset or advance with the observed fingertips")
                    observed_dwell = expected_next_dwell
                    expected_frame = torch.where(
                        reached,
                        torch.minimum(previous_frame + cfg.subgoal_stride, task.bank.lengths[motion_ids] - 1),
                        previous_frame,
                    )
                    if not torch.equal(decision["next_frame"][active], expected_frame[active]):
                        raise AssertionError("Subgoal advanced without success or failed to advance on success")
                    active_errors = tip_error[active]
                    active_error_sum += active_errors.mean(dim=-1).sum().item()
                    active_error_count += int(active.sum().item())
                    max_error_m = max(max_error_m, active_errors.max().item())
                    goals += (reached & active).long()
                    steps += active.long()
                    just_completed = active & reached & (previous_frame >= task.bank.lengths[motion_ids] - 1)
                    if (just_completed & ~terminated).any():
                        raise AssertionError("Final subgoal reached without same-step termination")
                    completed |= just_completed
                    timed_out |= active & terminated & ~just_completed
                    truncated_by_episode |= active & truncated
                    done |= active & (terminated | truncated)

            rows = []
            for i, record in enumerate(task.bank.records):
                rows.append(
                    {
                        "motion": record["source"],
                        "frames": int(task.bank.lengths[i].item()),
                        "steps": int(steps[i].item()),
                        "goals_reached": int(goals[i].item()),
                        "completed": bool(completed[i].item()),
                        "terminated_without_completion": bool(timed_out[i].item()),
                        "truncated_by_episode": bool(truncated_by_episode[i].item()),
                        "unfinished_at_step_cap": bool((~done[i]).item()),
                    }
                )
            summary = {
                "task_id": RIGHT_HAND_CLEAN_ID,
                "split": args.split,
                "seed": args.seed,
                "num_motions": len(rows),
                "physics_hz": 480,
                "policy_hz": 30,
                "decimation": cfg.decimation,
                "usd": cfg.scene.robot.spawn.usd_path,
                "library_path": cfg.library_path,
                "goal_tolerance_mm": cfg.goal_tolerance_m * 1000,
                "goal_dwell_steps": cfg.goal_dwell_steps,
                "max_goal_steps": cfg.max_goal_steps,
                "subgoal_stride": cfg.subgoal_stride,
                "initial_fk_max_error_mm": initial_tip_error * 1000,
                "total_goals_reached": int(goals.sum().item()),
                "motions_with_goal_reached": int((goals > 0).sum().item()),
                "completed_motions": int(completed.sum().item()),
                "terminated_without_completion": int(timed_out.sum().item()),
                "truncated_by_episode": int(truncated_by_episode.sum().item()),
                "unfinished_at_step_cap": int((~done).sum().item()),
                "active_mean_tip_error_mm": 1000 * active_error_sum / max(active_error_count, 1),
                "active_max_tip_error_mm": 1000 * max_error_m,
                "motions": rows,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({k: v for k, v in summary.items() if k != "motions"}, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
