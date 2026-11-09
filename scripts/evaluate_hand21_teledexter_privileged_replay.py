"""Audit privileged q_ref control under the current Hand 2.1 TELEDEXTER task."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
from dexgym.algorithms.goal_replay_audit import GoalReplayAudit
from dexgym.tasks import HAND21_TELEDEXTER_ID
from dexgym.tasks.direct.hand21_ppo.logic import action_toward_joint_target
from dexgym.tasks.direct.hand21_ppo.env_cfg import WujiHand21TeleDexterCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))]


def summarize(events: list[dict], *, censored: int, initial_coverage: int, motion_count: int) -> dict:
    groups = defaultdict(list)
    for event in events:
        groups[event["span"]].append(event)
    by_span = {}
    for span, rows in sorted(groups.items()):
        outcomes = {key: sum(row["outcome"] == key for row in rows) for key in (
            "reached", "budget_timeout", "other_termination", "episode_truncation"
        )}
        reached = [row for row in rows if row["outcome"] == "reached"]
        first = [row["first_within_step"] for row in reached if row["first_within_step"] is not None]
        full = [row["elapsed_steps"] for row in reached]
        by_span[str(span)] = dict(
            count=len(rows), **outcomes,
            timeout_rate=outcomes["budget_timeout"] / len(rows),
            success_rate=outcomes["reached"] / len(rows),
            first_within_p50=percentile(first, 0.5), first_within_p90=percentile(first, 0.9),
            full_reach_p50=percentile(full, 0.5), full_reach_p90=percentile(full, 0.9),
            reached_within_budget=sum(row["first_within_step"] is not None for row in reached),
        )
    return dict(
        completed_goals=len(events), censored_goals=censored,
        initial_motion_coverage=initial_coverage, motion_count=motion_count,
        by_span=by_span,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--stage", choices=("initial", "full"), required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    torch.manual_seed(args.seed)
    load_external_tasks()
    cfg = WujiHand21TeleDexterCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.split = args.split
    cfg.seed = args.seed
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    if abs(cfg.decimation * cfg.sim.dt - 1 / 30) > 1e-9:
        raise AssertionError("Replay requires the real 30-Hz policy loop")
    args.output.mkdir(parents=True, exist_ok=True)
    event_path = args.output / f"{args.split}_{args.stage}_events.csv"
    summary_path = args.output / f"{args.split}_{args.stage}_summary.json"
    with launch_simulation(cfg):
        env = gym.make(HAND21_TELEDEXTER_ID, cfg=cfg)
        try:
            task = env.unwrapped
            if args.num_envs < task.bank.motion_count:
                raise ValueError("Use at least one environment per motion for initial coverage")
            if args.stage == "full":
                task.curriculum_steps[:] = cfg.curriculum_anneal_steps
            # Cover every segment on the first explicit reset; later resets use
            # the task's ordinary random RSI sampling, curriculum, and switching.
            sampler = task.bank.sample_reference_starts
            used = False

            def initially_balanced(count, *, first_fraction=0.9):
                nonlocal used
                if used:
                    return sampler(count, first_fraction=first_fraction)
                used = True
                ids = torch.arange(count, device=task.device) % task.bank.motion_count
                choices = torch.minimum(
                    torch.ceil(task.bank.lengths[ids] * first_fraction).long(),
                    task.bank.lengths[ids] - 1,
                )
                frames = torch.floor(torch.rand(count, device=task.device) * choices).long()
                return ids, frames

            task.bank.sample_reference_starts = initially_balanced
            env.reset()
            task.bank.sample_reference_starts = sampler
            audit = GoalReplayAudit(task.num_envs, task.device)
            decision = {}
            original_get_dones = task._get_dones

            def audited_get_dones():
                old = dict(
                    span=task.goal_span.clone(), age=task.goal_age.clone(),
                    budget=task.fail_budget.clone(), dwell=task.dwell_required.clone(),
                    motion_id=task.motion_ids.clone(), frame=task.goal_frame.clone(),
                )
                within = ((task._current_tips() - task.goal_tips).norm(dim=-1)
                          < cfg.goal_tolerance_m).all(dim=-1)
                terminated, truncated = original_get_dones()
                decision.clear()
                decision.update(old, within=within, reached=task.last_reached.clone(),
                                fail_frames=task.fail_frames.clone(),
                                terminated=terminated.clone(), truncated=truncated.clone())
                return terminated, truncated

            task._get_dones = audited_get_dones
            events = []
            fieldnames = ["global_step", "env_id", "motion_id", "goal_frame", "span", "outcome",
                          "elapsed_steps", "first_within_step", "fail_frames", "fail_budget", "dwell_required"]
            with event_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for step in range(args.steps):
                    desired = task.bank.joints(task.motion_ids, task.goal_frame)
                    action = action_toward_joint_target(
                        desired, task.target_q, scale=cfg.action_delta_rad,
                        deadzone=cfg.action_deadzone,
                    )
                    env.step(action)
                    step_events = audit.record_step(
                        span=decision["span"], age=decision["age"], within=decision["within"],
                        reached=decision["reached"], terminated=decision["terminated"],
                        truncated=decision["truncated"], fail_frames=decision["fail_frames"],
                        fail_budget=decision["budget"], dwell_required=decision["dwell"],
                    )
                    if step_events:
                        ids = torch.tensor([row["env_id"] for row in step_events], device=task.device)
                        motions = decision["motion_id"][ids].cpu().tolist()
                        frames = decision["frame"][ids].cpu().tolist()
                        for row, motion, frame in zip(step_events, motions, frames):
                            row.update(global_step=step + 1, motion_id=motion, goal_frame=frame)
                        writer.writerows(step_events)
                        events.extend(step_events)
                    if (step + 1) % 100 == 0:
                        print(f"REPLAY_PROGRESS split={args.split} stage={args.stage} "
                              f"step={step + 1}/{args.steps} goals={len(events)}", flush=True)
            summary = summarize(events, censored=task.num_envs,
                                initial_coverage=task.bank.motion_count,
                                motion_count=task.bank.motion_count)
            summary.update(split=args.split, stage=args.stage, seed=args.seed,
                           num_envs=args.num_envs, steps=args.steps,
                           event_file=str(event_path), dt_policy=cfg.decimation * cfg.sim.dt,
                           failure_tolerance_scale=cfg.failure_tolerance_scale,
                           dwell_range=[cfg.dwell_min_steps, cfg.dwell_max_steps],
                           cross_probability=cfg.cross_trajectory_probability)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"REPLAY_DONE {summary_path} goals={len(events)} "
                  f"span1={summary['by_span'].get('1')}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
