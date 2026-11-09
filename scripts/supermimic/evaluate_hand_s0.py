"""Held-out deterministic Newton evaluation: real subgoals or all-clip fixed clock."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from isaaclab.app import launch_simulation
from dexgym.algorithms.goal_replay_audit import GoalReplayAudit
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.tasks.direct.right_hand_tracking.env import RightHandTrackingDirectEnv
from dexgym.tasks.direct.right_hand_tracking.subgoals import HandGoals, unsafe_hand_state
from dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg import WujiHand21SuperMimicCfg
from dexgym.tasks.direct.right_hand_tracking.supermimic_collector import HandNewtonCollector
from scripts.supermimic.train_hand_s0 import components, physics_contract


def summarize_errors(values):
    error = torch.cat(values)
    worst = error.max(-1).values
    return {'sampled_frames': len(error), 'mean_tip_error_mm': float(error.mean() * 1000),
            'p95_worst_finger_mm': float(torch.quantile(worst, .95) * 1000),
            'per_finger_mean_mm': (error.mean(0) * 1000).tolist(),
            'all_fingers_coverage': {str(mm): float((worst < mm / 1000).float().mean()) for mm in (5, 10, 15)},
            'per_finger_coverage': {str(mm): float((error < mm / 1000).float().mean()) for mm in (5, 10, 15)}}


class FixedClockHandEnv(RightHandTrackingDirectEnv):
    """Evaluation only; exactly one current target frame per control step.

    No dwell or tracking-error reset in this diagnostic. The separate subgoal
    evaluator below retains the actual training MDP without overrides.
    """
    evaluation_motion_ids = None

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self.evaluation_motion_ids is None:
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        motion = self.evaluation_motion_ids[env_ids]
        start = torch.zeros_like(motion)
        q = self.home_q[None].repeat(len(env_ids), 1)
        q[:, self.right_ids] = self.bank.joints(motion, start)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.robot.actuators.target_command.set_position_index(value=q, env_ids=env_ids)
        self.full_target[env_ids], self.target_q[env_ids] = q, q[:, self.right_ids]
        self.motion_ids[env_ids], self.goal_frame[env_ids] = motion, 1
        self.goal_tips[env_ids] = self.bank.fingertips(motion, start + 1)

    def _get_dones_teledexter(self):
        current = self._current_tips()
        self.fixed_clock_actual_tips = current.detach().clone()
        self.fixed_clock_target_tips = self.goal_tips.detach().clone()
        self.fixed_clock_errors = torch.linalg.vector_norm(current - self.goal_tips, dim=-1).detach().clone()
        q = self.robot.data.joint_pos.torch[:, self.right_ids]
        qd = self.robot.data.joint_vel.torch[:, self.right_ids]
        failed = unsafe_hand_state(q, qd, self.joint_limits)
        finished = self.goal_frame >= self.bank.lengths[self.motion_ids] - 1
        self._step_reward = torch.zeros(self.num_envs, device=self.device)
        self.goal_frame = torch.minimum(self.goal_frame + 1, self.bank.lengths[self.motion_ids] - 1)
        self.goal_tips = self.bank.fingertips(self.motion_ids, self.goal_frame)
        return failed, finished


@torch.no_grad()
def evaluate_subgoals(env, robot, task, actor, args):
    env.curriculum_steps[:] = env.cfg.curriculum_anneal_steps
    collector = HandNewtonCollector(env, robot, task, action_noise_std=0.)
    audit = GoalReplayAudit(env.num_envs, env.device)
    events, errors = [], []
    for _ in range(args.steps):
        batch = collector.step(actor)
        old = HandGoals.unpack(batch.task_state.data)
        next_goals = HandGoals.unpack(batch.next_task_state.data)
        within = (env.last_hand_errors < env.cfg.goal_tolerance_m).all(-1)
        events.extend(audit.record_step(
            span=old.goal_span, age=old.goal_age, within=within, reached=env.last_hand_reached,
            terminated=batch.terminated, truncated=batch.truncated, fail_frames=next_goals.fail_frames,
            fail_budget=old.fail_budget, dwell_required=old.dwell_required))
        errors.append(env.last_hand_errors.cpu())
    outcomes = {name: sum(e['outcome'] == name for e in events) for name in
                ('reached', 'budget_timeout', 'other_termination', 'episode_truncation')}
    settled = outcomes['reached'] + outcomes['budget_timeout'] + outcomes['other_termination']
    first = [e['first_within_step'] for e in events if e['outcome'] == 'reached' and e['first_within_step'] is not None]
    report = {**summarize_errors(errors), 'completed_targets': len(events), 'outcomes': outcomes,
              'censored_at_stop': env.num_envs, 'success_over_settled_targets': outcomes['reached'] / settled if settled else None,
              'median_first_within_steps_successes': float(torch.tensor(first).float().median()) if first else None,
              'success_over_all_finalized_targets': outcomes['reached'] / len(events) if events else None}
    with (args.output / 'goal_events.csv').open('w', newline='') as f:
        if events:
            writer = csv.DictWriter(f, fieldnames=list(events[0]))
            writer.writeheader()
            writer.writerows(events)
    return report


@torch.no_grad()
def evaluate_fixed_clock(env, robot, task, actor, args):
    errors, records = [], []
    expected_frames = int((env.bank.lengths - 1).sum())
    for start in range(0, env.bank.motion_count, env.num_envs):
        count = min(env.num_envs, env.bank.motion_count - start)
        ids = torch.arange(start, start + env.num_envs, device=env.device).clamp_max(env.bank.motion_count - 1)
        env.evaluation_motion_ids = ids
        env.reset()
        previous = torch.zeros((env.num_envs, 20), device=env.device)
        obs = task.observation(robot.physical_state(env), previous, task.physical_task_state(env))
        history = obs[:, None].expand(-1, task.spec.history_length, -1).clone()
        active = torch.arange(env.num_envs, device=env.device) < count
        seen = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        failed_clips = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        actual_trace, target_trace = [], []
        maximum_steps = int(env.bank.lengths[ids[:count]].max()) - 1
        for _ in range(maximum_steps):
            state = robot.physical_state(env)
            task_state = task.physical_task_state(env)
            action = actor(history, task.goal(state, task_state))
            _, _, terminated, truncated, _ = env.step(action.tanh().clone())
            errors.append(env.fixed_clock_errors[active].cpu())
            actual_trace.append(env.fixed_clock_actual_tips.cpu())
            target_trace.append(env.fixed_clock_target_tips.cpu())
            seen += active.long()
            failed_clips |= active & terminated
            active &= ~(terminated | truncated)
            if not active.any():
                break
            previous = torch.where((terminated | truncated)[:, None], 0., action)
            obs = task.observation(robot.physical_state(env), previous, task.physical_task_state(env))
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
        actual_trace, target_trace = torch.stack(actual_trace), torch.stack(target_trace)
        for lane in range(count):
            frames = int(seen[lane])
            lag_errors = [float(torch.linalg.vector_norm(
                actual_trace[lag:frames, lane] - target_trace[:frames-lag, lane], dim=-1).mean())
                for lag in range(min(6, frames))]
            records.append({'motion_id': start + lane, 'file': env.bank.records[start + lane]['file'],
                            'frames_evaluated': int(seen[lane]), 'frames_expected': int(env.bank.lengths[start + lane]) - 1,
                            'unsafe_failure': bool(failed_clips[lane]),
                            'best_alignment_lag_steps_0_to_5': min(range(len(lag_errors)), key=lag_errors.__getitem__)})
        print(json.dumps({'fixed_clock_clips_done': len(records), 'total': env.bank.motion_count}), flush=True)
    report = {**summarize_errors(errors), 'clips': len(records), 'expected_frames': expected_frames,
              'unsafe_failed_clips': sum(r['unsafe_failure'] for r in records),
              'median_best_alignment_lag_steps_0_to_5': float(torch.tensor([
                  r['best_alignment_lag_steps_0_to_5'] for r in records]).float().median()),
              'lag_note': 'Offline alignment diagnostic only; primary errors are uncompensated at current target time',
              'missing_frames_after_failure': expected_frames - sum(r['frames_evaluated'] for r in records)}
    with (args.output / 'clip_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/supermimic/hand_s0_train.json'))
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('subgoals', 'fixed_clock'), required=True)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--policy', choices=('learned', 'zero'), default='learned')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = json.loads(args.config.read_text())
    cfg = WujiHand21SuperMimicCfg()
    cfg.library_path, cfg.scene.num_envs, cfg.seed, cfg.split = str(args.library), args.num_envs, args.seed, 'validation'
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    torch.manual_seed(args.seed)
    cls = FixedClockHandEnv if args.mode == 'fixed_clock' else RightHandTrackingDirectEnv
    with launch_simulation(cfg):
        env = cls(cfg)
        try:
            robot, task, modules = components(env, experiment)
            checkpoint = CoreCheckpoint(git_commit='evaluation', physics_config=physics_contract(cfg, robot),
                reference_hashes={'manifest': hashlib.sha256(args.library.read_bytes()).hexdigest()})
            provenance = checkpoint.load_actor_for_evaluation(args.checkpoint, robot=robot, task=task, modules=modules)
            policy = (modules.actor if args.policy == 'learned' else
                      lambda history, goal: history.new_zeros((len(history), 20)))
            report = (evaluate_fixed_clock if args.mode == 'fixed_clock' else evaluate_subgoals)(env, robot, task, policy, args)
            report.update({'mode': args.mode, 'split': cfg.split, 'seed': cfg.seed, 'num_envs': env.num_envs,
                           'eligible_clips': env.bank.motion_count, 'provenance': provenance, 'exploration_noise': 0.,
                           'normalizers_updated': False, 'policy': args.policy})
            (args.output / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
