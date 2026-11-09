"""Task-specific non-regression: event timing, Actor causality and sparse model gradient."""

from types import SimpleNamespace

import torch
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.tasks.direct.right_hand_tracking.supermimic_robot import Hand21DynamicsAdapter
from dexgym.tasks.direct.right_hand_tracking.supermimic_task import Hand21Task
from dexgym.tasks.direct.right_hand_tracking.supermimic_model import HandWorldModel
from dexgym.tasks.direct.right_hand_tracking.subgoals import HandGoals, advance_hand_goals, paper_hand_reward


def setup():
    robot = Hand21DynamicsAdapter()
    q = (robot.fk.limits.sum(-1) / 2)[None].repeat(2, 1)
    bank = SimpleNamespace(lengths=torch.tensor([100, 100]),
        fingertips=lambda ids, frames: robot.fk(q[:len(ids)]),
        joints=lambda ids, frames: q[:len(ids)],
        sample_other_at_frame=lambda ids, frames: (1 - ids, torch.ones_like(ids, dtype=torch.bool)))
    cfg = SimpleNamespace(goal_tolerance_m=.015, curriculum_stride_min=20, curriculum_stride_max=40,
        curriculum_anneal_steps=25600, curriculum_sigma_min=.7, cross_trajectory_probability=1.,
        cross_trajectory_failure_budget=300, failure_tolerance_scale=1.5, dwell_min_steps=5,
        dwell_max_steps=15, reward_dense_scale=.1, reward_time_cost=.1, episode_length_s=30.)
    z = torch.zeros(2, dtype=torch.long)
    goals = HandGoals(z.clone(), z + 10, z + 4, z + 4, z + 5, z.clone(), (z+6).float(),
                      z.clone(), z.clone(), z.clone(), z.bool())
    ts = TaskState(z.clone(), z.clone(), z.clone(), goals.pack())
    state = torch.cat((q, torch.zeros_like(q), q), -1)
    return robot, Hand21Task(robot, bank, cfg), goals, ts, state


def test_cross_motion_switch_preserves_old_reward_and_control_clock():
    robot, task, goals, ts, state = setup()
    result = advance_hand_goals(robot.fk(state[:, :20]), goals, task.bank, task.cfg)
    assert result.crossed.all()
    assert result.goals.motion_ids.tolist() == [1, 1]
    assert result.goals.goal_frame.tolist() == [10, 10]
    assert result.goals.fail_budget.tolist() == [300., 300.]
    assert goals.goal_span.tolist() == [4, 4]  # input is immutable
    # Complete hand dense reward is .52-.1 at zero error; sparse is 9*3.75.
    torch.testing.assert_close(task.exact_reward(state, ts), torch.full((2,), 34.17))


def test_actor_does_not_read_reference_joint_values_or_privileged_counters():
    robot, task, goals, ts, state = setup()
    action = torch.zeros(2, 20)
    before = task.observation(state, action, ts)
    task.bank.joints = lambda *args: (_ for _ in ()).throw(AssertionError('q_ref leaked to Actor'))
    changed = ts.data.clone()
    changed[:, 2:] += 3
    other = TaskState(ts.task_id, ts.motion_id, ts.frame + 50, changed)
    torch.testing.assert_close(task.observation(state, action, other), before)
    assert task.goal(state, other).shape == (2, 1, 15)


def test_learned_reach_is_differentiable_but_real_success_remains_exact():
    robot, task, goals, ts, state = setup()
    config = NetworkSpec(world_hidden=(32,), goal_hidden=(32,), actor_hidden=(32,), critic_hidden=(32,))
    model = HandWorldModel(robot.spec, config, task)
    moved = state.clone()
    moved[:, :20] += .1
    moved.requires_grad_(True)
    step = task.transition(state, moved, state[:, :20]*0, ts, state[:, :20]*0)
    gradient, = torch.autograd.grad(step.reward.sum(), moved)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert step.next_task_state.frame.tolist() == [1, 1]
    assert task.critic_features(state, ts).shape == (2, 162)
    assert task.termination_features(state, ts).shape == (2, 26)


def test_physics_threshold_labels_survive_submicron_fk_rounding():
    robot, task, goals, ts, state = setup()
    errors = torch.full((2, 5), .015 - 1e-7)
    actual = task.exact_reward(state, ts, fingertip_errors=errors)
    errors_outside = torch.full((2, 5), .015 + 1e-7)
    missed = task.exact_reward(state, ts, fingertip_errors=errors_outside)
    assert torch.all(actual - missed > 8.)
    expected = paper_hand_reward(errors, torch.ones(2, dtype=torch.bool), goals.goal_span,
                                 robot.fk.landmarks(state[:, :20]), task.goal_landmarks(ts))
    torch.testing.assert_close(actual, expected)
