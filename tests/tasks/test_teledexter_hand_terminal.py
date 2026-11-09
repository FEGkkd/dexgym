"""Paper-style failure budgets and cross-trajectory switches act on the current transition."""

from types import SimpleNamespace

import torch
from dexgym.tasks.direct.hand21_ppo.env import Hand21PPOEnv
from dexgym.tasks.direct.hand21_ppo.env_cfg import WujiHand21TeleDexterCfg


def _task(*, cross_probability: float) -> SimpleNamespace:
    cfg = WujiHand21TeleDexterCfg()
    cfg.cross_trajectory_probability = cross_probability
    q = torch.zeros((2, 20))
    return SimpleNamespace(
        cfg=cfg,
        num_envs=2,
        device="cpu",
        robot=SimpleNamespace(data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=q),
            joint_vel=SimpleNamespace(torch=torch.zeros_like(q)),
        )),
        bank=SimpleNamespace(
            lengths=torch.tensor([100, 100]),
            fingertips=lambda ids, frames: torch.zeros((len(ids), 5, 3)),
            sample_other_at_frame=lambda ids, frames: (1 - ids, torch.ones_like(ids, dtype=torch.bool)),
        ),
        motion_ids=torch.zeros(2, dtype=torch.long),
        goal_frame=torch.full((2,), 10, dtype=torch.long),
        goal_span=torch.ones(2, dtype=torch.long),
        goal_tips=torch.zeros((2, 5, 3)),
        dwell=torch.zeros(2, dtype=torch.long),
        dwell_required=torch.full((2,), 5, dtype=torch.long),
        fail_frames=torch.zeros(2, dtype=torch.long),
        fail_budget=torch.full((2,), 1.5),
        curriculum_steps=torch.zeros(2, dtype=torch.long),
        goal_age=torch.zeros(2, dtype=torch.long),
        goals_reached=torch.zeros(2, dtype=torch.long),
        completed=torch.zeros(2, dtype=torch.bool),
        last_reached=torch.zeros(2, dtype=torch.bool),
        last_crossed=torch.zeros(2, dtype=torch.bool),
        joint_limits=torch.tensor([[-1.0, 1.0]] * 20),
        right_ids=list(range(20)),
        episode_length_buf=torch.ones(2, dtype=torch.long),
        max_episode_length=900,
        extras={},
    )


def test_failure_budget_terminates_on_the_frame_that_exceeds_it():
    task = _task(cross_probability=0.0)
    task.fail_frames[0] = 1
    current = torch.zeros((2, 5, 3))
    current[0, 0, 0] = 0.03
    task._current_tips = lambda: current
    terminated, truncated = Hand21PPOEnv._get_dones_teledexter(task)
    assert terminated.tolist() == [True, False]
    assert not truncated.any()
    assert task.fail_frames.tolist() == [2, 0]


def test_cross_switch_keeps_frame_and_grants_paper_failure_budget():
    task = _task(cross_probability=1.0)
    task.dwell[:] = 4
    task._current_tips = lambda: torch.zeros((2, 5, 3))
    terminated, _ = Hand21PPOEnv._get_dones_teledexter(task)
    assert not terminated.any()
    assert task.last_crossed.tolist() == [True, True]
    assert task.motion_ids.tolist() == [1, 1]
    assert task.goal_frame.tolist() == [10, 10]
    assert task.fail_budget.tolist() == [300.0, 300.0]
    assert torch.all(task._step_reward > 0)
    assert torch.all((task.dwell_required >= 5) & (task.dwell_required <= 15))


def test_sparse_reward_belongs_to_goal_completed_before_switch():
    task = _task(cross_probability=1.0)
    task.goal_span[:] = 4
    task.dwell[:] = task.dwell_required - 1
    task._current_tips = lambda: torch.zeros((2, 5, 3))

    Hand21PPOEnv._get_dones_teledexter(task)
    torch.testing.assert_close(task._step_reward, torch.full((2,), 33.75))
    assert task.last_crossed.tolist() == [True, True]
    assert task.goal_span.tolist() == [0, 0]

    task.cfg.cross_trajectory_probability = 0.0
    task.dwell[:] = task.dwell_required - 1
    Hand21PPOEnv._get_dones_teledexter(task)
    torch.testing.assert_close(task._step_reward, torch.full((2,), 375.0))
    assert task.last_crossed.tolist() == [False, False]
