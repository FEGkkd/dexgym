"""Per-frame Hand 2.1 rewards and terminals use the current reference frame."""

from types import SimpleNamespace

import torch
from dexgym.tasks.direct.hand21_ppo.env import Hand21PPOEnv
from dexgym.tasks.direct.hand21_ppo.env_cfg import WujiHand21TeleDexterCfg


def _task() -> SimpleNamespace:
    cfg = WujiHand21TeleDexterCfg()
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
        dwell_required=torch.ones(2, dtype=torch.long),
        fail_frames=torch.zeros(2, dtype=torch.long),
        fail_budget=torch.full((2,), 45.0),
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
    task = _task()
    task.fail_frames[:] = torch.tensor([45, 44])
    current = torch.zeros((2, 5, 3))
    current[:, 0, 0] = 0.03
    task._current_tips = lambda: current
    terminated, truncated = Hand21PPOEnv._get_dones_teledexter(task)
    assert terminated.tolist() == [True, False]
    assert not truncated.any()
    assert task.fail_frames.tolist() == [46, 45]


def test_each_successful_frame_advances_without_random_cross_switch():
    task = _task()
    task._current_tips = lambda: torch.zeros((2, 5, 3))
    terminated, _ = Hand21PPOEnv._get_dones_teledexter(task)
    assert not terminated.any()
    assert task.last_crossed.tolist() == [False, False]
    assert task.motion_ids.tolist() == [0, 0]
    assert task.goal_frame.tolist() == [11, 11]
    assert task.fail_budget.tolist() == [45.0, 45.0]
    assert torch.all(task._step_reward > 0)
    assert task.dwell_required.tolist() == [1, 1]


def test_sparse_reward_belongs_to_scored_frame_before_advance():
    task = _task()
    task.bank.fingertips = lambda ids, frames: torch.where(
        (frames == 10)[:, None, None], torch.zeros((len(ids), 5, 3)), torch.full((len(ids), 5, 3), 0.03)
    )
    task._current_tips = lambda: torch.zeros((2, 5, 3))

    Hand21PPOEnv._get_dones_teledexter(task)
    torch.testing.assert_close(task._step_reward, torch.full((2,), 22.5))
    assert task.goal_frame.tolist() == [11, 11]
    assert task.goal_span.tolist() == [1, 1]

    Hand21PPOEnv._get_dones_teledexter(task)
    assert torch.all(task._step_reward < 22.5)
    assert task.last_reached.tolist() == [False, False]
