"""TELEDEXTER's hand-only curriculum, hold, sparse reward, and failure budget."""

import torch
from dexgym.tasks.direct.hand21_ppo.logic import (
    curriculum_stride_upper,
    sample_goal_span,
    teledexter_hand_dwell,
    teledexter_hand_reward,
    update_failure_count,
)


def test_cubic_curriculum_expands_uniformly_sampled_stride_bounds():
    progress = torch.tensor([0, 12800, 25600, 50000])
    upper = curriculum_stride_upper(progress)
    assert upper[0] == 20 and upper[2] == upper[3] == 40
    assert 20 < upper[1] < 40
    for _ in range(10):
        spans = sample_goal_span(upper)
        assert torch.all(spans >= 1) and torch.all(spans <= upper)


def test_per_goal_hold_duration_and_out_of_tolerance_failure_count():
    current = torch.zeros((2, 5, 3))
    target = torch.zeros_like(current)
    required = torch.tensor([5, 15])
    dwell = torch.zeros(2, dtype=torch.long)
    for _ in range(4):
        _, within, reached, dwell = teledexter_hand_dwell(current, target, dwell, required, tolerance_m=0.02)
        assert within.tolist() == [True, True] and not reached.any()
    _, within, reached, dwell = teledexter_hand_dwell(current, target, dwell, required, tolerance_m=0.02)
    assert reached.tolist() == [True, False]
    failures = update_failure_count(torch.tensor([3, 2]), within, reached)
    assert failures.tolist() == [0, 2]
    current[1, 0, 0] = 0.03
    _, within, reached, dwell = teledexter_hand_dwell(current, target, dwell, required, tolerance_m=0.02)
    assert dwell[1] == 0 and not within[1]
    assert update_failure_count(failures, within, reached).tolist() == [0, 3]


def test_sparse_bonus_uses_completed_goal_span_or_cross_trajectory_weight():
    errors = torch.zeros((2, 5))
    reward = teledexter_hand_reward(
        errors,
        reached=torch.tensor([True, True]),
        goal_span=torch.tensor([4, 0]),
    )
    torch.testing.assert_close(reward, torch.tensor([33.75, 375.0]))
