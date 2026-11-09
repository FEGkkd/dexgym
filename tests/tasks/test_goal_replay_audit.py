"""Privileged replay outcomes use the active goal's timing and failure budget."""

import torch

from dexgym.algorithms.goal_replay_audit import GoalReplayAudit


def test_span_one_second_outside_frame_times_out_before_dwell() -> None:
    audit = GoalReplayAudit(num_envs=2, device="cpu")
    first = audit.record_step(
        span=torch.tensor([1, 2]),
        age=torch.tensor([0, 0]),
        within=torch.tensor([False, True]),
        reached=torch.tensor([False, False]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        fail_frames=torch.tensor([1, 0]),
        fail_budget=torch.tensor([1.5, 3.0]),
        dwell_required=torch.tensor([5, 5]),
    )
    assert first == []
    second = audit.record_step(
        span=torch.tensor([1, 2]),
        age=torch.tensor([1, 1]),
        within=torch.tensor([False, True]),
        reached=torch.tensor([False, True]),
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, False]),
        fail_frames=torch.tensor([2, 0]),
        fail_budget=torch.tensor([1.5, 3.0]),
        dwell_required=torch.tensor([5, 2]),
    )
    assert [(x["span"], x["outcome"], x["elapsed_steps"], x["first_within_step"]) for x in second] == [
        (1, "budget_timeout", 2, None),
        (2, "reached", 2, 1),
    ]
    assert audit.first_within_step.tolist() == [0, 0]


def test_cross_target_tracks_first_entry_until_reached() -> None:
    audit = GoalReplayAudit(num_envs=1, device="cpu")
    audit.record_step(
        span=torch.tensor([0]), age=torch.tensor([0]), within=torch.tensor([True]),
        reached=torch.tensor([False]), terminated=torch.tensor([False]), truncated=torch.tensor([False]),
        fail_frames=torch.tensor([0]), fail_budget=torch.tensor([300.0]), dwell_required=torch.tensor([2]),
    )
    events = audit.record_step(
        span=torch.tensor([0]), age=torch.tensor([1]), within=torch.tensor([True]),
        reached=torch.tensor([True]), terminated=torch.tensor([False]), truncated=torch.tensor([False]),
        fail_frames=torch.tensor([0]), fail_budget=torch.tensor([300.0]), dwell_required=torch.tensor([2]),
    )
    assert [(x["outcome"], x["elapsed_steps"], x["first_within_step"]) for x in events] == [
        ("reached", 2, 1)
    ]
