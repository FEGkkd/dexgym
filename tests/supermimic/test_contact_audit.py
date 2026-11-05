"""A collision in one Newton world must not terminate its neighbors."""

import pytest
import torch
from dexgym.tasks.marvin_supermimic.contact_audit import contact_counts, unsafe_arm_states


def test_contact_counts_keep_worlds_isolated_and_ignore_unused_slots():
    world_ids = torch.tensor([1, 1, 3, 0, 0, 0], dtype=torch.int32)
    counts = contact_counts(world_ids, active_count=3, num_envs=4)
    assert counts.tolist() == [0, 2, 0, 1]
    assert counts.gt(0).tolist() == [False, True, False, True]


def test_contact_counts_reject_invalid_active_world_id():
    with pytest.raises(ValueError, match="world"):
        contact_counts(torch.tensor([0, 4], dtype=torch.int32), active_count=2, num_envs=4)


def test_contact_counts_reject_active_count_past_buffer():
    with pytest.raises(ValueError, match="count"):
        contact_counts(torch.tensor([0, 1], dtype=torch.int32), active_count=3, num_envs=4)


def test_only_contacting_environment_terminates_when_arm_states_are_legal():
    q = torch.zeros(3, 14)
    qd = torch.zeros_like(q)
    tcp_error = torch.zeros(3, 2)
    counts = contact_counts(torch.tensor([1], dtype=torch.int32), active_count=1, num_envs=3)
    unsafe = unsafe_arm_states(q, qd, -torch.ones(14), torch.ones(14), tcp_error, counts)
    assert unsafe.tolist() == [False, True, False]


def test_tcp_and_speed_failures_remain_environment_local():
    q = torch.zeros(3, 14)
    qd = torch.zeros_like(q)
    qd[2, 5] = 5.1
    tcp_error = torch.zeros(3, 2)
    tcp_error[0, 1] = 0.121
    unsafe = unsafe_arm_states(q, qd, -torch.ones(14), torch.ones(14), tcp_error, torch.zeros(3))
    assert unsafe.tolist() == [True, False, True]
