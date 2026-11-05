import torch
import pytest
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay


def test_eight_step_windows_do_not_cross_episode_or_wrap_boundary():
    replay = MotionReplay(48, 3, "cpu")
    for t in range(90):
        state = torch.full((3, 42), float(t))
        action = torch.zeros(3, 14)
        next_state = torch.full((3, 42), float(t + 1))
        done = torch.tensor([t % 13 == 12, t % 19 == 18, False])
        replay.append(state, action, next_state, torch.zeros(3, 10, 56), torch.full((3,), t), done)
    batch = replay.sample_windows(64, 8)
    assert batch["state"].shape == (64, 8, 42)
    valid = batch["mask"].bool()
    torch.testing.assert_close(batch["next_state"][valid, 0], batch["state"][valid, 0] + 1)
    assert ((batch["frame"][:, 1:] - batch["frame"][:, :-1] - 1)[valid[:, 1:]] == 0).all()
    assert replay.bytes_allocated < 1_000_000
    assert replay.sample_starts(4)["history"].shape == (4, 10, 56)
    phase = replay.sample_starts(16, frame_range=(70, 80))
    assert torch.all((phase["frame"] >= 70) & (phase["frame"] < 80))


def test_terminal_at_window_end_is_kept_for_termination_head_without_crossing_reset():
    replay = MotionReplay(40, 1, "cpu")
    for t in range(9):
        state = torch.full((1, 42), float(t))
        action = torch.zeros(1, 14)
        physical_next = torch.full((1, 42), float(t + 1 if t != 7 else 77))
        terminated = torch.tensor([t == 7])
        replay.append(
            state,
            action,
            physical_next,
            torch.zeros(1, 10, 56),
            torch.tensor([t]),
            terminated,
            terminated=terminated,
        )
    batch = replay.sample_windows(8, 8, frame_range=(0, 1))
    assert batch["frame"][:, 0].eq(0).all()
    assert batch["mask"].all()
    assert batch["terminated"][:, -1].all()
    assert not batch["terminated"][:, :-1].any()
    assert batch["next_state"][:, -1, 0].eq(77).all()


def test_rare_single_valid_window_is_sampled_without_rejection_failure():
    replay = MotionReplay(200, 32, "cpu")
    for t in range(200):
        done = torch.ones(32, dtype=torch.bool)
        if 100 <= t < 107:
            done[7] = False
        replay.append(
            torch.zeros(32, 42),
            torch.zeros(32, 14),
            torch.zeros(32, 42),
            torch.zeros(32, 10, 56),
            torch.full((32,), t),
            done,
        )
    batch = replay.sample_windows(128, 8, require_full=True)
    assert batch["frame"][:, 0].eq(100).all()
    assert batch["mask"].all()
    assert batch["terminated"][:, -1].all()


def test_full_window_validation_rejects_only_short_episodes():
    replay = MotionReplay(8, 1, "cpu")
    for t in range(8):
        replay.append(torch.zeros(1, 42), torch.zeros(1, 14), torch.zeros(1, 42),
                      torch.zeros(1, 10, 56), torch.tensor([t]), torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError):
        replay.sample_windows(1, 8, require_full=True)
