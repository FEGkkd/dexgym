import torch
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
    torch.testing.assert_close(batch["next_state"][:, :, 0], batch["state"][:, :, 0] + 1)
    torch.testing.assert_close(batch["frame"][:, 1:] - batch["frame"][:, :-1], torch.ones(64, 7, dtype=torch.long))
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
    batch = replay.sample_windows(8, 8)
    assert batch["frame"][:, 0].eq(0).all()
    assert batch["terminated"][:, -1].all()
    assert not batch["terminated"][:, :-1].any()
    assert batch["next_state"][:, -1, 0].eq(77).all()
