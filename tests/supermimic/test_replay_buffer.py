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
