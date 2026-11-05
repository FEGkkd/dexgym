"""Fast-changing references expose one-frame mistakes at state arrival."""

import torch
import pytest

from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.algorithms.supermimic.transition import ReferenceTimeIndex, arrival_reference_features
from dexgym.motion.motion_manager import MotionManager


def test_reference_time_index_distinguishes_current_from_future():
    frame = ReferenceTimeIndex(torch.tensor([2, 5]), torch.tensor([9, 9]))
    torch.testing.assert_close(frame.next_frame(), torch.tensor([3, 6]))
    torch.testing.assert_close(frame.future_frames(3), torch.tensor([[3, 4, 5], [6, 7, 8]]))
    with pytest.raises(ValueError):
        ReferenceTimeIndex(torch.tensor([8]), torch.tensor([9])).next_frame()


def test_current_reference_is_not_next_frames_future_goal():
    motion = object.__new__(MotionManager)
    motion.device = torch.device("cpu")
    motion.length = 5
    motion.data = {
        "q_arm": torch.arange(5).float()[:, None].repeat(1, 14),
        "link_pos": torch.arange(5).float()[:, None, None, None].repeat(1, 2, 3, 3),
        "link_quat_xyzw": torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(5, 2, 3, 1),
        "link_linear_vel": torch.zeros(5, 2, 3, 3),
        "link_angular_vel": torch.zeros(5, 2, 3, 3),
    }
    motion.data["link_rotmat"] = torch.eye(3).repeat(5, 2, 3, 1, 1)
    frame = torch.tensor([1])
    assert motion.reference_at(frame)["link_pos"][0, 0, 0, 0] == 1
    assert motion.reference_features_at(frame)[0, 0] == 1
    assert motion.future_goal_at(frame, 1)[0, 0, 0] == 2
    assert motion.reference_features_at(frame + 1)[0, 0] == 2
    assert motion.future_goal_at(frame + 1, 1)[0, 0, 0] == 3
    assert arrival_reference_features(motion, frame)[0, 0] == 2


@pytest.mark.parametrize("valid_length", [1, 2, 3, 5, 8])
def test_masked_replay_retains_short_terminal_episode(valid_length):
    replay = MotionReplay(12, 1, "cpu")
    for step in range(valid_length):
        state = torch.full((1, 42), float(step))
        replay.append(
            state, torch.zeros(1, 14), state + 1,
            torch.zeros(1, 10, 56), torch.tensor([step]),
            torch.tensor([step == valid_length - 1]),
            terminated=torch.tensor([step == valid_length - 1]),
        )
    batch = replay.sample_windows(32, 8)
    assert batch["mask"].shape == (32, 8)
    assert (batch["mask"][:, 0] == 1).all()
    assert (batch["mask"].sum(1) <= valid_length).all()
    assert (batch["mask"].sum(1) >= 1).all()
    terminal = batch["terminated"] & batch["mask"].bool()
    assert (terminal.sum(1) <= 1).all()


def test_partial_reset_and_ring_wrap_never_join_episodes():
    replay = MotionReplay(8, 2, "cpu")
    for step in range(20):
        done = torch.tensor([step % 2 == 0, step == 17])
        state = torch.full((2, 42), float(step))
        replay.append(state, torch.zeros(2, 14), state + 1,
                      torch.zeros(2, 10, 56), torch.tensor([step, step]), done)
    batch = replay.sample_windows(128, 8)
    valid = batch["mask"].bool()
    assert ((batch["next_state"][..., 0] - batch["state"][..., 0] - 1)[valid] == 0).all()
    assert ((batch["frame"][:, 1:] - batch["frame"][:, :-1] - 1)[valid[:, 1:]] == 0).all()
    assert (batch["episode"][:, 0, None] == batch["episode"])[valid].all()
