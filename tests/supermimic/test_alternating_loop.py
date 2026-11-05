"""The aligned trainer must repeatedly interleave physics and learning."""

import torch
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.algorithms.supermimic.training_loop import AlternatingLoop, prepare_model_windows


def test_each_ready_iteration_collects_then_updates_model_actor_and_critic():
    events = []
    loop = AlternatingLoop(world_updates_per_iteration=1, actor_updates_per_iteration=1)
    for _ in range(3):
        loop.run_one(
            collect=lambda: events.append("Newton"),
            replay_ready=lambda: True,
            model_update=lambda: events.append("WM8"),
            actor_ready=lambda: True,
            actor_update=lambda: events.append("BPTT32"),
            critic_update=lambda: events.append("TDlambda"),
        )
    assert events == ["Newton", "WM8", "BPTT32", "TDlambda"] * 3
    assert loop.counts == {"collections": 3, "world_updates": 3, "actor_updates": 3, "critic_updates": 3}


def test_warmup_and_model_trust_gate_never_skip_real_collection():
    events = []
    loop = AlternatingLoop(world_updates_per_iteration=2, actor_updates_per_iteration=1)
    ready = False
    trusted = False
    callbacks = {
        "collect": lambda: events.append("Newton"),
        "replay_ready": lambda: ready,
        "model_update": lambda: events.append("WM8"),
        "actor_ready": lambda: trusted,
        "actor_update": lambda: events.append("BPTT32"),
        "critic_update": lambda: events.append("TDlambda"),
    }
    loop.run_one(**callbacks)
    ready = True
    loop.run_one(**callbacks)
    trusted = True
    loop.run_one(**callbacks)
    assert events == ["Newton", "Newton", "WM8", "WM8", "Newton", "WM8", "WM8", "BPTT32", "TDlambda"]


def test_model_update_waits_for_episode_safe_windows_after_physics_warmup():
    replay = MotionReplay(40, 2, "cpu")
    for t in range(40):
        replay.append(
            torch.zeros(2, 42),
            torch.zeros(2, 14),
            torch.zeros(2, 42),
            torch.zeros(2, 10, 56),
            torch.full((2,), t),
            torch.ones(2, dtype=torch.bool),
        )
    assert prepare_model_windows(replay, batch=4, updates=1, env_range=(0, 2)) is None
    for t in range(8):
        replay.append(
            torch.zeros(2, 42),
            torch.zeros(2, 14),
            torch.zeros(2, 42),
            torch.zeros(2, 10, 56),
            torch.full((2,), t),
            torch.zeros(2, dtype=torch.bool),
        )
    windows = prepare_model_windows(replay, batch=4, updates=1, env_range=(0, 2))
    assert windows is not None and len(windows) == 1
    assert windows[0]["state"].shape == (4, 8, 42)
