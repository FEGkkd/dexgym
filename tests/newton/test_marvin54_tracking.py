"""Regression checks for the tracking phase and the shared replay split."""

from types import SimpleNamespace

import torch

from dexgym.algorithms.supermimic.dex_reference.buffer import ReplayBuffer
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine
from dexgym.tasks.direct.marvin54_tracking.policy import MotionTrackingActor
from dexgym.tasks.direct.marvin54_tracking.task import MotionLibrary, TrackingTask


def test_home_action_does_not_depend_on_reference_phase():
    task = TrackingTask.__new__(TrackingTask)
    task.action_mode = "home"
    task.home = torch.linspace(-0.5, 0.5, 54)
    task.nominal = task.home
    task.lo, task.hi = torch.full((54,), -1.0), torch.full((54,), 1.0)
    task.scale = torch.maximum(task.home - task.lo, task.hi - task.home)
    memory = {"frame": torch.tensor([7, 19]), "action": torch.zeros(2, 54)}
    action = torch.zeros(2, 54)
    target, executed = task.control_target(action, memory, None)
    assert torch.equal(target, task.home.expand_as(target))
    assert torch.equal(executed, target)
    action[0] = 1.0
    action[1] = -1.0
    target, _ = task.control_target(action, memory, None)
    assert torch.allclose(target[0], task.hi)
    assert torch.allclose(target[1], task.lo)


def test_reference_preview_and_step_stay_inside_the_clip():
    library = MotionLibrary.__new__(MotionLibrary)
    library.end = torch.tensor([2, 2, 2, 5, 5, 5])
    task = TrackingTask.__new__(TrackingTask)
    task.library = library
    phase = torch.tensor([1, 2, 3, 4])
    memory = {"frame": phase, "action": torch.zeros(4, 54)}
    assert library.at(phase, 5).tolist() == [2, 2, 5, 5]
    assert task.current_goal(memory).tolist() == [2, 2, 4, 5]
    later = task.advance_memory(None, memory, memory["action"], None, None, None)
    assert task.decode_memory(task.encode_memory(later))["frame"].tolist() == [2, 2, 4, 5]
    assert torch.equal(later["frame"], task.replay_failure_memory(memory, memory["action"])["frame"])
    assert task.imagination_start_mask(memory, 2).tolist() == [False, False, True, False]
    large = {"frame": torch.tensor([2**24 + 1]), "action": torch.zeros(1, 54)}
    assert torch.equal(task.decode_memory(task.encode_memory(large))["frame"], large["frame"])


def test_minimum_length_clip_reserves_the_full_imagination_window():
    library = MotionLibrary.__new__(MotionLibrary)
    library.frames = 33
    library.end = torch.full((33,), 32)
    library.reset_limit = torch.zeros(33, dtype=torch.long)
    task = TrackingTask.__new__(TrackingTask)
    task.library = library
    assert task.imagination_start_mask({"frame": library.sample(128)}, 32).all()


def test_actor_imagination_starts_exclude_world_model_holdout():
    buffer = ReplayBuffer(capacity=4, num_envs=4, device="cpu", holdout_fraction=0.25)
    for i in range(7):
        buffer.add({"valid": torch.ones(4, dtype=torch.bool), "x": torch.full((4, 1), float(i))})
    engine = SuperMimicEngine.__new__(SuperMimicEngine)
    engine.buffer = buffer
    engine.device = "cpu"
    engine.cfg = SimpleNamespace(pi_batch=256)
    engine.task = SimpleNamespace(simulator_context=torch.arange(4)[:, None])
    starts = engine.sample_starts()
    assert starts["env"].max().item() == 2
    assert starts["x"].min().item() >= 3
    assert torch.equal(starts["sim_context"][:, 0], starts["env"])


def test_replay_failure_features_use_the_post_transition_phase():
    library = MotionLibrary.__new__(MotionLibrary)
    library.end = torch.full((6,), 5)
    library.qpos = torch.arange(6).float()[:, None].repeat(1, 54)
    library.qvel = torch.zeros_like(library.qpos)
    task = TrackingTask.__new__(TrackingTask)
    task.library = library
    engine = SuperMimicEngine.__new__(SuperMimicEngine)
    engine.task = task
    engine.fail_head = SimpleNamespace(features=lambda x, context: context)
    memory = {"frame": torch.tensor([2]), "action": torch.full((1, 54), -0.25)}
    action = torch.full((1, 1, 54), 0.5)
    replay = {
        "x1": torch.zeros(1, 1, 108), "m": task.encode_memory(memory)[:, None],
        "action": action, "aux": torch.zeros(1, 1, 1),
    }
    features = engine.replay_failure_features(replay, torch.zeros(1, 1))
    assert features[0, 0, 108].item() == 4
    assert features[0, 0, -1].item() == 0.5


def test_motion_actor_reconstructs_the_same_goals_from_replay_phase():
    library = MotionLibrary.__new__(MotionLibrary)
    library.end = torch.tensor([2, 2, 2, 5, 5, 5])
    library.qpos = torch.arange(6).float()[:, None].expand(6, 54)
    library.qvel = library.qpos + 100
    task = TrackingTask.__new__(TrackingTask)
    task.policy_arch = "motion-encoder"
    task.library = library
    task.history_steps, task.history_frame_dim, task.history_dim = 10, 162, 1620
    task.future_offsets = torch.arange(1, 11)
    history = torch.arange(1620).float().expand(2, -1).reshape(2, 10, 162)
    memory = {"frame": torch.tensor([0, 4])}
    actor_input = task.observation_from_history(history, memory)
    assert actor_input.shape == (2, 2700)
    goals = actor_input[:, 1620:].reshape(2, 10, 108)
    assert goals[0, :, 0].tolist() == [1.0] + [2.0] * 9
    assert goals[1, :, 54].tolist() == [105.0] * 10
    saved = task.replay_observation(actor_input)
    assert torch.equal(task.observation_from_history(task.history_from_obs(saved), memory), actor_input)


def test_motion_encoder_and_decoder_both_receive_actor_gradients():
    actor = MotionTrackingActor(1620, 1080, 54)
    obs = torch.randn(2, 2700)
    action = actor(obs)
    assert action.shape == (2, 54) and action.abs().max() <= 1
    action.square().sum().backward()
    for module in (actor.encoder, actor.decoder):
        assert module[0].weight.grad.abs().sum() > 0


def test_hard_contact_label_and_uncapped_imagination_probability():
    task = TrackingTask.__new__(TrackingTask)
    task.contact_dim = 3
    depth = torch.tensor([0.0, 0.0006, 0.003, 0.004])
    labels = task.contact_labels(depth)
    assert labels.tolist() == [[0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1]]
    assert torch.equal(task.imagined_contact_failure(labels * 0.75), labels[:, 2] * 0.75)
    per_channel = torch.tensor([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]])
    assert task.world_model_contact_loss(per_channel, torch.tensor([0.25, 1.0])).tolist() == [4.0, 7.0]
    task.contact_dim = 2
    assert task.imagined_contact_failure(task.contact_labels(depth)) is None
