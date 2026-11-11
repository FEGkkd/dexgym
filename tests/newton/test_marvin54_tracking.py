"""Regression checks for the tracking phase and the shared replay split."""

from types import SimpleNamespace

import pytest
import torch

from dexgym.algorithms.supermimic.dex_reference.buffer import ReplayBuffer
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine
from dexgym.algorithms.supermimic.dex_reference.networks import FailureHead
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


def test_first_event_action_has_gradient_on_both_sides_of_home():
    task = TrackingTask.__new__(TrackingTask)
    task.action_mode = "home"
    task.first_event = True
    task.home = task.nominal = torch.full((54,), 0.8)
    task.lo, task.hi = torch.full((54,), -1.0), torch.full((54,), 1.0)
    task.lower_travel, task.upper_travel = task.home - task.lo, task.hi - task.home
    task.scale = torch.maximum(task.lower_travel, task.upper_travel)
    action = torch.stack((torch.full_like(task.home, -0.8), torch.full_like(task.home, 0.8))).requires_grad_()
    target, _ = task.control_target(action, None, None)
    target.sum().backward()
    assert torch.allclose(action.grad[0], task.lower_travel)
    assert torch.allclose(action.grad[1], task.upper_travel)
    task.library = MotionLibrary.__new__(MotionLibrary)
    task.library.end = torch.tensor([1, 1])
    task.library.qpos = target.detach()
    memory = {"frame": torch.tensor([0])}
    inverse = task.baseline_action(memory)
    assert torch.allclose(task.control_target(inverse, memory, None)[0], target[1:2])


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
    engine.transition_termination = False
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


def test_termination_head_uses_state_and_executed_servo_target():
    engine = SuperMimicEngine.__new__(SuperMimicEngine)
    engine.transition_termination = True
    engine.fail_head = FailureHead(context_dim=54, state_dim=108)
    x = torch.randn(2, 3, 108)
    u = torch.randn(2, 3, 54)
    features = engine.replay_failure_features({"x": x, "u": u}, None)
    assert torch.equal(features[..., :108], x)
    assert torch.equal(features[..., 108:], u)
    assert engine.fail_head(features).shape == (2, 3)


def test_contact_reward_labels_and_clip_end():
    depth = torch.tensor([0.0, 0.0006, 0.003, 0.004])
    assert TrackingTask.contact_labels(depth).tolist() == [[0, 0], [1, 1], [1, 1], [1, 1]]
    task = TrackingTask.__new__(TrackingTask)
    task.transition_termination = True
    task.library = MotionLibrary.__new__(MotionLibrary)
    task.library.end = torch.tensor([2, 2, 2])
    memory = {"frame": torch.tensor([0, 1])}
    assert task.termination(torch.zeros(2, 108), memory).tolist() == [0.0, 1.0]


def test_tracking_termination_uses_worst_wrist_and_known_clip_end():
    task = TrackingTask.__new__(TrackingTask)
    task.first_event = True
    task.tracking_failure_m = 0.25
    task.library = MotionLibrary.__new__(MotionLibrary)
    task.library.end = torch.tensor([2, 2, 2])
    task.library.ref_wrist_pos = torch.zeros(3, 2, 3)
    task.geometry = SimpleNamespace(pose=lambda q: (q[:, :6].reshape(-1, 2, 3), None, None, None, None))
    x = torch.zeros(2, 108)
    x[0, 3] = 0.25
    memory = {"frame": torch.tensor([0, 1])}
    assert task.wrist_tracking_error(x, memory).tolist() == [0.25, 0.0]
    assert task.termination(x, memory).tolist() == [0.5, 1.0]


def test_first_safety_substep_latches_each_environment():
    pytest.importorskip("isaaclab")
    from dexgym.tasks.direct.marvin54_tracking.native import (
        COLLISION, JOINT_LIMIT, NONFINITE, SUBSTEPS, NewtonTrackingEnv,
    )

    task = TrackingTask.__new__(TrackingTask)
    task.first_event = True
    task.joint_lo, task.joint_hi = torch.full((54,), -0.5), torch.full((54,), 0.5)
    env = NewtonTrackingEnv.__new__(NewtonTrackingEnv)
    env.task, env.device, env.num_envs = task, "cpu", 4
    env._state = torch.zeros(4, 108)
    tick = 0

    def step(render):
        nonlocal tick
        tick += 1

    def physical_state():
        q = torch.full((4, 108), tick / 100)
        if tick == 3:
            q[2, 0] = 0.6
        if tick == 1:
            q[3, 0] = float("nan")
        return q

    env.sim = SimpleNamespace(step=step)
    env.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None)
    env._physics_state = physical_state
    env._actuator_effort = lambda: torch.full((4, 54), float(tick))
    env.contact_depth = lambda: torch.tensor([0.004 if tick == 2 else 0.005 if tick > 2 else 0.0,
                                             0.003, 0.0, 0.0])
    state, torque, depth, substeps, flags = env._physics_interval()
    assert tick == SUBSTEPS
    assert substeps.tolist() == [2, 0, 3, 1]
    assert flags.tolist() == [COLLISION, 0, JOINT_LIMIT, NONFINITE]
    assert torch.allclose(depth, torch.tensor([0.004, 0.003, 0.0, 0.0]))
    assert torch.allclose(state[:3, 1], torch.tensor([0.02, SUBSTEPS / 100, 0.03]))
    assert torque[:, 0].tolist() == [2.0, float(SUBSTEPS), 3.0, 1.0]


def test_partial_safety_event_remains_a_positive_replay_sample():
    buffer = ReplayBuffer(capacity=3, num_envs=1, device="cpu", holdout_fraction=0)
    for safety, full_step in ((True, False), (False, True)):
        buffer.add({
            "valid": torch.tensor([True]), "done": torch.tensor([True]), "terminated": torch.tensor([True]),
            "safety_failure": torch.tensor([safety]), "full_step": torch.tensor([full_step]),
        })
    sampled = buffer.sample(batch=8, failure_frac=1.0, failure_horizon=1)
    assert sampled["safety_failure"].all()
    assert not sampled["full_step"].any()
    assert sampled["mask"].all()
