from types import SimpleNamespace

import torch

from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine
from dexgym.algorithms.supermimic.dex_reference.networks import FailureHead, WorldModel
from dexgym.algorithms.supermimic.dex_reference.reset_sampling import ReturnResetSampler
from dexgym.tasks.direct.marvin54_s0.task import HandCentricTask, HandGoalLibrary, VelocityIntegratedJointCodec
from dexgym.tasks.direct.marvin54_tracking.task import JointCodec, MotionLibrary


def test_preview_critic_uses_future_within_clip_and_remaining_steps():
    task = HandCentricTask.__new__(HandCentricTask)
    task.library = MotionLibrary.__new__(MotionLibrary)
    task.library.end = torch.tensor([2, 2, 2, 5, 5, 5])
    task.future_offsets = torch.tensor([1, 2, 3])
    task.critic_context = "preview"
    task.target_memory = 0.0
    task.goal_features = lambda frames: frames.float().unsqueeze(-1)
    state = torch.zeros(2, 108)
    memory = {"frame": torch.tensor([0, 4]), "action": torch.zeros(2, 54)}

    result = task.context(state, memory, None, None)

    torch.testing.assert_close(result[:, -4:], torch.tensor([[1., 2., 2., 2.], [5., 5., 5., 1.]]))


def test_value_bound_caps_only_high_estimates():
    engine = SimpleNamespace(cfg=SuperMimicConfig(max_step_reward=HandCentricTask.max_step_reward))
    value = torch.tensor([-1., 9., 1000.], requires_grad=True)
    bounded = SuperMimicEngine.limit_value(engine, value)

    torch.testing.assert_close(bounded, torch.tensor([-1., 9., 954.]))
    bounded.sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([1., 1., 0.]))


def test_return_resets_focus_on_failures_and_keep_uniform_coverage():
    library = SimpleNamespace(
        start=torch.tensor([0, 3, 6]), end=torch.tensor([2, 2, 2, 5, 5, 5, 8, 8, 8]),
        reset_limit=torch.tensor([1, 1, 1, 4, 4, 4, 7, 7, 7]), frames=9,
    )
    sampler = ReturnResetSampler(library, 3, gamma=0.99, reward_max=1.0, window=1)
    sampler.record(torch.tensor([1, 4, 7]), torch.tensor([0.5, 1., 0.]),
                   torch.tensor([True, True, True]), torch.tensor([False, True, False]),
                   torch.tensor([False, False, True]))

    assert sampler.weights[0] > sampler.weights[1]
    assert (sampler.weights >= 0.5 / 3).all()
    torch.testing.assert_close(sampler.value, torch.tensor([-0.5, 1., 1.]), atol=1e-5, rtol=1e-5)
    frames = sampler.sample(64)
    assert (frames <= library.reset_limit[frames]).all()
    restored = ReturnResetSampler(library, 3, gamma=0.99, reward_max=1.0, window=1)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.weights, sampler.weights)
    torch.testing.assert_close(restored.draws, sampler.draws)


def test_structured_world_model_integrates_measured_control_interval():
    codec = VelocityIntegratedJointCodec
    q = torch.zeros(2, 54)
    qd = torch.ones_like(q)
    state = torch.cat((q, qd), -1)
    delta = torch.full_like(qd, 0.5)
    expected = torch.cat((torch.full_like(q, 0.0255), qd + delta), -1)
    torch.testing.assert_close(codec.integrate(state, delta), expected)

    model = WorldModel(108, 2, codec, 54, [0.001] * 108, hidden=(16,), amp=False,
                       predict_contact_depth=True)
    next_state, contact = model(state, q, state)
    torch.testing.assert_close(next_state, torch.cat((torch.full_like(q, 0.02), qd), -1))
    assert contact.shape == (2, 3)


def test_early_physical_contact_supervises_contact_with_full_interval_state_loss():
    engine = SuperMimicEngine.__new__(SuperMimicEngine)
    engine.cfg = SuperMimicConfig(wm_window=1, wm_batch=2, contact_partial_intervals=True)
    engine.task = SimpleNamespace(
        contact_dim=2, codec=JointCodec, simulator_context=torch.zeros(2, 1),
        replay_push=lambda x, push, aux: (x, aux),
    )
    engine.transition_termination = True
    engine.wm = WorldModel(108, 2, JointCodec, 54, [0.01] * 108, hidden=(8,), amp=False)
    engine.step_wm = engine.wm
    engine.wm_features = lambda x, sim, aux: x
    engine.fail_head = FailureHead(54, 108)
    w = dict(
        env=torch.arange(2), aux=torch.zeros(2, 1, 0), x=torch.zeros(2, 1, 108),
        x1=torch.stack((torch.zeros(1, 108), torch.ones(1, 108))),
        u=torch.zeros(2, 1, 54), push=torch.zeros(2, 1, 108),
        mask=torch.ones(2, 1), full_step=torch.tensor([[True], [False]]),
        contact=torch.tensor([[[0., 0.]], [[1., 1.]]]),
        terminated=torch.tensor([[False], [True]]), importance=torch.ones(2, 1),
    )
    engine.buffer = SimpleNamespace(sample=lambda *args, **kwargs: w)

    observed = engine.fit_world_model(holdout=True)
    engine.cfg.contact_partial_intervals = False
    complete = engine.fit_world_model(holdout=True)

    assert observed["partial_contact_steps"] == 1
    assert observed["contact_digit_loss"] > complete["contact_digit_loss"]
    assert observed["state_loss"] == complete["state_loss"] == 0


def test_fingertip_velocity_stops_at_clip_boundary():
    library = HandGoalLibrary.__new__(HandGoalLibrary)
    library.start = torch.tensor([0, 3])
    library.end = torch.tensor([2, 2, 2, 5, 5, 5])
    pos = torch.tensor([0., 0.02, 0.04, 1., 1.02, 1.04])
    library.ref_tip_pos = pos[:, None, None, None].expand(6, 2, 5, 3)
    library.ref_tip_rot = torch.eye(3).expand(6, 2, 5, 3, 3)

    velocities = library.reference_velocities(torch.tensor([[1, 2, 3, 4, 5]]))

    torch.testing.assert_close(velocities[0, :, 0], torch.tensor([1., 0., 1., 1., 0.]))
    torch.testing.assert_close(velocities[..., 30:], torch.zeros_like(velocities[..., 30:]))
