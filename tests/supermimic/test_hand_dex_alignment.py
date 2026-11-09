"""Numerical and dataflow oracles for the four v4 Dex alignments."""

from types import SimpleNamespace

import pytest
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.imagination import critic_td_lambda_batch
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.tasks.direct.hand21_supermimic.supermimic_cfg import default_experiment, load_hand_configuration
from dexgym.tasks.direct.hand21_supermimic.supermimic_dex_trainer import HandDexTrainer
from dexgym.tasks.direct.hand21_supermimic.supermimic_normalization import EmpiricalNorm, RunningNorm
from dexgym.tasks.direct.hand21_supermimic.supermimic_runner import components


def small_components(version="hand21_s0_v4"):
    experiment = default_experiment()
    experiment["version"] = version
    for key in ("world_hidden", "actor_hidden", "critic_hidden", "termination_hidden"):
        experiment["network"][key] = [32]
    cfg, _ = load_hand_configuration()
    return components(SimpleNamespace(device="cpu", bank=None, cfg=cfg), experiment)


@pytest.mark.parametrize("normalizer", [RunningNorm, EmpiricalNorm])
def test_statistics_match_population_and_resume(normalizer):
    torch.manual_seed(73)
    samples = torch.randn(31, 5) * torch.tensor([0.0, 0.001, 0.1, 1.0, 10.0]) + 2
    norm = normalizer(5)
    norm.update(samples[:11])
    restored = normalizer(5)
    restored.load_state_dict(norm.state_dict())
    norm.update(samples[11:])
    restored.update(samples[11:])
    mean, var = samples.mean(0), samples.var(0, unbiased=False)
    scale = (var + 0.01**2).sqrt() if normalizer is RunningNorm else var.sqrt() + 0.01
    torch.testing.assert_close(norm(samples), (samples - mean) / scale, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(restored(samples), norm(samples))
    assert norm.count.item() == 31


def test_running_norm_cap_and_finite_filter_survive_resume():
    norm = RunningNorm(1, max_count=3)
    norm.update(torch.tensor([[1.0], [float("nan")], [3.0]]))
    restored = RunningNorm(1, max_count=3)
    restored.load_state_dict(norm.state_dict())
    restored.update(torch.tensor([[5.0], [100.0]]))
    assert restored.count == 3
    torch.testing.assert_close(restored.mean, torch.tensor([3.0]))
    torch.testing.assert_close(restored.var, torch.tensor([8 / 3]))


def test_actor_normalizes_each_history_coordinate_and_noise_after_tanh():
    robot, task, modules = small_components()
    actor = modules.actor
    history = torch.randn(4, 10, 125) + torch.arange(10)[None, :, None] * 10
    goal = torch.randn(4, 1, 15)
    actor.update_norms(history, goal)
    assert actor.input_norm.count == 4
    torch.testing.assert_close(actor.input_norm._mean[0, :1250].reshape(10, 125), history.mean(0))
    with torch.no_grad():
        actor.decoder[-1].weight.zero_()
        actor.decoder[-1].bias.fill_(2)
    action = actor(history, goal)
    noise = torch.full_like(action, -0.3)
    expected = (torch.tanh(torch.tensor(2.0)) - 0.3).expand_as(action)
    torch.testing.assert_close(robot.exploration_action(action, noise), expected)
    assert not torch.allclose(expected, torch.tanh(torch.tensor(1.7)).expand_as(action))
    torch.testing.assert_close(robot.bounded_action(action), action)
    assert torch.equal(robot.exploration_action(action, torch.ones_like(action)), torch.ones_like(action))
    assert actor.input_norm.count == 4  # forward/evaluation cannot update stats


def test_world_uses_executed_targets_and_preserves_action_gradient():
    robot, task, modules = small_components()
    world = modules.world_model
    q = robot.fk.limits.mean(-1)[None].repeat(2, 1)
    state = torch.cat((q, torch.zeros_like(q), q), -1)
    context = torch.empty(2, 0)
    # Two different actions inside the same deadzone imply identical WM input.
    observed = []
    handle = world.input_norm.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach().clone()))
    world(state, torch.full((2, 20), 0.01), context, robot)
    world(state, torch.full((2, 20), 0.09), context, robot)
    handle.remove()
    torch.testing.assert_close(observed[0], observed[1])
    torch.testing.assert_close(observed[0][:, -20:], q)
    with torch.no_grad():
        world.network[-1].weight.normal_(0, 0.01)
    action = torch.full((2, 20), 0.4, requires_grad=True)
    result = world(state, action, context, robot)
    torch.testing.assert_close(result[:, 40:], robot.executed_command(state, action))
    (grad,) = torch.autograd.grad(result[:, :40].sum(), action)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def loss_window(world):
    batch, steps = 2, 8
    state = torch.zeros(batch, steps, 60)
    target = state.clone()
    target[..., :40] = world.error_floor
    mask = torch.ones(batch, steps, dtype=torch.bool)
    mask[1, 3:] = False
    return SimpleNamespace(
        state=state,
        next_state=target,
        action=torch.zeros(batch, steps, 20),
        executed_command=torch.full((batch, steps, 20), 0.123),
        dynamics_context=torch.empty(batch, steps, 0),
        mask=mask,
    )


def test_world_loss_uses_logged_commands_floor_and_time_sum():
    robot, task, modules = small_components()
    world = modules.world_model
    world.delta_norm.var.zero_()
    window = loss_window(world)
    seen = []
    handle = world.input_norm.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    loss, errors = world.dynamics_loss(window, robot)
    handle.remove()
    # Each active step has unit mean normalized squared error: (8+3)/2.
    torch.testing.assert_close(loss, torch.tensor(5.5))
    assert torch.equal(errors[~window.mask], torch.zeros_like(errors[~window.mask]))
    assert len(seen) == 8
    assert all(torch.equal(x[:, -20:], window.executed_command[:, 0]) for x in seen)
    loss.backward()
    assert torch.isfinite(world.network[-1].bias.grad).all()
    assert world.network[-1].bias.grad.abs().sum() > 0
    # Raising one dimension's measured scale reduces only that dimension.
    world.delta_norm.var[0] = (world.error_floor[0] * 2) ** 2 - world.delta_norm.eps**2
    scaled, _ = world.dynamics_loss(window, robot)
    torch.testing.assert_close(scaled, torch.tensor(5.5 * (39 + 0.25) / 40))


def test_reach_head_loss_stays_mean_bce_on_real_labels():
    robot, task, modules = small_components()
    world = modules.world_model
    task.reach_features = lambda arrival, old: arrival[:, :7]
    window = loss_window(world)
    z = torch.zeros(2, 8, dtype=torch.long)
    data = torch.zeros(2, 8, 11)
    next_data = data.clone()
    next_data[0, 0, 9] = 1  # one real reached event, independent of predictions
    window.task_state = TaskState(z, z, z, data)
    window.next_task_state = TaskState(z, z, z, next_data)
    dynamics, _ = world.dynamics_loss(window, robot)
    total, _ = world.autoregressive_loss(window, robot)
    logits = world.reach_head(window.next_state[window.mask][:, :7]).squeeze(-1)
    labels = next_data[..., 9][window.mask]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    torch.testing.assert_close(total - dynamics, expected)
    assert world.last_event_metrics["reach_label_rate"] == pytest.approx(1 / 11)
    world.zero_grad()
    total.backward()
    assert world.reach_head[-1].bias.grad is not None


def test_critic_refresh_includes_terminal_and_rebuilds_lambda_targets():
    _, _, modules = small_components()
    critic = modules.critic
    trainer = object.__new__(HandDexTrainer)
    trainer.critic = critic
    features = torch.randn(3, 2, 162) + 4
    terminal = torch.randn(2, 162) + 10
    rollout = SimpleNamespace(
        critic_features=features,
        terminal_features=terminal,
        values=torch.full((4, 2), 999.0),
        rewards=torch.ones(3, 2),
        terminations=torch.zeros(3, 2),
    )
    _, stale = critic_td_lambda_batch(rollout)
    trainer.refresh_critic_values(rollout)
    all_inputs = torch.cat((features, terminal[None]))
    assert critic.input_norm.count == 8
    torch.testing.assert_close(critic.input_norm.mean, all_inputs.mean((0, 1)))
    torch.testing.assert_close(rollout.values, critic(all_inputs))
    assert not rollout.values.requires_grad
    _, fresh = critic_td_lambda_batch(rollout)
    assert not torch.allclose(stale, fresh)


def test_real_normalizers_receive_logged_command_and_arrival_only():
    robot, task, modules = small_components()
    trainer = object.__new__(HandDexTrainer)
    trainer.robot, trainer.task = robot, task
    trainer.world_model, trainer.actor, trainer.critic = modules.world_model, modules.actor, modules.critic
    trainer.termination_head = modules.termination
    trainer.train_env_range = (0, 2)
    task.termination_features = lambda state, arrival: torch.zeros(len(state), 26)
    z = torch.zeros(2, dtype=torch.long)
    transition = SimpleNamespace(
        state=torch.zeros(2, 60),
        next_state=torch.ones(2, 60),
        executed_command=torch.full((2, 20), 0.3),
        dynamics_context=torch.empty(2, 0),
        next_task_state=TaskState(z, z, z, torch.zeros(2, 11)),
    )
    trainer._update_normalizers_from_real(transition)
    torch.testing.assert_close(modules.world_model.input_norm.mean[-20:], torch.full((20,), 0.3))
    torch.testing.assert_close(modules.world_model.delta_norm.mean, torch.ones(40))
    torch.testing.assert_close(modules.termination.input_norm.mean[:60], torch.ones(60))
    assert modules.termination.input_norm.count == 2
    assert modules.actor.input_norm.count == modules.critic.input_norm.count == 0


def test_checkpoint_contract_describes_new_semantics_and_preserves_v3():
    manifests = {}
    for version in ("hand21_s0_v3", "hand21_s0_v4"):
        robot, task, modules = small_components(version)
        trainer = SimpleNamespace(robot=robot, task=task)
        manifests[version], _ = CoreCheckpoint._manifest(trainer, vars(modules), {}, {})
    old, new = manifests.values()
    assert old["networks"]["actor"]["output_activation"] == "linear_action_v1"
    assert new["networks"]["actor"]["output_activation"] == "tanh_action_v1"
    norms = new["networks"]["world_model"]["normalizers"]
    assert norms["input_norm"]["eps"] == 0.01
    assert norms["delta_norm"]["eps"] == 0.0001
    assert new["networks"]["actor"]["normalizers"]["input_norm"]["dimension"] == 1265
    assert new["robot_contract"]["world_action_input"] == "executed_joint_target"
    assert CoreCheckpoint._first_difference(old, new) is not None


@pytest.mark.parametrize("training", [True, False])
def test_collector_uses_current_stats_and_bounded_previous_action(training, monkeypatch):
    from dexgym.tasks.direct.hand21_supermimic.supermimic_collector import HandNewtonCollector

    robot, _, modules = small_components()
    q = robot.fk.limits.mean(-1)[None].repeat(2, 1)
    physical = torch.cat((q, torch.zeros_like(q), q), -1)
    z = torch.zeros(2, dtype=torch.long)
    ts = TaskState(z, z, z, torch.zeros(2, 11))
    actor = modules.actor
    with torch.no_grad():
        actor.decoder[-1].weight.zero_()
        actor.decoder[-1].bias.fill_(2)
    goal = robot.fk(q).reshape(2, 1, 15)
    env = SimpleNamespace(
        last_hand_control_frame=z,
        last_hand_goal_data=ts.data,
        last_hand_errors=torch.zeros(2, 5),
        last_hand_failure_reason=z,
        last_hand_reached=z.bool(),
    )
    task = SimpleNamespace(
        physical_task_state=lambda _: ts,
        goal=lambda *_: goal,
        hard_termination=lambda *_: z.bool(),
        exact_reward=lambda *_, **__: z.float(),
        observation=lambda state, action, _: torch.cat((action, torch.zeros(2, 105)), -1),
    )
    robot.physical_state = lambda _: physical.clone()
    robot.dynamics_context = lambda _: torch.empty(2, 0)
    norm_count_at_step = []

    def step(action):
        norm_count_at_step.append(actor.input_norm.count.item())
        env.last_hand_state = physical.clone()
        env.last_hand_state[:, 40:] = robot.executed_command(physical, action)
        expected = (torch.tanh(torch.tensor(2.0)) + 0.1).clamp(-1, 1)
        torch.testing.assert_close(action, expected.expand_as(action))
        return None, z.float(), z.bool(), z.bool(), {}

    env.step = step
    collector = object.__new__(HandNewtonCollector)
    collector.env = collector.native = env
    collector.robot, collector.task, collector.num_envs = robot, task, 2
    collector.action_noise_std, collector.update_actor_normalization = 0.1, training
    collector.previous_action = torch.zeros(2, 20)
    collector.history = torch.randn(2, 10, 125)
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.ones_like(x))
    transition = collector.step(actor)
    assert norm_count_at_step == [2 if training else 0]
    assert actor.input_norm.count == (2 if training else 0)
    torch.testing.assert_close(collector.previous_action, transition.action)
    torch.testing.assert_close(collector.history[:, -1, :20], transition.action)
    torch.testing.assert_close(transition.executed_command, env.last_hand_state[:, 40:])


@pytest.mark.parametrize("amplitude, expected_gradient", [(0.05, 0.0), (0.4, 0.1)])
def test_exact_deadzone_has_no_surrogate_gradient(amplitude, expected_gradient):
    robot, _, _ = small_components()
    q = robot.fk.limits.mean(-1)[None]
    state = torch.cat((q, q * 0, q), -1)
    action = torch.full_like(q, amplitude, requires_grad=True)
    command = robot.executed_command(state, action)
    (gradient,) = torch.autograd.grad(command.sum(), action)
    torch.testing.assert_close(gradient, torch.full_like(action, expected_gradient))


def test_strict_evaluation_accepts_v4_and_rejects_v3(tmp_path):
    from dexgym.tasks.direct.hand21_supermimic.supermimic_schedule import HandTrainerConfig

    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    for version in ("hand21_s0_v3", "hand21_s0_v4"):
        robot, task, modules = small_components(version)
        trainer = SimpleNamespace(
            robot=robot,
            task=task,
            actor=modules.actor,
            world_model=modules.world_model,
            critic=modules.critic,
            termination_head=modules.termination,
            config=HandTrainerConfig(),
            optimizers={},
            iteration=0,
            counts={},
            allowed_horizon=32,
            trust_report={},
            observation_noise=None,
        )
        manager.save(tmp_path / (version + ".pt"), trainer)
    robot, task, modules = small_components()
    manager.load_actor_for_evaluation(tmp_path / "hand21_s0_v4.pt", robot=robot, task=task, modules=modules)
    with pytest.raises(ValueError, match="contract"):
        manager.load_actor_for_evaluation(tmp_path / "hand21_s0_v3.pt", robot=robot, task=task, modules=modules)
