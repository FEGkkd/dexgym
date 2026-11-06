"""Actor sensing noise must not enter physical state, reward or model targets."""

import torch

from dexgym.algorithms.supermimic.core.observation_noise import (
    ObservationNoiseModel, ObservationNoiseSpec)
from dexgym.tasks.supermimic_toy import ToyCollector, ToyGoalTask
from dexgym.tasks.supermimic_toy import make_toy_trainer


class RecordingActor:
    def __init__(self):
        self.seen = None

    def __call__(self, history, goal):
        self.seen = history.clone()
        return torch.zeros(len(history), 1)


def test_disabled_noise_is_bitwise_identity_and_seeded_uniform_is_reproducible():
    clean = torch.ones(128, 3)
    disabled = ObservationNoiseModel(ObservationNoiseSpec.disabled("toy_obs_v1", 3), seed=17)
    assert torch.equal(disabled(clean), clean)
    spec = ObservationNoiseSpec("uniform", "toy_obs_v1", (0.1, 0.0, 0.2),
                                ("m", "m/s", "unitless"))
    first = ObservationNoiseModel(spec, seed=17)
    second = ObservationNoiseModel(spec, seed=17)
    third = ObservationNoiseModel(spec, seed=18)
    a, b, c = first(clean), second(clean), third(clean)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.equal(a[:, 1], clean[:, 1])
    assert torch.all((a[:, 0] - 1).abs() <= 0.1)
    assert torch.equal(first(clean, mode="eval"), clean)


def test_toy_real_actor_history_is_noisy_but_world_targets_are_clean():
    task = ToyGoalTask("position")
    spec = ObservationNoiseSpec("uniform", task.spec.goal_schema,
                                (0.1, 0.1, 0.1), ("m", "m/s", "action"))
    noise = ObservationNoiseModel(spec, seed=432)
    collector = ToyCollector(task, 8, exploration_std=0, observation_noise=noise)
    actor = RecordingActor()
    expected = collector.history.clone()
    transition = collector.step(actor)
    assert torch.equal(actor.seen, expected)
    assert torch.equal(transition.history, expected)
    assert torch.equal(transition.state, torch.zeros_like(transition.state))
    assert torch.equal(transition.next_state, torch.zeros_like(transition.next_state))
    assert torch.equal(transition.reward, task.reward(
        transition.state, transition.next_state, transition.action,
        transition.next_task_state))


def test_noisy_imagination_retains_actor_bptt_and_one_norm_count_per_real_frame():
    torch.manual_seed(19)
    task = ToyGoalTask("position")
    spec = ObservationNoiseSpec("gaussian", task.spec.goal_schema,
                                (0.02, 0.02, 0.0), ("m", "m/s", "action"))
    trainer = make_toy_trainer("position", num_envs=8,
                               observation_noise=ObservationNoiseModel(spec, seed=432))
    for _ in range(12):
        trainer.collect()
        trainer.update_world_model()
    trainer.allowed_horizon = 1
    result = trainer.update_actor_critic()
    assert result["updated"]
    assert result["actor_grad_norm"] > 0
    assert int(trainer.actor.observation_norm.count) == 12 * 6
    assert int(trainer.world_model.input_norm.count) == 12 * 6


def test_noise_contract_and_generator_resume_exactly(tmp_path):
    from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint

    task = ToyGoalTask("position")
    spec = ObservationNoiseSpec("uniform", task.spec.goal_schema,
                                (0.1, 0.2, 0.0), ("m", "m/s", "action"))
    source = make_toy_trainer("position", num_envs=8,
                              observation_noise=ObservationNoiseModel(spec, seed=123))
    source.collect()
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    path = manager.save(tmp_path / "noisy.pt", source, resume_mode="full")
    clean = torch.zeros(8, 3)
    expected = source.observation_noise(clean)
    target = make_toy_trainer("position", num_envs=8,
                              observation_noise=ObservationNoiseModel(spec, seed=123))
    manager.load(path, target)
    assert torch.equal(expected, target.observation_noise(clean))
    assert torch.equal(source.collector.history, target.collector.history)
    assert torch.equal(source.actor.observation_norm.count,
                       target.actor.observation_norm.count)
