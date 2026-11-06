"""A learned termination loss must update its own parameters."""

import pytest
import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.tasks.supermimic_toy import ToyCollector, ToyGoalTask, ToyRobot


class LearnedToyTask(ToyGoalTask):
    termination_mode = "learned"

    def termination_features(self, next_state, next_task_state):
        return next_task_state.data


def make_trainer(*, learned=True, include_head=True, wrong_actor=False):
    robot = ToyRobot()
    task = LearnedToyTask("position") if learned else ToyGoalTask("position")
    modules = NetworkFactory.build(robot.spec, task.spec, NetworkSpec(
        world_hidden=(16,), goal_hidden=(8,), actor_hidden=(16,),
        critic_hidden=(8,), termination_hidden=(16,), latent_dim=4))
    world_parameters = list(modules.world_model.parameters())
    if include_head:
        world_parameters += list(modules.termination.parameters())
    if wrong_actor:
        world_parameters += list(modules.actor.parameters())
    optimizers = {
        "world_model": torch.optim.Adam(world_parameters, lr=0.01),
        "actor": torch.optim.Adam(modules.actor.parameters(), lr=0.001),
        "critic": torch.optim.Adam(modules.critic.parameters(), lr=0.001),
    }
    trainer = SuperMimicTrainer(
        collector=ToyCollector(task, 8, exploration_std=0),
        replay=SchemaReplay(64, 8, robot.spec, task.spec, "cpu"),
        robot=robot, task=task, modules=modules, optimizers=optimizers,
        trust_evaluator=ModelTrustEvaluator(TrustConfig(min_distinct_starts=1)),
        config=TrainerConfig(world_batch=16, imagination_batch=4,
                             world_warmup_updates=1000),
        train_env_range=(0, 6), validation_env_range=(6, 8))
    return trainer


def test_missing_learned_head_or_cross_owned_actor_is_rejected():
    with pytest.raises(ValueError, match="termination"):
        make_trainer(include_head=False)
    with pytest.raises(ValueError, match="actor"):
        make_trainer(wrong_actor=True)


def test_learned_head_changes_and_bce_falls_on_positive_and_negative_real_labels():
    torch.manual_seed(432)
    trainer = make_trainer()
    for _ in range(12):
        transition = trainer.collector.step(trainer.actor)
        # Positive and negative hard labels originate at arrival, never at reset.
        transition.terminated[:] = transition.next_task_state.data[:, 0] > 0
        trainer.replay.append(transition)
    head_before = [p.detach().clone() for p in trainer.termination_head.parameters()]
    losses = [trainer.update_world_model()["loss"] for _ in range(60)]
    assert any(not torch.equal(a, b) for a, b in zip(head_before, trainer.termination_head.parameters()))
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
    assert trainer.counts["world_updates"] == 60


def test_analytic_mode_does_not_require_termination_optimizer():
    assert make_trainer(learned=False, include_head=False).termination_head is None


def test_learned_head_optimizer_state_survives_checkpoint_resume(tmp_path):
    from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint

    trainer = make_trainer()
    for _ in range(8):
        transition = trainer.collector.step(trainer.actor)
        transition.terminated[:] = transition.next_task_state.data[:, 0] > 0
        trainer.replay.append(transition)
    trainer.update_world_model()
    assert all(parameter in trainer.optimizers["world_model"].state
               for parameter in trainer.termination_head.parameters())
    path = CoreCheckpoint(git_commit="test", reference_hashes={}).save(
        tmp_path / "learned.pt", trainer, resume_mode="replay")
    restored = make_trainer()
    CoreCheckpoint(git_commit="test", reference_hashes={}).load(path, restored)
    for before, after in zip(trainer.termination_head.parameters(),
                             restored.termination_head.parameters()):
        torch.testing.assert_close(before, after)
        assert restored.optimizers["world_model"].state[after]["step"] >= 1
    assert restored.update_world_model()["updated"]
