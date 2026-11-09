"""Boundary, resume and architecture checks for the hand-local Dex preset."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.tasks.direct.hand21_supermimic.supermimic_cfg import default_experiment
from dexgym.tasks.direct.hand21_supermimic.supermimic_networks import HandCritic, HandDirectActor
from dexgym.tasks.direct.hand21_supermimic.supermimic_robot import Hand21DynamicsAdapter
from dexgym.tasks.direct.hand21_supermimic.supermimic_schedule import HandSuperMimicTrainer, HandTrainerConfig
from dexgym.tasks.direct.hand21_supermimic.supermimic_task import Hand21Task
from dexgym.tasks.supermimic_toy import make_toy_trainer
from torch import nn


def small_trainer():
    old = make_toy_trainer("position", num_envs=8, world_hidden=16, world_batch=8, imagination_batch=4)
    config = asdict(old.config)
    config.update(actor_update_schedule="every_iteration", world_warmup_updates=0)
    modules = SimpleNamespace(
        world_model=old.world_model, actor=old.actor, critic=old.critic, termination=old.termination_head
    )
    return HandSuperMimicTrainer(
        collector=old.collector,
        replay=old.replay,
        robot=old.robot,
        task=old.task,
        modules=modules,
        optimizers=old.optimizers,
        trust_evaluator=None,
        config=HandTrainerConfig(**config),
        train_env_range=(0, 8),
        validation_env_range=None,
    )


def test_exact_collection_ratio_and_independent_lr():
    cfg = HandTrainerConfig()
    assert [cfg.collection_steps(i) for i in range(6)] == [2, 3, 2, 3, 2, 3]
    assert sum(cfg.collection_steps(i) for i in range(10000)) == 25000
    assert cfg.collection_steps(499) == 3
    assert cfg.collection_steps(500) == 2
    assert cfg.learning_rate(0) == cfg.lr
    assert cfg.learning_rate(125000) == pytest.approx(cfg.lr * 0.515)
    assert cfg.learning_rate(250000) == pytest.approx(cfg.lr * 0.03)
    assert cfg.learning_rate(320000) == cfg.learning_rate(250000)


def test_prefill_and_500_iteration_boundary_survive_resume(tmp_path):
    trainer = small_trainer()
    assert trainer.prefill() == 64
    assert trainer.iteration == 0
    assert trainer.counts["real_transitions"] == 64 * 8
    assert trainer.counts["world_updates"] == trainer.counts["actor_updates"] == 0
    # Boundary fixture: no claim that this toy ran 499 training iterations.
    trainer.iteration = 499
    before = [p.detach().clone() for p in trainer.actor.parameters()]
    last_wm_only = trainer.run_iteration()
    assert trainer.iteration == 500
    assert last_wm_only["training_phase"] == "world_model_only"
    assert last_wm_only["collection_steps"] == 3
    assert not last_wm_only["actor"][0]["updated"]
    assert trainer.counts["world_updates"] > 0
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.actor.parameters()))
    assert trainer.counts["critic_updates"] == 0
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    path = manager.save(tmp_path / "boundary.pt", trainer, resume_mode="fresh_physics")
    resumed = small_trainer()
    manager.load(path, resumed)
    resumed.prefill()
    result = resumed.run_iteration()
    assert resumed.iteration == 501
    assert result["training_phase"] == "actor_critic"
    assert result["collection_steps"] == 2
    assert result["actor"][0]["updated"]
    assert result["actor"][0]["horizon"] == 32
    assert resumed.counts["actor_updates"] == 1
    assert resumed.counts["critic_updates"] > 0
    assert result["lr"] == resumed.config.learning_rate(500)
    from dataclasses import replace

    mismatch = small_trainer()
    mismatch.config = replace(mismatch.config, utd=0.5)
    with pytest.raises(ValueError, match="Trainer configuration"):
        manager.load(path, mismatch)
    mismatch.config = replace(resumed.config, lr_decay_iters=1000)
    with pytest.raises(ValueError, match="Trainer configuration"):
        manager.load(path, mismatch)


def test_dex_networks_use_current_goal_and_small_orthogonal_output():
    experiment = default_experiment()
    assert experiment["version"] == "hand21_s0_v3"
    fields = experiment["network"].copy()
    for key in ("world_hidden", "goal_hidden", "actor_hidden", "critic_hidden", "termination_hidden"):
        fields[key] = tuple(fields[key])
    network = NetworkSpec(**fields)
    robot = Hand21DynamicsAdapter()
    task = Hand21Task(robot, None, SimpleNamespace())
    actor = HandDirectActor(robot.spec, task.spec, network)
    critic = HandCritic(task.spec, network)
    assert not hasattr(actor, "goal_encoder")
    assert [m.out_features for m in actor.decoder if isinstance(m, nn.Linear)] == [512, 256, 128, 20]
    assert [m.out_features for m in critic.network if isinstance(m, nn.Linear)] == [512, 512, 256, 128, 1]
    assert not any(isinstance(m, nn.LayerNorm) for m in actor.modules())
    final = actor.decoder[-1]
    assert torch.allclose(final.weight @ final.weight.T, torch.eye(20) * 1e-4, atol=1e-8)
    assert torch.count_nonzero(final.bias) == 0
    history = torch.randn(2, 10, 125)
    goal = torch.randn(2, 1, 15, requires_grad=True)
    action = actor(history, goal)
    action.sum().backward()
    assert torch.isfinite(action).all() and goal.grad.abs().sum() > 0
    assert network.world_hidden == (2048, 2048, 2048)
    cfg = HandTrainerConfig(**experiment["trainer"])
    assert (cfg.world_updates_per_iteration, cfg.world_batch, cfg.imagination_batch) == (10, 2048, 1024)


@pytest.mark.parametrize(
    "kwargs", [{"utd": 0}, {"utd": float("nan")}, {"warmup_steps": 1}, {"wm_only_iters": -1}, {"lr_decay_iters": 0}]
)
def test_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        HandTrainerConfig(**kwargs)
