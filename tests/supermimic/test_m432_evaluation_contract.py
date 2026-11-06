"""Independent inference can change sampling protocol but not executable semantics."""

import pytest
import torch

from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.tasks.supermimic_toy import make_toy_trainer


TRAIN_PHYSICS = {
    "backend": "toy", "num_envs": 16, "seed": 11,
    "reset_joint_perturb_rad": 0.01, "reset_random_reference_start": True,
    "physics_dt": 0.2, "policy_dt": 0.2, "controller_kp_source": "v1",
}


def test_inference_accepts_explicit_protocol_overrides_without_changing_policy(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    manager = CoreCheckpoint(git_commit="test", reference_hashes={}, physics_config=TRAIN_PHYSICS)
    path = manager.save(tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    history = torch.randn(2, 2, 3)
    goal = torch.randn(2, 1, 1)
    expected = source.actor(history, goal).detach().clone()
    changed = dict(TRAIN_PHYSICS, num_envs=1, seed=42,
                   reset_joint_perturb_rad=0.03,
                   reset_random_reference_start=False)
    evaluation = CoreCheckpoint(git_commit="test", reference_hashes={}, physics_config=changed)
    report = evaluation.load_actor_for_evaluation(
        path, robot=target.robot, task=target.task,
        modules=type("Modules", (), {"actor": target.actor,
                                     "world_model": target.world_model,
                                     "critic": target.critic,
                                     "termination": None})())
    assert report["evaluation_protocol"]["num_envs"] == 1
    assert report["evaluation_protocol"]["seed"] == 42
    assert torch.equal(expected, target.actor(history, goal))
    assert torch.equal(source.actor.observation_norm.count,
                       target.actor.observation_norm.count)
    with pytest.raises(ValueError, match="physics"):
        evaluation.load(path, target)


def test_inference_ignores_training_collection_protocol_but_resume_rejects_it(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    training_physics = dict(TRAIN_PHYSICS, collection_contract={
        "version": "dual_source_geometry_trust_v2", "exploration_sigma": 0.8})
    path = CoreCheckpoint(git_commit="test", reference_hashes={},
                          physics_config=training_physics).save(tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    manager = CoreCheckpoint(git_commit="test", reference_hashes={},
                             physics_config=dict(TRAIN_PHYSICS, num_envs=1, seed=42))
    modules = type("Modules", (), {"actor": target.actor,
                                   "world_model": target.world_model,
                                   "critic": target.critic, "termination": None})()
    report = manager.load_actor_for_evaluation(
        path, robot=target.robot, task=target.task, modules=modules)
    assert report["evaluation_overrides"]["num_envs"] == 1
    assert report["evaluation_overrides"]["collection_contract"] is None
    with pytest.raises(ValueError, match="physics"):
        manager.load(path, target)


@pytest.mark.parametrize("key,value", [
    ("physics_dt", 0.01), ("controller_kp_source", "v2"),
])
def test_inference_rejects_changed_execution_contract(tmp_path, key, value):
    source = make_toy_trainer("position", num_envs=8)
    path = CoreCheckpoint(git_commit="test", reference_hashes={},
                          physics_config=TRAIN_PHYSICS).save(tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    changed = dict(TRAIN_PHYSICS, **{key: value})
    manager = CoreCheckpoint(git_commit="test", reference_hashes={}, physics_config=changed)
    with pytest.raises(ValueError, match=key):
        manager.load_actor_for_evaluation(
            path, robot=target.robot, task=target.task,
            modules=type("Modules", (), {"actor": target.actor,
                                         "world_model": target.world_model,
                                         "critic": target.critic,
                                         "termination": None})())


def test_inference_rejects_same_shape_changed_actor_and_task_semantics(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    manager = CoreCheckpoint(git_commit="test", reference_hashes={},
                             physics_config=TRAIN_PHYSICS)
    path = manager.save(tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    modules = type("Modules", (), {"actor": target.actor,
                                   "world_model": target.world_model,
                                   "critic": target.critic,
                                   "termination": None})()
    target.actor.goal_latent_tanh = False
    with pytest.raises(ValueError, match="goal_latent_tanh"):
        manager.load_actor_for_evaluation(path, robot=target.robot,
                                          task=target.task, modules=modules)
    target.actor.goal_latent_tanh = True
    original = target.task.checkpoint_contract
    target.task.checkpoint_contract = lambda: dict(original(), position_weight=3.0)
    with pytest.raises(ValueError, match="position_weight"):
        manager.load_actor_for_evaluation(path, robot=target.robot,
                                          task=target.task, modules=modules)
