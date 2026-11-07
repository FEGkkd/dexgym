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


def test_inference_may_change_reference_start_sampler_bound_but_resume_may_not(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    training_physics = dict(TRAIN_PHYSICS, max_reference_start_frame=80)
    path = CoreCheckpoint(git_commit="test", reference_hashes={},
                          physics_config=training_physics).save(tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    evaluation = CoreCheckpoint(git_commit="test", reference_hashes={},
                                physics_config=dict(TRAIN_PHYSICS, max_reference_start_frame=0))
    modules = type("Modules", (), {"actor": target.actor,
                                   "world_model": target.world_model,
                                   "critic": target.critic, "termination": None})()
    report = evaluation.load_actor_for_evaluation(
        path, robot=target.robot, task=target.task, modules=modules)
    assert report["evaluation_overrides"]["max_reference_start_frame"] == 0
    with pytest.raises(ValueError, match="physics"):
        evaluation.load(path, target)


def test_longer_reference_episode_is_explicit_evaluation_protocol_only(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    path = CoreCheckpoint(git_commit="test", reference_hashes={},
                          physics_config=dict(TRAIN_PHYSICS, episode_length_s=4.0)).save(
                              tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    evaluation = CoreCheckpoint(git_commit="test", reference_hashes={},
                                physics_config=dict(TRAIN_PHYSICS, episode_length_s=12.0))
    modules = type("Modules", (), {"actor": target.actor,
                                   "world_model": target.world_model,
                                   "critic": target.critic, "termination": None})()
    report = evaluation.load_actor_for_evaluation(
        path, robot=target.robot, task=target.task, modules=modules)
    assert report["evaluation_overrides"]["episode_length_s"] == 12.0
    with pytest.raises(ValueError, match="physics"):
        evaluation.load(path, target)


def test_causal_stream_evaluation_can_explicitly_override_target_bank_not_training_resume(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    old_contract = source.task.checkpoint_contract
    source.task.checkpoint_contract = lambda: dict(
        old_contract(), target_stream="causal_fixed_base_50hz_v1",
        reference_hashes={"bank": "train"})
    path = CoreCheckpoint(git_commit="test", reference_hashes={"bank": "train"},
                          physics_config=TRAIN_PHYSICS).save(tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    target.task.inference_reference_agnostic = True
    new_contract = target.task.checkpoint_contract
    target.task.checkpoint_contract = lambda: dict(
        new_contract(), target_stream="causal_fixed_base_50hz_v1",
        reference_hashes={"bank": "unseen"})
    manager = CoreCheckpoint(git_commit="test", reference_hashes={"bank": "unseen"},
                             physics_config=TRAIN_PHYSICS)
    modules = type("Modules", (), {"actor": target.actor,
                                   "world_model": target.world_model,
                                   "critic": target.critic, "termination": None})()
    with pytest.raises(ValueError, match="Reference hashes"):
        manager.load_actor_for_evaluation(
            path, robot=target.robot, task=target.task, modules=modules)
    report = manager.load_actor_for_evaluation(
        path, robot=target.robot, task=target.task, modules=modules,
        allow_reference_override=True)
    assert report["target_source_override"]["evaluation"] == {"bank": "unseen"}
    incompatible = CoreCheckpoint(
        git_commit="test", reference_hashes={"bank": "unseen"},
        physics_config=dict(TRAIN_PHYSICS, controller_kp_source="different"))
    with pytest.raises(ValueError, match="controller_kp_source"):
        incompatible.load_actor_for_evaluation(
            path, robot=target.robot, task=target.task, modules=modules,
            allow_reference_override=True)
    with pytest.raises(ValueError, match="Reference"):
        manager.load(path, target)


def test_explicit_new_reference_initialization_copies_models_but_not_training_state(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    old_contract = source.task.checkpoint_contract
    source.task.checkpoint_contract = lambda: dict(
        old_contract(), target_stream="causal_fixed_base_50hz_v1",
        reference_hashes={"bank": "old"})
    with torch.no_grad():
        next(source.world_model.parameters()).add_(0.25)
        next(source.actor.parameters()).add_(0.25)
    path = CoreCheckpoint(git_commit="test", reference_hashes={"bank": "old"},
                          physics_config=TRAIN_PHYSICS).save(tmp_path / "old.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    target.task.inference_reference_agnostic = True
    new_contract = target.task.checkpoint_contract
    target.task.checkpoint_contract = lambda: dict(
        new_contract(), target_stream="causal_fixed_base_50hz_v1",
        reference_hashes={"bank": "new"})
    manager = CoreCheckpoint(git_commit="test", reference_hashes={"bank": "new"},
                             physics_config=TRAIN_PHYSICS)
    result = manager.initialize_modules_for_new_reference(path, target)
    assert result["target_source_override"]["evaluation"] == {"bank": "new"}
    for name in ("world_model", "actor", "critic"):
        for key, value in getattr(source, name).state_dict().items():
            torch.testing.assert_close(value, getattr(target, name).state_dict()[key])
    assert target.iteration == 0 and target.counts["world_updates"] == 0
    assert target.allowed_horizon == 0 and target.trust_report == {}
    assert all(not optimizer.state for optimizer in target.optimizers.values())


def test_dynamics_only_transfer_keeps_fresh_actor_and_critic(tmp_path):
    source = make_toy_trainer("position", num_envs=8)
    source_contract = source.task.checkpoint_contract
    source.task.checkpoint_contract = lambda: dict(
        source_contract(), target_stream="causal_fixed_base_50hz_v1",
        reference_hashes={"bank": "old"})
    with torch.no_grad():
        next(source.world_model.parameters()).add_(0.5)
    path = CoreCheckpoint(git_commit="test", reference_hashes={"bank": "old"},
                          physics_config=TRAIN_PHYSICS).save(tmp_path / "model.pt", source)
    target = make_toy_trainer("position", num_envs=8)
    target.task.inference_reference_agnostic = True
    target_contract = target.task.checkpoint_contract
    target.task.checkpoint_contract = lambda: dict(
        target_contract(), target_stream="causal_fixed_base_50hz_v1",
        reference_hashes={"bank": "new"})
    before_actor = {key: value.clone() for key, value in target.actor.state_dict().items()}
    before_critic = {key: value.clone() for key, value in target.critic.state_dict().items()}
    report = CoreCheckpoint(git_commit="test", reference_hashes={"bank": "new"},
                            physics_config=TRAIN_PHYSICS).initialize_dynamics_for_new_reference(
                                path, target)
    assert report["transfer_kind"] == "compatible_world_model_fresh_actor_v1"
    assert report["actor_source"] == "fresh_random_init"
    assert report["world_model_source"] == str(path)
    assert report["policy_source"] is None
    for key, value in source.world_model.state_dict().items():
        torch.testing.assert_close(target.world_model.state_dict()[key], value)
    for key, value in before_actor.items():
        torch.testing.assert_close(target.actor.state_dict()[key], value)
    for key, value in before_critic.items():
        torch.testing.assert_close(target.critic.state_dict()[key], value)
    assert target.allowed_horizon == 0 and target.counts["real_transitions"] == 0


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
