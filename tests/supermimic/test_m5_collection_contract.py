from types import SimpleNamespace

from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.tasks.marvin_supermimic.physics_contract import marvin_se3_physics_config


def _cfg():
    return SimpleNamespace(
        scene=SimpleNamespace(num_envs=16), seed=51, sim=SimpleNamespace(dt=0.002),
        decimation=10, episode_length_s=4.0, audit_contacts_each_physics_step=True,
        goal_train_seed=152, goal_eval_seed=100152, curriculum_level="near",
        evaluation_mode=False, collection_contract_version=None)


def _robot():
    return SimpleNamespace(native=SimpleNamespace(
        controller_kp_source=100.0, controller_kd_source=10.0,
        controller_damping_scale_source=0.5))


def test_partitioned_actor_collection_has_immutable_checkpoint_contract():
    cfg = _cfg()
    old = marvin_se3_physics_config(cfg, _robot())
    assert "collection_contract" not in old
    cfg.collection_contract_version = "deterministic_trust_lanes_v1"
    cfg.exploration_env_count = 6
    cfg.exploration_rho = 0.95
    cfg.exploration_sigma = 0.8
    new = marvin_se3_physics_config(cfg, _robot())
    assert new["collection_contract"]["exploration_env_count"] == 6
    assert "collection_contract" in CoreCheckpoint._first_difference(old, new)
    cfg.collection_contract_version = "dual_source_geometry_trust_v2"
    cfg.train_env_count = 12
    cfg.validation_exploration_env_count = 2
    cfg.noisy_trust_config = {"metric": "normalized", "require_baseline_improvement": True}
    cfg.geometry_trust_config = {"max_tcp_position_p95_m": (0.005,) * 3}
    dual = marvin_se3_physics_config(cfg, _robot())
    assert dual["collection_contract"]["validation_exploration_env_count"] == 2
    assert "collection_contract" in CoreCheckpoint._first_difference(new, dual)
    cfg.collection_contract_version = "dual_source_geometry_trust_v3_full32"
    cfg.minimum_actor_horizon = 32
    full = marvin_se3_physics_config(cfg, _robot())
    assert full["collection_contract"]["minimum_actor_horizon"] == 32
    assert "collection_contract" in CoreCheckpoint._first_difference(dual, full)
