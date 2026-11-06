"""Shared weights require the same actual physics, controller and integration contract."""

from dataclasses import replace

import pytest
import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, TaskSpec
from dexgym.tasks.supermimic_toy import ToyRobot
from dexgym.tasks.supermimic_toy import ToyCollector, ToyGoalTask
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig


class ContractRobot(ToyRobot):
    def __init__(self, *, action_scale=1.0, gain=1.0, context_schema="damping_v1",
                 asset_sha256="toy_asset_v1"):
        self.spec = replace(ToyRobot.spec, asset_sha256=asset_sha256)
        self.action_scale, self.gain, self.context_schema = action_scale, gain, context_schema

    def checkpoint_contract(self):
        return {"action_scale": self.action_scale, "gain": self.gain,
                "context_schema": self.context_schema,
                "context_model_range": [0.8, 1.0],
                "integration": "additive_delta_v1"}


TRACKING = TaskSpec("tracking", 3, 2, 1, 1, 1, 3, "toy_position_v1")
REACH = TaskSpec("reach", 4, 2, 2, 1, 2, 4, "toy_reach_v1")
NETWORK = NetworkSpec(world_hidden=(16,), goal_hidden=(8,), actor_hidden=(8,),
                      critic_hidden=(8,), latent_dim=4)


def test_compatible_dynamics_share_weights_and_normalizers_across_goal_schemas():
    robot = ContractRobot()
    source = NetworkFactory.build(robot.spec, TRACKING, NETWORK, robot_adapter=robot)
    target = NetworkFactory.build(robot.spec, REACH, NETWORK,
                                  robot_adapter=ContractRobot(),
                                  shared_world_model=source.world_model)
    assert target.world_model is source.world_model
    assert target.world_model.input_norm is source.world_model.input_norm
    assert target.world_model.delta_norm is source.world_model.delta_norm
    assert target.actor is not source.actor


@pytest.mark.parametrize("change", [
    {"action_scale": 0.5}, {"gain": 2.0},
    {"context_schema": "friction_v1"}, {"asset_sha256": "asset_v2"},
    {"physics_dt": 0.01},
])
def test_same_dimensions_with_different_dynamics_reject_shared_model(change):
    source_robot = ContractRobot()
    source = NetworkFactory.build(source_robot.spec, TRACKING, NETWORK,
                                  robot_adapter=source_robot)
    physics_dt = change.pop("physics_dt", None)
    target_robot = ContractRobot(**change)
    if physics_dt is not None:
        target_robot.spec = replace(target_robot.spec, physics_dt=physics_dt)
    with pytest.raises(ValueError, match="Shared World Model"):
        NetworkFactory.build(target_robot.spec, REACH, NETWORK,
                             robot_adapter=target_robot,
                             shared_world_model=source.world_model)


def test_shared_model_requires_explicit_physics_adapter():
    source = NetworkFactory.build(ToyRobot.spec, TRACKING, NETWORK)
    with pytest.raises(ValueError, match="Shared World Model"):
        NetworkFactory.build(ToyRobot.spec, REACH, NETWORK,
                             shared_world_model=source.world_model)


def test_shared_model_rejects_changed_world_normalizer_semantics():
    robot = ContractRobot()
    source = NetworkFactory.build(robot.spec, TRACKING, NETWORK, robot_adapter=robot)
    with pytest.raises(ValueError, match="Shared World Model"):
        NetworkFactory.build(robot.spec, REACH,
                             replace(NETWORK, normalizer_min_std=0.1),
                             robot_adapter=robot,
                             shared_world_model=source.world_model)


def test_nonzero_context_requires_schema_and_model_range():
    class MissingRange(ContractRobot):
        def checkpoint_contract(self):
            value = super().checkpoint_contract()
            value.pop("context_model_range")
            return value

    robot = MissingRange()
    with pytest.raises(ValueError, match="context schema and declared model range"):
        NetworkFactory.build(robot.spec, TRACKING, NETWORK, robot_adapter=robot)


def test_two_toy_goal_schemas_train_same_world_weights_with_one_optimizer():
    torch.manual_seed(432)
    robot = ContractRobot()
    task_a, task_b = ToyGoalTask("position"), ToyGoalTask("position_velocity")
    first = NetworkFactory.build(robot.spec, task_a.spec, NETWORK, robot_adapter=robot)
    second = NetworkFactory.build(robot.spec, task_b.spec, NETWORK,
                                  robot_adapter=ContractRobot(),
                                  shared_world_model=first.world_model)
    world_optimizer = torch.optim.Adam(first.world_model.parameters(), lr=0.002)

    def trainer(task, modules, world):
        return SuperMimicTrainer(
            collector=ToyCollector(task, 8, exploration_std=0.1),
            replay=SchemaReplay(64, 8, robot.spec, task.spec, "cpu"),
            robot=robot, task=task, modules=modules,
            optimizers={"world_model": world,
                        "actor": torch.optim.Adam(modules.actor.parameters(), lr=0.002),
                        "critic": torch.optim.Adam(modules.critic.parameters(), lr=0.002)},
            trust_evaluator=ModelTrustEvaluator(TrustConfig(min_distinct_starts=1)),
            config=TrainerConfig(world_batch=8, imagination_batch=4,
                                 world_warmup_updates=0,
                                 critic_epochs=1, critic_minibatches=2),
            train_env_range=(0, 6), validation_env_range=(6, 8))

    a = trainer(task_a, first, world_optimizer)
    with pytest.raises(ValueError, match="one common World Model optimizer"):
        trainer(task_b, second, torch.optim.Adam(second.world_model.parameters(), lr=0.002))
    b = trainer(task_b, second, world_optimizer)
    assert a.world_model is b.world_model
    for item in (a, b):
        for _ in range(10):
            item.collect()
            item.update_world_model()
        item.allowed_horizon = 1
        assert item.update_actor_critic()["updated"]
        assert item.counts["critic_updates"] > 0
    assert all(parameter in world_optimizer.state
               for parameter in first.world_model.parameters())
