"""Different robot and task dimensions must share one network factory."""

import pytest
import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, RobotSpec, TaskSpec


@pytest.mark.parametrize(
    ("robot", "task", "history_shape", "goal_shape", "critic_width"),
    [
        (RobotSpec("marvin", 42, 14, 28, 0, "robot_fixed_v3", "arm42_v1"),
         TaskSpec("tracking", 56, 10, 90, 10, 0, 146, "six_link_v1"),
         (3, 10, 56), (3, 10, 90), 146),
        (RobotSpec("marvin", 42, 14, 28, 0, "robot_fixed_v3", "arm42_v1"),
         TaskSpec("reach", 42, 2, 6, 1, 6, 48, "dual_tcp_pos_v1"),
         (3, 2, 42), (3, 1, 6), 48),
        (RobotSpec("toy", 3, 1, 3, 1, "toy_delta_v1", "toy3_v1"),
         TaskSpec("toy_target", 4, 3, 2, 2, 2, 5, "toy_goal_v1"),
         (3, 3, 4), (3, 2, 2), 5),
    ],
)
def test_factory_builds_multiple_dimensions(robot, task, history_shape, goal_shape, critic_width):
    modules = NetworkFactory.build(robot, task, NetworkSpec(
        world_hidden=(32, 32), goal_hidden=(32,), actor_hidden=(32,),
        critic_hidden=(16,), latent_dim=8))
    assert modules.actor(torch.zeros(history_shape), torch.zeros(goal_shape)).shape == (3, robot.action_dim)
    assert modules.critic(torch.zeros(3, critic_width)).shape == (3,)
    assert modules.world_model.predict_delta(
        torch.zeros(3, robot.state_dim), torch.zeros(3, robot.action_dim),
        torch.zeros(3, robot.context_dim)).shape == (3, robot.delta_dim)
    assert modules.termination.logits(torch.zeros(3, robot.state_dim),
                                      torch.zeros(3, task.goal_dim)).shape == (3,)
    assert modules.actor.goal_encoder[0].in_features == task.goal_horizon * task.goal_dim
    assert modules.critic.network[0].in_features == task.critic_dim


def test_factory_rejects_invalid_specs_and_tensor_shapes():
    with pytest.raises(ValueError, match="delta_dim"):
        RobotSpec("bad", 2, 1, 3, 0, "a", "s")
    with pytest.raises(ValueError, match="dimension"):
        TaskSpec("bad", 0, 2, 6, 1, 0, 8, "g")
    robot = RobotSpec("toy", 3, 1, 3, 0, "a", "s")
    task = TaskSpec("goal", 4, 2, 6, 1, 6, 9, "g")
    modules = NetworkFactory.build(robot, task, NetworkSpec(
        world_hidden=(8,), goal_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,), latent_dim=4))
    with pytest.raises(ValueError, match="history"):
        modules.actor(torch.zeros(2, 3, 4), torch.zeros(2, 1, 6))
    with pytest.raises(ValueError, match="goal"):
        modules.actor(torch.zeros(2, 2, 4), torch.zeros(2, 1, 90))
    with pytest.raises(ValueError, match="Critic"):
        modules.critic(torch.zeros(2, 10))


def test_actor_normalizers_are_schema_specific_and_restorable():
    robot = RobotSpec("toy", 3, 1, 3, 0, "a", "s")
    task = TaskSpec("goal", 4, 2, 6, 1, 6, 9, "g")
    config = NetworkSpec(world_hidden=(8,), goal_hidden=(8,), actor_hidden=(8,),
                         critic_hidden=(8,), latent_dim=4)
    actor = NetworkFactory.build(robot, task, config).actor
    actor.update_norms(torch.full((3, 2, 4), 2.0), torch.full((3, 1, 6), 4.0))
    restored = NetworkFactory.build(robot, task, config).actor
    restored.load_state_dict(actor.state_dict(), strict=True)
    torch.testing.assert_close(actor(torch.ones(2, 2, 4), torch.ones(2, 1, 6)),
                               restored(torch.ones(2, 2, 4), torch.ones(2, 1, 6)))
    assert restored.observation_norm.mean.shape == (4,)
    assert restored.goal_norm.mean.shape == (6,)
