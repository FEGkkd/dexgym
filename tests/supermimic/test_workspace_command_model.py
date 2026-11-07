"""Position-servo dynamics can predict a bounded residual around the command."""

from types import SimpleNamespace

import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory, StateDeltaWorldModel
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, TaskSpec
from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot


def test_command_centered_world_model_integrates_exact_servo_command_at_zero_residual():
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                            action_schema="arm_joint_delta_v1")
    network = NetworkSpec(world_hidden=(16,))
    model = StateDeltaWorldModel(robot.spec, network,
                                 integration_version="command_centered_residual_v2")
    state = torch.zeros(2, 42)
    state[:, :14] = robot.action_map.origin
    state[:, 28:42] = robot.action_map.origin
    action = torch.ones(2, 14, requires_grad=True)
    next_state = model(state, action, state.new_empty((2, 0)), robot)
    command = robot.executed_command(state, action)
    torch.testing.assert_close(next_state[:, :14], command)
    torch.testing.assert_close(next_state[:, 28:42], command)
    next_state[:, :14].sum().backward()
    assert torch.isfinite(action.grad).all()
    assert float(action.grad.abs().min()) > 0


def test_command_centered_delta_norm_uses_actual_q_minus_executed_command():
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                            action_schema="arm_joint_delta_v1")
    model = StateDeltaWorldModel(robot.spec, NetworkSpec(world_hidden=(16,)),
                                 integration_version="command_centered_residual_v2")
    state = torch.zeros(4, 42)
    state[:, :14] = robot.action_map.origin
    state[:, 28:42] = robot.action_map.origin
    action = torch.ones(4, 14)
    command = robot.executed_command(state, action)
    next_state = state.clone()
    next_state[:, :14] = command + 0.002
    next_state[:, 14:28] = 0.2
    next_state[:, 28:42] = command
    model.update_norms(state, action, state.new_empty((4, 0)), next_state, robot)
    torch.testing.assert_close(model.delta_norm.mean[:14], torch.full((14,), 0.002), atol=1e-6, rtol=0)
    torch.testing.assert_close(model.delta_norm.mean[14:], torch.full((14,), 0.2), atol=1e-6, rtol=0)


def test_legacy_delta_integration_stays_the_default():
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                            action_schema="arm_joint_delta_v1")
    model = StateDeltaWorldModel(robot.spec, NetworkSpec(world_hidden=(16,)))
    state = torch.zeros(1, 42)
    state[:, :14] = robot.action_map.origin
    state[:, 28:42] = robot.action_map.origin
    next_state = model(state, torch.ones(1, 14), state.new_empty((1, 0)), robot)
    torch.testing.assert_close(next_state[:, :14], state[:, :14])
    assert not torch.equal(next_state[:, 28:42], state[:, 28:42])


def test_factory_marks_command_centered_model_as_different_dynamics_contract():
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                            action_schema="arm_joint_delta_v1")
    task = TaskSpec("test", 3, 2, 1, 1, 0, 3, "goal_v1")
    network = NetworkSpec(world_hidden=(16,))
    legacy = NetworkFactory.build(robot.spec, task, network, robot_adapter=robot)
    centered = NetworkFactory.build(
        robot.spec, task, network, robot_adapter=robot,
        world_integration_version="command_centered_residual_v2")
    assert legacy.world_model.integration_version == "robot_adapter_delta_v1"
    assert centered.world_model.integration_version == "command_centered_residual_v2"
    assert legacy.world_model.dynamics_signature[1] != centered.world_model.dynamics_signature[1]


def test_command_residual_cannot_invent_large_position_servo_error():
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                            action_schema="arm_joint_delta_v1")
    state = torch.zeros(1, 42)
    state[:, :14] = robot.action_map.origin
    state[:, 28:42] = robot.action_map.origin
    predicted = torch.zeros(1, 28, requires_grad=True)
    predicted.data[:, :14] = 0.2
    action = torch.ones(1, 14)
    next_state = robot.integrate_command_centered(
        state, predicted, action, state.new_empty((1, 0)))
    commanded = robot.executed_command(state, action)
    assert float((next_state[:, :14] - commanded).detach().abs().max()) <= 0.005001
    next_state[:, :14].sum().backward()
    assert torch.isfinite(predicted.grad).all()


def test_training_config_explicitly_selects_new_integration_and_rejects_unknown():
    import pytest
    from scripts.supermimic.train_continuous_se3 import _model_integration_version

    assert _model_integration_version({}) == "robot_adapter_delta_v1"
    assert _model_integration_version({
        "world_model_integration": "command_centered_residual_v2"
    }) == "command_centered_residual_v2"
    with pytest.raises(ValueError, match="integration"):
        _model_integration_version({"world_model_integration": "unknown"})
