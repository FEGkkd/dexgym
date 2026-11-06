import pytest

from dexgym.algorithms.supermimic.registry import PluginRegistry
from dexgym.tasks.supermimic_toy import ToyGoalTask, ToyRobot


def test_registry_accepts_two_goal_schemas_and_rejects_duplicates():
    registry = PluginRegistry()
    registry.register_robot("damped_point", ToyRobot)
    registry.register_task("position", lambda: ToyGoalTask("position"))
    registry.register_task("position_velocity", lambda: ToyGoalTask("position_velocity"))
    assert registry.robot("damped_point").spec.state_dim == 2
    assert registry.task("position").spec.goal_dim == 1
    assert registry.task("position_velocity").spec.goal_dim == 2
    with pytest.raises(ValueError, match="duplicate"):
        registry.register_task("position", ToyGoalTask)
    with pytest.raises(KeyError, match="Unknown"):
        registry.task("missing")
