from types import SimpleNamespace

import pytest

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.algorithms.supermimic.core.dynamics_contract import world_model_dynamics_signature
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot
from dexgym.tasks.marvin_supermimic.se3_reach_hold import MarvinBimanualSE3ReachHold


def test_delta_action_has_distinct_dynamics_signature_and_cannot_share_legacy_model():
    source = SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES)
    old = MarvinCoreRobot(source, "cpu", action_schema="robot_fixed_v3")
    delta = MarvinCoreRobot(source, "cpu", action_schema="arm_joint_delta_v1")
    network = NetworkSpec(world_hidden=(16,), goal_hidden=(16,),
                          actor_hidden=(16,), critic_hidden=(16,))
    assert world_model_dynamics_signature(old, network)[1] != world_model_dynamics_signature(delta, network)[1]
    old_modules = NetworkFactory.build(old.spec, MarvinBimanualSE3ReachHold(old).spec,
                                       network, robot_adapter=old)
    with pytest.raises(ValueError, match="robot/action/context schema"):
        NetworkFactory.build(delta.spec, MarvinBimanualSE3ReachHold(delta).spec,
                             network, robot_adapter=delta,
                             shared_world_model=old_modules.world_model)
