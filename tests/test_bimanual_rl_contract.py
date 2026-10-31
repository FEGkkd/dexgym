"""Phase 1 task registration and observation schema remain explicit."""

import gymnasium as gym

from dexteroushands_lab.tasks.direct.bimanual_bottle.task_contract import (
    ACTION_DIM, ACTOR_FIELDS, CRITIC_FIELDS, actor_dim, critic_dim, schema_dict,
)
from dexteroushands_lab.tasks.direct.bimanual_bottle import register_tasks
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from isaaclab.envs import DirectRLEnv


def test_action_and_observation_dimensions_are_schema_derived():
    assert ACTION_DIM == 52
    assert actor_dim() == sum(ACTOR_FIELDS.values())
    assert critic_dim() == actor_dim() + sum(CRITIC_FIELDS.values())
    assert "contact_forces" not in ACTOR_FIELDS
    assert "left_bottle_forces" in CRITIC_FIELDS
    schema = schema_dict()
    assert schema["actor"]["oracle_object_state"] is True
    assert schema["actor"]["dimension"] == actor_dim()
    assert schema["critic"]["dimension"] == critic_dim()
    assert schema["quaternion_convention"] == "xyzw"


def test_both_tasks_are_registered_with_shared_environment():
    register_tasks()
    hold = gym.spec("DexterousHands-WujiHand2-Bimanual-GraspHold-Direct-v0")
    turn = gym.spec("DexterousHands-WujiHand2-Bimanual-CapTurn30-Direct-v0")
    assert hold.entry_point == turn.entry_point
    assert hold.kwargs["env_cfg_entry_point"] != turn.kwargs["env_cfg_entry_point"]
    assert issubclass(BimanualBottleDirectEnv, DirectRLEnv)
