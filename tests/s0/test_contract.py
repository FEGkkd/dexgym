"""S0 actor input and Gym registration are explicit deployment contracts."""

import json
from importlib.resources import files

import dexgym.tasks  # noqa: F401
import gymnasium as gym


def test_s0_registered_without_replacing_cartpole_tasks():
    assert "DexGym-Marvin-Bimanual-SE3-Direct-v0" in gym.registry
    assert "DexGym-Joint-Tracking-Direct-v0" in gym.registry
    assert "DexGym-Joint-Tracking-Manager-v0" in gym.registry


def test_actor_contract_has_73_features_and_no_goal_joint_angles():
    schema = json.loads(files("dexgym.tasks.marvin_s0").joinpath("observation_schema.json").read_text())
    assert sum(item["size"] for item in schema["fields"]) == 73
    assert all("q_goal" not in item["name"] for item in schema["fields"])
    assert schema["quaternion_order"] == "xyzw"
    action = json.loads(files("dexgym.tasks.marvin_s0").joinpath("action_schema.json").read_text())
    assert action["action_size"] == 14
    assert action["policy_period_s"] == 0.01
    assert action["real_robot_safety_certified"] is False
