"""Changing one hand task must not change the other task's executable rules."""

import ast
from pathlib import Path

import gymnasium as gym
from dexgym.tasks import HAND21_SUPERMIMIC_ID, HAND21_TELEDEXTER_ID
from dexgym.tasks.direct.hand21_ppo import env as ppo_env
from dexgym.tasks.direct.hand21_ppo import logic as ppo_logic
from dexgym.tasks.direct.hand21_ppo import subgoals as ppo_rules
from dexgym.tasks.direct.hand21_ppo.env_cfg import WujiHand21TeleDexterCfg
from dexgym.tasks.direct.hand21_supermimic import env as sm_env
from dexgym.tasks.direct.hand21_supermimic import logic as sm_logic
from dexgym.tasks.direct.hand21_supermimic import subgoals as sm_rules
from dexgym.tasks.direct.hand21_supermimic.env_cfg import WujiHand21SuperMimicCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def test_registry_uses_different_environments_and_standalone_configs():
    assert gym.spec(HAND21_TELEDEXTER_ID).entry_point.endswith("hand21_ppo.env:Hand21PPOEnv")
    assert gym.spec(HAND21_SUPERMIMIC_ID).entry_point.endswith("hand21_supermimic.env:Hand21SuperMimicEnv")
    assert WujiHand21TeleDexterCfg.__bases__ == (DirectRLEnvCfg,)
    assert WujiHand21SuperMimicCfg.__bases__ == (DirectRLEnvCfg,)
    config = load_cfg_from_registry(HAND21_TELEDEXTER_ID, "rsl_rl_cfg_entry_point")
    assert type(config).__module__.startswith("dexgym.tasks.direct.hand21_ppo.")
    assert config.max_iterations == 50000
    assert config.actor.distribution_cfg.init_std == 1.0


def test_changes_to_ppo_rules_do_not_propagate_to_supermimic(monkeypatch):
    ppo = WujiHand21TeleDexterCfg()
    ppo.goal_tolerance_m = 0.010
    assert WujiHand21SuperMimicCfg().goal_tolerance_m == 0.015
    assert ppo_rules.advance_hand_goals is not sm_rules.advance_hand_goals
    assert ppo_logic.sample_goal_span is not sm_logic.sample_goal_span
    monkeypatch.setattr(ppo_env, "advance_hand_goals", None)
    monkeypatch.setattr(ppo_rules, "sample_goal_span", None)
    assert sm_env.advance_hand_goals is sm_rules.advance_hand_goals
    assert sm_rules.sample_goal_span is sm_logic.sample_goal_span


def test_changes_to_supermimic_config_do_not_propagate_to_ppo():
    sm = WujiHand21SuperMimicCfg()
    ppo = WujiHand21TeleDexterCfg()
    sm.goal_tolerance_m = 0.010
    sm.scene.num_envs = 99
    assert ppo.goal_tolerance_m == 0.015 and ppo.scene.num_envs == 16
    assert sm.scene is not ppo.scene
    assert sm.scene.robot is not ppo.scene.robot


def test_task_packages_do_not_import_each_other_or_legacy_task_implementation():
    for module, forbidden in [(ppo_env, "hand21_supermimic"), (sm_env, "hand21_ppo")]:
        directory = Path(module.__file__).parent
        for source in directory.rglob("*.py"):
            for node in ast.walk(ast.parse(source.read_text())):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                for name in names:
                    assert forbidden not in name and "right_hand_tracking" not in name, (source, name)
