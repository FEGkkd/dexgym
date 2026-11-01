import gymnasium as gym

from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn import register_tasks


def test_unified_task_is_a_separate_registered_direct_env():
    register_tasks()
    spec = gym.spec("DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0")
    assert "bimanual_pregrasp_capturn" in spec.entry_point
    assert "bimanual_pregrasp_capturn" in spec.kwargs["env_cfg_entry_point"]
    assert "bimanual_pregrasp_capturn" in spec.kwargs["rsl_rl_cfg_entry_point"]
