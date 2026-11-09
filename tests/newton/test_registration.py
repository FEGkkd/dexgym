import gymnasium as gym
from dexgym.tasks import (
    CARTPOLE_TASK_IDS,
    HAND21_SUPERMIMIC_ID,
    HAND21_TELEDEXTER_ID,
    TASK_IDS,
)


def test_task_registration():
    assert len(TASK_IDS) == 14
    for task_id in TASK_IDS:
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"]
        if task_id in CARTPOLE_TASK_IDS or task_id == HAND21_TELEDEXTER_ID:
            assert spec.kwargs["rsl_rl_cfg_entry_point"]
        else:
            assert "rsl_rl_cfg_entry_point" not in spec.kwargs


def test_hand_supermimic_has_its_own_algorithm_configuration():
    spec = gym.spec(HAND21_SUPERMIMIC_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":WujiHand21SuperMimicCfg")
    assert spec.kwargs["supermimic_cfg_entry_point"].endswith(":default_experiment")
    assert "rsl_rl_cfg_entry_point" not in spec.kwargs
    assert gym.spec(HAND21_TELEDEXTER_ID).kwargs["rsl_rl_cfg_entry_point"].endswith(":WujiHand21TeleDexterPPOCfg")
