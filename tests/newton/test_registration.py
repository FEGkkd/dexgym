import gymnasium as gym
from dexgym.tasks import (
    CARTPOLE_TASK_IDS,
    HAND21_TELEDEXTER_ID,
    TASK_IDS,
)


def test_task_registration():
    assert len(TASK_IDS) == 13
    for task_id in TASK_IDS:
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"]
        if task_id in CARTPOLE_TASK_IDS or task_id == HAND21_TELEDEXTER_ID:
            assert spec.kwargs["rsl_rl_cfg_entry_point"]
        else:
            assert "rsl_rl_cfg_entry_point" not in spec.kwargs
