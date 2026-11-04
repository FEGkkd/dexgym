import gymnasium as gym
from dexgym.tasks import TASK_IDS


def test_task_registration():
    assert len(TASK_IDS) == 2
    for task_id in TASK_IDS:
        spec = gym.spec(task_id)
        assert spec.kwargs["rsl_rl_cfg_entry_point"]
        assert spec.kwargs["env_cfg_entry_point"]
