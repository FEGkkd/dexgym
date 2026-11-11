import gymnasium as gym
from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.tasks import HAND2_CUBE_SUPERMIMIC_ID, TASK_IDS
from dexgym.tasks.direct.hand2_cube_supermimic.env_cfg import Hand2CubeSuperMimicCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def test_cube_is_the_only_registered_dexgym_task():
    registered = {key for key in gym.registry if key.startswith("DexGym-")}
    assert registered == {HAND2_CUBE_SUPERMIMIC_ID}
    assert TASK_IDS == (HAND2_CUBE_SUPERMIMIC_ID,)


def test_cube_registration_loads_environment_and_agent_configs():
    cube = gym.spec(HAND2_CUBE_SUPERMIMIC_ID)
    assert cube.entry_point == "dexgym.tasks.direct.hand2_cube_supermimic.env:Hand2CubeSuperMimicEnv"
    assert set(cube.kwargs) == {"env_cfg_entry_point", "supermimic_cfg_entry_point"}
    env_config = load_cfg_from_registry(cube.id, "env_cfg_entry_point")
    agent_config = load_cfg_from_registry(cube.id, "supermimic_cfg_entry_point")
    assert isinstance(env_config, Hand2CubeSuperMimicCfg)
    assert isinstance(agent_config, SuperMimicConfig)
