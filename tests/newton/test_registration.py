import gymnasium as gym
from dexgym.tasks import CARTPOLE_TASK_IDS, HAND2_CUBE_SUPERMIMIC_ID, HAND21_TELEDEXTER_ID, TASK_IDS


def test_task_registration():
    assert len(TASK_IDS) == 11
    for task_id in TASK_IDS:
        spec = gym.spec(task_id)
        assert spec.kwargs['env_cfg_entry_point']
        if task_id in CARTPOLE_TASK_IDS or task_id == HAND21_TELEDEXTER_ID:
            assert spec.kwargs['rsl_rl_cfg_entry_point']
        else:
            assert 'rsl_rl_cfg_entry_point' not in spec.kwargs


def test_cube_is_the_only_registered_supermimic_task():
    registered = {key for key in gym.registry if key.startswith('DexGym-') and 'SuperMimic' in key}
    assert registered == {HAND2_CUBE_SUPERMIMIC_ID}
    assert not any(key.startswith('DexGym-Marvin-') for key in gym.registry)
    cube = gym.spec(HAND2_CUBE_SUPERMIMIC_ID)
    assert cube.entry_point == 'dexgym.tasks.direct.hand2_cube_supermimic.env:Hand2CubeSuperMimicEnv'
    assert cube.kwargs['supermimic_cfg_entry_point'].endswith(':default_experiment')


def test_independent_hand_ppo_remains_loadable():
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from dexgym.tasks.direct.hand21_ppo.env_cfg import WujiHand21TeleDexterCfg
    from isaaclab.envs import DirectRLEnvCfg

    assert WujiHand21TeleDexterCfg.__bases__ == (DirectRLEnvCfg,)
    config = load_cfg_from_registry(HAND21_TELEDEXTER_ID, 'rsl_rl_cfg_entry_point')
    assert type(config).__module__.startswith('dexgym.tasks.direct.hand21_ppo.')
    assert config.max_iterations == 50000
    assert config.actor.distribution_cfg.init_std == 1.0
