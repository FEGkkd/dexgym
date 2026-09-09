import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.agents.rsl_rl_ppo_cfg import (
    WujiHandReorientationActorCfg,
    WujiHandReorientationCriticCfg,
    WujiHandReorientationPPORunnerCfg,
)


def test_wuji_ppo_network_defaults_match_reference():
    cfg = WujiHandReorientationPPORunnerCfg()
    assert cfg.actor.hidden_dims == [512, 256, 128]
    assert cfg.critic.hidden_dims == [512, 512, 256, 128]
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}


def test_wuji_actor_and_critic_are_independently_subclassable():
    actor = WujiHandReorientationActorCfg(hidden_dims=[64, 32])
    critic = WujiHandReorientationCriticCfg(hidden_dims=[128, 64])
    assert actor.hidden_dims == [64, 32]
    assert critic.hidden_dims == [128, 64]


def test_top_level_entrypoints_are_importable():
    from rsl_rl.base import play, train

    assert callable(train.main)
    assert callable(play.main)
    assert callable(play._disable_training_enhancements)
