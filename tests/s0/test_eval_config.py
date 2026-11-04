"""Evaluation must pass the same RSL-RL config migration as official play."""

from dexgym.tasks.marvin_s0.ppo_cfg import MarvinS0PPOCfg, compatible_agent_cfg
from isaaclab.utils import to_dict


def test_eval_config_removes_legacy_stochastic_fields():
    cfg = compatible_agent_cfg(MarvinS0PPOCfg())
    assert "stochastic" not in to_dict(cfg)["actor"]
    assert "stochastic" not in to_dict(cfg)["critic"]
