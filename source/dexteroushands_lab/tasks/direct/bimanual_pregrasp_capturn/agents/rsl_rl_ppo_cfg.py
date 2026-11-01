"""One 52D Gaussian actor with an asymmetric privileged critic."""

from isaaclab.utils.configclass import configclass

from dexteroushands_lab.tasks.direct.bimanual_bottle.agents.rsl_rl_ppo_cfg import BimanualBottlePPORunnerCfg


@configclass
class UnifiedPPORunnerCfg(BimanualBottlePPORunnerCfg):
    experiment_name = "dexteroushands_pregrasp_capturn30_phase1b"
    max_iterations = 1000
