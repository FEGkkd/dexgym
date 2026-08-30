from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents.rsl_rl_ppo_cfg import (
    AllegroCubePPORunnerCfg,
)


@configclass
class ManagerBasedAllegroReposePPORunnerCfg(AllegroCubePPORunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    experiment_name = "dexteroushands_allegro_managerbased"
