from isaaclab.utils.configclass import configclass
from isaaclab_assets import ALLEGRO_HAND_CFG

import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg


@configclass
class ManagerBasedAllegroReposeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    """Manager-Based baseline with local ownership of the task configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.clone_in_fabric = True


@configclass
class ManagerBasedAllegroReposeEnvCfg_PLAY(ManagerBasedAllegroReposeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None
