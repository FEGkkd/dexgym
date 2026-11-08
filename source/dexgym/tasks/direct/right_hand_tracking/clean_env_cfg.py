"""Independent 30 Hz clean-bank right-hand tracking task."""

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg

from .drive_experiment_cfg import TUNED_FULL_CONTACT_USD
from .env_cfg import REPO_ROOT, RightHandTrackingDirectCfg, RightHandTrackingPhysicsCfg


@configclass
class RightHandCleanSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = make_robot_cfg(usd_path=TUNED_FULL_CONTACT_USD)


@configclass
class RightHandCleanTrackingCfg(RightHandTrackingDirectCfg):
    """Keep the original tracking contract with the screened bank and tuned drive."""

    decimation = 16
    episode_length_s = 30.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 480, render_interval=16, physics=RightHandTrackingPhysicsCfg())
    scene: RightHandCleanSceneCfg = RightHandCleanSceneCfg(num_envs=16, env_spacing=2.0)
    library_path: str = str(REPO_ROOT / "data/processed/hand_only_right_collision_clean/manifest.json")
