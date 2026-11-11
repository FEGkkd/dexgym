"""Isolated right-hand drive diagnostics; never used by the PPO task registry."""

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg

from .env_cfg import RightHandTrackingDirectCfg


@configclass
class HandOnlyDriveSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = make_robot_cfg()


@configclass
class HandOnlyDriveExperimentCfg(RightHandTrackingDirectCfg):
    scene: HandOnlyDriveSceneCfg = HandOnlyDriveSceneCfg(num_envs=16, env_spacing=2.0)


def make_drive_experiment_cfg(
    *, physics_dt: float, num_envs: int, hand_contacts: bool, profile: str = "source"
) -> HandOnlyDriveExperimentCfg:
    """Build a diagnostic-only configuration with corrected angular drive units."""
    if physics_dt <= 0 or num_envs < 1:
        raise ValueError("physics_dt and num_envs must be positive")
    if profile not in {"source", "hand_x2"}:
        raise ValueError("profile must be source or hand_x2")
    cfg = HandOnlyDriveExperimentCfg()
    cfg.sim.dt = physics_dt
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = make_robot_cfg(hand_contacts=hand_contacts)
    if profile == "hand_x2":
        actuator = cfg.scene.robot.actuators["all"]
        actuator.stiffness = {
            name: gain * (2.0 if name.startswith(("l_", "r_")) else 1.0) for name, gain in actuator.stiffness.items()
        }
        actuator.damping = {
            name: gain * (2.0 if name.startswith(("l_", "r_")) else 1.0) for name, gain in actuator.damping.items()
        }
    return cfg
