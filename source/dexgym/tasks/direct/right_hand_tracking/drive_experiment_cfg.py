"""Isolated right-hand drive diagnostics; never used by the PPO task registry."""

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg

from .env_cfg import RightHandTrackingDirectCfg

USD_DIR = Path(__file__).resolve().parents[3] / "assets/marvin_wuji/usd"
FULL_CONTACT_USD = USD_DIR / "hand_only_drive_corrected.usda"
NO_HAND_CONTACT_USD = USD_DIR / "hand_only_drive_corrected_no_hand_contacts.usda"
TUNED_FULL_CONTACT_USD = USD_DIR / "hand_only_drive_corrected_hand_x2.usda"
TUNED_NO_HAND_CONTACT_USD = USD_DIR / "hand_only_drive_corrected_hand_x2_no_hand_contacts.usda"


@configclass
class HandOnlyDriveSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = make_robot_cfg(usd_path=FULL_CONTACT_USD)


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
    usd = {
        ("source", True): FULL_CONTACT_USD,
        ("source", False): NO_HAND_CONTACT_USD,
        ("hand_x2", True): TUNED_FULL_CONTACT_USD,
        ("hand_x2", False): TUNED_NO_HAND_CONTACT_USD,
    }[(profile, hand_contacts)]
    cfg.scene.robot = make_robot_cfg(usd_path=usd)
    return cfg
