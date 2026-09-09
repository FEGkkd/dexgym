import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.wuji_hand_reorientation_env_cfg import (
    WujiCurriculumCfg,
    WujiDisturbanceCfg,
    WujiDomainRandomizationCfg,
    WujiHandReorientationDirectEnvCfg,
)


def test_wuji_config_groups_are_enabled_with_explicit_ranges():
    cfg = WujiHandReorientationDirectEnvCfg()
    assert cfg.enable_curriculum and cfg.curriculum.enabled
    assert cfg.enable_domain_randomization and cfg.domain_randomization.enabled
    assert cfg.enable_disturbance and cfg.disturbance.enabled
    assert cfg.domain_randomization.object_mass_scale == (0.4, 1.6)
    assert cfg.disturbance.interval_s[0] < cfg.disturbance.interval_s[1]


def test_wuji_config_groups_are_independent():
    cfg = WujiHandReorientationDirectEnvCfg()
    cfg.enable_curriculum = False
    cfg.domain_randomization = WujiDomainRandomizationCfg(enabled=False)
    cfg.disturbance = WujiDisturbanceCfg(enabled=False)
    cfg.curriculum = WujiCurriculumCfg(enabled=False)
    assert not cfg.enable_curriculum and not cfg.curriculum.enabled
    assert not cfg.domain_randomization.enabled
    assert not cfg.disturbance.enabled
