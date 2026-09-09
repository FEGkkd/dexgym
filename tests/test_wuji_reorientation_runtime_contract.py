import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from rsl_rl.base.play import _disable_training_enhancements
from rsl_rl import cli_args
from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.wuji_hand_reorientation_env_cfg import (
    WujiHandReorientationDirectEnvCfg,
)


def test_play_override_disables_all_training_enhancements():
    cfg = WujiHandReorientationDirectEnvCfg()
    _disable_training_enhancements(cfg)
    assert cfg.enable_curriculum is False
    assert cfg.enable_domain_randomization is False
    assert cfg.enable_disturbance is False
    assert cfg.curriculum.enabled is False
    assert cfg.domain_randomization.enabled is False
    assert cfg.disturbance.enabled is False
    assert cfg.observation_noise_enabled is False


def test_runtime_document_keeps_usd_boundary_explicit():
    doc = Path(__file__).resolve().parents[1] / "docs" / "wuji_hand_reorientation_training.md"
    text = doc.read_text()
    assert "不会转换或修改 USD" in text
    assert "enable_domain_randomization" in text


def test_cli_feature_gates_disable_nested_groups():
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    args = parser.parse_args(
        ["--disable_curriculum", "--disable_domain_randomization", "--disable_disturbance"]
    )
    cfg = WujiHandReorientationDirectEnvCfg()
    cli_args.apply_env_cfg_overrides(cfg, args)
    assert not cfg.enable_curriculum and not cfg.curriculum.enabled
    assert not cfg.enable_domain_randomization and not cfg.domain_randomization.enabled
    assert not cfg.enable_disturbance and not cfg.disturbance.enabled
