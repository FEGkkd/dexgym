from types import SimpleNamespace

import pytest

from dexteroushands_lab.tasks.direct.bimanual_bottle.variant_config import (
    configure_bimanual_hand_variant,
)


@pytest.mark.parametrize("task,old_probability", [
    ("grasp_hold", 0.65), ("cap_turn_30", 1.0),
    ("pregrasp_cap_turn_30", 0.0),
])
def test_beta1_replay_restores_historical_reset_distribution(task, old_probability):
    cfg = SimpleNamespace(task_kind=task, hand_variant="beta2",
                          stable_reset_probability=0.0, pregrasp_bank_path=None,
                          pregrasp_split_key="local_shape_seed",
                          pregrasp_split_manifest_path=None)
    configure_bimanual_hand_variant(cfg, "beta1")
    assert cfg.hand_variant == "beta1"
    assert cfg.stable_reset_probability == old_probability
    if task == "pregrasp_cap_turn_30":
        assert cfg.pregrasp_bank_path == "docs/assets/phase1b_pregrasp_bank.json"
        assert cfg.pregrasp_split_key == "right_natural_source_seed"
        assert cfg.pregrasp_split_manifest_path == "docs/assets/phase1b_pregrasp_split.json"


def test_beta2_selection_clears_beta1_pregrasp_manifest():
    cfg = SimpleNamespace(task_kind="pregrasp_cap_turn_30", hand_variant="beta1",
                          stable_reset_probability=0.0,
                          pregrasp_bank_path="docs/assets/phase1b_pregrasp_bank.json",
                          pregrasp_split_key="right_natural_source_seed",
                          pregrasp_split_manifest_path="docs/assets/phase1b_pregrasp_split.json")
    configure_bimanual_hand_variant(cfg, "beta2")
    assert cfg.hand_variant == "beta2"
    assert cfg.pregrasp_bank_path is None
    assert cfg.pregrasp_split_key == "local_shape_seed"
    assert cfg.pregrasp_split_manifest_path == "docs/assets/beta2_phase1b_pregrasp_split.json"


def test_unknown_hand_variant_is_rejected():
    with pytest.raises(ValueError, match="hand variant"):
        configure_bimanual_hand_variant(SimpleNamespace(), "beta3")
