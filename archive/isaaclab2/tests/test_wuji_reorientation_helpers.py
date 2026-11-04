import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.curriculum import CurriculumState
from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.disturbance import DisturbanceScheduler
from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.randomization import DomainRandomizationState
from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.wuji_hand_reorientation_env_cfg import (
    WujiCurriculumCfg,
    WujiDisturbanceCfg,
    WujiDomainRandomizationCfg,
)


def test_disabled_helpers_return_neutral_state():
    curriculum = CurriculumState(WujiCurriculumCfg(enabled=False), 2, "cpu")
    progress = curriculum.update(torch.tensor([True, False]), torch.tensor([1, 2]))
    assert torch.equal(progress["scale"], torch.ones(2))
    disturbance = DisturbanceScheduler(WujiDisturbanceCfg(enabled=False), 2, "cpu")
    disturbance.reset(torch.tensor([0, 1]), 0.0)
    trigger, direction, velocity = disturbance.step(torch.zeros(2), torch.ones(2))
    assert not trigger.any() and torch.count_nonzero(direction) == 0
    assert torch.count_nonzero(velocity) == 0
    randomization = DomainRandomizationState(WujiDomainRandomizationCfg(enabled=False), 2, "cpu")
    randomization.reset(torch.tensor([0, 1]))
    assert torch.equal(randomization.ratios, torch.ones(2, 6))
    assert torch.count_nonzero(randomization.encoder_bias) == 0


def test_randomization_reset_isolated_by_env_ids():
    state = DomainRandomizationState(WujiDomainRandomizationCfg(), 3, "cpu")
    state.reset(torch.tensor([0, 1]))
    before = state.ratios[2].clone()
    state.reset(torch.tensor([0]))
    assert torch.equal(state.ratios[2], before)


def test_curriculum_success_threshold_changes_scale():
    state = CurriculumState(WujiCurriculumCfg(success_count_threshold=2), 1, "cpu")
    state.update(torch.tensor([True]), torch.tensor([10]))
    state.update(torch.tensor([True]), torch.tensor([10]))
    assert state.scale.item() > 1.0


def test_disturbance_trigger_records_velocity():
    cfg = WujiDisturbanceCfg(warmup_time_s=0.0, interval_s=(0.1, 0.1), speed_range=(0.1, 0.1))
    scheduler = DisturbanceScheduler(cfg, 1, "cpu")
    scheduler.reset(torch.tensor([0]), 0.0)
    trigger, direction, velocity = scheduler.step(torch.tensor([0.2]), torch.tensor([1.0]))
    assert trigger.item()
    assert torch.isclose(direction.norm(), torch.ones(()), atol=1e-5)
    assert torch.isclose(velocity[0, :3].norm(), torch.tensor(0.1), atol=1e-5)
