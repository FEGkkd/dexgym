"""Independent contract checks for per-environment mimic-joint resistance."""

import torch

from dexteroushands_lab.assets.bottle.config import BottleConfig
from dexteroushands_lab.assets.bottle.mimic_resistance import BatchedMimicResistance


def test_tight_thread_resists_small_displacement_and_partial_reset_is_isolated():
    cfg = BottleConfig()
    model = BatchedMimicResistance(cfg, 2, device="cpu")
    theta = torch.tensor([0.02, 0.0])
    terms = model.step(theta, torch.zeros(2), torch.zeros(2))
    assert terms.torque_nm[0].item() < 0
    assert terms.torque_nm[1].item() == 0
    assert not terms.sliding.any()
    before = model.stick_reference_rad[1].clone()
    model.reset(torch.tensor([0]), torch.tensor([0.25]))
    assert model.stick_reference_rad[0].item() == torch.tensor(0.25).item()
    assert model.stick_reference_rad[1].item() == before.item()
    assert model.step(torch.tensor([0.25, 0.0]), torch.zeros(2), torch.zeros(2)).torque_nm[0].item() == 0


def test_looser_engagement_has_lower_breakaway_and_sliding_reduces_stick_torque():
    cfg = BottleConfig()
    model = BatchedMimicResistance(cfg, 2, device="cpu")
    theta = torch.tensor([0.20, 0.20])
    speed = torch.tensor([0.2, 0.2])
    axial = torch.tensor([0.0, cfg.thread_pitch * cfg.thread_turns * 0.9])
    terms = model.step(theta, speed, axial)
    assert terms.breakaway_limit_nm[0] > terms.breakaway_limit_nm[1]
    assert terms.engagement_fraction[0] > terms.engagement_fraction[1]
    assert terms.sliding.all()
    assert torch.allclose(terms.breakaway_nm, torch.zeros(2))
    assert torch.all(terms.running_nm < 0)
