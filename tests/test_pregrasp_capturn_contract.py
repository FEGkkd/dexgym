import math

import torch

from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.contract import (
    capture_hold_anchor, valid_turn_step, bounded_turn_progress,
)
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.reward_terms import (
    unified_reward_terms, accumulate_reward_terms, SUCCESS_BONUS,
)


def test_anchor_latches_once_and_never_follows_later_drift():
    reset = torch.tensor([[0., 0., .5, 0., 0., 0., 1.]])
    caught = reset.clone()
    caught[0, 2] -= .008
    anchor, latched = capture_hold_anchor(reset, torch.tensor([False]), caught,
                                           torch.tensor([True]))
    assert latched.item()
    assert torch.allclose(anchor, caught)
    drifted = caught.clone()
    drifted[0, 2] -= .03
    unchanged, latched = capture_hold_anchor(anchor, latched, drifted,
                                              torch.tensor([False]))
    assert latched.item()
    assert torch.allclose(unchanged, caught)


def test_success_requires_stability_and_rejects_angle_jump():
    opening = torch.tensor([math.pi / 6, math.pi / 6, math.pi / 6])
    previous = torch.tensor([.45, .45, -.5])
    expected = opening * .003 / (2 * math.pi)
    good = valid_turn_step(opening, previous, expected, expected,
                           torch.zeros(3), torch.tensor([True, False, True]),
                           goal_rad=math.pi / 6)
    assert good.tolist() == [True, False, False]


def test_contact_or_angle_anomalies_never_pass_success():
    opening = torch.tensor([float("nan"), math.pi / 6, 6.74])
    previous = torch.tensor([0., .45, 0.])
    axial = torch.tensor([0., .00025, .003 * 6.74 / (2 * math.pi)])
    expected = axial.clone()
    supported = torch.tensor([True, False, True])
    accepted = valid_turn_step(opening, previous, axial, expected,
                               torch.zeros(3), supported,
                               goal_rad=math.pi / 6)
    assert accepted.tolist() == [False, False, False]
    progress = bounded_turn_progress(opening, previous, supported,
                                     goal_rad=math.pi / 6)
    assert progress[0] < 0 and progress[1] == 0 and progress[2] < 0


def test_positive_progress_caps_at_goal_and_needs_support():
    previous = torch.tensor([.50, .50, .50, .50])
    current = torch.tensor([.60, .60, .40, 6.74])
    supported = torch.tensor([True, False, False, True])
    progress = bounded_turn_progress(current, previous, supported,
                                      goal_rad=math.pi / 6)
    assert 0 < progress[0] < 1
    assert progress[1] == 0
    assert progress[2] < 0
    assert progress[3] <= 0


def test_no_turn_dense_return_cannot_outscore_success():
    one = torch.ones(1)
    zero = torch.zeros(1)
    terms = unified_reward_terms(
        one, one, one, one, zero, zero, zero, zero, one.bool(),
        one.bool(), zero, torch.zeros(1, 52), dt=.02,
        goal_rad=math.pi / 6, pitch_tolerance_m=.00005)
    # Approach is a signed potential difference: this 1-point jump occurs
    # once, then its value is zero while proximity is held fixed.
    persistent = sum(value for name, value in terms.items() if name != "approach")
    assert (terms["approach"] + 250 * persistent).item() < SUCCESS_BONUS
    assert terms["stability"].item() > 0
    released = unified_reward_terms(
        one, zero, one, one, zero, zero, zero, zero, zero.bool(),
        one.bool(), zero, torch.zeros(1, 52), dt=.02,
        goal_rad=math.pi / 6, pitch_tolerance_m=.00005)
    assert released["stability"].item() == 0


def test_contact_coverage_profile_remains_bounded_below_success():
    one = torch.ones(1)
    zero = torch.zeros(1)
    terms = unified_reward_terms(
        one, one, one, one, zero, zero, zero, zero, one.bool(),
        one.bool(), zero, torch.zeros(1, 52), dt=.02,
        goal_rad=math.pi / 6, pitch_tolerance_m=.00005,
        profile="contact_coverage")
    persistent = sum(value for name, value in terms.items() if name != "approach")
    assert (terms["approach"] + 250 * persistent).item() < SUCCESS_BONUS
    assert terms["left_stability"].item() > 0
    assert terms["right_contact_dense"].item() > 0
    assert terms["right_proximity_hold"].item() > 0


def test_reward_accumulator_accepts_profile_specific_terms():
    sums = {"contact": torch.zeros(2)}
    terms = {"contact": torch.ones(2), "left_stability": torch.full((2,), .5)}
    accumulate_reward_terms(sums, terms)
    assert torch.allclose(sums["contact"], torch.ones(2))
    assert torch.allclose(sums["left_stability"], torch.full((2,), .5))
