import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.reward_terms import (
    digit_contact_score, signed_progress_reward, thread_opening_progress,
    terminal_failure_penalty,
    suspended_bottle,
    grasp_contact_reward, physically_supported_turn_reward,
    physically_valid_grasp, thread_error_penalty,
)


def test_contact_score_is_saturated_and_requires_correct_object_view():
    names = ("l_thumb_distal", "l_index_finger_distal", "l_middle_finger_distal")
    force = torch.tensor([[1., 0., 1.], [1000., 0., 1000.]])
    score = digit_contact_score(force, names, ("thumb", "index_finger", "middle_finger"))
    assert torch.all(score >= 0)
    assert torch.all(score <= 1)
    assert score[1] <= 1
    assert score[0] < score[1]
    assert digit_contact_score(torch.zeros_like(force), names,
                               ("thumb", "index_finger", "middle_finger")).sum() == 0


def test_balanced_tripod_contact_requires_thumb_without_losing_partial_signal():
    names = ("r_thumb_distal", "r_index_finger_distal", "r_middle_finger_distal")
    forces = torch.tensor([[0.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
    score = digit_contact_score(
        forces, names, ("thumb", "index_finger", "middle_finger"),
        aggregation="balanced")
    assert 0.0 < score[0] < score[1]


def test_grasp_success_rejects_large_thread_constraint_error():
    true = torch.tensor([True, True])
    constraint_error = torch.tensor([0.0005, 0.00425])
    valid = physically_valid_grasp(true, true, true, true, constraint_error)
    assert valid.tolist() == [True, False]


def test_thread_error_penalty_is_zero_inside_tolerance_and_bounded_outside():
    error = torch.tensor([0.0005, 0.001, 0.002, 0.00425, 0.1, float("nan")])
    penalty = thread_error_penalty(error)
    assert penalty[0] == 0 and penalty[1] == 0
    assert 0 > penalty[2] > penalty[3] > -1
    assert -1 <= penalty[4] < penalty[3]
    assert penalty[5] == -1


def test_cap_progress_uses_real_relative_angle_and_penalizes_reverse():
    before = torch.zeros(3)
    actual = torch.tensor([0., 0.02, -0.02])
    score = signed_progress_reward(actual, before)
    assert score[0] == 0  # wrist-only motion gives zero real cap progress
    assert score[1] > 0
    assert score[2] < 0


def test_positive_turn_reward_requires_contact_and_pitch_consistency():
    opening = torch.tensor([.02, .02, .02, -.02])
    score = physically_supported_turn_reward(
        opening, torch.zeros_like(opening),
        torch.tensor([0., 0., .002, 0.]),
        torch.tensor([True, False, True, False]))
    assert score[0] > 0
    assert score[1] == 0  # no right-cap contact cannot earn opening reward
    assert score[2] == 0  # thread travel inconsistent with the pitch
    assert score[3] < 0  # reverse rotation remains penalized


def test_thread_progress_uses_each_reset_angle_and_axial_reference():
    theta = torch.tensor([-0.74, -0.74 + 0.5235987756])
    axial = torch.tensor([0.1451, 0.14535])
    opening, travel, expected = thread_opening_progress(
        theta, axial, torch.tensor([-0.74, -0.74]),
        torch.tensor([0.1451, 0.1451]), sign=1, pitch=0.003)
    assert opening[0].abs() < 1e-8
    assert travel[0].abs() < 1e-8
    assert torch.allclose(travel[1], expected[1], atol=1e-7)


def test_drop_has_explicit_terminal_cost_without_penalizing_success():
    failure = torch.tensor([False, True, False])
    assert terminal_failure_penalty(failure).tolist() == [0., -5., 0.]


def test_recovered_pregrasp_can_succeed_after_catching_falling_bottle():
    current = torch.tensor([[0., 0., .45], [0., 0., .10]])
    reset = torch.tensor([[0., 0., .50], [0., 0., .50]])
    orientation = torch.tensor([[0., 0., 0., 1.]] * 2)
    origin = torch.zeros(2, 3)
    valid = suspended_bottle(current, orientation, reset, orientation, origin)
    assert valid.tolist() == [True, False]


def test_grasp_reward_prefers_balanced_two_hand_contact():
    left = torch.tensor([0., .5])
    right = torch.tensor([1., 1.])
    score = grasp_contact_reward(left, right)
    assert score[0] > 0
    assert score[1] > score[0]
    assert grasp_contact_reward(torch.tensor([.5]), torch.tensor([.5])) > (
        grasp_contact_reward(torch.tensor([.5]), torch.tensor([0.])))
