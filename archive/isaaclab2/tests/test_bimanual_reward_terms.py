import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.reward_terms import (
    GRASP_SUCCESS_BONUS, CAP_TURN_SUCCESS_BONUS,
    time_integrated_contact_reward,
    digit_contact_score, signed_progress_reward, thread_opening_progress,
    terminal_failure_penalty,
    suspended_bottle,
    grasp_contact_reward, physically_supported_turn_reward,
    physically_valid_grasp, thread_error_penalty, cap_turn_physics_valid,
    cylinder_surface_proximity,
    bottle_hold_stability,
    grasp_hold_motion_penalty,
    grasp_contact_stability_bonus,
)


def test_success_outscores_five_seconds_of_maximal_dense_contact():
    """A policy cannot improve return merely by avoiding early success."""
    one = torch.ones(1)
    horizon_steps = 250  # 5 s at the configured 50 Hz policy rate
    for task, bonus in (("grasp_hold", GRASP_SUCCESS_BONUS),
                        ("cap_turn_30", CAP_TURN_SUCCESS_BONUS)):
        contact, hold = time_integrated_contact_reward(
            one, one, one, dt=0.02, task_kind=task)
        assert float((contact + hold) * horizon_steps) < bonus


def test_grasp_hold_motion_penalty_is_bounded_and_tracks_real_motion():
    velocity = torch.tensor([
        [0., 0., 0., 0., 0., 0.],
        [0.05, 0., 0., 0., 0., 0.5],
        [0.1, 0., 0., 0., 0., 1.0],
    ])
    cap_spin = torch.tensor([0., 0.2, 0.4])
    penalty = grasp_hold_motion_penalty(velocity, cap_spin, dt=0.02)
    assert penalty.shape == (3,)
    assert torch.isfinite(penalty).all()
    assert penalty[0] == 0
    assert penalty[2] < penalty[1] < penalty[0]
    extreme = grasp_hold_motion_penalty(
        torch.full((1, 6), 1.0e6), torch.tensor([1.0e6]), dt=0.02)
    assert torch.isfinite(extreme).all()
    assert extreme.item() > -0.25


def test_stability_bonus_requires_bilateral_contact_and_favors_quiet_hold():
    left = torch.tensor([1., 1., 1., 0.])
    right = torch.tensor([1., 1., 1., 1.])
    velocity = torch.tensor([
        [0., 0., 0., 0., 0., 0.],
        [0.05, 0., 0., 0., 0., 0.5],
        [0.1, 0., 0., 0., 0., 1.0],
        [0., 0., 0., 0., 0., 0.],
    ])
    spin = torch.tensor([0., 0.2, 0.4, 0.])
    bonus = grasp_contact_stability_bonus(left, right, velocity, spin, dt=0.02)
    assert torch.isfinite(bonus).all()
    assert bonus[0] > bonus[1] > bonus[2] > 0
    assert bonus[3] == 0
    assert bonus.max() <= 0.04
    contact, hold = time_integrated_contact_reward(
        torch.ones(1), torch.ones(1), torch.ones(1), dt=0.02,
        task_kind="grasp_hold")
    assert float((contact + hold + bonus[:1]) * 250) < GRASP_SUCCESS_BONUS


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


def test_balanced_contact_weight_makes_partial_site_contact_denser_without_changing_full_contact():
    names = ("r_thumb_distal", "r_index_finger_distal", "r_middle_finger_distal")
    partial = torch.tensor([[2.0, 0.0, 0.0]])
    full = torch.tensor([[2.0, 2.0, 2.0]])
    old_partial = digit_contact_score(
        partial, names, ("thumb", "index_finger", "middle_finger"),
        aggregation="balanced", balanced_weight=0.8)
    dense_partial = digit_contact_score(
        partial, names, ("thumb", "index_finger", "middle_finger"),
        aggregation="balanced", balanced_weight=0.2)
    old_full = digit_contact_score(
        full, names, ("thumb", "index_finger", "middle_finger"),
        aggregation="balanced", balanced_weight=0.8)
    dense_full = digit_contact_score(
        full, names, ("thumb", "index_finger", "middle_finger"),
        aggregation="balanced", balanced_weight=0.2)
    assert dense_partial.item() > old_partial.item() > 0
    assert torch.allclose(old_full, dense_full)


def test_sidewall_proximity_prefers_interior_tripod_sites_over_bottom_rim():
    # The diagnosed index collides at z=-12.47 mm on a 25 mm cap.
    near_sidewall = torch.tensor([[[0.019, 0., 0.], [0.019, 0., -0.002],
                                   [0.019, 0., 0.003], [0.5, 0., 0.], [0.5, 0., 0.]]])
    bottom_rim = near_sidewall.clone()
    bottom_rim[0, 1, 2] = -0.01247
    kwargs = dict(radius_m=0.019, half_height_m=0.0125, radial_scale_m=0.010,
                  axial_scale_m=0.005, axial_inset_m=0.005, digit_count=3)
    sidewall_score = cylinder_surface_proximity(near_sidewall, **kwargs)
    rim_score = cylinder_surface_proximity(bottom_rim, **kwargs)
    assert sidewall_score.item() > 0.99
    assert rim_score.item() < sidewall_score.item() - 0.1


def test_hold_stability_rejects_captured_spinning_drifting_false_success():
    anchor = torch.tensor([[0., 0., .5, 0., 0., 0., 1.]])
    captured = torch.tensor([[-.021675, .015589, .488844,
                              .040830, .091233, -.076073, .992080]])
    velocity = torch.tensor([[-.022099, .059330, -.041539,
                              1.305736, -1.662306, 1.560694]])
    assert not bottle_hold_stability(
        captured, velocity, anchor, cap_spin_velocity_rad_s=torch.tensor([4.25077]))[0]
    nearly_still = anchor.clone()
    nearly_still[0, 0] = .005
    assert bottle_hold_stability(
        nearly_still, torch.zeros_like(velocity), anchor,
        cap_spin_velocity_rad_s=torch.zeros(1))[0]
    # Cap turning is permitted only when the bottle itself remains stable.
    assert bottle_hold_stability(
        nearly_still, torch.zeros_like(velocity), anchor,
        cap_spin_velocity_rad_s=None)[0]


def test_hold_stability_checks_each_motion_limit_independently():
    anchor = torch.tensor([[0., 0., .5, 0., 0., 0., 1.]])
    rest = torch.zeros(1, 6)
    spin = torch.zeros(1)
    assert bottle_hold_stability(anchor, rest, anchor, spin).item()
    drifted = anchor.clone()
    drifted[0, 0] = .021
    assert not bottle_hold_stability(drifted, rest, anchor, spin).item()
    tilted = anchor.clone()
    tilted[0, 5] = .1
    tilted[0, 6] = (1 - .1 ** 2) ** .5
    assert not bottle_hold_stability(tilted, rest, anchor, spin).item()
    linear = rest.clone()
    linear[0, 0] = .051
    assert not bottle_hold_stability(anchor, linear, anchor, spin).item()
    angular = rest.clone()
    angular[0, 5] = .51
    assert not bottle_hold_stability(anchor, angular, anchor, spin).item()
    assert not bottle_hold_stability(anchor, rest, anchor, torch.tensor([.21])).item()
    assert bottle_hold_stability(anchor, rest, anchor, cap_spin_velocity_rad_s=None).item()


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


def test_cap_turn_requires_correct_signed_axial_travel_at_the_30_degree_scale():
    opening = torch.full((4,), torch.pi / 6)
    expected = torch.full((4,), 0.00025)
    actual = torch.tensor([0.00026, -0.00020, 0.00035, 0.000001])
    good = cap_turn_physics_valid(
        opening, actual, expected, torch.zeros(4),
        torch.ones(4, dtype=torch.bool), goal_rad=torch.pi / 6,
        pitch_tolerance_m=0.00005)
    assert good.tolist() == [True, False, False, False]


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
