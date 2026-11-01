import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.contact_math import (
    cap_sidewall_site_mask, gate_digit_forces_by_site, surface_slip_speed,
    cap_sidewall_manifold_forces,
)


def test_surface_slip_proxy_uses_tangential_relative_motion():
    tip = torch.tensor([[[.035, 0., 0.]]])
    obj = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])
    vel = torch.zeros(1, 6)
    weights = torch.ones(1, 1)
    assert surface_slip_speed(tip, torch.zeros_like(tip), obj, vel, weights).item() == 0
    assert surface_slip_speed(tip, torch.tensor([[[1., 0., 0.]]]), obj, vel,
                              weights).item() < 1e-6  # radial approach is not sliding
    assert surface_slip_speed(tip, torch.tensor([[[0., 1., 0.]]]), obj, vel,
                              weights).item() == 1


def test_cap_sidewall_site_mask_uses_cap_local_axis_after_tilt():
    # A +90 degree world-X rotation makes the cap's local Z axis world -Y.
    root = torch.tensor([[.2, .3, .4, 2**-.5, 0., 0., 2**-.5]])
    local = torch.tensor([[[.017, 0., 0.], [.017, 0., -.013],
                           [.020, 0., .003], [.030, 0., 0.]]])
    from dexteroushands_lab.assets.bottle.batched_thread import _rotate

    world = root[:, None, :3] + _rotate(root[:, None, 3:].expand(-1, 4, -1)
                                        .reshape(4, 4), local.reshape(4, 3)).reshape(1, 4, 3)
    eligible = cap_sidewall_site_mask(
        world, root, radius_m=.017, half_height_m=.0125,
        axial_inset_m=.002, radial_tolerance_m=.010)
    assert eligible.tolist() == [[True, False, True, False]]


def test_site_gate_rejects_measured_bottom_rim_index_contact():
    # Last frame of row-3 PhysX probe: index tip z=-13.47 mm, and its
    # collision point was on the cap's bottom rim at approximately -12.5 mm.
    cap = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])
    tips = torch.tensor([[[-.0076, -.0152, -.0077],
                          [-.0128, -.0120, -.01347],
                          [-.0204, -.0087, .00745]]])
    site = cap_sidewall_site_mask(
        tips, cap, radius_m=.017, half_height_m=.0125,
        axial_inset_m=.002, radial_tolerance_m=.010)
    assert site.tolist() == [[True, False, True]]
    forces = torch.tensor([[3.12, 2.73, 1.55, .7]])
    body_names = ("r_thumb_distal", "r_index_finger_distal",
                  "r_middle_finger_distal", "r_pinky_distal")
    gated = gate_digit_forces_by_site(
        forces, body_names, site,
        digits=("thumb", "index_finger", "middle_finger"))
    assert torch.allclose(gated, torch.tensor([[3.12, 0., 1.55, .7]]))


def test_batched_manifold_forces_use_true_cap_local_points_and_environment_diagonal():
    names = ("r_thumb_distal", "r_index_finger_distal", "r_middle_finger_distal")
    cap = torch.tensor([[0., 0., 0., 0., 0., 0., 1.],
                        [.1, .2, .3, 2**-.5, 0., 0., 2**-.5]])
    points = torch.tensor([[.017, 0., 0.], [.017, 0., -.0124], [.017, 0., .002],
                           [.017, 0., 0.], [.117, .2, .3], [.117, .196, .3],
                           [.117, .205, .3], [0., 0., 0.]])
    forces = torch.tensor([[2.], [5.], [3.], [100.], [4.], [5.], [6.], [0.]])
    # Installed PhysX/Warp returns CUDA uint32 count/start buffers.
    counts = torch.tensor([[1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.uint32)
    starts = torch.tensor([[0, 1, 2, 3, 0, 0], [0, 0, 0, 4, 5, 6]], dtype=torch.uint32)
    selected, invalid = cap_sidewall_manifold_forces(
        forces, points, counts, starts, cap, names,
        digits=("thumb", "index_finger", "middle_finger"),
        radius_m=.017, half_height_m=.0125,
        axial_inset_m=.001, radial_tolerance_m=.004,
        per_pair_capacity=2)
    assert torch.allclose(selected, torch.tensor([[2., 0., 3.], [4., 5., 6.]]))
    assert invalid.tolist() == [False, False]
    over_counts = counts.clone()
    over_counts[0, 0] = 3
    _, overflow = cap_sidewall_manifold_forces(
        torch.cat((forces, torch.zeros(2, 1))),
        torch.cat((points, torch.zeros(2, 3))),
        over_counts, starts, cap, names,
        digits=("thumb", "index_finger", "middle_finger"),
        radius_m=.017, half_height_m=.0125,
        axial_inset_m=.001, radial_tolerance_m=.004,
        per_pair_capacity=2)
    assert overflow.tolist() == [True, False]
