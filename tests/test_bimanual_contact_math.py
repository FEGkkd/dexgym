import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.contact_math import surface_slip_speed


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
