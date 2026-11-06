import math

import torch

from dexgym.algorithms.supermimic.rewards import so3_geodesic
from dexgym.motion.motion_manager import quaternion_xyzw_to_matrix


def _rotation_z(theta):
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    c, s = theta.cos(), theta.sin()
    return torch.stack((torch.stack((c, -s, zero)),
                        torch.stack((s, c, zero)),
                        torch.stack((zero, zero, one))))


def test_so3_geodesic_has_finite_fp32_gradient_near_zero_and_pi():
    for angle in (0.0, 1e-4, math.pi - 1e-3):
        theta = torch.tensor(angle, dtype=torch.float32, requires_grad=True)
        value = so3_geodesic(_rotation_z(theta), torch.eye(3))
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(theta.grad)
        if angle > 0:
            assert abs(float(value.detach()) - angle) < 1e-4
            assert abs(float(theta.grad) - 1.0) < 2e-3


def test_quaternion_sign_does_not_change_reference_rotation():
    q = torch.tensor([0.13, -0.24, 0.37, 0.88], dtype=torch.float32)
    torch.testing.assert_close(quaternion_xyzw_to_matrix(q), quaternion_xyzw_to_matrix(-q))
