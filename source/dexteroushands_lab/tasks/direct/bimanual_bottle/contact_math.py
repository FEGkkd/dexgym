"""Force-weighted fingertip-site slip proxy using live PhysX link velocity.

This is a diagnostic proxy: a tip site can differ from the collision point.
Exact contact-point friction would require a separate sparse PhysX contact
stream, which is not currently batched at 1024 environments.
"""

import torch

from dexteroushands_lab.assets.bottle.batched_thread import _rotate


def surface_slip_speed(tip_positions_w: torch.Tensor, tip_velocities_w: torch.Tensor,
                       object_pose_w: torch.Tensor, object_velocity_w: torch.Tensor,
                       contact_weights_n: torch.Tensor) -> torch.Tensor:
    n, tips, _ = tip_positions_w.shape
    if (tip_velocities_w.shape != (n, tips, 3)
            or object_pose_w.shape != (n, 7)
            or object_velocity_w.shape != (n, 6)
            or contact_weights_n.shape != (n, tips)):
        raise ValueError("Slip proxy tensors have inconsistent environment/tip axes")
    axis = _rotate(object_pose_w[:, 3:], torch.tensor(
        [0., 0., 1.], device=object_pose_w.device,
        dtype=object_pose_w.dtype).expand(n, 3))
    r = tip_positions_w - object_pose_w[:, None, :3]
    axial = (r * axis[:, None]).sum(-1, keepdim=True)
    radial = r - axial * axis[:, None]
    normal = radial / torch.linalg.vector_norm(radial, dim=-1, keepdim=True).clamp_min(1e-9)
    object_point_velocity = (object_velocity_w[:, None, :3]
                             + torch.cross(object_velocity_w[:, None, 3:].expand(-1, tips, -1),
                                           r, dim=-1))
    relative = tip_velocities_w - object_point_velocity
    tangential = relative - (relative * normal).sum(-1, keepdim=True) * normal
    speed = torch.linalg.vector_norm(tangential, dim=-1)
    weights = contact_weights_n.clamp_min(0)
    return (speed * weights).sum(-1) / weights.sum(-1).clamp_min(1e-9)
