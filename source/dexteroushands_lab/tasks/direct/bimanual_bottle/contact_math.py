"""Batched contact-point classification and fingertip-site slip diagnostics."""

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


def cap_sidewall_site_mask(tip_positions_w: torch.Tensor, cap_pose_w: torch.Tensor,
                           *, radius_m: float, half_height_m: float,
                           axial_inset_m: float, radial_tolerance_m: float) -> torch.Tensor:
    """Approximate batched sidewall *site* proxy, never an exact contact point.

    A distal body can touch the cap rim while its fingertip site is elsewhere.
    This mask rejects one measured rim case but has both false positives and
    false negatives in other poses. Use the manifold mode for physical claims.
    """
    if (tip_positions_w.ndim != 3 or tip_positions_w.shape[-1] != 3
            or cap_pose_w.shape != (tip_positions_w.shape[0], 7)):
        raise ValueError("Cap sidewall site tensors have incompatible shapes")
    if (radius_m <= 0 or half_height_m <= 0 or not 0 < axial_inset_m < half_height_m
            or radial_tolerance_m <= 0):
        raise ValueError("Cap sidewall site geometry must be positive and interior")
    n, count, _ = tip_positions_w.shape
    cap_q = cap_pose_w[:, 3:]
    inverse_q = torch.cat((-cap_q[:, :3], cap_q[:, 3:]), dim=-1)
    local = _rotate(inverse_q[:, None, :].expand(-1, count, -1).reshape(-1, 4),
                    (tip_positions_w - cap_pose_w[:, None, :3]).reshape(-1, 3))
    local = local.reshape(n, count, 3)
    radial_gap = torch.linalg.vector_norm(local[..., :2], dim=-1) - radius_m
    return ((local[..., 2].abs() <= half_height_m - axial_inset_m)
            & (radial_gap.abs() <= radial_tolerance_m)
            & torch.isfinite(local).all(dim=-1))


def gate_digit_forces_by_site(forces: torch.Tensor, body_names: tuple[str, ...],
                              site_mask: torch.Tensor, *, digits: tuple[str, ...]
                              ) -> torch.Tensor:
    """Retain other bodies; zero each requested digit's force if its site is outside."""
    if (forces.ndim != 2 or forces.shape[1] != len(body_names)
            or site_mask.shape != (forces.shape[0], len(digits))):
        raise ValueError("Finger force/site mask body order mismatch")
    gated = forces.clone()
    for digit_index, digit in enumerate(digits):
        indices = [index for index, body in enumerate(body_names) if digit in body]
        if not indices:
            raise ValueError(f"Finger body missing for sidewall gate: {digit}")
        gated[:, indices] *= site_mask[:, digit_index, None]
    return gated


def cap_sidewall_manifold_forces(force_buffer: torch.Tensor, point_buffer: torch.Tensor,
                                count_buffer: torch.Tensor, start_buffer: torch.Tensor,
                                cap_pose_w: torch.Tensor, body_names: tuple[str, ...],
                                *, digits: tuple[str, ...], radius_m: float,
                                half_height_m: float, axial_inset_m: float,
                                radial_tolerance_m: float,
                                per_pair_capacity: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum true PhysX cap-local sidewall point forces for each requested digit.

    PhysX filters are ordered by environment, then body. The contact count
    matrix has every sensor/filter combination; only diagonal environment
    blocks describe a local cap-to-hand pair. No GPU tensor is read by CPU.
    ``invalid`` flags per-pair truncation or a filled global point buffer.
    """
    n = cap_pose_w.shape[0]
    body_count = len(body_names)
    if (n < 1 or body_count < 1 or cap_pose_w.shape != (n, 7)
            or point_buffer.ndim != 2 or point_buffer.shape[1] != 3
            or force_buffer.ndim != 2 or force_buffer.shape[0] != point_buffer.shape[0]
            or force_buffer.shape[1] not in (1, 3)
            or count_buffer.shape != (n, n * body_count)
            or start_buffer.shape != count_buffer.shape
            or point_buffer.shape[0] == 0 or per_pair_capacity < 1
            or radius_m <= 0 or half_height_m <= axial_inset_m
            or axial_inset_m <= 0 or radial_tolerance_m <= 0):
        raise ValueError("Invalid batched PhysX cap contact data or sidewall geometry")
    rows = torch.arange(n, device=count_buffer.device)[:, None]
    columns = rows * body_count + torch.arange(body_count, device=count_buffer.device)[None, :]
    # The installed Warp/PhysX API exposes uint32 CUDA buffers; PyTorch does
    # not implement advanced CUDA indexing on that dtype.
    counts_i64 = count_buffer.long()
    starts_i64 = start_buffer.long()
    local_count = counts_i64[rows, columns]
    local_start = starts_i64[rows, columns]
    offsets = torch.arange(per_pair_capacity, device=count_buffer.device)
    raw_index = local_start[..., None] + offsets
    in_range = ((offsets < local_count[..., None])
                & (raw_index >= 0) & (raw_index < point_buffer.shape[0]))
    indices = raw_index.clamp(0, point_buffer.shape[0] - 1)
    points = point_buffer[indices]
    q = cap_pose_w[:, 3:]
    inverse_q = torch.cat((-q[:, :3], q[:, 3:]), dim=-1)
    local = _rotate(
        inverse_q[:, None, None, :].expand(-1, body_count, per_pair_capacity, -1).reshape(-1, 4),
        (points - cap_pose_w[:, None, None, :3]).reshape(-1, 3),
    ).reshape(n, body_count, per_pair_capacity, 3)
    radial = torch.linalg.vector_norm(local[..., :2], dim=-1)
    sidewall = ((local[..., 2].abs() <= half_height_m - axial_inset_m)
                & ((radial - radius_m).abs() <= radial_tolerance_m)
                & torch.isfinite(local).all(dim=-1))
    force_n = torch.linalg.vector_norm(force_buffer[indices], dim=-1)
    force_n = torch.where(in_range & sidewall & torch.isfinite(force_n),
                          force_n, torch.zeros_like(force_n))
    per_body = force_n.sum(dim=-1)
    by_digit = []
    for digit in digits:
        selected = [index for index, name in enumerate(body_names) if digit in name]
        if not selected:
            raise ValueError(f"No PhysX cap contact filter for digit {digit}")
        by_digit.append(per_body[:, selected].sum(dim=-1))
    invalid = ((local_count > per_pair_capacity).any(dim=-1)
               | (counts_i64.sum() >= point_buffer.shape[0]))
    return torch.stack(by_digit, dim=-1), invalid
