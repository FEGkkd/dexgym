"""Exact object-to-hand PhysX contact matrices without per-step CPU loops.

The native filter API requires explicit partner paths. Its force matrix has
shape ``[N, N * body_count, 3]``; only the diagonal environment blocks are
valid. This O(N²) allocation is a measured backend limitation, so the caller
must benchmark memory before scaling to 1024 or more environments.
"""

from __future__ import annotations

import torch


def gather_local_contact_forces(matrix: torch.Tensor, *, num_bodies: int) -> torch.Tensor:
    if (matrix.ndim != 3 or matrix.shape[2] != 3 or num_bodies <= 0
            or matrix.shape[1] != matrix.shape[0] * num_bodies):
        raise ValueError("PhysX contact matrix does not match exact per-env filter order")
    n = matrix.shape[0]
    rows = torch.arange(n, device=matrix.device)[:, None]
    columns = rows * num_bodies + torch.arange(num_bodies, device=matrix.device)[None, :]
    return matrix[rows, columns]


class BatchedContactMonitor:
    """Left/right finger contact against both free objects, preserving body IDs."""

    def __init__(self, sim_view, num_envs: int, left_spec, right_spec, *, device: str,
                 force_floor_n: float = 0.01):
        if force_floor_n < 0:
            raise ValueError("Contact force floor must be nonnegative")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.force_floor_n = force_floor_n
        self.body_names = {"left": tuple(left_spec.body_names),
                           "right": tuple(right_spec.body_names)}
        self.views = {}
        self.ground_views = {}
        for obj in ("Bottle", "Cap"):
            for side, hand in (("left", "LeftHand"), ("right", "RightHand")):
                filters = [f"/World/envs/env_{env}/{hand}/{body}"
                           for env in range(num_envs) for body in self.body_names[side]]
                source = f"/World/envs/env_*/{obj}"
                view = sim_view.create_rigid_contact_view(
                    source, filter_patterns=filters, max_contact_data_count=8)
                expected_sources = [f"/World/envs/env_{env}/{obj}" for env in range(num_envs)]
                if (view is None or list(view.sensor_paths) != expected_sources
                        or any(list(row) != filters for row in view.filter_paths)):
                    raise RuntimeError(f"PhysX contact filter order mismatch for {obj}/{side}")
                self.views[(obj, side)] = view
            source = f"/World/envs/env_*/{obj}"
            ground = "/World/ground/GroundPlane/CollisionPlane"
            ground_view = sim_view.create_rigid_contact_view(
                source, filter_patterns=[ground], max_contact_data_count=8)
            expected_sources = [f"/World/envs/env_{env}/{obj}" for env in range(num_envs)]
            if (ground_view is None or list(ground_view.sensor_paths) != expected_sources
                    or any(list(row) != [ground] for row in ground_view.filter_paths)):
                raise RuntimeError(f"PhysX ground filter order mismatch for {obj}")
            self.ground_views[obj] = ground_view
        self.steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.active_steps = {key: torch.zeros(num_envs, device=device, dtype=torch.long)
                             for key in self.views}
        self.latest_forces = {key: torch.zeros(num_envs, len(self.body_names[key[1]]), 3,
                                               device=device) for key in self.views}
        self.latest_ground_force = {obj: torch.zeros(num_envs, 3, device=device)
                                    for obj in self.ground_views}

    def reset(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self.steps[ids] = 0
        for key in self.views:
            self.active_steps[key][ids] = 0
            self.latest_forces[key][ids] = 0
        for obj in self.ground_views:
            self.latest_ground_force[obj][ids] = 0

    def update(self, dt: float) -> dict[tuple[str, str], torch.Tensor]:
        import warp as wp

        self.steps += 1
        for key, view in self.views.items():
            matrix = wp.to_torch(view.get_contact_force_matrix(dt=dt))
            local = gather_local_contact_forces(matrix, num_bodies=len(self.body_names[key[1]]))
            self.latest_forces[key] = local
            self.active_steps[key] += (torch.linalg.vector_norm(local, dim=-1).amax(dim=-1)
                                       >= self.force_floor_n).long()
        for obj, view in self.ground_views.items():
            matrix = wp.to_torch(view.get_contact_force_matrix(dt=dt))
            if matrix.shape != (self.num_envs, 1, 3):
                raise RuntimeError(f"PhysX ground force shape mismatch for {obj}")
            self.latest_ground_force[obj] = matrix[:, 0]
        return self.latest_forces

    def force_magnitudes(self, obj: str, side: str) -> torch.Tensor:
        return torch.linalg.vector_norm(self.latest_forces[(obj, side)], dim=-1)

    def duration_ratio(self, obj: str, side: str) -> torch.Tensor:
        return self.active_steps[(obj, side)] / self.steps.clamp_min(1)

    def ground_contact(self, obj: str) -> torch.Tensor:
        return (torch.linalg.vector_norm(self.latest_ground_force[obj], dim=-1)
                >= self.force_floor_n)
