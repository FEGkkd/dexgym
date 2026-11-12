"""Differentiable clearance estimates for finger and palm collision geometry."""

import mujoco
import numpy as np
import torch


def segment_distance(a, b):
    """Return the closest distance between each pair of finite line segments."""
    p, q = a[..., 0, :], b[..., 0, :]
    u, v = a[..., 1, :] - p, b[..., 1, :] - q
    w = p - q
    uu, vv = u.square().sum(-1).clamp_min(1e-12), v.square().sum(-1).clamp_min(1e-12)
    uv, uw, vw = (u * v).sum(-1), (u * w).sum(-1), (v * w).sum(-1)
    determinant = (uu * vv - uv.square()).clamp_min(1e-14)
    s = (uv * vw - vv * uw) / determinant
    t = (uu * vw - uv * uw) / determinant
    inside = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    interior = w + s.clamp(0, 1)[..., None] * u - t.clamp(0, 1)[..., None] * v

    def endpoint(offset, axis, length):
        fraction = ((offset * axis).sum(-1) / length).clamp(0, 1)
        return (offset - fraction[..., None] * axis).square().sum(-1)

    distances = (
        torch.where(inside, interior.square().sum(-1), 1.0),
        endpoint(w, v, vv), endpoint(w + u, v, vv),
        endpoint(-w, u, uu), endpoint(v - w, u, uu),
    )
    return torch.stack(distances).amin(0).clamp_min(1e-12).sqrt()


class FingerClearance:
    def __init__(self, model, device):
        capsules, radius, bodies = [], [], []
        finger_pairs, thumb_ids, palm_bodies, palm_centers, palm_rotations, palm_half = [], [], [], [], [], []
        hand_ids = []

        def mesh_vertices(geom):
            i = model.geom_dataid[geom]
            start = model.mesh_vertadr[i]
            return model.mesh_vert[start:start + model.mesh_vertnum[i]].astype(np.float64)

        def rotation(geom):
            out = np.empty(9)
            mujoco.mju_quat2Mat(out, model.geom_quat[geom])
            return out.reshape(3, 3)

        for side in ("l", "r"):
            fingers = []
            for finger in range(1, 6):
                indices = []
                for link in ("link2", "link3", "link4", "tip_sensor"):
                    geom = model.geom(f"{side}_finger{finger}_{link}_col").id
                    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        r, half = model.geom_size[geom, :2]
                        line = np.array([[0, 0, -half], [0, 0, half]])
                    else:
                        vertices = mesh_vertices(geom)
                        center = vertices.mean(0)
                        offset = vertices - center
                        axis = np.linalg.svd(offset, full_matrices=False)[2][0]
                        distance = offset @ axis
                        r = np.quantile(np.linalg.norm(offset - distance[:, None] * axis, axis=-1), 0.85)
                        half = max(0., (distance.max() - distance.min()) / 2 - 0.7 * r)
                        middle = center + (distance.max() + distance.min()) / 2 * axis
                        line = middle + np.array([[-half], [half]]) * axis
                    indices.append(len(capsules))
                    capsules.append(line @ rotation(geom).T + model.geom_pos[geom])
                    radius.append(r)
                    bodies.append(model.geom_bodyid[geom])
                fingers.append(indices)
            for first, second in zip(fingers[:-1], fingers[1:], strict=True):
                finger_pairs.extend((a, b) for a in first for b in second)
            thumb_ids.append(fingers[0][1:])
            hand_ids.append([geom for finger in fingers for geom in finger])
            geom = model.geom(f"{side}_palm_collision").id
            verts = mesh_vertices(geom)
            rot = rotation(geom)
            low, high = verts.min(0), verts.max(0)
            palm_centers.append(rot @ ((low + high) / 2) + model.geom_pos[geom])
            palm_half.append((high - low) / 2)
            palm_rotations.append(rot)
            palm_bodies.append(model.geom_bodyid[geom])

        def tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)

        self.segments = tensor(capsules)
        self.radius = tensor(radius)
        self.bodies = torch.tensor(bodies, dtype=torch.long, device=device)
        self.first, self.second = torch.tensor(finger_pairs, dtype=torch.long, device=device).T
        cross_pairs = sorted({(a, b) for a in thumb_ids[0] for b in hand_ids[1]}
                             | {(a, b) for a in hand_ids[0] for b in thumb_ids[1]})
        self.cross_first, self.cross_second = torch.tensor(cross_pairs, dtype=torch.long, device=device).T
        self.thumb = torch.tensor(thumb_ids, dtype=torch.long, device=device)
        self.palms = torch.tensor(palm_bodies, dtype=torch.long, device=device)
        self.palm_center, self.palm_half, self.palm_rotation = map(tensor, (palm_centers, palm_half, palm_rotations))
        self.samples = torch.linspace(0, 1, 5, device=device).reshape(1, 1, 1, 5, 1)

    def gaps(self, p, r, *, between_hands=False):
        lines = self.segments @ r[:, self.bodies].transpose(-1, -2) + p[:, self.bodies, None]
        first, second = (self.cross_first, self.cross_second) if between_hands else (self.first, self.second)
        fingers = segment_distance(lines[:, first], lines[:, second])
        fingers = fingers - self.radius[first] - self.radius[second]
        thumb_ids = self.thumb.flip(0) if between_hands else self.thumb
        thumbs = lines[:, thumb_ids]
        points = torch.lerp(thumbs[..., :1, :], thumbs[..., 1:, :], self.samples)
        palm_r = r[:, self.palms]
        center = p[:, self.palms] + (palm_r @ self.palm_center[..., None]).squeeze(-1)
        palm_frame = palm_r @ self.palm_rotation
        local = (points - center[:, :, None, None]) @ palm_frame[:, :, None]
        d = local.abs() - self.palm_half[:, None, None]
        box_distance = d.clamp_min(0).norm(dim=-1) + d.amax(-1).clamp_max(0)
        palm = box_distance.amin(-1) - self.radius[thumb_ids]
        return torch.cat((fingers, palm.flatten(1)), dim=-1)

    @torch.no_grad()
    def calibrate(self, q, tree):
        # Calibrate against feasible robot poses to absorb mesh approximation error.
        gaps = []
        for batch in q.split(1024):
            p, r, _, _ = tree.forward(batch, torch.zeros_like(batch))
            gaps.append(self.gaps(p, r))
        self.margin = (torch.quantile(torch.cat(gaps), 0.02, dim=0) - 0.0005).clamp_max(0.0015)

    @torch.no_grad()
    def calibrate_between_hands(self, q, tree):
        columns = len(self.cross_first) + self.thumb.numel()
        self.cross_margin = q.new_empty(len(q), columns)
        for start in range(0, len(q), 2048):
            batch = q[start:start + 2048]
            p, r, _, _ = tree.forward(batch, torch.zeros_like(batch))
            # Preserve close but feasible hand coordination in the retargeted motion.
            self.cross_margin[start:start + len(batch)] = (self.gaps(p, r, between_hands=True) - 0.0005).clamp_max(0.002)

    def risk(self, gaps, frame=None):
        margin = self.margin if frame is None else self.cross_margin[frame]
        error = (margin - gaps).clamp_min(0).amax(dim=-1) / 0.004
        return error / (1 + error)
