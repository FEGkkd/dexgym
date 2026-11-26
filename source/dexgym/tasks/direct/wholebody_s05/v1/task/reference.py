"""Translate an admitted Teleop RMB batch into the exact Actor input boundary."""

from hashlib import sha256

import numpy as np
import torch

from .geometry import Geometry, GeometryVelocity, quaternion_xyzw_to_matrix
from .observation import GeometryMask, TargetFrame, VelocityMask


class RmbReference:
    """Only explicit, current validated RMB fields cross the task boundary."""

    def __init__(self, bank, *, device, train=True):
        if getattr(bank, "review_verified", False) is not True or bank.verify is not True:
            raise ValueError("Unqualified RMB is unavailable to the production task")
        self.split = "Train" if train else "Validation"
        self.device = device
        self.partitions = {row.key: row for row in bank.partitions(self.split)}
        self.generations = {
            key: int.from_bytes(sha256(key.encode()).digest()[:5], "little") for key in self.partitions
        }
        if len(set(self.generations.values())) != len(self.generations):
            raise ValueError("RMB generation namespace collision")

    def _tensor(self, values):
        return torch.as_tensor(np.stack(values, axis=0), device=self.device)

    def current(self, locations):
        if not locations:
            raise ValueError("Expected a nonempty batch of explicit partition/tick locations")
        frames = [(key, self.partitions[key].arrays, int(i)) for key, i in locations]
        for _, arrays, i in frames:
            if not 0 <= i < len(arrays["tick_ns"]):
                raise IndexError("RMB tick is outside its partition")
        def field(name):
            return self._tensor([array[name][i] for _, array, i in frames])

        def mapped(name):
            return field(f"mapped_{name}")

        def mask(name):
            return mapped(f"{name}_mask")

        g = Geometry(mapped("wrist_xyz"), quaternion_xyzw_to_matrix(mapped("wrist_xyzw")),
                     mapped("local_landmark_xyz"), mapped("world_tip_xyz"))
        geometry_valid = GeometryMask(mask("wrist_xyz") & mask("wrist_xyzw"), mask("world_tip_xyz"),
                                      mask("local_landmark_xyz"), mask("pinch_xyz"), mask("cross_xyz"),
                                      mask("relative_wrist_xyz") & mask("relative_wrist_xyzw"))
        velocity = GeometryVelocity(mapped("wrist_linear_W_m_s"), mapped("wrist_angular_W_rad_s"),
                                    mapped("tip_velocity_W_m_s"))
        changes = field("inputs_source_changed")
        velocity_known = mapped("geometry_velocity_valid")[:, None]
        velocity_valid = VelocityMask(velocity_known & changes[:, 2:] & geometry_valid.wrist,
                                      (velocity_known & (changes[:, :2] | changes[:, 2:]))[:, :, None]
                                      & geometry_valid.tips)
        nominal_valid = torch.stack((field("nominal_joint_valid").all(-1) & field("domain_nominal_legal"),
                                     field("nominal_velocity_valid").all(-1)), dim=-1)
        allowed = field("domain_mapped_valid") & field("inputs_safety_permission")
        generation = self._tensor([
            np.int64((self.generations[key] << 23) + int(array["audit_generation"][i]))
            for key, array, i in frames
        ])
        return TargetFrame(nominal=field("nominal_q_nominal"), nominal_velocity=field("nominal_qd_nominal"),
                           nominal_valid=nominal_valid, geometry=g, velocity=velocity, geometry_mask=geometry_valid,
                           velocity_mask=velocity_valid, stream_valid=field("inputs_stream_valid"),
                           age_seconds=field("inputs_stream_age_ns").float() * 1e-9,
                           age_known=field("audit_clock_eligible"), permitted=allowed,
                           pinch_active=mapped("pinch_active"), generation=generation, tick_ns=field("tick_ns"))
