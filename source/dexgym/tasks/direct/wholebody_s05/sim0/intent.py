"""Causal bimanual wrist targets for SIM0 scoring experiments."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


_ACTIVE = "ACTIVE"
_HOLD = "HOLD"
_REJOIN = "REJOINING"


@dataclass(frozen=True)
class IntentResult:
    """A valid target is fresh or in bounded-time reference re-entry."""

    wrist_xyz_m: np.ndarray | None
    wrist_rotation: np.ndarray | None
    mode: str
    valid: bool
    generation: int
    reason: str
    input_valid: bool
    reference_offset_m: np.ndarray
    relative_vector_error_m: float | None
    relative_orientation_error_rad: float | None
    rejoin_weight: float | None


def _value(config, canonical, signed_key):
    if canonical in config:
        return config[canonical]
    return config[signed_key]


def _array(x, shape, name):
    a = np.asarray(x, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a.copy()


def _rotations(x, shape, name):
    a = _array(x, shape, name)
    if (not np.allclose(np.swapaxes(a, -2, -1) @ a, np.eye(3), rtol=0, atol=1e-7)
            or not np.allclose(np.linalg.det(a), 1., rtol=0, atol=1e-7)):
        raise ValueError(f"{name} must be right-handed orthonormal rotations")
    return a


def _positive(x, name):
    f = float(x)
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return f


def _pair_valid(valid):
    mask = np.asarray(valid)
    if mask.dtype.kind != "b" or mask.shape not in ((), (2,)):
        raise ValueError("valid must be a boolean or two per-wrist booleans")
    return bool(np.all(mask))


def _interpolate_rotation(start, end, weight):
    delta = Rotation.from_matrix(start.transpose(0, 2, 1) @ end).as_rotvec()
    return start @ Rotation.from_rotvec(weight * delta).as_matrix()


class IntentMapper:
    """Left/right XYZ metres and unit XYZW into bimanual simulation targets.

    The position map uses one shared rigid transform. T2/T3 warp only the
    registered midpoint; the registered right-minus-left vector is unchanged
    in ACTIVE. T3 applies a persistent common translation from an explicit
    clutch release. During re-entry both wrists blend from the last emitted
    target, and transient relative geometry deviations are reported.
    """

    def __init__(self, config):
        self.kind = config["kind"]
        if self.kind not in {"T1", "T2", "T3"}:
            raise ValueError("kind must be T1, T2, or T3")
        self.orientation_mode = config.get("orientation_mode", "FULL_SHARED")
        if self.orientation_mode not in {"FULL_SHARED", "RELATIVE", "FIXED_ABSOLUTE"}:
            raise ValueError("unknown wrist orientation mode")
        self._shared_rotation = _rotations(
            _value(config, "shared_rotation", "world_rotation"), (3, 3), "shared_rotation")
        self._shared_translation = _array(
            _value(config, "shared_translation_m", "world_translation_m"), (3,), "shared_translation_m")
        self._mount = _rotations(
            _value(config, "wrist_mount_rotations", "tracker_to_tcp_orientation_mount"),
            (2, 3, 3), "wrist_mount_rotations")
        self._orientation_seed = _rotations(
            config.get("wrist_seed_rotations", self._shared_rotation @ self._mount),
            (2, 3, 3), "wrist_seed_rotations")
        self._absolute_factor = _rotations(
            config.get("absolute_calibration_rotations", np.broadcast_to(np.eye(3), (2, 3, 3))),
            (2, 3, 3), "absolute_calibration_rotations")
        if self.kind != "T1":
            self._center = _array(config["midpoint_center_m"], (3,), "midpoint_center_m")
            self._limits = _array(config["midpoint_limits_m"], (3,), "midpoint_limits_m")
            if (self._limits <= 0).any():
                raise ValueError("midpoint_limits_m must be positive")
        self._quat_tolerance = _positive(config.get("quaternion_norm_tolerance", 1e-3),
                                         "quaternion_norm_tolerance")
        if self._quat_tolerance >= 1:
            raise ValueError("quaternion norm tolerance must be less than 1")
        if self.kind == "T3":
            self._dt = _positive(config.get("dt_s", .02), "dt_s")
            self._duration = _positive(config.get("rejoin_duration_s", .25), "rejoin_duration_s")
            self._recovery_frames = config.get("recovery_valid_frames", 2)
            if (isinstance(self._recovery_frames, (bool, np.bool_))
                    or not isinstance(self._recovery_frames, (int, np.integer))
                    or self._recovery_frames < 1):
                raise ValueError("recovery_valid_frames must be a positive integer")
        self._input_reference = None
        self._hold_target = None
        self._blend_start = None
        self._blend_steps = 0
        self._offset = np.zeros(3)
        self._generation = 0
        self._clutched = False
        self._valid_count = 0
        self._needs_recovery = False

    def _pose_is_usable(self, pose):
        if not np.isfinite(pose).all():
            return False
        quat = pose[..., 3:]
        if (np.abs(quat) > 1. + self._quat_tolerance).any():
            return False
        lengths = np.linalg.norm(quat, axis=-1)
        return bool(np.all(np.abs(lengths - 1.) <= self._quat_tolerance))

    def _map_positions(self, pose):
        p = pose[..., :3] @ self._shared_rotation.T + self._shared_translation
        if self.kind != "T1":
            midpoint = p.mean(axis=-2, keepdims=True)
            bounded = self._center + self._limits * np.tanh((midpoint - self._center) / self._limits)
            p = bounded + (p - midpoint)
        return p

    def _map_rotations(self, human_rotation):
        if self.orientation_mode == "FULL_SHARED":
            return self._shared_rotation @ human_rotation @ self._mount
        if self.orientation_mode == "FIXED_ABSOLUTE":
            return self._absolute_factor @ human_rotation @ self._mount
        relative = self._input_reference.transpose(0, 2, 1) @ human_rotation
        return self._orientation_seed @ (self._mount.transpose(0, 2, 1) @ relative @ self._mount)

    def batch(self, pose):
        """Map prepared valid [N, 2, 7] T1/T2 inputs with no state mutation."""
        if self.kind == "T3" or self.orientation_mode == "RELATIVE":
            raise ValueError("clutch and relative reference modes require sequential step()")
        a = np.asarray(pose, dtype=float)
        if a.ndim != 3 or a.shape[1:] != (2, 7) or not self._pose_is_usable(a):
            raise ValueError("batch must contain finite [N, 2, 7] poses with unit XYZW")
        if len(a) == 0:
            return np.empty((0, 2, 3)), np.empty((0, 2, 3, 3))
        human_rotation = Rotation.from_quat(a[..., 3:].reshape(-1, 4)).as_matrix().reshape(-1, 2, 3, 3)
        return self._map_positions(a), self._map_rotations(human_rotation)

    def _result(self, mode, reason, input_valid, live_target=None, rejoin_weight=None):
        ref = self._hold_target
        p = None if ref is None else ref[0].copy()
        r = None if ref is None else ref[1].copy()
        dp = dq = None
        if ref is not None and live_target is not None:
            live_p, live_r = live_target
            dp = float(np.linalg.norm((p[1] - p[0]) - (live_p[1] - live_p[0])))
            dq = float(Rotation.from_matrix(
                (live_r[0].T @ live_r[1]).T @ (r[0].T @ r[1])).magnitude())
        return IntentResult(p, r, mode, mode != _HOLD, self._generation, reason,
                            input_valid, self._offset.copy(), dp, dq, rejoin_weight)

    def _start_rejoin(self, reason, live_target):
        self._blend_start = self._hold_target
        self._blend_steps = 0
        self._needs_recovery = False
        return self._result(_REJOIN, reason, True, live_target, 0.)

    def step(self, pose, valid, event=None):
        """Consume one tick. Event names are CLUTCH_ON and CLUTCH_OFF."""
        if event not in (None, "CLUTCH_ON", "CLUTCH_OFF"):
            raise ValueError("unknown clutch event")
        if event is not None and self.kind != "T3":
            raise ValueError("clutch events are defined only for T3")

        usable = _pair_valid(valid)
        a = np.asarray(pose, dtype=float) if usable else None
        if usable and (a.shape != (2, 7) or not self._pose_is_usable(a)):
            usable = False
        if event == "CLUTCH_ON":
            self._clutched = True
            self._blend_start = None
            self._valid_count = 0
            return self._result(_HOLD, "CLUTCH_ENGAGED", usable)
        if event == "CLUTCH_OFF":
            if not self._clutched:
                return self._result(_HOLD, "CLUTCH_RELEASE_WITHOUT_LATCH", usable)
            if not usable:
                return self._result(_HOLD, "CLUTCH_RELEASE_REQUIRES_VALID_PAIR", False)
        elif self._clutched:
            return self._result(_HOLD, "CLUTCH_HELD", usable)
        if not usable:
            self._valid_count = 0
            self._blend_start = None
            self._needs_recovery = self.kind == "T3" and self._hold_target is not None
            return self._result(_HOLD, "INVALID_INPUT", False)

        human_rotation = Rotation.from_quat(a[:, 3:]).as_matrix()
        if self.orientation_mode == "RELATIVE" and self._input_reference is None:
            self._input_reference = human_rotation.copy()
        base = self._map_positions(a)
        r = self._map_rotations(human_rotation)
        if event == "CLUTCH_OFF":
            self._clutched = False
            if self._hold_target is not None:
                self._offset = self._hold_target[0].mean(axis=0) - base.mean(axis=0)
                self._generation += 1
                return self._start_rejoin("CLUTCH_RELEASE_NEW_REFERENCE", (base + self._offset, r))

        live_target = base + self._offset, r
        if self._needs_recovery:
            self._valid_count += 1
            if self._valid_count < self._recovery_frames:
                return self._result(_HOLD, "RECOVERY_WAIT_VALID_PAIR", True, live_target)
            return self._start_rejoin("RECOVERY_REJOIN", live_target)
        if self._blend_start is not None:
            self._blend_steps += 1
            fraction = min(1., self._blend_steps * self._dt / self._duration)
            if fraction < 1.:
                alpha = fraction * fraction * (3. - 2. * fraction)
                start_p, start_r = self._blend_start
                p = (1. - alpha) * start_p + alpha * live_target[0]
                self._hold_target = p, _interpolate_rotation(start_r, r, alpha)
                return self._result(_REJOIN, "SMOOTH_REFERENCE_REENTRY", True, live_target, alpha)
            self._blend_start = None
            self._hold_target = live_target
            return self._result(_ACTIVE, "REENTRY_COMPLETE", True, live_target, 1.)

        self._hold_target = live_target
        return self._result(_ACTIVE, "MAPPED", True, live_target)
