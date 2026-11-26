"""Typed bindings for the pinned phase1 MuJoCo arm command kernel."""

import ctypes as C
from dataclasses import dataclass
from hashlib import sha1, sha256
from pathlib import Path

import numpy as np
import yaml


REAL_BLOB = "3ad6456c550992716e9c6d7f94395dd468c3ad0d"
SIM_BLOB = "14b7537d8d18ab198385dbce810abe7f29a75597"
MARVIN_CONFIG_SHA256 = "ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52"
_D = C.c_double
_P = C.POINTER(_D)
_SCALARS = (
    "fk_position_tolerance_m", "fk_orientation_tolerance_rad", "max_raw_joint_step_rad",
    "max_command_rate_rad_sec", "max_command_dt_sec", "max_tracking_error_rad",
    "hard_tracking_error_rad", "max_translation_rate_m_sec", "max_rotation_rate_rad_sec",
    "workspace_reserve_m", "workspace_transition_m", "joint_limit_margin_deg",
    "analytic_joint_rate_deg_sec", "analytic_max_step_dt_sec", "nullspace_activation_deg",
    "nullspace_rate_deg_sec", "nullspace_probe_deg", "arm_angle_limit_deg",
    "axis_projection_min_retained_ratio", "arm_angle_centering_weight", "tracker_timeout_sec", "zsp_angle_deg",
)


class _Profile(C.Structure):
    _fields_ = (
        [("lower", _D * 7), ("upper", _D * 7), ("solution_lower", _D * 7), ("solution_upper", _D * 7)]
        + [(name, _D) for name in _SCALARS]
        + [("left_zsp_para", _D * 6), ("right_zsp_para", _D * 6), ("dgr_deg", _D * 3),
           ("zsp_type", C.c_int), ("held_boundary_brake", C.c_int)]
    )


class _Arm(C.Structure):
    _fields_ = [
        ("joint_candidate", _D * 7), ("joint_command", _D * 7),
        ("step_target", _D * 16), ("workspace_target", _D * 16), ("attempt_target", _D * 16),
        ("arm_angle_command_rad", _D), ("workspace_projection_m", _D),
        ("position_lag_m", _D), ("rotation_lag_rad", _D), ("backoff_scale", _D),
        ("projected_axis", C.c_int), ("safety_code", C.c_int), ("flags", C.c_uint32),
    ]


@dataclass(frozen=True)
class NativeArmProposal:
    accepted: bool
    ik_target_accepted: bool
    analytic_braking: bool
    workspace_projected: bool
    rate_limited: bool
    analytic_exact_stop_used: bool
    pair_accepted: bool
    pair_hard_fault: bool
    joint_candidate: np.ndarray
    joint_command: np.ndarray
    step_target: np.ndarray
    workspace_target: np.ndarray
    attempt_target: np.ndarray
    arm_angle_command_rad: float
    workspace_projection_m: float
    position_lag_m: float
    rotation_lag_rad: float
    backoff_scale: float
    projected_axis: int
    safety_code: int
    safety_reason: str


def _read_overlay(path, git_blob):
    raw = path.read_bytes()
    actual = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if actual != git_blob:
        raise ValueError(f"Teleop overlay does not match fixed Git blob: {path.name}")
    return yaml.safe_load(raw)["whole_body_teleop"]["ros__parameters"]


def load_profile(phase1_repo):
    folder = Path(phase1_repo) / "src/whole_body_teleop/config"
    merged = {**_read_overlay(folder / "real.yaml", REAL_BLOB), **_read_overlay(folder / "sim.yaml", SIM_BLOB)}
    arm = {key.removeprefix("arm."): value for key, value in merged.items() if key.startswith("arm.")}
    required = {
        "control_pipeline": "manimux_analytic_ik_position", "control_rate_hz": 250.0,
        "driver_standby_service": "/tianji_mujoco_driver/standby", "kinematics_backend": "marvin",
        "ik_seed_mode": "previous_accepted_command", "ik_nsp_mode": "start_latched",
        "ik_arm_angle_bias_deg": 0.0, "analytic_ik_enabled": True, "analytic_source_nullspace_enabled": True,
        "axis_projection_enabled": True, "workspace_boundary_enabled": True, "independent_arm_execution": True,
        "continuous_servo_enabled": False, "ruckig_enabled": False, "feasible_path_enabled": False,
        "allow_identity_backend": False, "multi_seed_enabled": False, "task_projection_enabled": False,
        "local_redundancy_recovery_enabled": False, "nullspace_avoidance_enabled": False,
        "radial_overtravel_discard_enabled": False, "command_velocity_lowpass_hz": 0.0,
        "handoff_hold_sec": 0.0, "handoff_ramp_sec": 0.0, "backoff_scales": [1.0, .5, .25, .1],
    }
    if any(arm.get(key) != value for key, value in required.items()) or arm.get("elbow_guard_enabled", False):
        raise ValueError("The audited pure command seam no longer matches the simulation profile")
    if arm.get("marvin_left_serial", 0) != 0 or arm.get("marvin_right_serial", 1) != 1:
        raise ValueError("Unexpected side identity for the pinned native SDK")
    values = {
        **arm,
        "workspace_reserve_m": arm["workspace_boundary_reserve_m"],
        "workspace_transition_m": arm["workspace_boundary_transition_m"],
        "analytic_joint_rate_deg_sec": arm["analytic_joint_vmax_deg_s"] * arm["analytic_sdk_velocity_ratio"]
        / 100. * arm["analytic_command_rate_margin"],
        "zsp_angle_deg": arm["ik_zsp_angle_deg"],
    }
    result = _Profile(
        (_D * 7)(*arm["joint_lower_rad"]), (_D * 7)(*arm["joint_upper_rad"]),
        (_D * 7)(*arm.get("solution_lower_rad", arm["joint_lower_rad"])),
        (_D * 7)(*arm.get("solution_upper_rad", arm["joint_upper_rad"])),
        *(values[key] for key in _SCALARS),
        (_D * 6)(*arm["left_ik_zsp_para"]), (_D * 6)(*arm["right_ik_zsp_para"]),
        (_D * 3)(*arm["ik_dgr_deg"]), arm["ik_zsp_type"], int(arm["servo_held_boundary_brake_enabled"]),
    )
    if not all(np.isfinite(getattr(result, name)) for name in _SCALARS):
        raise ValueError("The pinned simulation profile contains nonfinite numeric values")
    return result


def _array(value, shape):
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Native Teleop expects a finite array with shape {shape}")
    return result


def _view(value, shape):
    return np.ctypeslib.as_array(value).copy().reshape(shape)


class TeleopNative:
    """Single-process stateless proposals with an explicit generation home."""

    def __init__(self, bridge_path, marvin_config_path, phase1_repo_path):
        self.handle = None
        self.initialized = False
        profile = load_profile(phase1_repo_path)
        config = Path(marvin_config_path).resolve()
        if sha256(config.read_bytes()).hexdigest() != MARVIN_CONFIG_SHA256:
            raise ValueError("The native Marvin configuration differs from the pinned simulation configuration")
        self.lib = C.CDLL(str(Path(bridge_path).resolve()))
        definitions = {
            "arm_output_size": ([], C.c_size_t), "profile_size": ([], C.c_size_t),
            "abi_version": ([], C.c_int),
            "open": ([C.c_char_p, C.POINTER(_Profile)], C.c_void_p),
            "close": ([C.c_void_p], None),
            "home": ([C.c_void_p, _P], C.c_int),
            "fk": ([C.c_void_p, C.c_int, _P, _P], C.c_int),
            "propose": ([C.c_void_p, _P, _P, _P, _P, _D, C.POINTER(_Arm)], C.c_int),
            "reason": ([C.c_void_p, C.c_int], C.c_char_p),
        }
        for name, (arguments, returns) in definitions.items():
            function = getattr(self.lib, "s05_candidate_" + name)
            function.argtypes, function.restype = arguments, returns
        if (self.lib.s05_candidate_abi_version() != 2
                or self.lib.s05_candidate_profile_size() != C.sizeof(_Profile)
                or self.lib.s05_candidate_arm_output_size() != C.sizeof(_Arm)):
            raise RuntimeError("Native Teleop bridge ABI size differs")
        self.handle = self.lib.s05_candidate_open(bytes(config), C.byref(profile))
        if not self.handle:
            raise RuntimeError("Cannot initialize the pinned native Teleop core")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.handle:
            self.lib.s05_candidate_close(self.handle)
            self.handle = None
        self.initialized = False

    def home_reference(self, home):
        values = _array(home, (14,))
        self.initialized = bool(self.handle) and self.lib.s05_candidate_home(self.handle, values.ctypes.data_as(_P)) == 0
        if not self.initialized:
            raise RuntimeError("Native Teleop could not latch an FK/NSP reference at signed home")
        return np.zeros(2, dtype=np.float64)

    def fk(self, side, joints):
        if side not in (0, 1) or not self.handle:
            raise RuntimeError("Native Teleop FK requires an open, named side")
        q = _array(joints, (7,))
        out = np.empty((4, 4), dtype=np.float64)
        if self.lib.s05_candidate_fk(self.handle, side, q.ctypes.data_as(_P), out.ctypes.data_as(_P)):
            raise RuntimeError("Native Teleop FK failed")
        return _array(out, (4, 4))

    def propose(self, wrist_targets_base_tcp, measured_q, previous_q, angles, dt_s=.004):
        if not self.initialized or dt_s != .004:
            raise ValueError("Native Teleop proposal requires an initialized 250 Hz generation")
        values = [_array(value, shape) for value, shape in (
            (wrist_targets_base_tcp, (2, 4, 4)), (measured_q, (14,)), (previous_q, (14,)), (angles, (2,)),
        )]
        outputs = (_Arm * 2)()
        if self.lib.s05_candidate_propose(self.handle, *(value.ctypes.data_as(_P) for value in values), dt_s, outputs):
            raise RuntimeError("Native Teleop core failed to produce an inspectable proposal")
        results = []
        for side, row in enumerate(outputs):
            results.append(NativeArmProposal(
                *(bool(row.flags & (1 << bit)) for bit in range(8)),
                _view(row.joint_candidate, (7,)), _view(row.joint_command, (7,)),
                _view(row.step_target, (4, 4)), _view(row.workspace_target, (4, 4)), _view(row.attempt_target, (4, 4)),
                row.arm_angle_command_rad, row.workspace_projection_m, row.position_lag_m,
                row.rotation_lag_rad, row.backoff_scale, row.projected_axis, row.safety_code,
                self.lib.s05_candidate_reason(self.handle, side).decode("utf-8", errors="replace"),
            ))
        return tuple(results)
