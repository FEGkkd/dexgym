"""Causal production-MARVIN/SDK reference and independent path qualification."""
from contextlib import ExitStack
import ctypes
from enum import IntEnum
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.data.wuji_retarget import SDK_VERSION
from .mapping import sdk_keypoints


DOUBLE = ctypes.POINTER(ctypes.c_double)
FLOAT = ctypes.POINTER(ctypes.c_float)
GROUPS = tuple(slice(a, b) for a, b in ((0, 7), (7, 14), (14, 34), (34, 54)))


class NominalStatus(IntEnum):
    UNAVAILABLE = 0
    ACTIVE = 1
    RECOVERING = 2
    MARVIN_IK_FAILED = 3
    WUJI_SDK_FAILED = 4
    PATH_FAILED = 5
    DENIED_SOURCE = 6


class SDKFailure(RuntimeError):
    pass


def matrix(translation, rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def model_wrist(model_geometry, side):
    return matrix(model_geometry.wrists[side], model_geometry.rotations[side])


class Marvin:
    def __init__(self, bridge, config):
        self.lib = ctypes.CDLL(str(Path(bridge).resolve()))
        self.lib.s05_marvin_open.argtypes = [ctypes.c_char_p]
        self.lib.s05_marvin_open.restype = ctypes.c_void_p
        self.lib.s05_marvin_close.argtypes = [ctypes.c_void_p]
        self.lib.s05_marvin_close.restype = None
        self.lib.s05_marvin_fk.argtypes = [ctypes.c_void_p, ctypes.c_int, DOUBLE, DOUBLE]
        self.lib.s05_marvin_ik.argtypes = [ctypes.c_void_p, ctypes.c_int, DOUBLE, DOUBLE, DOUBLE]
        self.lib.s05_marvin_fk.restype = self.lib.s05_marvin_ik.restype = ctypes.c_int
        self.lib.s05_marvin_last_error.argtypes = []
        self.lib.s05_marvin_last_error.restype = ctypes.c_char_p
        self.lib.s05_marvin_workspace_probe.argtypes = [ctypes.c_void_p, ctypes.c_int, DOUBLE, DOUBLE]
        self.lib.s05_marvin_workspace_probe.restype = ctypes.c_int
        self.handle = self.lib.s05_marvin_open(str(Path(config).resolve()).encode())
        if not self.handle:
            raise SDKFailure("Marvin configuration could not be loaded")

    def close(self):
        if self.handle:
            self.lib.s05_marvin_close(self.handle)
            self.handle = None

    def fk(self, side, joints):
        joints = np.ascontiguousarray(joints, dtype=np.float64)
        result = np.empty((4, 4), dtype=float)
        status = self.lib.s05_marvin_fk(self.handle, side, joints.ctypes.data_as(DOUBLE),
                                        result.ctypes.data_as(DOUBLE))
        if status != 0 or not np.isfinite(result).all():
            raise SDKFailure(f"Marvin FK failed with status {status}")
        return result

    def workspace_probe(self, side, target):
        target = np.ascontiguousarray(target, dtype=np.float64)
        result = np.empty(19, dtype=float)
        status = self.lib.s05_marvin_workspace_probe(self.handle, side, target.ctypes.data_as(DOUBLE),
                                                     result.ctypes.data_as(DOUBLE))
        if status or not np.isfinite(result).all():
            return None
        return {"offset_m": float(result[0]), "radius_max_m": float(result[1]), "radius_request_m": float(result[2]),
                "projected_pose": result[3:].reshape(4, 4)}

    def ik(self, side, target, seed):
        target = np.ascontiguousarray(target, dtype=np.float64)
        seed = np.ascontiguousarray(seed, dtype=np.float64)
        result = np.empty(7, dtype=float)
        status = self.lib.s05_marvin_ik(self.handle, side, target.ctypes.data_as(DOUBLE),
                                        seed.ctypes.data_as(DOUBLE), result.ctypes.data_as(DOUBLE))
        if status != 0 or not np.isfinite(result).all():
            reason = self.lib.s05_marvin_last_error().decode("utf-8", errors="replace")
            raise SDKFailure(f"Marvin IK failed with status {status}: {reason}")
        return result


class Wuji:
    """Hold one SDK initialization and one resettable session per hand."""
    def __init__(self, library):
        self.lib = lib = ctypes.CDLL(str(Path(library).resolve()))
        lib.wuji_version.argtypes = []
        lib.wuji_version.restype = ctypes.c_char_p
        lib.wuji_init.argtypes = [ctypes.c_void_p]
        lib.wuji_init.restype = ctypes.c_int32
        lib.wuji_shutdown.argtypes = []
        lib.wuji_shutdown.restype = None
        lib.wuji_retarget_session_create.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
        lib.wuji_retarget_session_create.restype = ctypes.c_int32
        lib.wuji_retarget_session_step.argtypes = [ctypes.c_void_p, FLOAT, FLOAT]
        lib.wuji_retarget_session_step.restype = ctypes.c_int32
        lib.wuji_retarget_session_reset.argtypes = [ctypes.c_void_p]
        lib.wuji_retarget_session_reset.restype = ctypes.c_int32
        lib.wuji_retarget_session_free.argtypes = [ctypes.c_void_p]
        lib.wuji_retarget_session_free.restype = None
        version = lib.wuji_version()
        if version != SDK_VERSION:
            raise SDKFailure(f"Pinned Wuji SDK version mismatch: {version!r}")
        if lib.wuji_init(None):
            raise SDKFailure("Wuji SDK initialization failed")
        self.sessions = []
        try:
            for side in range(2):
                handle = ctypes.c_void_p()
                if lib.wuji_retarget_session_create(1, side, ctypes.byref(handle)):
                    raise SDKFailure(f"Cannot create Wuji session for side {side}")
                self.sessions.append(handle)
        except BaseException:
            self.close()
            raise

    def reset(self):
        for handle in self.sessions:
            if self.lib.wuji_retarget_session_reset(handle):
                raise SDKFailure("Failed to reset Wuji session")

    def close(self):
        if self.sessions is not None:
            for handle in self.sessions:
                self.lib.wuji_retarget_session_free(handle)
            self.lib.wuji_shutdown()
            self.sessions = None

    def step(self, side, points):
        if points.shape != (21, 3) or points.dtype != np.float32 or not points.flags.c_contiguous:
            raise ValueError("SDK expects a contiguous named 21 x 3 float32 sample")
        if not np.isfinite(points).all():
            raise SDKFailure("Wuji input is nonfinite")
        result = np.empty(20, dtype=np.float32)
        status = self.lib.wuji_retarget_session_step(self.sessions[side], points.ctypes.data_as(FLOAT),
                                                     result.ctypes.data_as(FLOAT))
        if status or not np.isfinite(result).all():
            raise SDKFailure(f"Wuji retarget failed with status {status}")
        return result


class Nominal:
    def __init__(self, robot, marvin, hand, step_limit, swept_proof):
        self.robot = robot
        self.marvin = marvin
        self.hand = hand
        self.swept_proof = swept_proof
        self.step_limit = np.asarray(step_limit, dtype=float)
        if self.step_limit.shape != (54,) or not np.isfinite(self.step_limit).all() or not (self.step_limit > 0).all():
            raise ValueError("Missing signed positive step limits for all 54 joints")
        home = robot.geometry(robot.home)
        self.bases = []
        self.wrist_to_native = []
        for side, body in enumerate(("Base_L", "Base_R")):
            mark = robot.data.body(body)
            base = matrix(mark.xpos, mark.xmat.reshape(3, 3))
            sdk_home = marvin.fk(side, robot.home[GROUPS[side]])
            wrist_home = model_wrist(home, side)
            tool = np.linalg.solve(wrist_home, base @ sdk_home)
            self.bases.append(base)
            self.wrist_to_native.append(tool)
        self.validation = self.validate_transforms()
        self.generation = -1
        self.cut()

    def validate_transforms(self):
        errors = []
        for side in range(2):
            group = GROUPS[side]
            for axis in range(7):
                for step in (-0.04, 0.04):
                    q = self.robot.home.copy()
                    q[group.start + axis] += step
                    direct = self.bases[side] @ self.marvin.fk(side, q[group])
                    model = model_wrist(self.robot.geometry(q), side) @ self.wrist_to_native[side]
                    pos = float(np.linalg.norm(direct[:3, 3] - model[:3, 3]))
                    angle = float(Rotation.from_matrix(direct[:3, :3].T @ model[:3, :3]).magnitude())
                    errors.append({"side": side, "axis": axis, "pulse_rad": step,
                                   "position_error_m": pos, "orientation_error_rad": angle})
        return errors

    def cut(self):
        # New input epochs cannot use old IK/retarget state, including held references.
        self.generation += 1
        self.last = self.robot.home.copy()
        self.last_source = np.full(2, -1, dtype=np.int64)
        self.filtered_hand = [None, None]
        self.last_raw_hand = [None, None]
        self.consecutive = 0
        self.ik_failures = []
        self.hand.reset()

    def nominal_path_legal(self, start, goal):
        # Box and linear J6/J7 envelopes are convex. Collision uses the
        # same conservative swept-configuration bound as the physical task.
        return self.robot.legality(goal)["feasible"] and self.swept_proof.permits(start, goal)

    def step(self, target, common, order, source_index):
        raw = np.zeros(54, dtype=float)
        valid = np.zeros(4, dtype=bool)
        status = np.full(4, NominalStatus.ACTIVE, dtype="<u2")
        result = self.last.copy()
        self.ik_failures = []
        wrist_r = Rotation.from_quat(target["wrist_xyzw"]).as_matrix()
        for side in range(2):
            group = GROUPS[side]
            world_wrist = matrix(target["wrist_xyz"][side], wrist_r[side])
            wrist = np.linalg.solve(self.bases[side], world_wrist @ self.wrist_to_native[side])
            try:
                raw[group] = self.marvin.ik(side, wrist, self.last[group])
            except SDKFailure as error:
                status[side] = NominalStatus.MARVIN_IK_FAILED
                envelope = self.marvin.workspace_probe(side, wrist)
                self.ik_failures.append({"side": side, "reason": str(error), "workspace":
                    {key: val for key, val in envelope.items() if key != "projected_pose"} if envelope else None})
            else:
                result[group] = raw[group]
                valid[side] = True

            hand_group = GROUPS[side + 2]
            try:
                if source_index[side] != self.last_source[side] or self.filtered_hand[side] is None:
                    points = sdk_keypoints(common[side], order[side], side)
                    retarget = self.hand.step(side, points)
                    previous = self.filtered_hand[side]
                    self.filtered_hand[side] = retarget.copy() if previous is None else 0.3 * retarget + 0.7 * previous
                    self.last_raw_hand[side] = retarget.copy()
                    self.last_source[side] = source_index[side]
                raw[hand_group] = self.last_raw_hand[side]
                result[hand_group] = self.filtered_hand[side]
            except SDKFailure:
                status[side + 2] = NominalStatus.WUJI_SDK_FAILED
            else:
                valid[side + 2] = True

        bounded = np.clip(np.clip(result, *self.robot.bounds.T),
                          self.last - self.step_limit, self.last + self.step_limit)
        limited = np.abs(result - bounded) > 1e-10
        if not valid.all():
            self.consecutive = 0
            status[status == NominalStatus.ACTIVE] = NominalStatus.RECOVERING
        elif not self.nominal_path_legal(self.last, bounded):
            self.consecutive = 0
            status[:] = NominalStatus.PATH_FAILED
        else:
            self.last = bounded
            self.consecutive += 1
            if self.consecutive < 2:
                status[:] = NominalStatus.RECOVERING
        return self.last.copy(), raw, valid, status, limited


def sdk_context(bridge, config, library):
    """Return an exit stack plus one Marvin and one Wuji read-only computation session."""
    stack = ExitStack()
    try:
        marvin = Marvin(bridge, config)
        stack.callback(marvin.close)
        hand = Wuji(library)
        stack.callback(hand.close)
        return stack, marvin, hand
    except BaseException:
        stack.close()
        raise
