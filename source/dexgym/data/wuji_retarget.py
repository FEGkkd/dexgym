"""Pinned WujiHand2 SDK retargeting and right-hand fingertip FK."""

import ctypes
from pathlib import Path

import numpy as np

SDK_VERSION = b"2026.10.26"
FIRMWARE_JOINT_NAMES = tuple(f"r_{finger}J{joint}" for finger in ("TH", "FF", "MF", "RF", "LF") for joint in range(4))
ROBOT_RIGHT_JOINT_NAMES = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    *(
        f"r_{finger}_{joint}"
        for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")
        for joint in ("mcp_flex", "mcp_abd", "pip", "dip")
    ),
)
TIP_SITE_NAMES = (
    "r_thumb_tip",
    "r_index_finger_tip",
    "r_middle_finger_tip",
    "r_ring_finger_tip",
    "r_pinky_tip",
)


class WujiRetargetSession:
    """Call the same Hand2 C SDK retarget API used by whole_body_teleop."""

    def __init__(self, library_path: str | Path, *, side: str = "right"):
        handedness = {"left": 0, "right": 1}[side]
        self._library = ctypes.CDLL(str(Path(library_path).expanduser().resolve()))
        lib = self._library
        lib.wuji_version.argtypes = []
        lib.wuji_version.restype = ctypes.c_char_p
        lib.wuji_init.argtypes = [ctypes.c_void_p]
        lib.wuji_init.restype = ctypes.c_int32
        lib.wuji_shutdown.argtypes = []
        lib.wuji_shutdown.restype = None
        lib.wuji_retarget_session_create.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
        lib.wuji_retarget_session_create.restype = ctypes.c_int32
        lib.wuji_retarget_session_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.wuji_retarget_session_step.restype = ctypes.c_int32
        lib.wuji_retarget_session_reset.argtypes = [ctypes.c_void_p]
        lib.wuji_retarget_session_reset.restype = ctypes.c_int32
        lib.wuji_retarget_session_free.argtypes = [ctypes.c_void_p]
        lib.wuji_retarget_session_free.restype = None
        version = lib.wuji_version()
        if version != SDK_VERSION:
            raise RuntimeError(f"Wuji SDK version mismatch: expected {SDK_VERSION!r}, got {version!r}")
        if lib.wuji_init(None) != 0:
            raise RuntimeError("wuji_init failed")
        self._session = ctypes.c_void_p()
        status = lib.wuji_retarget_session_create(1, handedness, ctypes.byref(self._session))
        if status != 0:
            lib.wuji_shutdown()
            raise RuntimeError(f"wuji_retarget_session_create failed: {status}")

    def step(self, keypoints: np.ndarray) -> np.ndarray:
        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (21, 3):
            raise ValueError("Wuji SDK expects 21 XYZ landmarks")
        if not np.isfinite(points).all():
            raise ValueError("Wuji SDK landmarks contain nonfinite values")
        points = np.ascontiguousarray(points)
        output = np.empty(20, dtype=np.float32)
        status = self._library.wuji_retarget_session_step(
            self._session,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status != 0 or not np.isfinite(output).all():
            raise RuntimeError(f"wuji_retarget_session_step failed: {status}")
        return output

    def reset(self) -> None:
        status = self._library.wuji_retarget_session_reset(self._session)
        if status != 0:
            raise RuntimeError(f"wuji_retarget_session_reset failed: {status}")

    def close(self) -> None:
        if self._session:
            self._library.wuji_retarget_session_free(self._session)
            self._session = ctypes.c_void_p()
            self._library.wuji_shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class OfficialWujiRetargetSession:
    """Official wuji-sdk Python RetargetSession, independent of hand hardware."""

    def __init__(self):
        from wuji_sdk import Handedness, HandModel, RetargetSession

        self._session = RetargetSession.for_hand(HandModel.WujiHand2, side=Handedness.Right)

    def step(self, keypoints: np.ndarray) -> np.ndarray:
        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("Official Wuji SDK expects finite MediaPipe keypoints[21,3]")
        q = np.asarray(self._session.step(np.ascontiguousarray(points)), dtype=np.float32)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise RuntimeError("Official Wuji SDK returned invalid right-hand joints")
        return q

    def reset(self) -> None:
        self._session.reset()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class RightHandFK:
    """Evaluate the right hand's official fingertip sites in its wrist frame."""

    def __init__(self, xml_path: str | Path):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
        if self.model.nu != 20 or self.model.nq != 20:
            raise ValueError("Expected a standalone 20-DoF right WujiHand2 model")
        self.data = mujoco.MjData(self.model)
        self.q_indices = []
        lower, upper = [], []
        for name in FIRMWARE_JOINT_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Missing official Wuji actuator {name}")
            jid = int(self.model.actuator_trnid[aid, 0])
            self.q_indices.append(int(self.model.jnt_qposadr[jid]))
            lower.append(float(self.model.jnt_range[jid, 0]))
            upper.append(float(self.model.jnt_range[jid, 1]))
        if len(set(self.q_indices)) != 20:
            raise ValueError("Wuji firmware joint mapping is not one-to-one")
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_wrist")
        self.tip_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in TIP_SITE_NAMES]
        if self.wrist_id < 0 or any(index < 0 for index in self.tip_ids):
            raise ValueError("Missing right-wrist or fingertip sites")

    def clamp(self, q: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(q, dtype=np.float32)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Expected 20 finite right-hand joint angles")
        bounded = np.clip(values, self.lower, self.upper)
        return bounded, int(np.count_nonzero(np.abs(values - bounded) > 1e-6))

    def tips(self, q: np.ndarray) -> np.ndarray:
        values = np.asarray(q, dtype=np.float32)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Expected 20 finite right-hand joint angles")
        self.data.qpos[:] = 0.0
        self.data.qpos[self.q_indices] = values
        self._mj.mj_forward(self.model, self.data)
        wrist_pos = self.data.xpos[self.wrist_id]
        wrist_rot = self.data.xmat[self.wrist_id].reshape(3, 3)
        sites = self.data.site_xpos[self.tip_ids]
        return ((sites - wrist_pos) @ wrist_rot).astype(np.float32)


def retarget_sequence(
    keypoints: np.ndarray, session: WujiRetargetSession, fk: RightHandFK
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Retarget one 30-Hz motion, then derive reachable fingertip targets."""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or len(points) == 0:
        raise ValueError("Expected a nonempty sequence[T,21,3]")
    session.reset()
    q = np.empty((len(points), 20), dtype=np.float32)
    tips = np.empty((len(points), 5, 3), dtype=np.float32)
    clip_count = 0
    for index, frame in enumerate(points):
        q[index], count = fk.clamp(session.step(frame))
        tips[index] = fk.tips(q[index])
        clip_count += count
    max_step = float(np.max(np.abs(np.diff(q, axis=0)))) if len(q) > 1 else 0.0
    return q, tips, {"frames": len(q), "clipped_joint_values": clip_count, "max_joint_step_rad": max_step}
