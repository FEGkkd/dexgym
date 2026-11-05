"""50 Hz Marvin motion reference access and SO(3) helpers."""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


def bvh_to_marvin_rotation() -> np.ndarray:
    """CMU +X anatomical right, +Y up, -Z forward to Marvin -X right, +Z up, -Y forward."""
    return np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def smooth_so3(quat_xyzw: np.ndarray, *, sigma_frames: float) -> np.ndarray:
    """Continuous S3 chordal Gaussian filter; never filters Euler axes independently."""
    quat = np.asarray(quat_xyzw, dtype=np.float64).copy()
    if quat.ndim != 2 or quat.shape[1] != 4 or sigma_frames < 0:
        raise ValueError("Expected [T,4] quaternions and nonnegative sigma")
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1
    if sigma_frames == 0:
        return quat
    smooth = gaussian_filter1d(quat, sigma_frames, axis=0, mode="nearest")
    smooth /= np.linalg.norm(smooth, axis=1, keepdims=True)
    for i in range(1, len(smooth)):
        if np.dot(smooth[i], smooth[i - 1]) < 0:
            smooth[i] *= -1
    return smooth


def so3_distance_deg(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    a = Rotation.from_quat(a_xyzw)
    b = Rotation.from_quat(b_xyzw)
    return np.rad2deg((a * b.inv()).magnitude())


class MarvinReference:
    """Read a validated or preview NPZ; hold the final pose past its endpoint."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)
        self.time = self.data["time"]
        if len(self.time) < 2 or not np.allclose(self.time, np.arange(len(self.time)) * 0.02, atol=1e-9):
            raise ValueError("Reference must have an exact 50 Hz time axis")
        if int(self.data["fps"]) != 50:
            raise ValueError("Reference fps must be 50")

    def at(self, time_s: float) -> dict[str, np.ndarray]:
        t = float(np.clip(time_s, 0, self.time[-1]))
        lo = min(int(t / 0.02), len(self.time) - 1)
        hi = min(lo + 1, len(self.time) - 1)
        alpha = (t - self.time[lo]) / 0.02 if lo != hi else 0.0
        result = {}
        for key in ("q_arm", "qd_arm", "tcp_pos", "link_pos"):
            if key in self.data:
                result[key] = (1 - alpha) * self.data[key][lo] + alpha * self.data[key][hi]
        for key in ("tcp_quat_xyzw", "link_quat_xyzw"):
            if key in self.data:
                pair = self.data[key][[lo, hi]]
                if lo == hi:
                    result[key] = pair[0]
                else:
                    flat = pair.reshape(2, -1, 4)
                    result[key] = np.stack(
                        [
                            Slerp([0, 1], Rotation.from_quat(flat[:, i]))([alpha]).as_quat()[0]
                            for i in range(flat.shape[1])
                        ]
                    ).reshape(pair.shape[1:])
        return result

    def future_links(self, time_s: float, *, count: int = 10) -> dict[str, np.ndarray]:
        if count <= 0:
            raise ValueError("count must be positive")
        frames = [self.at(time_s + (i + 1) * 0.02) for i in range(count)]
        return {
            "pos": np.stack([frame["link_pos"] for frame in frames]),
            "quat_xyzw": np.stack([frame["link_quat_xyzw"] for frame in frames]),
        }
