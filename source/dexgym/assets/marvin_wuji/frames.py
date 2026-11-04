"""Explicit MuJoCo WXYZ to Isaac Lab 3.0 XYZW frame comparison helpers."""

import numpy as np


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    array = np.asarray(quat, dtype=np.float64)
    if array.shape[-1] != 4:
        raise ValueError("Quaternion last dimension must be four")
    return array[..., [1, 2, 3, 0]]


def angular_distance_deg(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Sign-invariant geodesic distance, numerically stable near zero."""
    a = np.asarray(a_xyzw, dtype=np.float64)
    b = np.asarray(b_xyzw, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.rad2deg(2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot))
