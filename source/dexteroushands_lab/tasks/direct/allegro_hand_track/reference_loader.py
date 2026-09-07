"""Pure-Python loading and validation for DexTrack GRAB references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


REQUIRED_KEYS = (
    "robot_delta_states_weights_np",
    "object_transl",
    "object_rot_quat",
)


class GrabReferenceError(ValueError):
    """Raised when a DexTrack reference file violates the data contract."""


@dataclass(frozen=True)
class GrabReference:
    """Validated one-trajectory reference tensors.

    The quaternion field intentionally retains the source ``xyzw`` convention.
    Conversion to a simulator convention belongs at the task boundary.
    """

    hand_target: torch.Tensor
    object_translation: torch.Tensor
    object_quaternion_xyzw: torch.Tensor

    def __post_init__(self) -> None:
        frame_count = self.hand_target.shape[0]
        if self.hand_target.ndim != 2 or self.hand_target.shape[1] != 22:
            raise GrabReferenceError("hand_target must have shape [T, 22]")
        if self.object_translation.shape != (frame_count, 3):
            raise GrabReferenceError("object_translation must have shape [T, 3]")
        if self.object_quaternion_xyzw.shape != (frame_count, 4):
            raise GrabReferenceError("object_quaternion_xyzw must have shape [T, 4]")
        if frame_count < 1:
            raise GrabReferenceError("reference must contain at least one frame")
        for name, value in (
            ("hand_target", self.hand_target),
            ("object_translation", self.object_translation),
            ("object_quaternion_xyzw", self.object_quaternion_xyzw),
        ):
            if not torch.isfinite(value).all():
                raise GrabReferenceError(f"{name} must contain only finite values")

    @property
    def num_frames(self) -> int:
        return int(self.hand_target.shape[0])

    def frame(
        self, index: torch.Tensor | int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return clamped hand, object translation, and quaternion frames."""

        indices = torch.as_tensor(index, dtype=torch.long, device=self.hand_target.device)
        indices = indices.clamp(0, self.num_frames - 1)
        return (
            self.hand_target[indices],
            self.object_translation[indices],
            self.object_quaternion_xyzw[indices],
        )


def _as_float_array(value: object, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise GrabReferenceError(f"{key} must contain only finite values")
    return array


def _validate_dictionary(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise GrabReferenceError("reference file must contain a dictionary")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise GrabReferenceError(f"missing required key(s): {', '.join(missing)}")
    return payload


def _validate_frame_arrays(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand = _as_float_array(payload["robot_delta_states_weights_np"], "robot_delta_states_weights_np")
    translation = _as_float_array(payload["object_transl"], "object_transl")
    quaternion = _as_float_array(payload["object_rot_quat"], "object_rot_quat")
    if hand.ndim != 2 or hand.shape[1] != 22:
        raise GrabReferenceError("robot_delta_states_weights_np must have shape [T, 22]")
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise GrabReferenceError("object_transl must have shape [T, 3]")
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise GrabReferenceError("object_rot_quat must have shape [T, 4]")
    frame_counts = {hand.shape[0], translation.shape[0], quaternion.shape[0]}
    if len(frame_counts) != 1:
        raise GrabReferenceError("reference arrays must have the same frame count")
    if hand.shape[0] < 1:
        raise GrabReferenceError("reference must contain at least one frame")
    norms = np.linalg.norm(quaternion, axis=1)
    if not np.isfinite(norms).all() or np.any(norms < 1.0e-8):
        raise GrabReferenceError("object_rot_quat contains an unusable quaternion norm")
    quaternion = quaternion / norms[:, None]
    return hand, translation, quaternion


def _find_indices(names: Sequence[str], requested: Iterable[str], label: str) -> list[int]:
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names contain duplicate entries")
    requested = tuple(requested)
    if len(set(requested)) != len(requested):
        raise ValueError(f"{label} names contain duplicate entries")
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in requested if name not in positions]
    if missing:
        raise ValueError(f"unknown {label} name(s): {', '.join(missing)}")
    return [positions[name] for name in requested]


def allegro_source_to_sim_joint_order(
    values: np.ndarray,
    *,
    source_names: Sequence[str] | None = None,
    destination_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Map a serialized 16-joint row by names, never by an implicit permutation."""

    values = np.asarray(values)
    if values.shape[-1] != 16:
        raise ValueError("Allegro finger joint values must have 16 entries")
    if source_names is None or destination_names is None:
        return values.copy()
    if len(source_names) != 16 or len(destination_names) != 16:
        raise ValueError("source and destination joint orders must contain 16 names")
    indices = _find_indices(source_names, destination_names, "joint")
    return values[..., indices]


def load_grab_reference(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    source_names: Sequence[str] | None = None,
    destination_names: Sequence[str] | None = None,
) -> GrabReference:
    """Load and validate one DexTrack ``.npy`` scalar dictionary."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GRAB reference file does not exist: {path}")
    try:
        payload = np.load(path, allow_pickle=True).item()
    except Exception as exc:  # pragma: no cover - numpy supplies varied errors
        raise GrabReferenceError(f"unable to read reference dictionary: {path}") from exc
    payload = _validate_dictionary(payload)
    hand, translation, quaternion = _validate_frame_arrays(payload)
    if source_names is not None or destination_names is not None:
        if source_names is None or destination_names is None:
            raise ValueError("source_names and destination_names must be provided together")
        hand = hand.copy()
        hand[..., 6:] = allegro_source_to_sim_joint_order(
            hand[..., 6:],
            source_names=source_names,
            destination_names=destination_names,
        )
    return GrabReference(
        hand_target=torch.as_tensor(hand, dtype=torch.float32, device=device),
        object_translation=torch.as_tensor(translation, dtype=torch.float32, device=device),
        object_quaternion_xyzw=torch.as_tensor(
            quaternion, dtype=torch.float32, device=device
        ),
    )
