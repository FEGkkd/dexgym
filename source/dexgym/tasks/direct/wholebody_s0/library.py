"""Read and validate wholebody_s0 reference clips for later training."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import SCHEMA

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class MotionLibrary:
    root: Path
    manifest: dict

    @classmethod
    def open(cls, path):
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["schema"] != SCHEMA or manifest["kind"] != "reference_motion_library":
            raise ValueError(f"Unsupported wholebody_s0 library: {path}")
        names = manifest["joint_names"]
        if len(set(names)) != 54 or len(names) != 54:
            raise ValueError("Motion Library requires 54 unique joint names")
        return cls(path.resolve().parent, manifest)

    def entries(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}")
        return [entry for entry in self.manifest["clips"] if entry["split"] == split]

    def load_clip(self, entry):
        path = (self.root / entry["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Motion path leaves library: {path}")
        with np.load(path, allow_pickle=False) as clip:
            values = {key: clip[key] for key in clip.files}
        n = entry["frames"]
        fixed_shapes = {
            "q_ref": (n, 54), "qd_ref": (n, 54), "time_ns": (n,),
            "tracker_pose": (n, 2, 7), "target_tcp_world": (n, 2, 7),
            "robot_wrist_pose": (n, 2, 7), "robot_fingertip_pose": (n, 2, 5, 7),
        }
        for field, shape in fixed_shapes.items():
            if values[field].shape != shape or not np.isfinite(values[field]).all():
                raise ValueError(f"Invalid {field} in {path}")
        for side in ("left", "right"):
            manus = values[f"manus_{side}_pose"]
            if manus.ndim != 3 or manus.shape[0] != n or manus.shape[2] != 7 or not np.isfinite(manus).all():
                raise ValueError(f"Invalid {side} MANUS skeleton in {path}")
        if values["joint_names"].tolist() != self.manifest["joint_names"]:
            raise ValueError(f"Joint order disagrees with manifest: {path}")
        if values["source_episode"].item() != entry["source_episode"]:
            raise ValueError(f"Episode identity disagrees with manifest: {path}")
        bounds = np.asarray(self.manifest["joint_limits_rad"])
        if np.any(values["q_ref"] < bounds[:, 0] - 1e-6) or np.any(values["q_ref"] > bounds[:, 1] + 1e-6):
            raise ValueError(f"Joint target exceeds model range: {path}")
        if n < 2 or not np.all(np.diff(values["time_ns"]) == 1_000_000_000 // self.manifest["rate_hz"]):
            raise ValueError(f"Broken reference clock in {path}")
        if values["time_ns"][0] != entry["start_time_ns"]:
            raise ValueError(f"Start time disagrees with manifest: {path}")
        return values
