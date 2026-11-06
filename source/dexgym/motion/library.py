"""Named, hashed, variable-length Marvin references with GPU batched lookup."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .motion_manager import MotionManager


class MotionLibrary:
    def __init__(self, registry: dict[str, str | Path], device: str | torch.device):
        if not registry:
            raise ValueError("Motion library needs at least one named clip")
        self.device = torch.device(device)
        self.names = tuple(registry)
        self.clips = []
        self.hashes = {}
        fps = []
        for name, path in registry.items():
            path = Path(path)
            clip = MotionManager(path, 1, self.device, motion_id=name)
            self.clips.append(clip)
            self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as data:
                frame_rate = int(data["fps"]) if "fps" in data else round(1 / float(np.diff(data["time"]).mean()))
            if frame_rate != 50:
                raise ValueError(f"{name}: declared FPS must equal 50 Hz policy rate")
            fps.append(frame_rate)
        joint_names = self.clips[0].joint_names_arm
        link_names = self.clips[0].link_names
        if any(clip.joint_names_arm != joint_names or clip.link_names != link_names for clip in self.clips):
            raise ValueError("Motion clips must share joint order and TCP frame definitions")
        self.joint_names_arm = joint_names
        self.link_names = link_names
        self.lengths = torch.tensor([clip.length for clip in self.clips], device=self.device, dtype=torch.long)
        self.fps = torch.tensor(fps, device=self.device, dtype=torch.long)
        self.curriculum_weights = torch.ones(len(self.clips), device=self.device)

    def _check(self, motion_id: torch.Tensor, frame: torch.Tensor):
        if motion_id.shape != frame.shape or motion_id.ndim != 1:
            raise ValueError("Motion IDs and frames must be aligned vectors")
        if bool(((motion_id < 0) | (motion_id >= len(self.clips))).any()):
            raise ValueError("Unknown motion ID")
        if bool(((frame < 0) | (frame >= self.lengths[motion_id])).any()):
            raise ValueError("Reference frame is outside its motion")

    def reference_at(self, motion_id: torch.Tensor, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        self._check(motion_id, frame)
        template = self.clips[0].reference_at(torch.zeros(1, device=self.device, dtype=torch.long))
        result = {key: torch.empty((len(frame), *value.shape[1:]), dtype=value.dtype, device=self.device)
                  for key, value in template.items()}
        for index, clip in enumerate(self.clips):
            selected = motion_id == index
            if bool(selected.any()):
                for key, value in clip.reference_at(frame[selected]).items():
                    result[key][selected] = value
        return result

    def reference_features_at(self, motion_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        self._check(motion_id, frame)
        result = torch.empty((len(frame), 90), device=self.device)
        for index, clip in enumerate(self.clips):
            selected = motion_id == index
            if bool(selected.any()):
                result[selected] = clip.reference_features_at(frame[selected])
        return result

    def future_goal_at(self, motion_id: torch.Tensor, frame: torch.Tensor, count: int = 10) -> torch.Tensor:
        self._check(motion_id, frame)
        if count < 1:
            raise ValueError("Goal window must contain future frames")
        result = torch.empty((len(frame), count, 90), device=self.device)
        for index, clip in enumerate(self.clips):
            selected = motion_id == index
            if bool(selected.any()):
                result[selected] = clip.future_goal_at(frame[selected], count)
        return result

    def sample_reset(self, batch: int, *, horizon: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        eligible = self.lengths > horizon
        if batch < 1 or horizon < 1 or not bool(eligible.any()):
            raise ValueError("No motion supports the requested reset horizon")
        weights = torch.where(eligible, self.curriculum_weights, 0)
        if not bool((weights > 0).any()):
            raise ValueError("No positive curriculum weight for eligible motions")
        motion_id = torch.multinomial(weights, batch, replacement=True)
        frame = (torch.rand(batch, device=self.device) * (self.lengths[motion_id] - horizon)).long()
        return motion_id, frame

    def metadata(self) -> list[dict]:
        return [{"motion_id": name, "frames": int(self.lengths[i]), "fps": int(self.fps[i]),
                 "sha256": self.hashes[name]} for i, name in enumerate(self.names)]
