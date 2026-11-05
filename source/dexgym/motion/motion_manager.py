"""GPU-resident reference frames and vectorized future goal windows."""

from pathlib import Path

import numpy as np
import torch


def quaternion_xyzw_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """Return first two rotation columns, as continuous six-dimensional features."""
    q = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = q.unbind(-1)
    col0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)), dim=-1)
    col1 = torch.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)
    return torch.cat((col0, col1), dim=-1)


def quaternion_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    six = quaternion_xyzw_to_6d(quat)
    first, second = six[..., :3], six[..., 3:]
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


class MotionManager:
    """One motion now; `motion_id` distinguishes future library entries."""

    def __init__(self, path: str | Path, num_envs: int, device: torch.device | str, motion_id: str | None = None):
        with np.load(path, allow_pickle=False) as npz:
            arrays = {
                name: npz[name]
                for name in (
                    "q_arm",
                    "qd_arm",
                    "q_full",
                    "link_pos",
                    "link_quat_xyzw",
                    "link_linear_vel",
                    "link_angular_vel",
                    "tcp_pos",
                    "tcp_quat_xyzw",
                    "tcp_linear_vel",
                    "tcp_angular_vel",
                    "time",
                )
            }
            self.link_names = tuple(tuple(row) for row in npz["link_names"])
            self.joint_names_arm = tuple(npz["joint_names_arm"])
            self.motion_id = motion_id or str(npz["motion_id"])
        self.device = torch.device(device)
        self.length = len(arrays["time"])
        self.frame = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.data = {
            key: torch.as_tensor(value.copy(), device=self.device, dtype=torch.float32) for key, value in arrays.items()
        }
        self.data["link_rotmat"] = quaternion_xyzw_to_matrix(self.data["link_quat_xyzw"])
        if self.data["q_arm"].shape != (self.length, 14) or self.link_names != (
            ("Link3_L", "Link4_L", "tcp_L"),
            ("Link3_R", "Link4_R", "tcp_R"),
        ):
            raise ValueError("Unexpected Marvin bimanual motion schema")

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | int = 0):
        frame = torch.as_tensor(frames, device=self.device, dtype=torch.long)
        self.frame[env_ids] = frame.clamp(0, self.length - 1)

    def advance(self):
        self.frame.add_(1).clamp_(max=self.length - 1)

    def current(self, name: str):
        return self.data[name][self.frame]

    def future(self, count: int = 10):
        return self.future_at(self.frame, count)

    def future_at(self, frame: torch.Tensor, count: int = 10):
        offsets = torch.arange(1, count + 1, device=self.device)
        raw = frame[:, None] + offsets
        index = raw.clamp(max=self.length - 1)
        return index, raw >= self.length

    def future_links(self, count: int = 10):
        return self.future_links_at(self.frame, count)

    def future_links_at(self, frame: torch.Tensor, count: int = 10):
        indices, beyond = self.future_at(frame, count)
        result = {
            key: self.data[key][indices]
            for key in ("link_pos", "link_quat_xyzw", "link_linear_vel", "link_angular_vel")
        }
        for key in ("link_linear_vel", "link_angular_vel"):
            result[key] = torch.where(beyond[..., None, None, None], torch.zeros_like(result[key]), result[key])
        return result

    def goal(self, count: int = 10):
        return self.goal_at(self.frame, count)

    def goal_at(self, frame: torch.Tensor, count: int = 10):
        links = self.future_links_at(frame, count)
        pos = links["link_pos"]
        quat6 = quaternion_xyzw_to_6d(links["link_quat_xyzw"])
        vel = links["link_linear_vel"]
        omega = links["link_angular_vel"]
        return torch.cat((pos, quat6, vel, omega), dim=-1).reshape(len(frame), count, 90)
