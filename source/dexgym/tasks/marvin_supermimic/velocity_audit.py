"""Compare sampled Newton joint velocity with physical substep displacement."""

import torch


class SubstepPeakRecorder:
    """Record per-environment peak arm speed across each policy step's substeps."""

    def __init__(self, *, num_envs: int, joints: int, substeps: int):
        if min(num_envs, joints, substeps) < 1:
            raise ValueError("Substep audit dimensions must be positive")
        self.num_envs, self.joints, self.substeps = num_envs, joints, substeps
        self._current = None
        self._samples = 0
        self._peaks = []

    def begin(self):
        if self._current is not None:
            raise RuntimeError("Previous policy step has no final velocity sample")
        self._samples = 0

    def sample(self, velocity: torch.Tensor):
        if (velocity.shape != (self.num_envs, self.joints)
                or not bool(torch.isfinite(velocity).all())):
            raise ValueError("Newton substep arm velocity has invalid shape or value")
        peak = velocity.detach().abs().amax(-1)
        self._current = peak.clone() if self._current is None else torch.maximum(self._current, peak)
        self._samples += 1

    def finish(self, velocity: torch.Tensor):
        self.sample(velocity)
        if self._samples != self.substeps + 1:
            raise RuntimeError("Newton velocity audit missed a physical substep")
        self._peaks.append(self._current.cpu())
        self._current = None

    def peaks(self) -> torch.Tensor:
        if not self._peaks:
            raise ValueError("Newton velocity audit has no complete policy step")
        return torch.stack(self._peaks)


def substep_velocity_metrics(position: torch.Tensor, velocity: torch.Tensor,
                             *, dt: float) -> dict:
    if (position.ndim != 2 or position.shape != velocity.shape
            or len(position) < 2 or dt <= 0
            or not bool(torch.isfinite(position).all() and torch.isfinite(velocity).all())):
        raise ValueError("Substep position, velocity or time contract is invalid")
    finite_difference = (position[1:] - position[:-1]) / dt
    net = (position[-1] - position[0]) / ((len(position) - 1) * dt)
    return {
        "sample_count": len(position),
        "instantaneous_abs_max_rad_s": float(velocity.abs().max()),
        "substep_displacement_abs_max_rad_s": float(finite_difference.abs().max()),
        "net_displacement_speed_rad_s": float(net.abs().max()),
        "joint_instantaneous_abs_p95_rad_s":
            torch.quantile(velocity.abs(), 0.95, dim=0).tolist(),
        "joint_substep_displacement_abs_p95_rad_s":
            torch.quantile(finite_difference.abs(), 0.95, dim=0).tolist(),
        "joint_net_displacement_speed_rad_s": net.abs().tolist(),
        "joint_position_range_rad": (position.max(0).values - position.min(0).values).tolist(),
    }
