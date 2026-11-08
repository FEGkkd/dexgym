"""Contact-aware refinement of Wuji SDK right-hand joint trajectories.

The SDK output is immutable input. Optimization uses the same right-hand MJCF
collision meshes and fingertip sites as the original retargeting FK. Frames
without penetration are left alone; penetrating frames are optimized with their
temporal neighbors fixed, followed by another sweep for local consistency.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import minimize

from .wuji_retarget import RightHandFK


@dataclass(frozen=True)
class PostOptWeights:
    """Dimensionless weights after normalizing each residual by its scale."""

    tip: float = 20.0
    sdk: float = 0.1
    collision: float = 1.0
    smooth: float = 0.1
    tip_scale_m: float = 0.01
    joint_scale_rad: float = 0.1
    collision_scale_m: float = 0.002
    max_sweeps: int = 2
    max_iterations_per_frame: int = 40
    trigger_penetration_m: float = 0.0005
    min_step_cap_rad: float = 0.2
    step_slack_rad: float = 0.01


class HandPostOptimizer:
    """Refine SDK poses while preserving tips and removing self-penetration."""

    def __init__(self, fk: RightHandFK, weights: PostOptWeights):
        self.fk = fk
        self.weights = weights
        self._site_jac = np.zeros((3, fk.model.nv), dtype=np.float64)
        self._geom_jac_1 = np.zeros_like(self._site_jac)
        self._geom_jac_2 = np.zeros_like(self._site_jac)

    def _forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fk = self.fk
        fk.data.qpos[:] = 0.0
        fk.data.qpos[fk.q_indices] = q
        mujoco.mj_forward(fk.model, fk.data)
        wrist_pos = fk.data.xpos[fk.wrist_id]
        wrist_rot = fk.data.xmat[fk.wrist_id].reshape(3, 3)
        tips = (fk.data.site_xpos[fk.tip_ids] - wrist_pos) @ wrist_rot
        return tips, wrist_rot

    def audit(self, q: np.ndarray) -> dict[str, float | int]:
        self._forward(np.asarray(q, dtype=np.float64))
        distances = [float(contact.dist) for contact in self.fk.data.contact[: self.fk.data.ncon]]
        return {
            "penetrating_contacts": sum(distance < -1e-6 for distance in distances),
            "deepest_penetration_mm": -1000.0 * min([0.0, *distances]),
        }

    def frame_loss_and_grad(
        self, q: np.ndarray, q_sdk: np.ndarray, tip_sdk: np.ndarray, q_prev: np.ndarray, q_next: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Evaluate four losses and analytic pose/contact Jacobians for one frame."""
        w = self.weights
        fk = self.fk
        tips, wrist_rot = self._forward(np.asarray(q, dtype=np.float64))
        grad = np.zeros(20, dtype=np.float64)
        tip_delta = tips - tip_sdk
        value = w.tip * float(np.sum(tip_delta**2)) / w.tip_scale_m**2
        for index, site in enumerate(fk.tip_ids):
            mujoco.mj_jacSite(fk.model, fk.data, self._site_jac, None, site)
            jac = wrist_rot.T @ self._site_jac[:, fk.q_indices]
            grad += 2.0 * w.tip / w.tip_scale_m**2 * (tip_delta[index] @ jac)

        q_delta = q - q_sdk
        value += w.sdk * float(q_delta @ q_delta) / w.joint_scale_rad**2
        grad += 2.0 * w.sdk / w.joint_scale_rad**2 * q_delta

        # Central second difference and local velocity keep corrected frames
        # consistent with both neighbors without changing the source sequence.
        acceleration = q_prev - 2.0 * q + q_next
        velocity_before = q - q_prev
        velocity_after = q_next - q
        smooth = acceleration @ acceleration + 0.25 * (
            velocity_before @ velocity_before + velocity_after @ velocity_after
        )
        value += w.smooth * float(smooth) / w.joint_scale_rad**2
        grad += w.smooth / w.joint_scale_rad**2 * (-4.0 * acceleration + 0.5 * (velocity_before - velocity_after))

        for contact in fk.data.contact[: fk.data.ncon]:
            depth = max(-float(contact.dist), 0.0)
            if depth <= 0:
                continue
            body1 = int(fk.model.geom_bodyid[contact.geom1])
            body2 = int(fk.model.geom_bodyid[contact.geom2])
            mujoco.mj_jac(fk.model, fk.data, self._geom_jac_1, None, contact.pos, body1)
            mujoco.mj_jac(fk.model, fk.data, self._geom_jac_2, None, contact.pos, body2)
            # MuJoCo's normal points from geom2 to geom1, hence distance
            # derivative is n dot (J2 - J1); checked against finite differences.
            distance_jac = contact.frame[:3] @ (self._geom_jac_2[:, fk.q_indices] - self._geom_jac_1[:, fk.q_indices])
            value += w.collision * depth**2 / w.collision_scale_m**2
            grad -= 2.0 * w.collision * depth / w.collision_scale_m**2 * distance_jac
        return value, grad

    def optimize_sequence(self, q_sdk: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return optimized joints, FK tips, and audit metrics for one motion."""
        source = np.asarray(q_sdk, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 20 or len(source) == 0 or not np.isfinite(source).all():
            raise ValueError("Expected finite q_sdk[T,20]")
        q = source.copy()
        tip_sdk = np.stack([self._forward(frame)[0].copy() for frame in source])
        initial_depth = np.array([self.audit(frame)["deepest_penetration_mm"] for frame in source])
        active = np.flatnonzero(initial_depth > self.weights.trigger_penetration_m * 1000)
        original_max_step = float(np.abs(np.diff(source, axis=0)).max()) if len(source) > 1 else 0.0
        step_cap = max(original_max_step + self.weights.step_slack_rad, self.weights.min_step_cap_rad)
        failures = 0
        infeasible_temporal_frames = 0
        for _ in range(self.weights.max_sweeps):
            changed = 0
            for frame in active:
                before = q[frame].copy()
                prev_q = q[max(frame - 1, 0)].copy()
                next_q = q[min(frame + 1, len(q) - 1)].copy()
                low = self.fk.lower.astype(float).copy()
                high = self.fk.upper.astype(float).copy()
                if frame > 0:
                    low = np.maximum(low, prev_q - step_cap)
                    high = np.minimum(high, prev_q + step_cap)
                if frame + 1 < len(q):
                    low = np.maximum(low, next_q - step_cap)
                    high = np.minimum(high, next_q + step_cap)
                bounds = list(zip(low, high))
                if np.any(low > high):
                    # Earlier optimized neighbors can make the local hard step
                    # constraints inconsistent. Keep this frame and audit
                    # its collision later instead of dropping the whole clip.
                    infeasible_temporal_frames += 1
                    continue
                solution = minimize(
                    self.frame_loss_and_grad,
                    before,
                    args=(source[frame], tip_sdk[frame], prev_q, next_q),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={"maxiter": self.weights.max_iterations_per_frame, "ftol": 1e-8},
                )
                if not np.isfinite(solution.x).all():
                    failures += 1
                    continue
                q[frame] = solution.x
                changed += int(np.max(np.abs(q[frame] - before)) > 1e-6)
            if changed == 0:
                break
        tips = np.stack([self._forward(frame)[0].copy() for frame in q]).astype(np.float32)
        final_depth = np.array([self.audit(frame)["deepest_penetration_mm"] for frame in q])
        tip_change = np.linalg.norm(tips - tip_sdk, axis=-1) * 1000.0
        fallback_reasons = []
        if final_depth.max() > initial_depth.max() + 0.1:
            fallback_reasons.append("max_penetration_worse")
        if tip_change.max() > 12.0:
            fallback_reasons.append("tip_change_over_12mm")
        if fallback_reasons:
            # Keep the complete SDK motion rather than quietly publishing a
            # collision or fingertip regression from a nonconvex local solve.
            q = source.copy()
            tips = tip_sdk.astype(np.float32)
            final_depth = initial_depth.copy()
            tip_change = np.zeros_like(tip_change)
        metrics = {
            "frames": len(q),
            "attempted_frames": len(active),
            "optimized_frames": 0 if fallback_reasons else len(active),
            "fallback_reasons": fallback_reasons,
            "initial_penetrating_frames": int(np.count_nonzero(initial_depth > 0.001)),
            "final_penetrating_frames": int(np.count_nonzero(final_depth > 0.001)),
            "initial_max_penetration_mm": float(initial_depth.max()),
            "final_max_penetration_mm": float(final_depth.max()),
            "final_frames_over_0p5mm": int(np.count_nonzero(final_depth > 0.5)),
            "max_tip_change_mm": float(tip_change.max()),
            "max_joint_change_rad": float(np.max(np.abs(q - source))),
            "source_max_joint_step_rad": original_max_step,
            "max_joint_step_rad": float(np.abs(np.diff(q, axis=0)).max()) if len(q) > 1 else 0.0,
            "joint_step_cap_rad": step_cap,
            "optimization_nonfinite_failures": failures,
            "infeasible_temporal_frames": infeasible_temporal_frames,
        }
        return q.astype(np.float32), tips, metrics
