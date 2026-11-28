"""Native nominal proposals on a deterministic offline simulation clock."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation

from .mapping import sdk_keypoints
from .nominal import GROUPS, SDKFailure
from .teleop_candidate import (
    ARM_COMMAND_STEP_RAD, ARM_DT_S, ARM_FK_POSITION_TOLERANCE_M, ARM_FK_ROTATION_TOLERANCE_RAD,
    COMMAND_CHANGE_TOLERANCE_RAD, HAND_COMMAND_STEP_RAD, HAND_FILTER_ALPHA,
)


class OfflineNominalStatus(IntEnum):
    ACTIVE = 1
    LIMITED_ACTIVE = 2
    NATIVE_ARM_HOLD = 3
    NATIVE_ARM_FALLBACK = 4
    WUJI_SDK_HOLD = 5
    WUJI_LIMIT_HOLD = 6
    HELD_BY_START_STAGE = 7
    NATIVE_PAIR_HOLD = 8
    NATIVE_PAIR_FAILED = 9
    INPUT_MISSING = 10


@dataclass(frozen=True)
class OfflineNominalProposal:
    q_requested: np.ndarray
    q_branch: np.ndarray
    status: np.ndarray
    native_arms: tuple
    hand_limited: np.ndarray
    native_tcp_target: np.ndarray | None
    hard_fault: bool
    raw_hand_q: np.ndarray
    filtered_hand_q: np.ndarray
    raw_hand_valid: np.ndarray
    hand_failure_reason: tuple[str | None, str | None]


def _array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite offline nominal input of shape {shape}")
    return result


def _poses(value, count):
    poses = _array(value, (count, 4, 4))
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
        raise ValueError("Offline nominal input requires proper SE(3) transforms")
    return poses


class OfflineTeleopNominal:
    """Reuse the native arm kernel and Wuji SDK, without a driver or command publisher.

    The caller owns the original time grid, current simulated feedback, the last
    certified command and branch, the START stage and the independent sweep proof.
    The hand filter advances on available observations even if an output is
    refused. Command and branch advance only by the caller after certification.
    """

    def __init__(self, robot, kernel, hand, frames):
        self.kernel = kernel
        self.hand = hand
        self.set_frames(frames)
        self.bounds = _array(robot.bounds, (54, 2)).copy()
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Offline nominal requires 54 nonempty joint ranges")
        self.ready = False

    def set_frames(self, frames):
        """Install the one locked map for a new root or explicit generation."""
        frames = _poses(frames, 7).copy()
        self.base_from_human = np.linalg.inv(frames[1:3]) @ frames[0]
        self.tracker_to_native_tcp = frames[3:5] @ frames[5:7]
        self.ready = False

    def reset(self, home_q):
        """Latch signed/observed simulator Home and reset both Wuji sessions."""
        self.ready = False
        joints = _array(home_q, (54,))
        if np.any(joints < self.bounds[:, 0]) or np.any(joints > self.bounds[:, 1]):
            raise ValueError("Offline nominal reset is outside the named joint ranges")
        branch = _array(self.kernel.home_reference(joints[:14].copy()), (2,)).copy()
        self.hand.reset()
        self.last_source = [None, None]
        self.raw_hand = np.zeros((2, 20), dtype=float)
        self.filtered_hand = np.zeros_like(self.raw_hand)
        self.hand_valid = np.zeros(2, dtype=bool)
        self.hand_filter_initialized = np.zeros(2, dtype=bool)
        self.hand_failure_reason = [None, None]
        self.refresh_tick_ns = None
        self.ready = True
        return branch

    def refresh(self, target, common, order, source_index, *, tick_ns):
        """Consume only currently available, root-stable MANUS input identities.

        None or a negative integer means unavailable. HMB row or partition
        indices alone cannot identify input samples across storage partitions.
        """
        if not self.ready:
            raise RuntimeError("Offline nominal generation requires an explicit reset")
        if isinstance(tick_ns, (bool, np.bool_)) or int(tick_ns) != tick_ns:
            raise ValueError("Simulated input cutoff must be an integer nanosecond")
        if self.refresh_tick_ns is not None and tick_ns < self.refresh_tick_ns:
            raise ValueError("Simulated input cutoff reversed within a generation")
        if any(len(pair) != 2 for pair in (common, order, source_index)):
            raise ValueError("Offline nominal refresh requires two hand inputs")
        if target is not None:
            _poses(target, 2)
        for side, token in enumerate(source_index):
            if token is None or (isinstance(token, (int, np.integer)) and token < 0):
                self.hand_valid[side] = False
                self.hand_failure_reason[side] = "UNAVAILABLE_SIGNED_MANUS"
                continue
            hash(token)
            if token == self.last_source[side]:
                continue
            self.last_source[side] = token
            self.hand_valid[side] = False
            self.hand_failure_reason[side] = None
            try:
                value = self.hand.step(side, sdk_keypoints(common[side], order[side], side))
            except SDKFailure as error:
                self.raw_hand[side].fill(0)
                self.hand_failure_reason[side] = str(error)
                continue
            value = _array(value, (20,))
            self.raw_hand[side] = value
            if self.hand_filter_initialized[side]:
                self.filtered_hand[side] *= 1 - HAND_FILTER_ALPHA
                self.filtered_hand[side] += HAND_FILTER_ALPHA * value
            else:
                self.filtered_hand[side] = value
            self.hand_filter_initialized[side] = self.hand_valid[side] = True
        self.refresh_tick_ns = int(tick_ns)

    def mapped_native_tcp(self, target):
        return self.base_from_human @ _poses(target, 2) @ self.tracker_to_native_tcp

    def _stationary_unmet(self, side, q, target):
        realized = self.kernel.fk(side, q)
        distance = np.linalg.norm(realized[:3, 3] - target[:3, 3])
        angle = Rotation.from_matrix(realized[:3, :3].T @ target[:3, :3]).magnitude()
        return distance > ARM_FK_POSITION_TOLERANCE_M or angle > ARM_FK_ROTATION_TOLERANCE_RAD

    def propose(self, target, measured_q, previous_q, branch, dt=ARM_DT_S, *, hand_axes=None):
        """Return pre-proof proposals and candidate branch at one 4 ms substep."""
        if not self.ready or dt != ARM_DT_S:
            raise ValueError("Offline nominal proposal requires a reset 250 Hz generation")
        measured = _array(measured_q, (54,))
        previous = _array(previous_q, (54,))
        command, proposed_branch = previous.copy(), _array(branch, (2,)).copy()
        enabled = np.ones(40, dtype=bool) if hand_axes is None else np.asarray(hand_axes)
        if enabled.shape != (40,) or enabled.dtype != np.bool_:
            raise ValueError("START hand-stage mask must be bool[40]")
        limited = np.zeros(40, dtype=bool)
        native_tcp = None if target is None else self.mapped_native_tcp(target)
        native = ()
        hard_fault = False
        status = np.full(4, OfflineNominalStatus.INPUT_MISSING, dtype=np.uint16)

        if native_tcp is not None:
            native = tuple(self.kernel.propose(native_tcp, measured[:14].copy(), previous[:14].copy(),
                                               proposed_branch.copy(), dt_s=dt))
            if len(native) != 2:
                raise ValueError("Expected two named native arm records")
            accepted = native[0].pair_accepted
            hard_fault = native[0].pair_hard_fault
            if (accepted != native[1].pair_accepted or hard_fault != native[1].pair_hard_fault
                    or (accepted and hard_fault)):
                raise ValueError("Native paired command state is inconsistent")
            status.fill(OfflineNominalStatus.NATIVE_PAIR_FAILED if hard_fault
                        else OfflineNominalStatus.NATIVE_PAIR_HOLD)
            if accepted:
                status[:2] = self._propose_arms(previous, command, proposed_branch, native, native_tcp)
                status[2:] = self._propose_hands(previous, command, limited, enabled)

        return OfflineNominalProposal(
            command, proposed_branch, status, native, limited, native_tcp, hard_fault,
            self.raw_hand.copy(), self.filtered_hand.copy(), self.hand_valid.copy(),
            tuple(self.hand_failure_reason),
        )

    def _propose_arms(self, previous, command, branch, native, tcp):
        statuses = []
        for side, row in enumerate(native):
            group = GROUPS[side]
            q = _array(row.joint_command, (7,))
            fallback = row.analytic_braking or row.analytic_exact_stop_used
            if not row.accepted or (not row.ik_target_accepted and not fallback):
                statuses.append(OfflineNominalStatus.NATIVE_ARM_HOLD)
                continue
            change = np.max(np.abs(q - previous[group]))
            if change > ARM_COMMAND_STEP_RAD + 1e-9 or not np.isfinite(row.arm_angle_command_rad):
                raise ValueError("Native arm proposal violates the 4 ms shape or rate contract")
            if not fallback and change <= COMMAND_CHANGE_TOLERANCE_RAD and self._stationary_unmet(side, q, tcp[side]):
                statuses.append(OfflineNominalStatus.NATIVE_ARM_HOLD)
                continue
            command[group] = q
            branch[side] = row.arm_angle_command_rad
            if fallback:
                statuses.append(OfflineNominalStatus.NATIVE_ARM_FALLBACK)
            elif row.rate_limited or row.workspace_projected:
                statuses.append(OfflineNominalStatus.LIMITED_ACTIVE)
            else:
                statuses.append(OfflineNominalStatus.ACTIVE)
        return statuses

    def _propose_hands(self, previous, command, limited, enabled):
        statuses = []
        low, high = self.bounds.T
        for side in range(2):
            group = GROUPS[side + 2]
            hand = slice(side * 20, (side + 1) * 20)
            axes = enabled[hand]
            if not axes.any():
                statuses.append(OfflineNominalStatus.HELD_BY_START_STAGE)
                continue
            if not self.hand_valid[side]:
                statuses.append(OfflineNominalStatus.WUJI_SDK_HOLD)
                continue
            prior, desired = previous[group], self.filtered_hand[side]
            bounded = np.clip(desired, np.maximum(low[group], prior - HAND_COMMAND_STEP_RAD),
                              np.minimum(high[group], prior + HAND_COMMAND_STEP_RAD))
            limited[hand] = axes & (np.abs(bounded - desired) > 1e-10)
            command[group] = np.where(axes, bounded, prior)
            stationary = np.max(np.abs(bounded[axes] - prior[axes])) <= COMMAND_CHANGE_TOLERANCE_RAD
            unmet = np.max(np.abs(bounded[axes] - desired[axes])) > COMMAND_CHANGE_TOLERANCE_RAD
            if stationary and unmet:
                statuses.append(OfflineNominalStatus.WUJI_LIMIT_HOLD)
                command[group] = prior
            elif limited[hand].any():
                statuses.append(OfflineNominalStatus.LIMITED_ACTIVE)
            else:
                statuses.append(OfflineNominalStatus.ACTIVE)
        return statuses
