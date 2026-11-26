"""Causal cached nominal command candidate for paired S0-5 design verification."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation

from .mapping import sdk_keypoints
from .nominal import GROUPS, SDKFailure, matrix
from .schema import TICK_NS


SCOPE = "CANDIDATE_ONLY_EXISTING_TELEOP_NOMINAL"
ARM_RATE_RAD_S = 0.9047786842338604
ARM_SUBSTEP_NS = 4_000_000
ARM_SUBSTEPS = TICK_NS // ARM_SUBSTEP_NS
ARM_DT_S = ARM_SUBSTEP_NS / 1e9
ARM_COMMAND_STEP_RAD = ARM_RATE_RAD_S * ARM_DT_S
ARM_FK_POSITION_TOLERANCE_M = 0.00001
ARM_FK_ROTATION_TOLERANCE_RAD = 0.003490658503988659
HAND_COMMAND_STEP_RAD = 0.10 / ARM_SUBSTEPS
HAND_FILTER_ALPHA = 0.30
RECOVERY_TICKS = 2


class CandidateStatus(IntEnum):
    ACTIVE = 1
    RECOVERING = 2
    NATIVE_ARM_HOLD = 3
    WUJI_SDK_HOLD = 4
    WHOLE_BODY_PATH_HOLD = 5


@dataclass(frozen=True)
class CandidateSubstep:
    issue_tick_ns: int
    q_issued: np.ndarray
    arm_branches_rad: np.ndarray
    status: np.ndarray
    hand_candidate_limit: np.ndarray
    hand_limit_applied: np.ndarray
    native_arms: tuple
    whole_body_path_accepted: bool


@dataclass(frozen=True)
class CandidateTick:
    scope: str
    generation: int
    input_tick_ns: int
    q_nominal: np.ndarray
    qd_nominal: np.ndarray
    qd_valid: bool
    status: np.ndarray
    raw_hand_q_rad: np.ndarray
    raw_hand_valid: np.ndarray
    hand_failure_reason: tuple[str | None, str | None]
    native_tcp_target: np.ndarray
    substeps: tuple[CandidateSubstep, ...]

    @property
    def q_valid(self):
        return bool(np.all(self.status == CandidateStatus.ACTIVE))


class TeleopCandidate:
    """Advance the native arm kernel on one frozen target per available HMB tick.

    The native setpoint is calculated from previous accepted virtual commands
    in place of real feedback. It has no driver, dynamics or release authority.
    Whole-body signed swept safety has final admission at each 4 ms
    substep. Per-side native state advances only alongside admitted commands.
    """

    def __init__(self, robot, kernel, hand, swept_proof, wrist_frames):
        self.robot = robot
        self.kernel = kernel
        self.hand = hand
        self.swept_proof = swept_proof
        self.wrist_frames = tuple(
            (np.asarray(base, dtype=float).copy(), np.asarray(tool, dtype=float).copy())
            for base, tool, *_ in wrist_frames
        )
        if len(self.wrist_frames) != 2 or any(
            pose.shape != (4, 4) or not np.isfinite(pose).all()
            for pair in self.wrist_frames for pose in pair
        ):
            raise ValueError("Signed per-side wrist/base and TCP transformations are required")
        self.generation = -1
        self.cut()

    def cut(self):
        """Start a new independent segment at the previously verified home."""
        home = np.asarray(self.robot.home, dtype=float).copy()
        if home.shape != (54,) or not np.isfinite(home).all() or not self._path_admitted(home, home):
            raise ValueError("Candidate generation requires a legal signed 54-axis home")
        branch = self.kernel.home_reference(home[:14].copy())
        branch = np.zeros(2) if branch is None else np.asarray(branch, dtype=float)
        if branch.shape != (2,) or not np.isfinite(branch).all():
            raise ValueError("Native Teleop returned an invalid START branch")
        self.hand.reset()
        self.generation += 1
        self.previous = home
        self.branch_rad = branch.copy()
        self.last_source = np.full(2, -1, dtype=np.int64)
        self.raw_hand = np.zeros((2, 20), dtype=float)
        self.filtered_hand = np.zeros((2, 20), dtype=float)
        self.hand_valid = np.zeros(2, dtype=bool)
        self.hand_failure_reason = [None, None]
        self.hand_filter_initialized = np.zeros(2, dtype=bool)
        self.previous_tick_ns = None
        self.previous_tick_active = False
        self.consecutive_active = 0

    def _path_admitted(self, start, end):
        return self.robot.legality(end)["feasible"] and self.swept_proof.permits(start, end)

    def _wrist_targets(self, target):
        position = np.asarray(target["wrist_xyz"], dtype=float)
        quaternion = np.asarray(target["wrist_xyzw"], dtype=float)
        if (position.shape != (2, 3) or quaternion.shape != (2, 4)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()):
            raise ValueError("Expected two finite frozen M1 wrist poses")
        orientation = Rotation.from_quat(quaternion).as_matrix()
        out = np.asarray([
            np.linalg.solve(base, matrix(position[side], orientation[side]) @ tool)
            for side, (base, tool) in enumerate(self.wrist_frames)
        ])
        out.flags.writeable = False
        return out

    def _advance_hand_input(self, common, order, source_index):
        if len(source_index) < 2 or any(source_index[side] < 0 for side in range(2)):
            raise ValueError("Candidate requires the two contemporaneous signed MANUS source indices")
        for side in range(2):
            if source_index[side] == self.last_source[side]:
                continue
            self.last_source[side] = source_index[side]
            self.hand_valid[side] = False
            self.hand_failure_reason[side] = None
            try:
                value = np.asarray(self.hand.step(side, sdk_keypoints(common[side], order[side], side)))
            except SDKFailure as error:
                self.raw_hand[side].fill(0.)
                self.hand_failure_reason[side] = str(error)
                continue
            if value.shape != (20,) or not np.isfinite(value).all():
                raise ValueError("Wuji candidate returned a malformed hand command")
            self.raw_hand[side] = value
            if self.hand_filter_initialized[side]:
                self.filtered_hand[side] *= 1. - HAND_FILTER_ALPHA
                self.filtered_hand[side] += HAND_FILTER_ALPHA * value
            else:
                self.filtered_hand[side] = value
            self.hand_filter_initialized[side] = self.hand_valid[side] = True

    def _settled(self, side, previous, target):
        realized = np.asarray(self.kernel.fk(side, previous.copy()), dtype=float)
        target = np.asarray(target, dtype=float)
        if (realized.shape != (4, 4) or target.shape != (4, 4)
                or not np.isfinite(realized).all() or not np.isfinite(target).all()):
            raise ValueError("Native Teleop returned an invalid FK or step target for a stationary command")
        distance = np.linalg.norm(realized[:3, 3] - target[:3, 3])
        angle = Rotation.from_matrix(realized[:3, :3].T @ target[:3, :3]).magnitude()
        return distance <= ARM_FK_POSITION_TOLERANCE_M and angle <= ARM_FK_ROTATION_TOLERANCE_RAD

    def _advance_substep(self, wrist_target, issue_tick_ns):
        previous = self.previous
        command = previous.copy()
        status = np.full(4, CandidateStatus.ACTIVE, dtype=np.uint16)
        limited = np.zeros(40, dtype=bool)
        branch = self.branch_rad.copy()
        native = tuple(self.kernel.propose(wrist_target, previous[:14].copy(), previous[:14].copy(),
                                           branch.copy(), dt_s=ARM_DT_S))
        if len(native) != 2:
            raise ValueError("Native Teleop candidate must return two per-arm records")
        for side, row in enumerate(native):
            arm = GROUPS[side]
            if not row.accepted or not row.ik_target_accepted or row.analytic_braking:
                status[side] = CandidateStatus.NATIVE_ARM_HOLD
                continue
            q = np.asarray(row.joint_command, dtype=float)
            if (q.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(row.arm_angle_command_rad)
                    or np.max(np.abs(q - previous[arm])) > ARM_COMMAND_STEP_RAD + 1e-9):
                raise ValueError("Native Teleop command violates candidate shape, finite or 4 ms rate contract")
            if np.array_equal(q, previous[arm]) and not self._settled(side, previous[arm], row.step_target):
                status[side] = CandidateStatus.NATIVE_ARM_HOLD
                continue
            command[arm] = q
            branch[side] = row.arm_angle_command_rad
        low, high = self.robot.bounds.T
        for side in range(2):
            hand = GROUPS[side + 2]
            if not self.hand_valid[side]:
                status[side + 2] = CandidateStatus.WUJI_SDK_HOLD
                continue
            desired = self.filtered_hand[side]
            command[hand] = np.clip(desired, np.maximum(low[hand], previous[hand] - HAND_COMMAND_STEP_RAD),
                                    np.minimum(high[hand], previous[hand] + HAND_COMMAND_STEP_RAD))
            limited[side * 20:(side + 1) * 20] = np.abs(command[hand] - desired) > 1e-10
        # A previously certified identical configuration needs no new path.
        allowed = np.array_equal(command, previous) or self._path_admitted(previous, command)
        if allowed:
            self.previous = command
            self.branch_rad = branch
        else:
            status[status == CandidateStatus.ACTIVE] = CandidateStatus.WHOLE_BODY_PATH_HOLD
        return CandidateSubstep(issue_tick_ns, self.previous.copy(), self.branch_rad.copy(), status, limited,
                                limited.copy() if allowed else np.zeros_like(limited), native, bool(allowed))

    def step(self, target, common, order, source_index, *, tick_ns):
        tick_ns = int(tick_ns)
        if self.previous_tick_ns is not None and tick_ns != self.previous_tick_ns + TICK_NS:
            raise ValueError("Discontinuous HMB tick requires an explicit candidate generation cut")
        wrist_target = self._wrist_targets(target)
        self._advance_hand_input(common, order, source_index)
        before = self.previous.copy()
        status = np.full(4, CandidateStatus.ACTIVE, dtype=np.uint16)
        substeps = []
        for index in range(ARM_SUBSTEPS):
            substep = self._advance_substep(wrist_target, tick_ns + index * ARM_SUBSTEP_NS)
            substeps.append(substep)
            first_failure = status == CandidateStatus.ACTIVE
            status[first_failure] = substep.status[first_failure]
        both_arms_and_hands_active = bool(np.all(status == CandidateStatus.ACTIVE))
        self.consecutive_active = self.consecutive_active + 1 if both_arms_and_hands_active else 0
        if self.consecutive_active < RECOVERY_TICKS:
            status[status == CandidateStatus.ACTIVE] = CandidateStatus.RECOVERING
        tick_active = bool(np.all(status == CandidateStatus.ACTIVE))
        velocity_valid = tick_active and self.previous_tick_active
        velocity = (self.previous - before) / (TICK_NS / 1e9) if velocity_valid else np.zeros_like(before)
        self.previous_tick_ns = tick_ns
        self.previous_tick_active = tick_active
        return CandidateTick(SCOPE, self.generation, tick_ns, self.previous.copy(), velocity,
                             velocity_valid, status, self.raw_hand.copy(), self.hand_valid.copy(),
                             tuple(self.hand_failure_reason),
                             wrist_target, tuple(substeps))
