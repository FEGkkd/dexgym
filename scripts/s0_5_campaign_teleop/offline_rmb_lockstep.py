"""Offline 250 Hz nominal / 500 Hz signed MuJoCo execution without ROS."""

from dataclasses import dataclass
from time import monotonic_ns

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT
from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineNominalStatus
from dexgym.tasks.direct.wholebody_s05.v1.rmb.teleop_candidate import ARM_SUBSTEPS
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import CpuServo, ServoProfile
from dexgym.tasks.direct.wholebody_s05.v1.task.safety import SafetyProjection
from swept_live_guard import Reason
from swept_live_guard_v2 import SweptEnvelopeV2


ACTIVE = (OfflineNominalStatus.ACTIVE, OfflineNominalStatus.LIMITED_ACTIVE)
FINGER_ORDER = (8, 5, 6, 7, 9, 0, 1, 2, 3, 4)
PROOF = {"max_queries": 8191, "max_depth": 18}
PROOF_NS = 2_000_000_000
START_FINGER_STEP_RAD = .015


@dataclass(frozen=True)
class Substep:
    proposal: object
    proof: str
    request_metrics: dict
    issued: np.ndarray | None
    interval: object
    used_hold: bool
    fault: str | None
    hold_proof: str | None = None


class Lockstep:
    """Advance only proved commands and save actual rather than predicted state."""

    def __init__(self, robot, nominal):
        self.robot = robot
        self.nominal = nominal
        self.profile = ServoProfile()
        self.physics = CpuServo(robot, 1, self.profile)
        self.safety = SafetyProjection(robot, device="cpu", max_step=[.027] * 14 + [.10] * 40,
                                       sweep=NativeSweep(robot, **PROOF))
        self.sweep = SweptEnvelopeV2(robot, **PROOF)
        self.issued = robot.home.copy()
        self.branch = nominal.reset(self.issued)
        self.fault = None

    def state(self):
        data = self.physics.data[0]
        return data.qpos[self.robot.q_indices].copy(), data.qvel[self.robot.v_indices].copy()

    def _proof(self, target, measured, *, engine=None):
        states = np.stack((self.issued, self.physics.filtered[0], measured, target))
        if not np.isfinite(states).all():
            return Reason.NONFINITE, {}
        low, high = states.min(axis=0), states.max(axis=0)
        if (np.any(low < self.robot.bounds[:, 0] - 1e-9)
                or np.any(high > self.robot.bounds[:, 1] + 1e-9)
                or np.any(np.maximum(self.robot.coupling * low, self.robot.coupling * high).sum(axis=1)
                          > COUPLING_LIMIT + 1e-9)):
            return Reason.LIMIT_OR_COUPLING, {}
        proof = self.sweep if engine is None else engine
        status = proof.prove(low, high, monotonic_ns() + PROOF_NS)
        return status, proof.last_metrics

    def advance(self, target, *, hand_axes=None, start_stage=False):
        if self.fault is not None:
            raise RuntimeError("No additional physical steps after an uncontained fault")
        measured, _ = self.state()
        proposal = self.nominal.propose(target, measured, self.issued, self.branch, hand_axes=hand_axes)
        if proposal.hard_fault:
            self.fault = "NATIVE_PAIR_HARD_FAULT"
            return Substep(proposal, "NATIVE_PAIR_HARD_FAULT", {}, None, None, True, self.fault)
        # CpuServo receives float32. Certify the exact vector it will receive.
        requested = proposal.q_requested.copy()
        if start_stage:
            previous = self.issued[14:]
            requested[14:] = np.clip(requested[14:], previous - START_FINGER_STEP_RAD,
                                     previous + START_FINGER_STEP_RAD)
        candidate = requested.astype(np.float32).astype(np.float64)
        proof, evidence = self._proof(candidate, measured)
        hold = target is None or proof != Reason.PERMIT
        hold_status = None
        if hold:
            hold_proof, _ = self._proof(self.issued, measured)
            hold_status = hold_proof.name
            if hold_proof != Reason.PERMIT:
                self.fault = f"HOLD_INFEASIBLE:{hold_proof.name}"
                return Substep(proposal, proof.name, evidence, None, None, True, self.fault, hold_status)
        else:
            self.issued = candidate
            self.branch = proposal.q_branch.copy()
        interval = self.physics.interval(torch.as_tensor(self.issued[None], dtype=torch.float32),
                                         torch.ones(1, dtype=torch.bool), self.safety, count=2)
        if interval.event[0] or interval.substeps[0] != 2:
            self.fault = f"PHYSICS:{int(interval.event[0])}"
        return Substep(proposal, proof.name, evidence, self.issued.copy(), interval, hold, self.fault, hold_status)

    def preposition(self, target, scoring_tcp, common, order, source, *, observe):
        """Advance frozen wrists, then digits, recording where dwell succeeds or fails."""
        self.nominal.refresh(target, common, order, source, tick_ns=0)
        enabled = np.zeros(40, dtype=bool)
        records = []
        for stage in range(11):
            if stage:
                finger = FINGER_ORDER[stage - 1]
                enabled[4 * finger:4 * finger + 4] = True
            dwell = 0
            maximum = 500 if stage == 0 else 175
            for tick in range(maximum):
                for _ in range(ARM_SUBSTEPS):
                    report = self.advance(target, hand_axes=enabled, start_stage=True)
                    observe(stage, tick, report, self.state())
                    if report.fault:
                        break
                q, qd = self.state()
                actual = self.robot.geometry(q)
                wrist_error = np.linalg.norm(actual.wrists - scoring_tcp[:, :3, 3], axis=1)
                wrist_angle = Rotation.from_matrix(
                    np.swapaxes(scoring_tcp[:, :3, :3], 1, 2) @ actual.rotations).magnitude()
                requested_hand = report.proposal.filtered_hand_q.reshape(-1)
                hand_ready = (np.max(abs(self.issued[14:][enabled] - requested_hand[enabled]), initial=0) <= .02
                              and np.max(abs(q[14:][enabled] - requested_hand[enabled]), initial=0) <= .10
                              and (not stage or report.proposal.raw_hand_valid[(FINGER_ORDER[stage - 1]) // 5]))
                ready = (not report.fault and not report.used_hold and hand_ready
                         and wrist_error.max() <= .01 and wrist_angle.max() <= np.deg2rad(5.)
                         and np.max(abs(qd[:14])) <= .10)
                dwell = dwell + 1 if ready else 0
                if dwell >= 5 or report.fault:
                    break
            records.append({"stage": stage, "elapsed_ticks": tick + 1, "dwell_ticks": dwell,
                            "held": report.used_hold, "proof": report.proof, "fault": report.fault,
                            "status": [OfflineNominalStatus(code).name for code in report.proposal.status],
                            "wrist_mm": (1000 * wrist_error).tolist(),
                            "wrist_deg": np.rad2deg(wrist_angle).tolist(),
                            "max_issued_finger_delta_rad":
                                float(np.max(abs(self.issued[14:][enabled] - requested_hand[enabled]), initial=0)),
                            "max_measured_finger_delta_rad":
                                float(np.max(abs(q[14:][enabled] - requested_hand[enabled]), initial=0)),
                            "sim_time_s": float(self.physics.data[0].time)})
            if dwell < 5:
                break
        return {"status": "START_REACHED" if len(records) == 11 and dwell >= 5 else "START_UNQUALIFIED",
                "stages": records}
