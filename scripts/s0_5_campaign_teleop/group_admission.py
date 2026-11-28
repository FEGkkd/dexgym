"""Four logical command groups, with a single simultaneous whole-body commit."""

from dataclasses import dataclass
from enum import IntEnum
from time import monotonic_ns

import numpy as np
import torch

from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineNominalStatus
from dexgym.tasks.direct.wholebody_s05.v1.rmb.teleop_candidate import ARM_SUBSTEPS, RECOVERY_TICKS
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import Interval
from dexgym.tasks.direct.wholebody_s05.v1.task.safety import Stop
from offline_rmb_lockstep import ACTIVE, Lockstep, Substep
from swept_live_guard import Reason


GROUPS = ("LEFT_ARM", "RIGHT_ARM", "LEFT_HAND", "RIGHT_HAND")
RECOVERY_SUBCYCLES = RECOVERY_TICKS * ARM_SUBSTEPS
RECOVERY_HAND_STEP_RAD = .015
FRACTIONS = (.5, .25)


class GroupStatus(IntEnum):
    ACTIVE = 1
    GROUP_HOLD = 2
    REJOINING = 3
    INPUT_HOLD = 4
    PHYSICS_FAULT = 5
    FEEDBACK_INVALID = 6
    NATIVE_FALLBACK = 7


def verified_groups(robot):
    index = {name: i for i, name in enumerate(robot.names)}
    result = tuple(np.asarray([index[name] for name in names]) for names in robot.groups)
    if (tuple(map(len, result)) != (7, 7, 20, 20)
            or not np.array_equal(np.concatenate(result), np.arange(54))
            or any(act.trnid[0] != joint.id for act, joint in zip(robot.actuators, robot.joints, strict=True))):
        raise ValueError("Signed model group names, transmissions or nominal ABI disagree")
    return result


@dataclass(frozen=True)
class Admission:
    combined: np.ndarray
    outcomes: tuple
    final: tuple
    hold: tuple | None
    issued: np.ndarray | None


def admit_groups(previous, requested, groups, eligible, recovering, prove):
    """Prove each trial and the final simultaneous command against one fixed snapshot."""
    combined = previous.copy()
    outcomes = []
    for group, indices in enumerate(groups):
        exact = requested[indices]
        delta = exact - previous[indices]
        capped = np.clip(delta, -RECOVERY_HAND_STEP_RAD, RECOVERY_HAND_STEP_RAD) if group >= 2 and recovering[group] else delta
        first = previous[indices] + capped
        trials = []
        chosen = None
        if eligible[group]:
            candidates = [first]
            if group >= 2:
                candidates.extend(previous[indices] + scale * (first - previous[indices]) for scale in FRACTIONS)
            for values in candidates:
                values = values.astype(np.float32).astype(np.float64)
                if trials and (np.array_equal(values, previous[indices])
                               or np.array_equal(values, trials[-1]["candidate"])):
                    continue
                candidate = combined.copy()
                candidate[indices] = values
                decision, metrics = prove(candidate)
                trials.append({"candidate": values.tolist(), "proof": decision.name, "metrics": metrics})
                if decision == Reason.PERMIT:
                    combined = candidate
                    chosen = "EXACT" if np.array_equal(values, exact) else "BOUNDED_REJOIN"
                    break
        outcomes.append({"group": GROUPS[group], "choice": chosen, "trials": trials,
                         "delta_max_rad": float(np.max(abs(combined[indices] - previous[indices])))})
    final = prove(combined)
    if final[0] == Reason.PERMIT:
        return Admission(combined, tuple(outcomes), final, None, combined)
    hold = prove(previous)
    return Admission(combined, tuple(outcomes), final, hold,
                     previous.copy() if hold[0] == Reason.PERMIT else None)


@dataclass(frozen=True)
class GroupedSubstep(Substep):
    groups: tuple = ()
    inputs: np.ndarray | None = None
    combined: np.ndarray | None = None
    physical_q: tuple = ()
    physical_qd: tuple = ()
    physical_steps: tuple = ()
    hold_metrics: dict | None = None


def combine_intervals(items):
    last = items[-1]
    return Interval(last.state, last.servo, torch.cat([i.applied for i in items], dim=1),
                    torch.cat([i.ctrl for i in items], dim=1), torch.cat([i.torque for i in items], dim=1),
                    torch.stack([i.contact for i in items]).amax(dim=0),
                    torch.stack([i.depth for i in items]).amax(dim=0),
                    torch.stack([i.minimum_gap for i in items]).amin(dim=0), sum(i.substeps for i in items),
                    torch.as_tensor([int(np.bitwise_or.reduce([int(i.event[0]) for i in items]))]))


class GroupedLockstep(Lockstep):
    """Keep signed START intact and opt into grouped admission for history only."""

    def __init__(self, robot, nominal):
        super().__init__(robot, nominal)
        self.indices = verified_groups(robot)
        self.recovery = np.full(4, RECOVERY_SUBCYCLES, dtype=np.int16)

    def _checked_proof(self, desired, measured):
        if not np.isfinite(desired).all():
            return Reason.NONFINITE, {"termination": "NONFINITE_PROPOSAL"}
        start = monotonic_ns()
        distances = self.sweep.native.pair_distances(desired)
        witness = int(np.argmin(distances))
        gap = float(distances[witness])
        if not np.isfinite(distances).all():
            decision, detail = Reason.INTERNAL, {"termination": "INVALID_ENDPOINT_GEOMETRY"}
        elif gap < self.robot.clearance_m:
            decision = Reason.COLLISION
            detail = {"termination": "PENETRATION" if gap < 0 else "CLEARANCE_VIOLATION",
                      "endpoint_gap_m": gap, "witness_pair_index": witness}
        else:
            decision, detail = self._proof(desired, measured)
        return decision, {**detail, "elapsed_ns": monotonic_ns() - start}

    def _physical_steps(self):
        samples, samples_v, results = [], [], []
        for _ in range(2):
            interval = self.physics.interval(torch.as_tensor(self.issued[None], dtype=torch.float32),
                                             torch.ones(1, dtype=torch.bool), self.safety, count=1)
            results.append(interval)
            if interval.substeps[0]:
                q, qd = self.state()
                samples.append(q)
                samples_v.append(qd)
            if interval.event[0] or interval.substeps[0] != 1:
                break
        return combine_intervals(results), tuple(samples), tuple(samples_v), tuple(results)

    def advance(self, target, *, hand_axes=None, start_stage=False):
        if start_stage:
            return super().advance(target, hand_axes=hand_axes, start_stage=True)
        if self.fault:
            raise RuntimeError("No physical step after fault")
        measured, qd = self.state()
        if not np.isfinite(np.stack((measured, qd, self.issued, self.physics.filtered[0]))).all():
            self.fault = "FEEDBACK_INVALID"
            return GroupedSubstep(None, self.fault, {}, None, None, True, self.fault)
        proposal = self.nominal.propose(target, measured, self.issued, self.branch)
        requested = proposal.q_requested.astype(np.float32).astype(np.float64)
        snapshot = np.stack((self.issued, self.physics.filtered[0], measured, requested))
        if proposal.hard_fault:
            self.fault = "NATIVE_PAIR_HARD_FAULT"
            return GroupedSubstep(proposal, self.fault, {}, None, None, True, self.fault, inputs=snapshot)
        eligible = np.isin(proposal.status, (*ACTIVE, OfflineNominalStatus.NATIVE_ARM_FALLBACK))
        admission = admit_groups(self.issued, requested, self.indices, eligible,
                                 self.recovery < RECOVERY_SUBCYCLES,
                                 lambda desired: self._checked_proof(desired, measured))
        final_reason, final_metrics = admission.final
        hold_reason, hold_metrics = admission.hold or (None, None)
        groups = admission.outcomes
        if admission.issued is None:
            self.fault = f"HOLD_INFEASIBLE:{hold_reason.name}"
            return GroupedSubstep(proposal, final_reason.name, final_metrics, None, None, True, self.fault,
                                  hold_reason.name, groups, snapshot, admission.combined, hold_metrics=hold_metrics)
        entire_hold = target is None or final_reason != Reason.PERMIT or not any(g["choice"] for g in groups)
        if entire_hold and final_reason == Reason.PERMIT:
            hold_reason, hold_metrics = final_reason, final_metrics
        self.issued = admission.issued.copy()
        if final_reason == Reason.PERMIT:
            for side in range(2):
                if groups[side]["choice"] == "EXACT":
                    self.branch[side] = proposal.q_branch[side]
        interval, positions, velocities, physical = self._physical_steps()
        if int(interval.event[0]) & Stop.BAD_FEEDBACK:
            self.fault = "FEEDBACK_INVALID"
        elif interval.event[0] or interval.substeps[0] != 2:
            self.fault = f"PHYSICS_FAULT:{int(interval.event[0])}"
        for i, entry in enumerate(groups):
            choice = entry["choice"] if final_reason == Reason.PERMIT else None
            exact_active = choice == "EXACT" and proposal.status[i] in ACTIVE
            if self.fault:
                status = GroupStatus.FEEDBACK_INVALID if self.fault == "FEEDBACK_INVALID" else GroupStatus.PHYSICS_FAULT
            elif target is None:
                status = GroupStatus.INPUT_HOLD
            elif choice is None:
                status = GroupStatus.GROUP_HOLD
            elif proposal.status[i] == OfflineNominalStatus.NATIVE_ARM_FALLBACK:
                status = GroupStatus.NATIVE_FALLBACK
            elif not exact_active or self.recovery[i] < RECOVERY_SUBCYCLES:
                status = GroupStatus.REJOINING
            else:
                status = GroupStatus.ACTIVE
            self.recovery[i] = min(self.recovery[i] + 1, RECOVERY_SUBCYCLES) if exact_active and not self.fault else 0
            entry["committed"] = choice
            entry["status"] = int(status)
            entry["label"] = f"{entry['group']}_ACTIVE" if status == GroupStatus.ACTIVE else status.name
            entry["rejoin_streak"] = int(self.recovery[i])
        return GroupedSubstep(proposal, final_reason.name, final_metrics, self.issued.copy(), interval,
                              entire_hold, self.fault, hold_reason.name if hold_reason is not None else None,
                              groups, snapshot, admission.combined, positions, velocities, physical, hold_metrics)
