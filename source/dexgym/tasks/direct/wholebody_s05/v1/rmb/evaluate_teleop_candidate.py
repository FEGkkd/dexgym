"""Paired, evidence-only DCR-008 evaluation of an offline Teleop nominal candidate."""

import argparse
from collections import Counter
from enum import IntEnum
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

from ...hmb_reader import HMBReader, SIMULATED, VERSION_FIELDS, eligible_history
from ...p02.model import RobotModel
from ..task.native_sweep import NativeSweep
from .audit_nominal_workspace import signed_inputs, wrist_scope
from .build import REGRESSION_SHA, scope_hashes
from .candidate_improvement import evaluate_candidate
from .mapping import IntentMapper, compiled_topologies
from .nominal import GROUPS, Wuji, matrix, model_wrist
from .reader import RobotReferenceBank
from .schema import GEOMETRY_SHAPES, SPLITS, TICK_NS, file_sha, write_json
from .teleop_candidate import ARM_SUBSTEP_NS, ARM_SUBSTEPS, CandidateStatus, TeleopCandidate


SCOPE = "DCR_008_CANDIDATE_ONLY"
QUANTIFICATION = "STATIC_SIGNED_MUJOCO_GEOMETRY_WITH_NO_SIMULATOR_ROLLOUT"
EXACT_M = EXACT_JOINT_RAD = 1e-6
EXACT_ROT_RAD = 1e-5
MINIMUM_COVERAGE = .85
EXPECTED = {"Train": (24, 228, 86_338, 85_382), "Validation": (8, 80, 30_275, 29_150)}
FRAME_NAMES = ("workspace_target", "step_target", "attempt_target")
NATIVE_FLAGS = (
    "accepted", "ik_target_accepted", "analytic_braking", "workspace_projected", "rate_limited",
    "analytic_exact_stop_used", "pair_accepted", "pair_hard_fault",
)


class Classification(IntEnum):
    ACTIVE_TRACKED = 1
    LIMITED_ACTIVE = 2
    HOLD = 3
    FALLBACK = 4
    FAILED = 5


def classify(accepted, changed, exact, limited, fallback):
    """An unchanged, unsettled group cannot earn active-tracking credit."""
    result = np.full(4, Classification.HOLD, dtype=np.uint8)
    result[accepted & exact & ~limited] = Classification.ACTIVE_TRACKED
    result[accepted & (changed | exact) & limited] = Classification.LIMITED_ACTIVE
    result[fallback] = Classification.FALLBACK
    return result


def pose_error(actual, desired):
    distance = np.linalg.norm(actual[:3, 3] - desired[:3, 3])
    angle = Rotation.from_matrix(actual[:3, :3].T @ desired[:3, :3]).magnitude()
    return float(distance), float(angle)


def native_frames(robot, kernel):
    """Derive the signed model-to-native TCP bridge without loading a second SDK handle."""
    home = robot.geometry(robot.home)
    frames = []
    witnesses = []
    for side, body in enumerate(("Base_L", "Base_R")):
        arm = GROUPS[side]
        mark = robot.data.body(body)
        base = matrix(mark.xpos, mark.xmat.reshape(3, 3))
        tool = np.linalg.solve(model_wrist(home, side), base @ kernel.fk(side, robot.home[arm]))
        frames.append((base, tool))
        errors = []
        for axis in range(arm.start, arm.stop):
            for pulse in (-.04, .04):
                q = robot.home.copy()
                q[axis] += pulse
                actual = kernel.fk(side, q[arm])
                expected = np.linalg.solve(base, model_wrist(robot.geometry(q), side) @ tool)
                errors.append(pose_error(actual, expected))
        maximum = np.asarray(errors).max(axis=0).tolist()
        if maximum[0] > .0015 or maximum[1] > np.deg2rad(.5):
            raise ValueError(f"Native/signed-model pose bridge failed on arm {side}: {maximum}")
        witnesses.append(maximum)
    return frames, witnesses


def authority_arrays(raw, count):
    return {
        "tick_ns": raw["tick_ns"][:count].copy(),
        "cutoff_ns": raw["cutoff_ns"][:count].copy(),
        "source_index": raw["source_index"][:count, :4].copy(),
        "capture_bounds_ns": raw["capture_bounds_ns"][:count, :4].copy(),
        "arrival_bounds_ns": raw["arrival_bounds_ns"][:count, :4].copy(),
        "age_ns": raw["age_ns"][:count, :4].copy(),
        "input_reason_code": raw["reason_code"][:count, :4].copy(),
        "segment": raw["segment_id"][:count].copy(),
        "hmb_joint_valid": raw["joint_valid"][:count].copy(),
        "hmb_five_frame": np.asarray([eligible_history(raw, i) for i in range(count)], dtype=bool),
        **{name: raw[name][:count, :4].copy() for name in VERSION_FIELDS},
    }


def output_arrays(raw, count):
    result = authority_arrays(raw, count)
    result.update(
        generation=np.full(count, -1, dtype=np.int32),
        mapping_valid=np.zeros(count, dtype=bool),
        pinch_active=np.zeros((count, 2, 4), dtype=bool),
        provider_observed=np.zeros(count, dtype=bool),
        q_nominal=np.zeros((count, 54)),
        qd_nominal=np.zeros((count, 54)),
        qd_valid=np.zeros(count, dtype=bool),
        provider_qd_nominal=np.zeros((count, 54)),
        provider_qd_valid=np.zeros(count, dtype=bool),
        q_changed=np.zeros((count, 54), dtype=bool),
        accepted=np.zeros((count, 4), dtype=bool),
        held=np.zeros((count, 4), dtype=bool),
        fallback=np.zeros((count, 4), dtype=bool),
        group_class=np.full((count, 4), Classification.FAILED, dtype=np.uint8),
        row_class=np.full(count, Classification.FAILED, dtype=np.uint8),
        sdk_status=np.zeros((count, 4), dtype=np.uint16),
        cause=np.full(count, "INPUT_NOT_ADMITTED", dtype="<U48"),
        path_accepted=np.zeros((count, ARM_SUBSTEPS), dtype=bool),
        issue_ns=np.full((count, ARM_SUBSTEPS), -1, dtype=np.int64),
        q_issued=np.zeros((count, ARM_SUBSTEPS, 54)),
        branch_rad=np.zeros((count, ARM_SUBSTEPS, 2)),
        substep_status=np.zeros((count, ARM_SUBSTEPS, 4), dtype=np.uint16),
        kernel_proposed=np.zeros((count, ARM_SUBSTEPS), dtype=bool),
        native_flags=np.zeros((count, ARM_SUBSTEPS, 2, len(NATIVE_FLAGS)), dtype=bool),
        native_code=np.zeros((count, ARM_SUBSTEPS, 2), dtype=np.int32),
        native_reason=np.zeros((count, ARM_SUBSTEPS, 2), dtype="<U256"),
        raw_ik_proposal=np.zeros((count, ARM_SUBSTEPS, 2, 7)),
        native_joint_command=np.zeros((count, ARM_SUBSTEPS, 2, 7)),
        native_tcp_target=np.zeros((count, 2, 4, 4)),
        projected_axis=np.full((count, ARM_SUBSTEPS, 2), -1, dtype=np.int8),
        backoff_scale=np.ones((count, ARM_SUBSTEPS, 2)),
        limit_applied=np.zeros((count, 54), dtype=bool),
        native_rate_limit_proposed=np.zeros((count, ARM_SUBSTEPS, 2), dtype=bool),
        native_rate_limit_applied=np.zeros((count, ARM_SUBSTEPS, 2), dtype=bool),
        hand_limit_proposed=np.zeros((count, ARM_SUBSTEPS, 40), dtype=bool),
        hand_limit_applied=np.zeros((count, ARM_SUBSTEPS, 40), dtype=bool),
        hand_raw_rad=np.zeros((count, 2, 20)),
        hand_request_rad=np.zeros((count, 2, 20)),
        hand_raw_valid=np.zeros((count, 2), dtype=bool),
        hand_reason=np.zeros((count, 2), dtype="<U256"),
        original_arm_pose_error=np.zeros((count, 2, 2)),
        issued_tcp_from_signed_fk=np.zeros((count, 2, 4, 4)),
        score_valid=np.zeros(count, dtype=bool),
        nominal_error_m=np.zeros(count),
        candidate_five_frame=np.zeros(count, dtype=bool),
        direct_observed=np.zeros(count, dtype=bool),
        direct_arm_sdk_valid=np.zeros((count, 2), dtype=bool),
        direct_arm_exact=np.zeros((count, 2), dtype=bool),
        direct_arm_error=np.zeros((count, 2, 2)),
        direct_nominal_legal=np.zeros(count, dtype=bool),
        direct_nominal_error_m=np.zeros(count),
        witness_sampled=np.zeros(count, dtype=bool),
        witness_ok=np.zeros(count, dtype=bool),
        witness_reason=np.full(count, "NOT_SAMPLED", dtype="<U64"),
        witness_q=np.zeros((count, 54)),
        witness_error_m=np.zeros(count),
        witness_error_available=np.zeros(count, dtype=bool),
    )
    for name in FRAME_NAMES:
        result[name] = np.zeros((count, ARM_SUBSTEPS, 2, 4, 4))
    for name, shape in GEOMETRY_SHAPES.items():
        result[f"mapped_{name}"] = np.zeros((count, *shape), dtype=np.float32)
    return result


def record_trace(fields, i, candidate):
    fields["provider_observed"][i] = True
    fields["q_nominal"][i] = candidate.q_nominal
    fields["provider_qd_nominal"][i] = candidate.qd_nominal
    fields["provider_qd_valid"][i] = candidate.qd_valid
    fields["sdk_status"][i] = candidate.status
    fields["hand_raw_rad"][i] = candidate.raw_hand_q_rad
    fields["hand_raw_valid"][i] = candidate.raw_hand_valid
    fields["hand_reason"][i] = [reason or "" for reason in candidate.hand_failure_reason]
    fields["native_tcp_target"][i] = candidate.native_tcp_target
    if len(candidate.substeps) != ARM_SUBSTEPS:
        raise ValueError("Nominal candidate did not provide every contracted substep")
    for j, step in enumerate(candidate.substeps):
        if step.issue_tick_ns != int(fields["tick_ns"][i]) + j * ARM_SUBSTEP_NS:
            raise ValueError("Nominal candidate moved substeps off the causal tick clock")
        fields["issue_ns"][i, j] = step.issue_tick_ns
        fields["q_issued"][i, j] = step.q_issued
        fields["branch_rad"][i, j] = step.arm_branches_rad
        fields["substep_status"][i, j] = step.status
        fields["path_accepted"][i, j] = step.whole_body_path_accepted
        fields["kernel_proposed"][i, j] = step.kernel_proposed
        fields["hand_limit_proposed"][i, j] = step.hand_candidate_limit
        fields["hand_limit_applied"][i, j] = step.hand_limit_applied
        fields["limit_applied"][i, 14:] |= step.hand_limit_applied
        for side, arm in enumerate(step.native_arms):
            fields["native_flags"][i, j, side] = [bool(getattr(arm, name)) for name in NATIVE_FLAGS]
            fields["native_code"][i, j, side] = arm.safety_code
            fields["native_reason"][i, j, side] = arm.safety_reason
            fields["raw_ik_proposal"][i, j, side] = arm.joint_candidate
            fields["native_joint_command"][i, j, side] = arm.joint_command
            fields["projected_axis"][i, j, side] = arm.projected_axis
            fields["backoff_scale"][i, j, side] = arm.backoff_scale
            for name in FRAME_NAMES:
                fields[name][i, j, side] = getattr(arm, name)
            rate_limited = arm.rate_limited
            limited = arm.workspace_projected or arm.backoff_scale < 1 or arm.projected_axis >= 0
            if arm.accepted and arm.ik_target_accepted:
                change = pose_error(arm.step_target, arm.workspace_target)
                rate_limited |= change[0] > EXACT_M or change[1] > EXACT_ROT_RAD
            admitted = (step.kernel_proposed and step.whole_body_path_accepted
                        and step.status[side] == CandidateStatus.ACTIVE)
            fields["native_rate_limit_proposed"][i, j, side] = step.kernel_proposed and rate_limited
            fields["native_rate_limit_applied"][i, j, side] = admitted and rate_limited
            if admitted and (limited or rate_limited):
                fields["limit_applied"][i, GROUPS[side]] = True


def direct_identity(part, raw, denied):
    arrays = part.arrays
    for key, data in (
        ("tick_ns", raw["tick_ns"]), ("inputs_pose_joint_valid", raw["joint_valid"]),
        ("audit_hmb_source_index", raw["source_index"][:, :4, 0]), ("audit_segment", raw["segment_id"]),
    ):
        if not np.array_equal(arrays[key], data):
            raise ValueError(f"Direct archive and signed HMB differ in {part.key}: {key}")
    if denied and arrays["domain_mapped_valid"].any():
        raise ValueError(f"Direct archive mapped an excluded lineage in {part.key}")


def independent_exact(fields, i, direct, robot, target):
    for name in GEOMETRY_SHAPES:
        if not np.array_equal(fields[f"mapped_{name}"][i], direct[f"mapped_{name}"][i]):
            raise ValueError(f"Independent hard M1 changed on tick {i}: {name}")
    if not np.array_equal(fields["pinch_active"][i], direct["mapped_pinch_active"][i]):
        raise ValueError(f"Immutable M1 pinch mask changed on tick {i}")
    fields["direct_observed"][i] = True
    sdk_valid = direct["nominal_raw_ik_valid"][i]
    fields["direct_arm_sdk_valid"][i] = sdk_valid
    for side in np.flatnonzero(sdk_valid):
        q = robot.home.copy()
        q[GROUPS[side]] = direct["nominal_raw_ik_arm"][i, side]
        desired = matrix(target["wrist_xyz"][side], Rotation.from_quat(target["wrist_xyzw"][side]).as_matrix())
        distance, angle = pose_error(model_wrist(robot.geometry(q), side), desired)
        fields["direct_arm_error"][i, side] = (distance, angle)
        fields["direct_arm_exact"][i, side] = distance <= EXACT_M and angle <= EXACT_ROT_RAD
    if direct["domain_nominal_legal"][i]:
        fields["direct_nominal_legal"][i] = True
        delta = robot.geometry(direct["nominal_q_nominal"][i]).tips - target["world_tip_xyz"]
        fields["direct_nominal_error_m"][i] = np.linalg.norm(delta, axis=-1).max()


def score_nominal(fields, i, robot, frames, target, state, previous, requested_hand):
    q = state.q_nominal
    geometry = robot.geometry(q)
    fields["score_valid"][i] = True
    fields["nominal_error_m"][i] = np.linalg.norm(geometry.tips - target["world_tip_xyz"], axis=-1).max()
    fields["hand_request_rad"][i] = requested_hand
    fields["q_changed"][i] = np.abs(q - previous) > EXACT_JOINT_RAD
    limited = np.asarray([fields["limit_applied"][i, group].any() for group in GROUPS])
    changed = np.asarray([fields["q_changed"][i, group].any() for group in GROUPS])
    exact = np.zeros(4, dtype=bool)
    for side, (base, tool) in enumerate(frames):
        desired = matrix(target["wrist_xyz"][side], Rotation.from_quat(target["wrist_xyzw"][side]).as_matrix())
        wrist = model_wrist(geometry, side)
        fields["issued_tcp_from_signed_fk"][i, side] = np.linalg.solve(base, wrist @ tool)
        distance, angle = pose_error(wrist, desired)
        fields["original_arm_pose_error"][i, side] = (distance, angle)
        exact[side] = distance <= EXACT_M and angle <= EXACT_ROT_RAD
        exact[side + 2] = np.max(np.abs(q[GROUPS[side + 2]] - requested_hand[side])) <= EXACT_JOINT_RAD
    accepted = (state.status == CandidateStatus.ACTIVE) & fields["path_accepted"][i].all()
    # Mark only a protected replacement actually accepted by the scheduler.
    # Its per-substep native braking proposal is separately retained.
    fallback = ((state.status == CandidateStatus.NATIVE_ARM_FALLBACK)
                | (fields["substep_status"][i] == CandidateStatus.NATIVE_ARM_FALLBACK).any(axis=0))
    categories = classify(accepted, changed, exact, limited, fallback)
    failed = ((state.status == CandidateStatus.NATIVE_PAIR_FAILED)
              | (fields["substep_status"][i] == CandidateStatus.NATIVE_PAIR_FAILED).any(axis=0))
    categories[failed] = Classification.FAILED
    fields["accepted"][i] = accepted
    fields["held"][i] = categories == Classification.HOLD
    fields["fallback"][i] = fallback
    fields["group_class"][i] = categories
    fields["row_class"][i] = categories.max()
    if fields["row_class"][i] <= Classification.LIMITED_ACTIVE:
        fields["cause"][i] = ""
    elif np.any(state.status != CandidateStatus.ACTIVE):
        fields["cause"][i] = CandidateStatus(int(state.status.max())).name
    else:
        fields["cause"][i] = "UNSETTLED_OR_NO_TRACKING_PROOF"


def causal_token(raw, i):
    return (int(raw["segment_id"][i]), *(tuple(raw[name][i, :4].tolist()) for name in VERSION_FIELDS))


def witness_offset(key, stride, length):
    # Sample independently of outputs, including partitions shorter than a stride.
    return (int.from_bytes(sha256(f"49:{key}".encode()).digest()[:8], "little") % min(stride, length)
            if stride and length else -1)


def evaluate_partition(raw, meta, mapper, candidate, key, *, denied, direct=None, limit=None, witness_stride=0):
    count = len(raw["tick_ns"]) if limit is None else min(limit, len(raw["tick_ns"]))
    fields = output_arrays(raw, count)
    if not np.array_equal(fields["tick_ns"], fields["cutoff_ns"]):
        raise ValueError("Signed HMB information cutoff must equal the proxy issue clock")
    if direct is not None:
        direct_identity(direct, raw, denied)
    supported = wrist_scope(raw, denied)
    generation, previous_token, previous_tick = -1, None, None
    offset = witness_offset(key, witness_stride, count)
    if offset >= 0:
        fields["witness_sampled"][offset::witness_stride] = True
        fields["witness_reason"][offset::witness_stride] = "NO_ADMITTED_M1"
    order_cache = {}
    for i in range(count):
        if not supported[i]:
            fields["cause"][i] = "DENIED_LINEAGE" if denied else "INPUT_NOT_ADMITTED"
            previous_token = None
            continue
        topo = tuple(int(x) for x in raw["topology_id"][i, :2])
        try:
            if topo not in order_cache:
                order_cache[topo] = compiled_topologies(meta, topo)
            map_order, sdk_order = order_cache[topo]
            mapped = mapper.map(raw["pose"][i, 2:4, 0], raw["pose"][i, :2], map_order)
        except (ValueError, KeyError, IndexError) as error:
            fields["cause"][i] = f"INVALID_MAPPING_{type(error).__name__}"
            previous_token = None
            continue
        target, active = mapped["mapped"], mapped["pinch_active"]
        for name, values in target.items():
            fields["mapped_" + name][i] = values
        fields["pinch_active"][i] = active
        fields["mapping_valid"][i] = True
        if direct is not None:
            independent_exact(fields, i, direct.arrays, candidate.robot, target)
        tick = int(fields["tick_ns"][i])
        token = causal_token(raw, i)
        if token != previous_token or tick != previous_tick + TICK_NS:
            candidate.cut()
            generation += 1
        fields["generation"][i] = generation
        previous = candidate.previous.copy()
        state = candidate.step(target, raw["sdk_common_pose"][i, :2], sdk_order,
                               fields["source_index"][i, :, 0], tick_ns=tick)
        record_trace(fields, i, state)
        score_nominal(fields, i, candidate.robot, candidate.wrist_frames, target, state, previous,
                      candidate.filtered_hand.copy())
        if (i > 0 and fields["generation"][i - 1] == generation
                and np.all(fields["row_class"][i - 1:i + 1] <= Classification.LIMITED_ACTIVE)):
            fields["qd_nominal"][i] = (fields["q_nominal"][i] - fields["q_nominal"][i - 1]) / (TICK_NS / 1e9)
            fields["qd_valid"][i] = True
        previous_token, previous_tick = token, tick
        last = slice(i - 4, i + 1)
        if (fields["hmb_five_frame"][i] and np.all(fields["generation"][last] == generation)
                and np.all(fields["row_class"][last] <= Classification.LIMITED_ACTIVE)):
            fields["candidate_five_frame"][i] = True
        if fields["witness_sampled"][i]:
            if fields["row_class"][i] > Classification.LIMITED_ACTIVE:
                fields["witness_reason"][i] = "NO_ACTIVE_NOMINAL"
                continue
            witness = evaluate_candidate(candidate.robot, state.q_nominal, target, active,
                                         sweep=candidate.swept_proof, search_steps=8)
            fields["witness_reason"][i] = witness["reason"] or witness["status"]
            if witness.get("trial_max_ten_tip_error_m") is not None:
                fields["witness_error_available"][i] = True
                fields["witness_error_m"][i] = witness["trial_max_ten_tip_error_m"]
            if witness["status"] == "WITNESSED":
                fields["witness_ok"][i] = True
                fields["witness_q"][i] = witness["witness_q_rad"]
    if direct is not None and not np.array_equal(fields["mapping_valid"], direct.arrays["domain_mapped_valid"][:count]):
        raise ValueError(f"Direct and candidate mapping-availability masks differ in {key}")
    return fields


def prefix_candidates(raw, key, denied):
    count = len(raw["tick_ns"])
    usable = np.flatnonzero(wrist_scope(raw, denied))
    usable = usable[usable + 3 < count]
    if not len(usable):
        return {}
    ordinary = usable[:max(1, min(12, len(usable)))]
    base = int(ordinary[witness_offset(key, 12, len(ordinary))]) + 3
    choices = {"ordinary": base}
    for old, new in zip(usable[:-1], usable[1:], strict=True):
        if new != old + 1 or causal_token(raw, old) != causal_token(raw, new):
            choices["generation_barrier"] = int(new) + 3
            break
    return choices


def compare_prefix(raw, meta, mapper, candidate, key, denied, fields, size):
    truncated = {name: values[:size] for name, values in raw.items()}
    replaced = dict(raw)
    for name in ("pose", "sdk_common_pose", "source_index", "capture_bounds_ns", "arrival_bounds_ns",
                 "age_ns", "segment_id", "pose_valid", "joint_valid", "safety_permission", *VERSION_FIELDS):
        changed = raw[name].copy()
        changed[size:] = -1 if changed.dtype.kind == "i" else 0
        replaced[name] = changed
    deleted_run = evaluate_partition(truncated, meta, mapper, candidate, key, denied=denied)
    replaced_run = evaluate_partition(replaced, meta, mapper, candidate, key, denied=denied, limit=size)
    compared = [name for name in fields if not name.startswith(("direct_", "witness_"))]
    comparison = {
        "partition": key, "prefix_ticks": size, "compared_fields": len(compared),
        "deletion_differences": [
            name for name in compared if fields[name][:size].tobytes() != deleted_run[name].tobytes()
        ],
        "replacement_differences": [
            name for name in compared if fields[name][:size].tobytes() != replaced_run[name].tobytes()
        ],
    }
    if comparison["deletion_differences"] or comparison["replacement_differences"]:
        raise ValueError(f"Candidate prefix comparisons failed: {comparison}")
    return comparison


def count_evidence(fields):
    hmb = fields["hmb_joint_valid"]
    candidate = fields["row_class"]
    observed = fields["direct_observed"]
    active = candidate <= Classification.LIMITED_ACTIVE
    exact = fields["direct_arm_exact"].all(axis=1)
    counters = Counter(
        planned=len(hmb), hmb_valid=int(hmb.sum()), mapped=int(fields["mapping_valid"].sum()),
        candidate_observed=int(fields["provider_observed"].sum()), direct_mapped_observed=int(observed.sum()),
        direct_lr_sdk=int(fields["direct_arm_sdk_valid"].all(axis=1).sum()), direct_lr_exact=int(exact.sum()),
        direct_nominal_legal=int(fields["direct_nominal_legal"].sum()),
        hmb_five_frame=int(fields["hmb_five_frame"].sum()),
        candidate_five_frame=int(fields["candidate_five_frame"].sum()),
        candidate_active=int(active.sum()), candidate_active_hmb=int((active & hmb).sum()),
        joint_limit_reported=int(fields["limit_applied"].any(axis=1).sum()),
        native_rate_limit_proposed=int(fields["native_rate_limit_proposed"].any(axis=(1, 2)).sum()),
        native_rate_limit_applied=int(fields["native_rate_limit_applied"].any(axis=(1, 2)).sum()),
        hand_slew_or_joint_limit_applied=int(fields["hand_limit_applied"].any(axis=(1, 2)).sum()),
        braking_proposed=int((fields["native_flags"][..., NATIVE_FLAGS.index("analytic_braking")]
                              & fields["kernel_proposed"][..., None]).any(axis=(1, 2)).sum()),
        exact_stop_proposed=int((fields["native_flags"][..., NATIVE_FLAGS.index("analytic_exact_stop_used")]
                                & fields["kernel_proposed"][..., None]).any(axis=(1, 2)).sum()),
        witness_selected=int(fields["witness_sampled"].sum()), witness_proven=int(fields["witness_ok"].sum()),
        static_5mm_witness=int((fields["witness_ok"] & (fields["witness_error_m"] <= .005)).sum()),
    )
    for category in Classification:
        mask = candidate == category
        counters[category.name] = int(mask.sum())
        counters[category.name + "_HMB_VALID"] = int((mask & hmb).sum())
        for index, role in enumerate(("left_arm", "right_arm", "left_hand", "right_hand")):
            counters[f"{role}_{category.name}"] = int((fields["group_class"][:, index] == category).sum())
    for side, label in enumerate(("left", "right")):
        counters[f"direct_{label}_sdk"] = int(fields["direct_arm_sdk_valid"][:, side].sum())
        counters[f"direct_{label}_exact"] = int(fields["direct_arm_exact"][:, side].sum())
    for reason in fields["cause"]:
        if reason:
            counters["cause:" + str(reason)] += 1
    for reason in fields["witness_reason"][fields["witness_sampled"]]:
        counters["witness:" + str(reason)] += 1
    if sum(counters[c.name] for c in Classification) != counters["planned"]:
        raise ValueError("Outcome partition violates planned denominator conservation")
    return counters


def summarize(counters, *, paired):
    hmb, planned = counters["hmb_valid"], counters["planned"]
    active = counters["candidate_active_hmb"] / hmb if hmb else None
    return {
        "counts": dict(counters),
        "rates_per_planned": {
            key.name: counters[key.name] / planned if planned else None for key in Classification
        },
        "rates_per_hmb_valid": {
            key.name: counters[key.name + "_HMB_VALID"] / hmb if hmb else None for key in Classification
        },
        "direct_nominal_legal_coverage_hmb": counters["direct_nominal_legal"] / hmb if paired and hmb else None,
        "direct_exact_wrist_coverage_hmb": counters["direct_lr_exact"] / hmb if paired and hmb else None,
        "candidate_active_coverage_hmb": active,
        "candidate_five_frame_coverage": (counters["candidate_five_frame"] / counters["hmb_five_frame"]
                                          if counters["hmb_five_frame"] else None),
        "shadow_85_percent_candidate_active": active >= MINIMUM_COVERAGE if active is not None else None,
        "official_coverage_pass": False,
    }


def quantiles(values):
    if not len(values):
        return {"n": 0, "p50_m": None, "p95_m": None, "p99_m": None}
    p50, p95, p99 = np.percentile(values, (50, 95, 99)).tolist()
    return {"n": len(values), "p50_m": p50, "p95_m": p95, "p99_m": p99}


def validate_bank(bank, identity):
    old = bank.manifest["identity"]
    expected = {
        "HMB_E1": identity["HMB_E1"], "HMB_CONFIG": identity["HMB_CONFIG"], "HMB_HANDOFF": identity["HMB_HANDOFF"],
        "M1": identity["M1"], "M0": identity["M0_CALIBRATION"],
    }
    if any(old.get(name) != value for name, value in expected.items()):
        raise ValueError("Direct reference bank does not use the same signed inputs")
    if bank.manifest["approved_for_training"] is not False:
        raise ValueError("Expected a separately retained evidence-only original nominal comparator")
    if any(not any(Path(name).name == basename and actual == digest for name, actual in old.items())
           for basename, digest in (("dual_arm_marvin.xml", identity["model"]),
                                    ("ccs_m6_40.MvKDCfg", identity["marvin_config"]),
                                    ("libwuji_sdk_c.so", identity["wuji_sdk"]))):
        raise ValueError("Direct reference bank uses a different signed model, Marvin config or Wuji SDK")


def open_reader(args, split):
    evidence = args.docs / "evidence"
    return HMBReader(evidence / "P01_E_E1/hmb_manifest.json", evidence / "P01_E_E1/production_plan.json",
                     evidence / "P01_E_E0/candidate_config.json", args.products, split, SIMULATED)


def compare_split(reader, args, split, mapper, candidate, bank, prefix_receipt):
    aggregate, per_root = Counter(), {}
    partitions = []
    fresh_errors, exact_errors, shared_fresh_errors, shared_exact_errors = [], [], [], []
    roots = set()
    for ordinal, key in enumerate(reader.keys()):
        if args.max_partitions is not None and ordinal >= args.max_partitions:
            break
        raw, meta, refs, _ = reader.read(key)
        root = meta["identity"]["root_episode_id"]
        roots.add(root)
        sources = scope_hashes(refs)
        denied = not sources.isdisjoint(REGRESSION_SHA)
        direct = bank.read(key, split) if bank else None
        if direct is not None and (direct.root != root or
                set(bank.index[key]["source_asset_sha256"]) != sources):
            raise ValueError(f"Direct reference lineage differs in {key}")
        fields = evaluate_partition(raw, meta, mapper, candidate, key, denied=denied, direct=direct,
                                    limit=args.max_ticks, witness_stride=args.witness_stride)
        checks = prefix_receipt.setdefault(root, {})
        if args.prefix_audit:
            for name, size in prefix_candidates(raw, key, denied).items():
                if name not in checks and size < len(fields["tick_ns"]):
                    checks[name] = compare_prefix(raw, meta, mapper, candidate, key, denied, fields, size)

        count = count_evidence(fields)
        aggregate.update(count)
        per_root.setdefault(root, Counter()).update(count)
        active = fields["row_class"] <= Classification.LIMITED_ACTIVE
        exact = fields["direct_nominal_legal"]
        fresh_scores = fields["nominal_error_m"]
        old_scores = fields["direct_nominal_error_m"]
        fresh_errors.extend(fresh_scores[active].tolist())
        exact_errors.extend(old_scores[exact].tolist())
        shared_fresh_errors.extend(fresh_scores[active & exact].tolist())
        shared_exact_errors.extend(old_scores[active & exact].tolist())

        folder = args.out / split
        folder.mkdir(parents=True, exist_ok=True)
        trace = folder / f"{ordinal:04d}.npz"
        if any(value.dtype.kind == "f" and not np.isfinite(value).all() for value in fields.values()):
            raise ValueError(f"Unmasked non-finite telemetry in {key}")
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **fields)
        evidence = {"partition": key, "root": root, "trace": trace.relative_to(args.out).as_posix(),
                    "trace_sha256": file_sha(trace), "source_asset_sha256": sorted(sources),
                    **summarize(count, paired=bool(bank))}
        write_json(trace.with_suffix(".json"), evidence)
        partitions.append(evidence)
        print(f"{split} partition {ordinal + 1}/{len(reader.keys())} {root} planned={count['planned']} "
              f"candidate_active={count['candidate_active']} direct_legal={count['direct_nominal_legal']}", flush=True)
    population = (len(roots), len(partitions), aggregate["planned"], aggregate["hmb_valid"])
    full = args.max_partitions is None and args.max_ticks is None
    if full and population != EXPECTED[split]:
        raise ValueError(f"{split} differs from signed denominators: {population} vs {EXPECTED[split]}")
    return {
        "full_signed_split": full, "root_count": len(roots), "partition_count": len(partitions),
        **summarize(aggregate, paired=bool(bank)),
        "scores": {
            "candidate_active": quantiles(fresh_errors), "direct_nominal_legal": quantiles(exact_errors),
            "candidate_active_on_jointly_legal_rows": quantiles(shared_fresh_errors),
            "direct_on_jointly_legal_rows": quantiles(shared_exact_errors),
        },
        "per_root": {root: summarize(count, paired=bool(bank)) for root, count in per_root.items()},
        "partitions": partitions,
    }


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("products", "model", "out", "teleop-bridge", "phase1-repo", "marvin-config", "marvin-vendor", "wuji-sdk"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--teleop-bridge-sha256", required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    parser.add_argument("--direct-bank", type=Path)
    parser.add_argument("--direct-bank-sha256")
    parser.add_argument("--split", choices=("Train", "Both"), default="Train")
    parser.add_argument("--max-partitions", type=int, help="Bounded Train smoke only")
    parser.add_argument("--max-ticks", type=int, help="Bounded Train smoke limit, per partition")
    parser.add_argument("--witness-stride", type=int, default=64, help="Seed-49 periodic signed rows per partition")
    parser.add_argument("--skip-prefix-audit", action="store_false", dest="prefix_audit")
    args = parser.parse_args()
    bounded = args.max_partitions is not None or args.max_ticks is not None
    if any(value is not None and value <= 0 for value in (args.max_partitions, args.max_ticks)):
        parser.error("Smoke limits must be positive")
    if (args.direct_bank is None) != (args.direct_bank_sha256 is None):
        parser.error("The exact reference bank and its manifest SHA must be supplied together")
    if bounded and args.split != "Train":
        parser.error("Bounded diagnostic tuning can only read Train")
    if not bounded and args.direct_bank is None:
        parser.error("Full results require the preserved original exact comparator")
    if args.witness_stride < 0 or (not bounded and args.witness_stride == 0):
        parser.error("Full results require a positive fixed witness sampling stride")
    if not bounded and not args.prefix_audit:
        parser.error("Full results require prefix controls")
    return args


def main():
    args = cli()
    from .teleop_native import TeleopNative

    started = monotonic()
    # Reuse the established asset/HMB identity verifier for the new native bridge.
    args.marvin_bridge = args.teleop_bridge
    args.marvin_bridge_sha256 = args.teleop_bridge_sha256
    reader, fit, calib, identity = signed_inputs(args)
    identity["teleop_core_bridge"] = identity.pop("marvin_bridge")
    identity["wuji_sdk"] = file_sha(args.wuji_sdk)
    identity["evaluation_source"] = file_sha(Path(__file__))
    identity["scheduler_source"] = file_sha(Path(__file__).with_name("teleop_candidate.py"))
    identity["review_protocol"] = file_sha(args.docs / "campaign/nominal_candidate_independent_protocol.md")
    bank = None
    if args.direct_bank:
        identity["direct_bank"] = args.direct_bank_sha256
        bank = RobotReferenceBank(args.direct_bank, expected_sha=identity["direct_bank"], verify=True,
                                  require_qualified=False)
        validate_bank(bank, identity)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "candidate_inputs.json", {
        "scope": SCOPE, "identity_sha256": identity, "split": args.split,
        "witness_selection_seed": 49, "witness_stride": args.witness_stride,
        "max_partitions": args.max_partitions, "max_ticks": args.max_ticks,
    })
    robot = RobotModel(args.model)
    mapper = IntentMapper(robot.geometry(robot.home), calib, fit)
    sweep = NativeSweep(robot)
    prefix_receipt = {}
    summaries = {}
    with TeleopNative(args.teleop_bridge, args.marvin_config, args.phase1_repo) as kernel:
        mapped = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()}
        loaded_vendor = {Path(path).resolve() for path in mapped if path.endswith("/libKine.so")}
        if loaded_vendor != {args.marvin_vendor.resolve()}:
            raise ValueError(f"Actual Marvin library mapping differs from the pinned vendor: {loaded_vendor}")
        frames, frame_witnesses = native_frames(robot, kernel)
        hand = Wuji(args.wuji_sdk)
        try:
            candidate = TeleopCandidate(robot, kernel, hand, sweep, frames)
            for split in SPLITS if args.split == "Both" else ("Train",):
                source = reader if split == "Train" else open_reader(args, split)
                summaries[split] = compare_split(source, args, split, mapper, candidate, bank, prefix_receipt)
        finally:
            hand.close()
    write_json(args.out / "candidate_prefix_controls.json", {
        "scope": SCOPE, "protocol": "truncate versus adversarial suffix with identical past",
        "roots": prefix_receipt,
        "ordinary_checked_roots": sum("ordinary" in checks for checks in prefix_receipt.values()),
        "generation_barrier_checked_roots": sum("generation_barrier" in checks for checks in prefix_receipt.values()),
    })
    write_json(args.out / "candidate_comparison.json", {
        "schema": "s05.dcr008.nominal_evaluation.v1", "scope": SCOPE,
        "identity_sha256": identity, "wall_s": monotonic() - started, "native_frame_witness_max_m_rad": frame_witnesses,
        "frozen_m1_scoring_preserved": True, "official_admission": False, "eligible_for_training": False,
        "formal_contract_frozen": False, "full_comparison": (args.split == "Both" and
            all(value["full_signed_split"] for value in summaries.values()) and bank is not None),
        "physical_witness_scope": QUANTIFICATION, "actual_d10_rollout": "NOT_RUN",
        "online_teleop_and_measured_feedback": "UNKNOWN_REQUIRES_ACTUAL_D1",
        "m1_continuity_gate": "STILL_FAILS_INDEPENDENTLY",
        "coverage_change": "original legal exact nominal rows versus independently admitted fresh active rows",
        "status_dictionary": {int(status): status.name for status in Classification},
        "candidate_native_flags_order": NATIVE_FLAGS, "split_results": summaries,
    })


if __name__ == "__main__":
    main()
