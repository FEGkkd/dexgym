"""Pair measured MuJoCo nominal feedback with the signed HMB control grid."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
from hashlib import file_digest, sha256
from heapq import heappop, heappush
import json
from pathlib import Path

import numpy as np
import yaml

from summarize_nominal_replay import ACTIVE, FEEDBACK, HISTORICAL, joint_values, quantiles, stamp


DCR_COMPARISON_SHA = "4f418fc020cb8ed41a718a2378a30ab362faf0a0b7b3fe621b16dac5c04761e9"
SCOPE = "SIMULATOR_NOMINAL_HMB_DIAGNOSTIC"
STATUS = {1: "ACTIVE_TRACKED", 2: "LIMITED_ACTIVE", 3: "HOLD", 4: "FALLBACK", 5: "FAILED"}
WRIST_POSITION_TOLERANCE_M = 1e-6
WRIST_ROTATION_TOLERANCE_RAD = 1e-5


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def one(rows, description):
    if len(rows) != 1:
        raise ValueError(f"Expected one {description}, got {len(rows)}")
    return rows[0]


def signed_feedback_names(robot):
    """Return driver feedback aliases in signed physical joint order."""
    arms = tuple(f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8))
    if len(robot.names) != 54 or len(set(robot.names)) != 54 or tuple(robot.names[:14]) != arms:
        raise ValueError("Unexpected signed Marvin joint names or model order")
    arm_aliases = tuple(f"{side}_j{i}" for side in ("left", "right") for i in range(1, 8))
    # RobotModel binds each actuator through its exact physical joint transmission.
    aliases = arm_aliases + tuple(actuator.name for actuator in robot.actuators[14:])
    if len(aliases) != 54 or len(set(aliases)) != 54 or not all(aliases):
        raise ValueError("Driver feedback aliases must bind one-to-one to all 54 signed joints")
    return aliases


def trace_evidence(path):
    timebases, models, input_schedules = [], [], []
    phases, segments, states = {}, [], defaultdict(dict)
    segment, ready, last_arrival = 0, False, -1
    content_digest = sha256()
    open_trace = gzip.open if path.suffix == ".gz" else open
    with open_trace(path, "rb") as stream:
        for line in stream:
            content_digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete replay trace tail")
            row = json.loads(line)
            arrival = row["mono_ns"]
            if arrival < last_arrival:
                raise ValueError("Replay trace observation order is not causal")
            last_arrival = arrival
            kind, stage = row["kind"], row["stage"]
            if kind == "timebase":
                timebases.append(row)
            if kind == "model":
                models.append(row)
            if kind == "input" and stage == HISTORICAL:
                input_schedules.append(row["original_schedule_ns"])
            topic, value = row.get("topic"), row.get("value")
            if kind == "output" and topic in ACTIVE:
                key, active_value = ACTIVE[topic]
                phases[topic] = value.get(key) == active_value
                new_ready = all(phases.get(name, False) for name in ACTIVE)
                if new_ready != ready:
                    segment += 1
                    ready = new_ready
                    segments.append((arrival, segment if ready else 0))
            if kind != "output" or stage != HISTORICAL or not ready or topic not in FEEDBACK:
                continue
            header = stamp(value.get("header", {}).get("stamp"))
            if header is None:
                continue
            entries = states[segment, header]
            angles = joint_values(value, FEEDBACK[topic])
            if topic.endswith("/state") and (
                value.get("connected") is not True or value.get("position_valid") != [True] * FEEDBACK[topic]
            ):
                angles = None
            entries[topic] = (arrival, angles) if topic not in entries else (arrival, None)
    frames, invalid_groups = [], 0
    for (session, header), messages in states.items():
        if messages.keys() != FEEDBACK.keys() or any(angles is None for _, angles in messages.values()):
            invalid_groups += 1
            continue
        merged = {name: q for _, angles in messages.values() for name, q in angles.items()}
        if len(merged) != 54:
            invalid_groups += 1
            continue
        completion = max(at for at, _ in messages.values())
        frames.append((completion, header, session, merged))
    return {
        "timebase": one(timebases, "recorded replay timebase"), "model": one(models, "observed simulator model"),
        "first_source_schedule": min(input_schedules), "last_source_schedule": max(input_schedules),
        "frames": sorted(frames, key=lambda row: (row[0], row[1], row[2])),
        "segments": segments, "incomplete_invalid_or_duplicate_measured_stamps": invalid_groups,
        "uncompressed_jsonl_sha256": content_digest.hexdigest(),
    }


class PastMeasured:
    """Select a complete measured state available at the signed planned cutoff."""

    def __init__(self, evidence, joint_names, max_age_ns, driver_names=None):
        self.timebase = evidence["timebase"]
        self.frames = iter(evidence["frames"])
        self.segment_times = [row[0] for row in evidence["segments"]]
        self.segment_ids = [row[1] for row in evidence["segments"]]
        self.last_source_schedule = evidence["last_source_schedule"]
        self.names = {frozenset(joint_names): tuple(joint_names)}
        if driver_names is not None:
            self.names[frozenset(driver_names)] = driver_names
        self.max_age_ns = max_age_ns
        self.pending = next(self.frames, None)
        self.current = {}
        self.eligible_on_header = []

    def get(self, source_cutoff):
        base = self.timebase
        offset = source_cutoff - base["origin_schedule_ns"]
        if offset < 0 or source_cutoff > self.last_source_schedule:
            return "NOT_RUN_OUTSIDE_REPLAYED_INPUT_TIMELINE", None, None, None
        cutoff_mono = base["origin_mono_ns"] + offset
        cutoff_ros = base["origin_log_ns"] + base["epoch_delta_ns"] + offset
        i = bisect_right(self.segment_times, cutoff_mono) - 1
        segment = self.segment_ids[i] if i >= 0 else 0
        while self.pending is not None and self.pending[0] <= cutoff_mono:
            arrival, header, session, names = self.pending
            heappush(self.eligible_on_header, (header, arrival, session, names))
            self.pending = next(self.frames, None)
        while self.eligible_on_header and self.eligible_on_header[0][0] <= cutoff_ros:
            header, arrival, session, names = heappop(self.eligible_on_header)
            previous = self.current.get(session)
            # Retain the newest measured header from all completely received
            # messages in this active lifecycle segment.
            if previous is None or header > previous[1]:
                self.current[session] = (arrival, header, names)
        if not segment:
            return "NOT_ACTIVE_AT_PLANNED_CUTOFF", None, None, None
        if segment not in self.current:
            return "NO_COMPLETE_MEASURED_STATE_BEFORE_CUTOFF", None, None, None
        arrival, header, names = self.current[segment]
        age = cutoff_ros - header
        if age < 0:
            return "MEASURED_HEADER_AFTER_CUTOFF_OR_UNKNOWN_EPOCH", None, age, cutoff_mono - arrival
        if age > self.max_age_ns or cutoff_mono - arrival > self.max_age_ns:
            return "STALE_MEASURED_STATE", None, age, cutoff_mono - arrival
        order = self.names.get(frozenset(names))
        if order is None:
            return "MEASURED_JOINT_NAMES_DIFFER_FROM_SIGNED_MODEL", None, age, cutoff_mono - arrival
        return "COHERENT_MEASURED", np.array([names[name] for name in order]), age, cutoff_mono - arrival


def normalized_assets(xml):
    from dexgym.tasks.direct.wholebody_s05.p02.model import asset_files

    return {Path(source).relative_to(xml.parent).as_posix(): sha for source, sha in asset_files(xml).items()}


def geometries_comparable(actual, robot):
    path = Path(actual["path"])
    if actual["sha256"] != digest(robot.path):
        return "UNSUPPORTED_SIMULATOR_MJCF_SHA_DIFFERS"
    if not path.is_file() or digest(path) != actual["sha256"]:
        return "UNSUPPORTED_SIMULATOR_FILE_NOT_REVERIFIABLE"

    if normalized_assets(path) != normalized_assets(robot.path):
        return "UNSUPPORTED_SIMULATOR_DEPENDENCY_BYTES_DIFFER"
    return "EXACT_MJCF_AND_DEPENDENCIES"


def score_tip(robot, q, target):
    return float(np.linalg.norm(robot.geometry(q).tips - target, axis=-1).max())


def masked_target(part, layer, i):
    key = f"{layer}_world_tip_xyz"
    value = part[key][i]
    return value if part[f"{key}_mask"][i].all() and np.isfinite(value).all() else None


def wrist_metrics(row, exact, i, robot, actual_robot, q, offline_q):
    from scipy.spatial.transform import Rotation

    live = actual_robot.geometry(q)
    signed_at_live = robot.geometry(q)
    xyz_error = np.linalg.norm(live.wrists - signed_at_live.wrists, axis=-1).max()
    rotations = Rotation.from_matrix(np.swapaxes(live.rotations, -1, -2) @ signed_at_live.rotations)
    angle_error = rotations.magnitude().max()
    row["simulator_vs_signed_wrist_fk_m"] = float(xyz_error)
    row["simulator_vs_signed_wrist_fk_rad"] = float(angle_error)
    if xyz_error > WRIST_POSITION_TOLERANCE_M or angle_error > WRIST_ROTATION_TOLERANCE_RAD:
        row["wrist_comparability"] = "FK_NOT_EQUIVALENT_ON_MEASURED_ROW"
        return
    row["wrist_comparability"] = "MEASURED_CONFIGURATION_WORLDFRAME_FK_MATCH"
    offline = robot.geometry(offline_q) if offline_q is not None else None
    for layer, label in (("M0_audit", "m0"), ("mapped", "m1")):
        xyz_key, quaternion_key = f"{layer}_wrist_xyz", f"{layer}_wrist_xyzw"
        if not exact[f"{xyz_key}_mask"][i].all() or not exact[f"{quaternion_key}_mask"][i].all():
            continue
        target = exact[xyz_key][i]
        orientation = Rotation.from_quat(exact[quaternion_key][i])
        for source, observed in (("online", live), ("dcr", offline)):
            if observed is None:
                continue
            angle = (orientation.inv() * Rotation.from_matrix(observed.rotations)).magnitude()
            row[f"{source}_{label}_wrist_pos_max_m"] = float(np.linalg.norm(observed.wrists - target, axis=-1).max())
            row[f"{source}_{label}_wrist_rot_max_rad"] = float(angle.max())


def planned_rows(root, reader, bank, dcr_index, dcr_path, clock, alignment, geometry_supported, robot, actual_wrist):
    for partition in root["hmb_partitions"]:
        key = partition["key"]
        raw, meta, _, _ = reader.read(key)
        bank_row = bank.read(key, root["split"])
        offline = dcr_index[key]
        if (meta["identity"]["root_episode_id"] != root["root"] or bank_row.root != root["root"]
                or offline["root"] != root["root"]):
            raise ValueError(f"Source lineage mismatch at {key}")
        npz = dcr_path.parent / offline["trace"]
        if digest(npz) != offline["trace_sha256"]:
            raise ValueError(f"Offline candidate trace SHA-256 differs: {npz}")
        with np.load(npz, allow_pickle=False) as dcr:
            exact = bank_row.arrays
            ticks, cutoffs = raw["tick_ns"], raw["cutoff_ns"]
            if (len(ticks) != partition["planned_ticks"] or not np.array_equal(ticks, cutoffs)
                    or not np.array_equal(ticks, exact["tick_ns"]) or not np.array_equal(ticks, dcr["tick_ns"])
                    or not np.array_equal(cutoffs, dcr["cutoff_ns"])
                    or not np.array_equal(raw["joint_valid"], dcr["hmb_joint_valid"])):
                raise ValueError(f"Three-way signed planned timeline mismatch: {key}")
            valid_mapping = dcr["mapping_valid"]
            direct_valid_mapping = exact["domain_mapped_valid"]
            if (not np.array_equal(valid_mapping, direct_valid_mapping) or not np.array_equal(
                    exact["mapped_world_tip_xyz"][valid_mapping], dcr["mapped_world_tip_xyz"][valid_mapping])):
                raise ValueError(f"Pinned M1 reference mismatch: {key}")
            for i, cutoff in enumerate(cutoffs):
                category = STATUS[int(dcr["row_class"][i])]
                reason, q, age_ns, receive_age_ns = alignment.get(clock + int(cutoff))
                row = {
                    "split": root["split"], "root": root["root"], "partition": key, "tick": i,
                    "hmb_tick_ns": int(ticks[i]), "hmb_cutoff_ns": int(cutoff), "hmb_joint_valid": bool(raw["joint_valid"][i]),
                    "replay_feedback": reason, "measurement_age_ns": age_ns, "completion_age_ns": receive_age_ns,
                    "replay_measured_q_rad": q.tolist() if q is not None else None,
                    "dcr_row_class": category, "dcr_provider_observed": bool(dcr["provider_observed"][i]),
                    "dcr_nominal_q_rad": dcr["q_nominal"][i].tolist() if dcr["provider_observed"][i] else None,
                }
                if q is not None and row["dcr_provider_observed"]:
                    row["online_measured_vs_dcr_nominal_max_abs_rad"] = float(np.abs(q - dcr["q_nominal"][i]).max())
                if q is not None and row["hmb_joint_valid"] and direct_valid_mapping[i]:
                    active = category in ("ACTIVE_TRACKED", "LIMITED_ACTIVE")
                    if actual_wrist is not None:
                        wrist_metrics(row, exact, i, robot, actual_wrist, q, dcr["q_nominal"][i] if active else None)
                    if geometry_supported:
                        for key_name, layer in (("m0", "M0_audit"), ("m1", "mapped")):
                            target = masked_target(exact, layer, i)
                            if target is not None:
                                row[f"online_{key_name}_tip_max_m"] = score_tip(robot, q, target)
                                if active:
                                    row[f"dcr_{key_name}_tip_max_m"] = score_tip(robot, dcr["q_nominal"][i], target)
                yield row


def main():
    from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, SIMULATED
    from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
    from dexgym.tasks.direct.wholebody_s05.v1.rmb.build import MODEL_MANIFEST_SHA, MODEL_SHA
    from dexgym.tasks.direct.wholebody_s05.v1.rmb.reader import RobotReferenceBank

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "cohort", "model", "direct-bank", "dcr-comparison", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    parser.add_argument("--max-age-ms", type=float, default=40., help="Diagnostic freshness window, default two HMB ticks")
    parser.add_argument("--wrist-only-diagnostic", action="store_true",
                        help="Independently compare actual and signed World wrist FK at each measured configuration")
    args = parser.parse_args()
    if not 0 < args.max_age_ms < 1000:
        parser.error("Specify a positive bounded diagnostic freshness window")
    cohort = json.loads(args.cohort.read_bytes())
    if cohort["schema"] != "s05.teleop.hmb_source_cohort.v1" or not cohort["native_assets_sha256_checked"]:
        raise ValueError("Cohort identity must be resolved and byte-verified")
    root = one([row for row in cohort["selected_roots"] if row["root"] == args.root], "signed root")
    replay_inputs = json.loads(args.trace.with_name("input.json").read_bytes())
    if replay_inputs["root"] != root or replay_inputs["cohort_receipt_sha256"] != digest(args.cohort):
        raise ValueError("Replay input receipt does not bind the same signed raw source and cohort")
    source_manifest = (Path(root["canonical_episode_uri"]) / "manifest.yaml").read_bytes()
    if sha256(source_manifest).hexdigest() != root["manifest_sha256"]:
        raise ValueError("Signed raw capture manifest SHA-256 differs")
    start_monotonic = yaml.safe_load(source_manifest)["start_monotonic_ns"]
    if type(start_monotonic) is not int or start_monotonic <= 0:
        raise ValueError("Signed raw capture manifest has no valid monotonic origin")
    if digest(args.dcr_comparison) != DCR_COMPARISON_SHA:
        raise ValueError("Sealed DCR-008 comparison identity differs")
    dcr_comparison = json.loads(args.dcr_comparison.read_bytes())
    dcr_index = {row["partition"]: row for row in dcr_comparison["split_results"][root["split"]]["partitions"]}
    bank = RobotReferenceBank(args.direct_bank, expected_sha=dcr_comparison["identity_sha256"]["direct_bank"],
                              verify=True, require_qualified=False)
    docs = args.docs / "evidence"
    hmb = HMBReader(docs / "P01_E_E1/hmb_manifest.json", docs / "P01_E_E1/production_plan.json",
                    docs / "P01_E_E0/candidate_config.json", cohort["signed_hmb_ubuntu_products"], root["split"], SIMULATED)
    if digest(args.model) != MODEL_SHA:
        raise ValueError("Use the signed independent P02 scoring XML")
    robot = RobotModel(args.model)
    asset_manifest = docs / "P02/refinement/model_assets_sha256.json"
    if digest(asset_manifest) != MODEL_MANIFEST_SHA:
        raise ValueError("Signed P02 scoring asset manifest differs")
    expected = {name.split("/generated/")[-1]: value for name, value in json.loads(asset_manifest.read_bytes()).items()}
    if normalized_assets(robot.path) != expected:
        raise ValueError("Actual scoring assets differ from signed model")
    trace = trace_evidence(args.trace)
    model_scope = geometries_comparable(trace["model"], robot)
    actual_wrist = None
    if args.wrist_only_diagnostic:
        path = Path(trace["model"]["path"])
        if digest(path) != trace["model"]["sha256"]:
            raise ValueError("Current simulator wrist XML differs from the live-run trace")
        actual_wrist = RobotModel(path)
        if actual_wrist.names != robot.names:
            raise ValueError("Actual and signed model joint orders differ; wrist diagnostic requires the same axes")
    driver_names = signed_feedback_names(robot)
    alignment = PastMeasured(trace, robot.names, round(args.max_age_ms * 1e6), driver_names)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, scores = Counter(), defaultdict(list)
    with (args.out / "paired_ticks.jsonl").open("x", encoding="utf-8") as output:
        for row in planned_rows(root, hmb, bank, dcr_index, args.dcr_comparison, start_monotonic, alignment,
                                model_scope == "EXACT_MJCF_AND_DEPENDENCIES", robot, actual_wrist):
            counts["planned"] += 1
            counts["signed_hmb_joint_valid"] += row["hmb_joint_valid"]
            counts[f"feedback:{row['replay_feedback']}"] += 1
            if row["hmb_joint_valid"]:
                counts[f"feedback_hmb_valid:{row['replay_feedback']}"] += 1
            counts[f"dcr:{row['dcr_row_class']}"] += 1
            if "wrist_comparability" in row:
                counts[f"wrist:{row['wrist_comparability']}"] += 1
            for key, value in row.items():
                if key.endswith(("_tip_max_m", "_wrist_pos_max_m", "_wrist_rot_max_rad", "_wrist_fk_m", "_wrist_fk_rad")
                        ) or key == "online_measured_vs_dcr_nominal_max_abs_rad":
                    scores[key].append(value)
            for layer in ("m0", "m1"):
                for statistic in ("tip_max_m", "wrist_pos_max_m", "wrist_rot_max_rad"):
                    field = f"{layer}_{statistic}"
                    if f"dcr_{field}" in row:
                        scores[f"common_online_{field}"].append(row[f"online_{field}"])
                        scores[f"common_online_minus_dcr_{field}"].append(
                            row[f"online_{field}"] - row[f"dcr_{field}"])
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    if counts["planned"] != root["planned_ticks"]:
        raise ValueError("Signed root denominator differs")
    output = {
        "schema": "s05.teleop.hmb_nominal_pair.v1", "scope": SCOPE, "root": root["root"], "split": root["split"],
        "geometry_comparability": model_scope, "max_age_ms": args.max_age_ms,
        "wrist_only_diagnostic": args.wrist_only_diagnostic,
        "feedback_timebase": "completed four-topic same-stamp feedback, historical input stage, as-of original HMB cutoff",
        "signed_cutoff_alignment": "ros_observer_origin_mono + capture_manifest_start_mono + cutoff_ns - collector_origin_mono",
        "comparison_semantics": "online measured q versus offline DCR commanded q, reported separately",
        "feedback_joint_pairs_in_signed_order": list(zip(robot.names, driver_names, strict=True)),
        "replay_trace_sha256": digest(args.trace), "cohort_sha256": digest(args.cohort),
        "replay_trace_sha256_scope": "stored input bytes, including gzip framing when present",
        "replay_trace_uncompressed_jsonl_sha256": trace["uncompressed_jsonl_sha256"],
        "dcr_comparison_sha256": DCR_COMPARISON_SHA, "output_ticks_sha256": digest(args.out / "paired_ticks.jsonl"),
        "simulator_reported_model_sha256": trace["model"]["sha256"], "signed_scoring_model_sha256": MODEL_SHA,
        "incomplete_invalid_or_duplicate_measured_stamps": trace["incomplete_invalid_or_duplicate_measured_stamps"],
        "counts": dict(counts), "scores": {name: quantiles(values) for name, values in scores.items()},
        "formal_d1_gate_authority": False, "training_admission": False,
    }
    with (args.out / "pair_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
