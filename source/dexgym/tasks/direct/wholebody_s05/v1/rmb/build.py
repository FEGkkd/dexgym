"""Build a per-partition HMB-derived reference candidate on the Ubuntu simulator host."""
import argparse
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

from ...hmb_reader import CONFIG_SHA, MANIFEST_SHA, HMBReader, SIMULATED, checked_json
from ...p02.model import RobotModel, asset_files
from ..task.native_sweep import NativeSweep
from .domains import THRESHOLDS_M, contiguous_segments, radial_envelope, radial_lower_bound, realized, residual_witness, threshold_hard
from .mapping import IntentMapper, compiled_topologies
from .nominal import Nominal, NominalStatus, sdk_context
from .schema import SCHEMA, SPLITS, TICK_NS, add_arrays, empty_geometry, file_sha, put_geometry, write_json, write_partition


FIT_SHA = "577195a8eb66699e11cf8bcf910383086672c384723d61fa90c13ed90652baa7"
CAL_SHA = "27671adc7f78d29ab282fabcf43d542f16e31e1481e51b2f7f3cd5c4121b5b2b"
HANDOFF_SHA = "cc1f21ead157e576113a184dba48a5d2596f4835ffdf53e554823cb67eb3fb97"
MODEL_SHA = "939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d"
MODEL_MANIFEST_SHA = "92c44cabe65b278ab8261c2659c2cfa8c670fad07fd7d3a640e8629bc5c06a0d"
REGRESSION_SHA = frozenset((
    "ffd5353918e38e6527b534aa3263baa20d7cb3f8a868d91f33b8644cbd437817",
    "cbac62bb6d58ce2fa159d547408cf3534a29c32ca8b0bba1cddc8c4bc99ba1f5",
))


def scope_hashes(references):
    return {item["record"]["asset_sha256"] for item in references}


def windows_json(mask, segment, generation):
    return [{"start": first, "stop": stop, "ticks": stop - first}
            for first, stop in contiguous_segments(mask, segment, generation)]


def output_fields(n):
    return {
        "audit_plan_ordinal": np.arange(n, dtype="<i8"),
        "audit_generation": np.full(n, -1, dtype="<i4"),
        "audit_segment": np.full(n, -1, dtype="<i4"),
        "audit_cause_code": np.zeros(n, dtype="<u2"),
        "inputs_source_changed": np.zeros((n, 4), dtype=bool),
        "nominal_q_nominal": np.zeros((n, 54), dtype="<f4"),
        "nominal_qd_nominal": np.zeros((n, 54), dtype="<f4"),
        "nominal_joint_valid": np.zeros((n, 54), dtype=bool),
        "nominal_velocity_valid": np.zeros((n, 54), dtype=bool),
        "nominal_status": np.zeros((n, 4), dtype="<u2"),
        "nominal_held": np.zeros((n, 4), dtype=bool),
        "nominal_limit_applied": np.zeros((n, 54), dtype=bool),
        "nominal_raw_sdk_hand": np.zeros((n, 2, 20), dtype="<f4"),
        "nominal_raw_sdk_valid": np.zeros((n, 2), dtype=bool),
        "nominal_raw_ik_arm": np.zeros((n, 2, 7), dtype="<f4"),
        "nominal_raw_ik_valid": np.zeros((n, 2), dtype=bool),
        "mapped_pinch_active": np.zeros((n, 2, 4), dtype=bool),
        "mapped_geometry_velocity_valid": np.zeros(n, dtype=bool),
        "mapped_wrist_linear_W_m_s": np.zeros((n, 2, 3), dtype="<f4"),
        "mapped_wrist_angular_W_rad_s": np.zeros((n, 2, 3), dtype="<f4"),
        "mapped_tip_velocity_W_m_s": np.zeros((n, 2, 5, 3), dtype="<f4"),
        "mapped_local_velocity_P_m_s": np.zeros((n, 2, 5, 4, 3), dtype="<f4"),
        "mapped_relative_linear_U_L_m_s": np.zeros((n, 3), dtype="<f4"),
        "mapped_relative_angular_U_L_rad_s": np.zeros((n, 3), dtype="<f4"),
        "domain_mapped_valid": np.zeros(n, dtype=bool),
        "domain_nominal_legal": np.zeros(n, dtype=bool),
        "domain_improvable": np.zeros(n, dtype=bool),
        "domain_hard": np.zeros(n, dtype=bool),
        "domain_strictly_excluded_at": np.zeros((n, 4), dtype=bool),
        "domain_unclassified_at": np.ones((n, 4), dtype=bool),
        "domain_radial_lower_bound_m": np.zeros((n, 2, 5), dtype="<f8"),
        "domain_residual_search_status": np.zeros(n, dtype="<u2"),
        "domain_static_witness_at": np.zeros((n, 2), dtype=bool),
    }


def fill_mapped_velocity(fields, index, previous, current):
    for name, key in (
        ("wrist_linear_W_m_s", "wrist_xyz"), ("local_velocity_P_m_s", "local_landmark_xyz"),
        ("tip_velocity_W_m_s", "world_tip_xyz"), ("relative_linear_U_L_m_s", "relative_wrist_xyz"),
    ):
        fields[f"mapped_{name}"][index] = (current[key] - previous[key]) / 0.020
    for name, key, spatial in (
        ("wrist_angular_W_rad_s", "wrist_xyzw", True), ("relative_angular_U_L_rad_s", "relative_wrist_xyzw", False),
    ):
        old = Rotation.from_quat(previous[key])
        new = Rotation.from_quat(current[key])
        delta = new * old.inv() if spatial else old.inv() * new
        fields[f"mapped_{name}"][index] = delta.as_rotvec() / 0.020
    fields["mapped_geometry_velocity_valid"][index] = True


def build_partition(raw, meta, model, mapper, nominal, envelope, *, denied, max_ticks, witness_steps):
    length = len(raw["tick_ns"]) if max_ticks is None else min(max_ticks, len(raw["tick_ns"]))
    fields = output_fields(length)
    nominal.generation = -1
    nominal.cut()
    take = slice(0, length)
    fields["tick_ns"] = raw["tick_ns"][take].astype("<i8", copy=True)
    fields["audit_hmb_source_index"] = raw["source_index"][take, :4, 0].astype("<i8", copy=True)
    fields["audit_capture_ns"] = raw["capture_bounds_ns"][take, :4, 0].astype("<i8", copy=True)
    fields["audit_receive_bound_ns"] = raw["arrival_bounds_ns"][take, :4].astype("<i8", copy=True)
    fields["audit_hmb_reason"] = raw["reason_code"][take, :4].astype("<u2", copy=True)
    fields["audit_clock_eligible"] = ((fields["audit_hmb_source_index"] >= 0)
        & (raw["capture_bounds_ns"][take, :4, 1] <= raw["tick_ns"][take, None])
        & (raw["arrival_bounds_ns"][take, :4, 1] <= raw["cutoff_ns"][take, None]))
    fields["inputs_stream_age_ns"] = raw["age_ns"][take, :4].astype("<i8", copy=True)
    for dest, original in (("stream_valid", "pose_valid"), ("pose_joint_valid", "joint_valid"),
                           ("safety_permission", "safety_permission")):
        fields[f"inputs_{dest}"] = (raw[original][take, :4] if original == "pose_valid" else raw[original][take]).copy()
    geometry = {key: empty_geometry(length) for key in ("mapped", "hard_m1_audit", "M0_audit", "nominal", "feasible")}
    hard_features = np.full((length, 4), np.nan)
    diagnostic_q = np.zeros((length, 54), dtype="<f4")
    diagnostic_valid = np.zeros(length, dtype=bool)
    continuity = []
    ik_failures = []
    failures = Counter()
    compiled = {}
    prev_split_segment, prev_q, prev_map, prev_source = None, None, None, np.full(4, -1, dtype=np.int64)
    prev_active = prev_unadjusted = None
    for i in range(length):
        source = fields["audit_hmb_source_index"][i]
        segment = int(raw["segment_id"][i])
        valid = bool(raw["joint_valid"][i] and raw["safety_permission"][i]
                     and raw["pose_valid"][i, :4].all() and fields["audit_clock_eligible"][i].all()
                     and segment >= 0 and not denied)
        fields["audit_segment"][i] = segment
        if not valid:
            reason = NominalStatus.DENIED_SOURCE if denied else NominalStatus.UNAVAILABLE
            fields["nominal_status"][i] = reason
            fields["audit_cause_code"][i] = reason
            failures[reason.name] += 1
            # A cut resets the physics-independent reference state before next use.
            prev_split_segment, prev_q, prev_map = None, None, None
            continue
        if segment != prev_split_segment:
            nominal.cut()
            prev_source = np.full(4, -1, dtype=np.int64)
            prev_q, prev_map = None, None
        prev_split_segment = segment
        fields["audit_generation"][i] = nominal.generation
        fields["inputs_source_changed"][i] = source != prev_source
        prev_source = source
        topology_key = tuple(int(x) for x in raw["topology_id"][i, :2])
        try:
            if topology_key not in compiled:
                compiled[topology_key] = compiled_topologies(meta, topology_key)
            mapped_order, sdk_order = compiled[topology_key]
            result = mapper.map(raw["pose"][i, 2:4, 0], raw["pose"][i, :2], mapped_order)
        except (ValueError, KeyError, IndexError) as error:
            fields["audit_cause_code"][i] = 7
            prev_split_segment, prev_q, prev_map = None, None, None
            failures[f"MAPPED_GEOMETRY_INVALID:{type(error).__name__}"] += 1
            continue
        mapped = result["mapped"]
        active = result["pinch_active"]
        for layer in ("mapped", "M0_audit", "hard_m1_audit"):
            put_geometry(geometry[layer], i, result[layer])
        fields["mapped_pinch_active"][i] = active
        fields["domain_mapped_valid"][i] = True
        if prev_map is not None:
            fill_mapped_velocity(fields, i, prev_map, mapped)
            if np.any(active != prev_active):
                extra = (mapped["world_tip_xyz"] - prev_map["world_tip_xyz"]
                         - (result["unadjusted"]["world_tip_xyz"] - prev_unadjusted))
                continuity.append({"tick": i, "enter": int((active & ~prev_active).sum()),
                    "leave": int((prev_active & ~active).sum()), "excess_m": float(np.linalg.norm(extra, axis=-1).max())})
        prev_map, prev_active, prev_unadjusted = mapped, active, result["unadjusted"]["world_tip_xyz"]

        radial = radial_lower_bound(mapped["world_tip_xyz"], envelope)
        fields["domain_radial_lower_bound_m"][i] = radial
        excluded = radial.max() > THRESHOLDS_M
        fields["domain_strictly_excluded_at"][i] = excluded
        fields["domain_unclassified_at"][i] = ~excluded

        q, raw_q, raw_valid, status, limits = nominal.step(mapped, raw["sdk_common_pose"][i, :2], sdk_order, source)
        for failure in nominal.ik_failures:
            ik_failures.append({"tick": i, **failure})
        fields["nominal_status"][i] = status
        fields["nominal_held"][i] = status != NominalStatus.ACTIVE
        fields["nominal_limit_applied"][i] = limits
        fields["nominal_raw_ik_arm"][i] = raw_q[:14].reshape(2, 7)
        fields["nominal_raw_sdk_hand"][i] = raw_q[14:].reshape(2, 20)
        fields["nominal_raw_ik_valid"][i] = raw_valid[:2]
        fields["nominal_raw_sdk_valid"][i] = raw_valid[2:]
        fields["nominal_q_nominal"][i] = q
        if not np.all(status == NominalStatus.ACTIVE):
            fields["audit_cause_code"][i] = status.max()
            failures[NominalStatus(status.max()).name] += 1
            prev_q = None
            continue
        nominal_geom = realized(model, q)
        put_geometry(geometry["nominal"], i, nominal_geom)
        fields["nominal_joint_valid"][i] = True
        fields["domain_nominal_legal"][i] = True
        if prev_q is not None:
            fields["nominal_qd_nominal"][i] = (q - prev_q) / 0.020
            fields["nominal_velocity_valid"][i] = True
        prev_q = q
        if i >= 5 and fields["domain_nominal_legal"][i - 5:i + 1].all() and fields["audit_generation"][i - 5] == nominal.generation:
            delta = mapped["wrist_xyz"] - geometry["mapped"]["wrist_xyz"][i - 5]
            angle = Rotation.from_quat(geometry["mapped"]["wrist_xyzw"][i - 5]).inv() * Rotation.from_quat(mapped["wrist_xyzw"])
            hard_features[i] = (
                np.linalg.norm(delta, axis=-1).max() / 0.1, angle.magnitude().max() / 0.1,
                np.linalg.norm(nominal_geom["world_tip_xyz"] - mapped["world_tip_xyz"], axis=-1).max(),
                np.linalg.norm(nominal_geom["pinch_xyz"] - mapped["pinch_xyz"], axis=-1).max(),
            )
        if witness_steps > 0:
            witness = residual_witness(model, q, mapped, active, witness_steps, nominal.swept_proof)
            within = witness.error_m <= THRESHOLDS_M
            fields["domain_unclassified_at"][i] = ~(within | excluded)
            fields["domain_improvable"][i] = witness.improved
            fields["domain_static_witness_at"][i] = within[1:3]
            fields["domain_residual_search_status"][i] = 1 if witness.improved or within.any() else 2
            # Nominal FK remains in the nominal layer. Only an independently
            # checked non-nominal candidate is published as optional geometry.
            if not np.array_equal(witness.q, q) and (witness.improved or within.any()):
                put_geometry(geometry["feasible"], i, witness.geometry)
            if witness.improved or within.any():
                diagnostic_q[i] = witness.q
                diagnostic_valid[i] = True
    fields["nominal_nominal_generation"] = fields["audit_generation"].copy()
    for key, layer in geometry.items():
        add_arrays(fields, key, layer)
    return fields, hard_features, continuity, dict(failures), diagnostic_q, diagnostic_valid, ik_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--marvin-bridge", type=Path, required=True)
    parser.add_argument("--marvin-config", type=Path, required=True)
    parser.add_argument("--wuji-sdk", type=Path, required=True)
    parser.add_argument("--extra-denylist", type=Path)
    parser.add_argument("--split", choices=(*SPLITS, "Both"), default="Both")
    parser.add_argument("--max-partitions", type=int)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--one-partition-per-root", action="store_true")
    parser.add_argument("--witness-steps", type=int, default=8, choices=(0, *range(4, 13)))
    parser.add_argument("--smooth-band-mm", type=float, nargs=2)
    parser.add_argument("--verify-prefixes", action="store_true",
                        help="Replay two preselected prefixes per root after truncation and replacement of all future values")
    parser.add_argument("--arm-step-rad", type=float, default=0.04)
    parser.add_argument("--hand-step-rad", type=float, default=0.10)
    args = parser.parse_args()
    if args.split == "Validation" and not args.max_partitions:
        parser.error("Freeze Train before building Validation and preserving hard thresholds")
    if args.max_partitions is not None and args.max_partitions <= 0:
        parser.error("--max-partitions must be positive")
    if args.max_ticks is not None and args.max_ticks <= 0:
        parser.error("--max-ticks must be positive")
    start = monotonic()
    if file_sha(args.model) != MODEL_SHA:
        parser.error("The MJCF XML does not match the signed P02 model")
    docs = args.docs
    p02 = docs / "evidence" / "P02"
    e1 = docs / "evidence" / "P01_E_E1"
    e0 = docs / "evidence" / "P01_E_E0"
    fr = docs / "evidence" / "P01_F" / "p02_handoff_manifest.json"
    frozen_handoff = checked_json(fr, HANDOFF_SHA)
    if args.products.resolve() != Path(frozen_handoff["ubuntu_products"]).resolve():
        parser.error("RMB must read the frozen signed official HMB product path")
    signed_model_assets = checked_json(p02 / "refinement" / "model_assets_sha256.json", MODEL_MANIFEST_SHA)
    archive_assets = {name.split("/generated/")[-1]: digest for name, digest in signed_model_assets.items()}
    model_dir = args.model.resolve().parent
    actual_assets = {Path(name).relative_to(model_dir).as_posix(): digest for name, digest in asset_files(args.model).items()}
    if actual_assets != archive_assets:
        parser.error("Actual MJCF dependencies differ from the signed P02 asset manifest")
    fitted = p02 / "refinement" / "fitted_candidates.json"
    calibration = p02 / "mapping_contract.json"
    fit = checked_json(fitted, FIT_SHA)
    calibrated = checked_json(calibration, CAL_SHA)
    # Additional independent reviewer-reviewed lineage exclusions can only add
    # known roots or source fingerprints to the exact two recording exclusions.
    deny_roots, deny_assets = set(), set(REGRESSION_SHA)
    if args.extra_denylist is not None:
        import json
        closure = json.loads(args.extra_denylist.read_bytes())
        deny_roots.update(closure["roots"])
        deny_assets.update(closure["asset_sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    model = RobotModel(args.model)
    native_assets = (args.marvin_bridge, args.marvin_config, args.wuji_sdk, args.model)
    identity = {str(path.resolve()): file_sha(path) for path in native_assets}
    identity.update({"HMB_E1": MANIFEST_SHA, "HMB_CONFIG": CONFIG_SHA, "HMB_HANDOFF": HANDOFF_SHA, "M1": FIT_SHA, "M0": CAL_SHA})
    write_json(args.out / "source_hashes.json", identity)
    write_json(args.out / "input_execution.json", {
        "mode": "OFFLINE_ONLY", "witness_steps": args.witness_steps, "split": args.split, "max_partitions": args.max_partitions,
        "max_ticks": args.max_ticks, "one_partition_per_root": args.one_partition_per_root,
        "arm_step_rad": args.arm_step_rad, "hand_step_rad": args.hand_step_rad,
        "smooth_band_mm": args.smooth_band_mm, "reviewer_supplied_lineage_manifest_sha256":
        file_sha(args.extra_denylist) if args.extra_denylist else None})
    band = tuple(float(x) / 1000 for x in args.smooth_band_mm) if args.smooth_band_mm else None
    mapper = IntentMapper(model.geometry(model.home), calibrated, fit, smooth_band=band)
    envelope = radial_envelope(model)
    stack, marvin, hand = sdk_context(args.marvin_bridge, args.marvin_config, args.wuji_sdk)
    partitions = []
    causal_audit = {}
    with stack:
        nominal = Nominal(model, marvin, hand, np.repeat([args.arm_step_rad, args.hand_step_rad], [14, 40]),
                          NativeSweep(model))
        worst_position = max(x["position_error_m"] for x in nominal.validation)
        worst_angle = max(x["orientation_error_rad"] for x in nominal.validation)
        write_json(args.out / "nominal_bridge.json", {
            "native_wrist_from_model": [x.tolist() for x in nominal.wrist_to_native], "validation": nominal.validation,
            "position_max_m": worst_position, "angle_max_rad": worst_angle,
            "reference_frame": "SIGNED_SIMULATION_W", "reference_command": "CACHED_CAUSAL_NOMINAL_REFERENCE"})
        if worst_position > 0.0015 or worst_angle > np.deg2rad(0.5):
            raise ValueError(f"Marvin/MJCF wrist parity exceeded preflight tolerance: {worst_position:.5f} m, {worst_angle:.5f} rad")
        splits = SPLITS if args.split == "Both" else (args.split,)
        for split in splits:
            reader = HMBReader(e1 / "hmb_manifest.json", e1 / "production_plan.json", e0 / "candidate_config.json",
                               args.products, split, SIMULATED)
            selected_roots = set()
            count = 0
            for key in reader.keys():
                if args.max_partitions is not None and count >= args.max_partitions:
                    break
                raw, meta, refs, _ = reader.read(key)
                root = meta["identity"]["root_episode_id"]
                if args.one_partition_per_root and root in selected_roots:
                    continue
                selected_roots.add(root)
                count += 1
                source_hashes = scope_hashes(refs)
                denied = root in deny_roots or not source_hashes.isdisjoint(deny_assets)
                arrays, features, continuity, failures, witness_q, witness_valid, ik_failures = build_partition(raw, meta, model, mapper, nominal, envelope,
                    denied=denied, max_ticks=args.max_ticks, witness_steps=args.witness_steps)
                if args.verify_prefixes and root not in causal_audit:
                    ticks = len(arrays["tick_ns"])
                    changed = np.flatnonzero(np.diff(arrays["audit_generation"]) != 0)
                    cut = int(changed[0]) + 2 if len(changed) else ticks // 2
                    candidates = sorted({min(ticks - 1, cut), ticks // 2} - {0})
                    causal_audit[root] = []
                    for prefix in candidates:
                        if not 0 < prefix < ticks:
                            continue
                        shortened = {name: content[:prefix] for name, content in raw.items()}
                        trimmed, *_ = build_partition(shortened, meta, model, mapper, nominal, envelope,
                                                      denied=denied, max_ticks=prefix, witness_steps=0)
                        altered = {name: content.copy() for name, content in raw.items()}
                        for name in ("pose", "sdk_common_pose", "source_index", "age_ns", "pose_valid", "segment_id"):
                            altered[name][prefix:] = 0
                        replaced, *_ = build_partition(altered, meta, model, mapper, nominal, envelope,
                                                       denied=denied, max_ticks=prefix, witness_steps=0)
                        checked = [name for name in trimmed if not name.startswith(("domain_", "feasible_"))]
                        changed_fields = [name for name in checked
                                          if not (trimmed[name].tobytes() == replaced[name].tobytes()
                                                  == arrays[name][:prefix].tobytes())]
                        causal_audit[root].append({"partition": key, "prefix_ticks": prefix,
                                                   "boundary_candidate": prefix == cut,
                                                   "compared_fields": len(checked), "differences": changed_fields})
                    write_json(args.out / f"causal_{len(causal_audit):02d}.json", {"root": root, "controls": causal_audit[root]})
                    if any(row["differences"] for row in causal_audit[root]):
                        raise AssertionError(f"Native causal reference prefix changed for {root}")
                partitions.append((split, key, root, arrays, features, continuity, failures, sorted(source_hashes), witness_q, witness_valid, ik_failures))
                print(f"{split} partition {count}: {key}, {len(arrays['tick_ns'])} planned, "
                      f"{int(arrays['domain_nominal_legal'].sum())} nominal, "
                      f"{int(arrays['domain_improvable'].sum())} improvable", flush=True)
    train_values = [feature for split, _, _, _, feature, *_ in partitions if split == "Train"]
    train_values = np.concatenate(train_values) if train_values else np.empty((0, 4))
    usable = train_values[np.isfinite(train_values).all(axis=1)]
    thresholds = np.percentile(usable, 90, axis=0) if len(usable) else None
    output = []
    summaries = {}
    witnesses = {}
    continuity_log = {}
    static_windows, dynamic_windows = [], []
    native_failures = {}
    for i, (split, key, root, arrays, features, continuity, failures, source_hashes, witness_q, witness_valid, ik_failures) in enumerate(partitions):
        segments = arrays["audit_segment"]
        generations = arrays["audit_generation"]
        legal = arrays["domain_nominal_legal"]
        eligible = windows_json(legal, segments, generations)
        if thresholds is not None:
            arrays["domain_hard"] = threshold_hard(features, thresholds,
                                                    [(w["start"], w["stop"]) for w in eligible])
        location = Path("partitions") / split / f"{i:04d}"
        inventory = write_partition(args.out / location, arrays)
        output.append({"split": split, "key": key, "root": root, "relative_path": location.as_posix(),
                       "arrays": inventory, "source_asset_sha256": source_hashes, "legal_windows": eligible})
        counts = summaries.setdefault(split, Counter())
        counts["planned_ticks"] += len(legal)
        counts["hmb_joint_valid"] += int(arrays["inputs_pose_joint_valid"].sum())
        counts["mapped_valid"] += int(arrays["domain_mapped_valid"].sum())
        counts["legal_nominal"] += int(legal.sum())
        counts["five_frame_histories"] += sum(max(0, w["ticks"] - 4) for w in eligible)
        counts["improvable"] += int(arrays["domain_improvable"].sum())
        counts["hard_ticks"] += int(arrays["domain_hard"].sum())
        counts["static_5mm_witness_ticks"] += int(arrays["domain_static_witness_at"][:, 0].sum())
        counts["strictly_excluded_10mm"] += int(arrays["domain_strictly_excluded_at"][:, 2].sum())
        counts.update(failures)
        witness_folder = Path("diagnostics") / "witness_commands" / f"{i:04d}"
        witness_inventory = write_partition(args.out / witness_folder, {
            "q_witness": witness_q, "q_witness_valid": witness_valid})
        witnesses[key] = {
            "root": root, "commands_only_for_diagnostics": witness_folder.as_posix(),
            "command_arrays": witness_inventory, "improvable_windows": windows_json(arrays["domain_improvable"], segments, generations),
            "static_5mm_ticks": np.flatnonzero(arrays["domain_static_witness_at"][:, 0]).tolist(),
            "dynamic_10mm_status": "NOT_RUN_REQUIRES_ACTUAL_V1_B_SERVO_WITNESS"}
        for window in eligible:
            if window["ticks"] >= 25:
                dynamic_windows.append({"partition": key, "root": root, "split": split, **window,
                                        "D10": "NOT_RUN_REQUIRES_ACTUAL_V1_B_SERVO_WITNESS"})
        # Sample one target at the midpoint of every disjoint 25-tick eligible
        # block, independent of the static witness outcome or future policy.
        for window in eligible:
            for target_tick in range(window["start"] + 12, window["stop"] - 12, 25):
                static_windows.append({"partition": key, "root": root, "split": split, "target_tick": target_tick,
                    "legal_static_5mm_witness": bool(arrays["domain_static_witness_at"][target_tick, 0])})
        continuity_log[key] = continuity
        native_failures[key] = ik_failures
    write_json(args.out / "domain_counts.json", summaries)
    write_json(args.out / "causal_tests.json", {"tested_roots": len(causal_audit), "checks": causal_audit})
    write_json(args.out / "threshold_audit.json", {
        "hard_thresholds_train_p90": thresholds.tolist() if thresholds is not None else None,
        "switches": continuity_log,
        "hard_columns": ["wrist_speed_m_s", "wrist_speed_rad_s", "nominal_tip_m", "nominal_pinch_m"]})
    write_json(args.out / "witness_ledger.json", witnesses)
    write_json(args.out / "native_ik_failures.json", native_failures)
    write_json(args.out / "static_windows.json", {"candidate_targets": static_windows,
                                               "selection_stride_ticks": 25})
    write_json(args.out / "dynamic_windows.json", {"candidate_windows": dynamic_windows})
    take_all = (args.split == "Both" and args.max_partitions is None and
                args.max_ticks is None and not args.one_partition_per_root)
    if take_all and (len(output) != 308 or sum(row["planned_ticks"] for row in summaries.values()) != 116613):
        raise ValueError("Full production denominator differs from the signed HMB")
    write_json(args.out / "rmb_manifest.json", {
        "schema": SCHEMA, "qualification": "READY_FOR_REVIEW" if take_all else "SMOKE_ONLY",
        "approved_for_training": False, "scope": "HISTORICAL_SIMULATION_ONLY",
        "build_wall_s": monotonic() - start, "joint_names": model.names, "model": model.contract(),
        "blocked_pending": ["independent_qualification", "prefix_equivalence", "complete_lineage_assessment"],
        "identity": identity, "partitions": output})


if __name__ == "__main__":
    main()
