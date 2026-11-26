"""Read-only full-Train upper bound on exact frozen-M1 wrist IK coverage."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...p02.model import RobotModel, asset_files
from .mapping import IntentMapper
from .nominal import GROUPS, Marvin, matrix, model_wrist
from .probe_nominal import PHASE1_REVISION
from .schema import file_sha, write_json


NATIVE_CONFIG_SHA = "ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52"
NATIVE_VENDOR_SHA = "f2aec6ebc1f41302fe0126284076211c044d6c7e1c2276dc919767254e0c597f"
BOUND_MARGIN_M = 1e-6
MIN_COVERAGE = 0.85


def radius_excess(workspace, outer):
    """Use the unprojected wrist radius and uncapped two-link length."""
    if workspace is None:
        return None
    radius = workspace["radius_request_m"]
    if not np.isfinite(radius) or radius < 0 or workspace["radius_max_m"] != outer:
        raise ValueError("Native workspace returned inconsistent radius evidence")
    return radius - outer


def wrist_scope(raw, denied):
    joint_valid = raw["joint_valid"]
    clock_eligible = ((raw["source_index"][:, :4, 0] >= 0)
                      & (raw["capture_bounds_ns"][:, :4, 1] <= raw["tick_ns"][:, None])
                      & (raw["arrival_bounds_ns"][:, :4, 1] <= raw["cutoff_ns"][:, None]))
    return (joint_valid & raw["safety_permission"] & raw["pose_valid"][:, :4].all(axis=1)
            & clock_eligible.all(axis=1) & (raw["segment_id"] >= 0) & (not denied))


def wrist_bridge(model, marvin):
    """Match the home-based transform and FK witness in nominal.py."""
    home = model.geometry(model.home)
    bridges, witnesses = [], []
    for side, body in enumerate(("Base_L", "Base_R")):
        group = GROUPS[side]
        body_pose = model.data.body(body)
        base = matrix(body_pose.xpos, body_pose.xmat.reshape(3, 3))
        native_home = marvin.fk(side, model.home[group])
        wrist_to_native = np.linalg.solve(model_wrist(home, side), base @ native_home)
        home_workspace = marvin.workspace_probe(side, native_home)
        if home_workspace is None:
            raise ValueError(f"No native workspace for reachable home on side {side}")
        outer = home_workspace["radius_max_m"]
        if not np.isfinite(outer) or outer <= 0:
            raise ValueError(f"Invalid native outer radius for side {side}")
        home_excess = radius_excess(home_workspace, outer)
        if home_excess > BOUND_MARGIN_M:
            raise ValueError(f"Native home violates nominal radius bound on side {side}")
        position_error_m, rotation_error_rad, max_fk_radius_m = 0., 0., home_excess + outer
        for axis in range(group.start, group.stop):
            for pulse in (-0.04, 0.04):
                q = model.home.copy()
                q[axis] += pulse
                native = marvin.fk(side, q[group])
                mapped = np.linalg.solve(base, model_wrist(model.geometry(q), side) @ wrist_to_native)
                position_error_m = max(position_error_m, float(np.linalg.norm(native[:3, 3] - mapped[:3, 3])))
                angle = Rotation.from_matrix(native[:3, :3].T @ mapped[:3, :3]).magnitude()
                rotation_error_rad = max(rotation_error_rad, float(angle))
                excess = radius_excess(marvin.workspace_probe(side, native), outer)
                if excess is None or excess > BOUND_MARGIN_M:
                    raise ValueError(f"Known native FK witness violates nominal radius bound on side {side}")
                max_fk_radius_m = max(max_fk_radius_m, excess + outer)
        if position_error_m > .0015 or rotation_error_rad > np.deg2rad(.5):
            raise ValueError(f"Native/MJCF wrist transform differs on side {side}")
        bridges.append((base, wrist_to_native, outer))
        witnesses.append({
            "side": side, "outer_radius_m": outer, "known_fk_radius_max_m": max_fk_radius_m,
            "fixture_count": 15, "position_error_max_m": position_error_m, "rotation_error_max_rad": rotation_error_rad,
        })
    return bridges, witnesses


def audit_partition(raw, denied, mapper, marvin, bridges):
    counts = Counter(
        planned_ticks=len(raw["tick_ns"]),
        hmb_joint_valid_ticks=int(raw["joint_valid"].sum()),
        denied_hmb_joint_valid_ticks=int(raw["joint_valid"].sum()) if denied else 0,
    )
    rows = np.flatnonzero(wrist_scope(raw, denied))
    counts["rmb_input_mask_eligible_ticks"] = len(rows)
    no_hand_shape = np.zeros((2, 5, 4, 3))
    for row in rows:
        try:
            # Wrist geometry in place() is independent of the hand landmarks.
            target = mapper.place("M1", raw["pose"][row, 2:4, 0], no_hand_shape)
            if not np.isfinite(target["wrist_xyz"]).all():
                raise ValueError("Nonfinite mapped wrist")
            orientations = Rotation.from_quat(target["wrist_xyzw"]).as_matrix()
        except ValueError:
            counts["wrist_mapping_unknown_ticks"] += 1
            continue
        excluded = []
        unknown = False
        for side, (base, wrist_to_native, outer) in enumerate(bridges):
            world = matrix(target["wrist_xyz"][side], orientations[side])
            wanted = np.linalg.solve(base, world @ wrist_to_native)
            workspace = marvin.workspace_probe(side, wanted)
            excess = radius_excess(workspace, outer)
            if excess is None:
                unknown = True
                counts[f"workspace_unknown_side_{side}_ticks"] += 1
                continue
            counts[f"workspace_known_side_{side}_ticks"] += 1
            if excess > BOUND_MARGIN_M:
                excluded.append(side)
                counts[f"strict_outside_side_{side}_ticks"] += 1
            elif workspace["offset_m"] > 1e-12:
                counts[f"inside_adjusted_by_controller_side_{side}_ticks"] += 1
        if excluded:
            counts["strict_outside_either_side_ticks"] += 1
            counts["strict_outside_both_sides_ticks"] += len(excluded) == 2
        elif unknown:
            counts["not_excluded_but_workspace_unknown_ticks"] += 1
        else:
            counts["both_sides_known_not_excluded_ticks"] += 1
    return counts


def upper_bound(counts):
    planned, valid = counts["planned_ticks"], counts["hmb_joint_valid_ticks"]
    strict = counts["strict_outside_either_side_ticks"]
    upper = valid - strict
    return {
        **counts,
        "exact_direct_ik_hmb_joint_valid_ticks_upper": upper,
        "exact_direct_ik_hmb_joint_valid_fraction_upper": upper / valid if valid else None,
        "exact_direct_ik_planned_fraction_upper": upper / planned if planned else None,
        "upper_bound_below_85_percent_of_hmb_joint_valid": 100 * upper < 85 * valid if valid else None,
    }


def signed_inputs(args):
    from ...hmb_reader import CONFIG_SHA, MANIFEST_SHA, PLAN_SHA, HMBReader, SIMULATED, checked_json
    from .build import CAL_SHA, FIT_SHA, HANDOFF_SHA, MODEL_MANIFEST_SHA, MODEL_SHA

    docs = args.docs / "evidence"
    p02 = docs / "P02"
    handoff = checked_json(docs / "P01_F/p02_handoff_manifest.json", HANDOFF_SHA)
    if args.products.resolve() != Path(handoff["ubuntu_products"]).resolve():
        raise ValueError("HMB products must be the signed official archive")
    for path, expected in ((args.model, MODEL_SHA), (args.marvin_config, NATIVE_CONFIG_SHA),
                           (args.marvin_vendor, NATIVE_VENDOR_SHA), (args.marvin_bridge, args.marvin_bridge_sha256)):
        if file_sha(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")
    signed_model = checked_json(p02 / "refinement/model_assets_sha256.json", MODEL_MANIFEST_SHA)
    archive_assets = {name.split("/generated/")[-1]: digest for name, digest in signed_model.items()}
    model_dir = args.model.resolve().parent
    actual_assets = {Path(name).relative_to(model_dir).as_posix(): digest for name, digest in asset_files(args.model).items()}
    if actual_assets != archive_assets:
        raise ValueError("Actual MJCF dependencies differ from the signed model")
    fit = checked_json(p02 / "refinement/fitted_candidates.json", FIT_SHA)
    calib = checked_json(p02 / "mapping_contract.json", CAL_SHA)
    e1 = docs / "P01_E_E1"
    reader = HMBReader(e1 / "hmb_manifest.json", e1 / "production_plan.json",
                       docs / "P01_E_E0/candidate_config.json", args.products, "Train", SIMULATED)
    identity = {
        "HMB_E1": MANIFEST_SHA, "HMB_PLAN": PLAN_SHA, "HMB_CONFIG": CONFIG_SHA, "HMB_HANDOFF": HANDOFF_SHA,
        "M1": FIT_SHA, "M0_CALIBRATION": CAL_SHA, "model": MODEL_SHA, "model_assets": MODEL_MANIFEST_SHA,
        "marvin_config": NATIVE_CONFIG_SHA, "marvin_vendor": NATIVE_VENDOR_SHA,
        "marvin_bridge": args.marvin_bridge_sha256, "teleop_revision": PHASE1_REVISION,
        "teleop_marvin_kine_sdk_cpp_blob": "86f841f28099af7949da70a99806795c770f9ee6",
    }
    return reader, fit, calib, identity


def audit(args):
    from .build import REGRESSION_SHA, scope_hashes

    reader, fit, calib, identity = signed_inputs(args)
    model = RobotModel(args.model)
    mapper = IntentMapper(model.geometry(model.home), calib, fit)
    marvin = Marvin(args.marvin_bridge, args.marvin_config)
    totals, sources, partitions = Counter(), {}, []
    try:
        vendor = str(args.marvin_vendor.resolve())
        mapped_libraries = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()}
        if vendor not in mapped_libraries:
            raise ValueError("Native Marvin bridge did not load the verified vendor library")
        bridges, witnesses = wrist_bridge(model, marvin)
        for key in reader.keys():
            raw, meta, references, _ = reader.read(key)
            root = meta["identity"]["root_episode_id"]
            denied = not scope_hashes(references).isdisjoint(REGRESSION_SHA)
            counts = audit_partition(raw, denied, mapper, marvin, bridges)
            totals.update(counts)
            sources.setdefault(root, Counter()).update(counts)
            partitions.append({"partition": key, "root": root, **upper_bound(counts)})
            if len(partitions) % 20 == 0:
                print(f"Audited {len(partitions)} Train partitions", flush=True)
    finally:
        marvin.close()
    signed_train = reader.manifest["counts"]["Train"]
    signed_counts = signed_train["counts"]
    observed = (len(sources), len(partitions), totals["planned_ticks"], totals["hmb_joint_valid_ticks"])
    expected = (signed_train["source_count"], signed_counts["partitions"], signed_counts["planned_ticks"],
                signed_counts["causal.joint_valid"])
    if observed != expected:
        raise ValueError(f"Full Train scope differs from signed denominators: {observed} != {expected}")
    return {
        "scope": "DIAGNOSTIC_ONLY_EXACT_M1_WRIST_IK", "split": "Train", "full_signed_split": True,
        "bound": "||p_base_tcp - R_base_tcp @ tcp_tool_offset - shoulder|| <= upper_link + lower_link",
        "bound_semantics": "Triangle inequality on the verified CCS shoulder-to-elbow and elbow-to-wrist links. "
                           "Necessary for the exact requested pose. Unreachable rows only decrease the upper bound; "
                           "unknown rows still count as possible. Neither finger reach nor projected tracking is scored.",
        "strict_margin_m": BOUND_MARGIN_M, "design_coverage_reference": MIN_COVERAGE,
        "identity": identity,
        "bridge_fk_witnesses": witnesses, "totals": upper_bound(totals),
        "per_root": {root: upper_bound(rows) for root, rows in sources.items()}, "partitions": partitions,
        "training_qualification": False, "published_rmb": False,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    cli.add_argument("--products", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--marvin-bridge", type=Path, required=True)
    cli.add_argument("--marvin-bridge-sha256", required=True)
    cli.add_argument("--marvin-config", type=Path, required=True)
    cli.add_argument("--marvin-vendor", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    write_json(args.out, audit(args))


if __name__ == "__main__":
    main()
