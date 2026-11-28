"""Compare the pinned native MANUS/Wuji callback with the signed offline hand refresh."""

import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes as ct
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np


REVISION = "6d0b0b5b2601618f1b0052d93ef2403eea9b064a"
TELEOP = Path("src/whole_body_teleop")
SDK_CPP = Path("src/wuji_sdk_cpp")
VENDOR = Path("third_party/wuji_c_api")
RETARGET = TELEOP / "src/retarget_node.cpp"
SMOOTHER = TELEOP / "include/manus_wuji_teleop/command_smoother.hpp"
DEFAULT_CONFIGS = [TELEOP / "config" / name for name in ("real.yaml", "sim.yaml")]
LOCK = VENDOR / "release-lock.json"
SDK_SOURCES = [SDK_CPP / "src" / name for name in ("retarget_session.cpp", "sdk_manager.cpp", "error.cpp")]
SIDES = ("left", "right")
STAGES = ("REJECTED", "MAPPED", "RETARGETED", "FILTERED", "PUBLISHED")
TICK_NS = 20_000_000
HMB_GRID_TOLERANCE_NS = 1_000


def checked_source(root, relative):
    expected = subprocess.check_output(["git", "-C", str(root), "show", f"{REVISION}:{relative.as_posix()}"])
    source = (root / relative).read_bytes()
    if expected != source:
        raise ValueError(f"Native source differs from {REVISION} at {relative}")
    return source


def tapped_callback(source):
    """Copy the original methods with deterministic clock and observation seams."""
    helpers = source.split("namespace {\n", 1)[1].split("\nRetargetNode::RetargetNode", 1)[0]
    methods = source.split("std::string RetargetNode::normalize_token", 1)[1]
    methods = "std::string RetargetNode::normalize_token" + methods.split("} // namespace manus_wuji_teleop", 1)[0]
    edits = (
        ("std::chrono::duration<double>(\n      std::chrono::steady_clock::now().time_since_epoch()).count()",
         "receive_time_sec_"),
        ("retarget_->step(keypoints(*msg));", "retarget_->step(tap_keypoints(*msg));\n    tap_raw(result.position);"),
        ("command_smoother_.update(result.position, receive_time_sec);",
         "command_smoother_.update(result.position, receive_time_sec);\n    tap_filtered(position);"),
        ("publisher_->publish(command);", "tap_publish(command);"),
    )
    for before, after in edits:
        if methods.count(before) != 1:
            raise ValueError("Pinned callback instrumentation point changed")
        methods = methods.replace(before, after)
    return "namespace {\n" + helpers + methods


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def native_build(root, build, cxx, sdk):
    header = root / VENDOR / "include/wuji_sdk.h"
    locks = json.loads(checked_source(root, LOCK))
    if digest(header) != locks["header_sha256"] or digest(sdk) != locks["library_sha256"]:
        raise ValueError("Wuji SDK header or library differs from the pinned release")
    files = (RETARGET, SMOOTHER, LOCK, *SDK_SOURCES, *DEFAULT_CONFIGS,
             *(p.relative_to(root) for p in sorted((root / SDK_CPP / "include/wuji_sdk_cpp").glob("*.hpp"))))
    source_hashes = {str(path): sha256(checked_source(root, path)).hexdigest() for path in files}
    build.mkdir()
    include = build / "parity_wuji_callback.inc"
    include.write_text(tapped_callback((root / RETARGET).read_text(encoding="utf-8")), encoding="utf-8")
    source = Path(__file__).with_suffix(".cpp").resolve()
    library = build / "libparity_wuji_probe.so"
    command = [cxx, "-std=c++20", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra", "-Wpedantic",
               "-I", str(build), "-I", str(root / TELEOP / "include"), "-I", str(root / SDK_CPP / "include"),
               "-I", str(header.parent), str(source), *(str(root / p) for p in SDK_SOURCES),
               str(sdk), f"-Wl,-rpath,{sdk.parent}", "-Wl,-z,defs", "-o", str(library)]
    subprocess.run(command, check=True)
    return library, {
        "teleop_revision": REVISION, "source_sha256": source_hashes,
        "wrapper_cpp_sha256": digest(source), "python_sha256": digest(Path(__file__)),
        "instrumented_callback_sha256": digest(include), "bridge_sha256": digest(library),
        "wuji_library": str(sdk), "wuji_release": locks, "build_command": command,
    }


def profile(root, parameters):
    import yaml

    merged = {}
    files = [root / p for p in DEFAULT_CONFIGS]
    if parameters is not None:
        files.append(parameters)
    for path in files:
        docs = yaml.safe_load(path.read_text(encoding="utf-8"))
        node = docs.get("/whole_body_teleop", docs.get("whole_body_teleop"))
        if node is None:
            raise ValueError(f"Missing whole_body_teleop parameters in {path}")
        merged.update(node["ros__parameters"])
    config = []
    for side in SIDES:
        config.append({key: merged[f"hand.{side}.{key}"] for key in (
            "manus_rotation_z_deg", "max_input_age_ms", "output_lp_alpha", "glove_id", "input_topic", "command_topic",
        )})
    return config, {str(path): digest(path) for path in files}


class RawNode(ct.Structure):
    _fields_ = [("node_id", ct.c_int32), ("chain_type", ct.c_char_p), ("joint_type", ct.c_char_p),
                ("xyz", ct.c_double * 3)]


class Output(ct.Structure):
    _fields_ = [("stage", ct.c_int32), ("ids", ct.c_int32 * 21), ("points", ct.c_float * 63),
                ("raw", ct.c_float * 20), ("filtered", ct.c_float * 20), ("error", ct.c_char * 512)]


class NativePair:
    def __init__(self, library, configs):
        self.library = lib = ct.CDLL(str(library))
        lib.s05_wuji_probe_create.argtypes = [ct.c_int, ct.c_double, ct.c_int, ct.c_double,
                                             ct.c_uint32, ct.POINTER(Output)]
        lib.s05_wuji_probe_create.restype = ct.c_void_p
        lib.s05_wuji_probe_destroy.argtypes = [ct.c_void_p]
        lib.s05_wuji_probe_destroy.restype = None
        lib.s05_wuji_probe_reset.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(Output)]
        lib.s05_wuji_probe_reset.restype = ct.c_int
        lib.s05_wuji_probe_step.argtypes = [ct.c_void_p, ct.POINTER(RawNode), ct.c_size_t, ct.c_char_p,
                                           ct.c_int32, ct.c_double, ct.c_int64, ct.c_int64, ct.c_int64,
                                           ct.c_int, ct.POINTER(Output)]
        lib.s05_wuji_probe_step.restype = ct.c_int
        self.handles = []
        try:
            for side, p in enumerate(configs):
                output = Output()
                handle = lib.s05_wuji_probe_create(side, p["manus_rotation_z_deg"], p["max_input_age_ms"],
                                                    p["output_lp_alpha"], p["glove_id"], ct.byref(output))
                if not handle:
                    raise RuntimeError(output.error.decode())
                self.handles.append(handle)
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles:
            self.library.s05_wuji_probe_destroy(handle)
        self.handles.clear()

    def reset(self, *, recreate=False):
        for handle in self.handles:
            output = Output()
            if self.library.s05_wuji_probe_reset(handle, recreate, ct.byref(output)):
                raise RuntimeError(output.error.decode())

    def step(self, row, receive_ns, *, reconstructed=True, gate_open=True):
        nodes = (RawNode * len(row["raw_nodes"]))(*(
            RawNode(n["node_id"], n["chain_type"].encode(), n["joint_type"].encode(), (ct.c_double * 3)(*n["xyz"]))
            for n in row["raw_nodes"]
        ))
        now_ns = 1_000_000_000 + receive_ns
        if reconstructed:
            header_ns, before_ns, after_ns = now_ns, now_ns, now_ns
        else:
            header_ns, before_ns, after_ns = (row[k] for k in (
                "callback_header_ns", "callback_ros_ns", "callback_after_ros_ns",
            ))
        message_side = row["declared_side"]
        output = Output()
        if self.library.s05_wuji_probe_step(
            self.handles[SIDES.index(row["side"])], nodes, len(nodes), message_side.encode(), row["glove_id"],
            receive_ns / 1_000_000_000, header_ns, before_ns, after_ns, gate_open, ct.byref(output),
        ):
            raise RuntimeError(output.error.decode())
        return {
            "stage": STAGES[output.stage], "published": output.stage == 4,
            "node_ids": list(output.ids) if output.stage >= 1 else None,
            "sdk_keypoints": np.ctypeslib.as_array(output.points).reshape(21, 3).tolist() if output.stage >= 1 else None,
            "raw_rad": list(output.raw) if output.stage >= 2 else None,
            "filtered_rad": list(output.filtered) if output.stage >= 3 else None,
            "callback_error": output.error.decode() or None,
        }


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def clock_of(tick):
    return tick.get("control_cutoff_ns", tick["tick_ns"])


def in_replay(tick):
    return tick.get("in_start_window") is not False


def bind_control_grid(ticks):
    anchor, index = None, 0
    for tick in ticks:
        if not in_replay(tick):
            continue
        if anchor is None:
            anchor = tick["tick_ns"]
            if not tick["nominal_pair_eligible"]:
                raise ValueError("First in-window signed row must be the valid START anchor")
        expected = anchor + index * TICK_NS
        if abs(tick["tick_ns"] - expected) > HMB_GRID_TOLERANCE_NS:
            raise ValueError(f"Unbridged HMB grid gap at {tick['partition']} tick {tick['partition_tick']}")
        tick["control_cutoff_ns"] = expected + HMB_GRID_TOLERANCE_NS
        index += 1
    if anchor is None:
        raise ValueError("No signed replay START anchor was supplied")


def check_inputs(events, ticks):
    by_id = {row["source_index"]: row for row in events}
    if not events or not ticks or len(by_id) != len(events):
        raise ValueError("Need unique MANUS events and at least one signed tick")
    for rows, clock in ((events, lambda row: row["receive_ns"]), (ticks, clock_of)):
        times = [clock(row) for row in rows]
        if times != sorted(times):
            raise ValueError("Input rows must follow the supplied causal schedule")
    for tick in ticks:
        for i, token in enumerate(tick["source_index"]):
            if token is not None and (by_id[token]["side"] != SIDES[i] or by_id[token]["receive_ns"] > clock_of(tick)):
                raise ValueError(f"Impossible raw source association at {tick['partition']} tick {tick['partition_tick']}")
    return by_id


def scheduled_a(native, events, out, *, actual_clocks=False):
    earliest = min(0, events[0]["receive_ns"])
    a = []
    with out.open("x", encoding="utf-8") as stream:
        for event in events:
            probe = native.step(event, event["receive_ns"] - earliest, reconstructed=not actual_clocks,
                                gate_open=event.get("callback_gate_open", True) if actual_clocks else True)
            row = {key: event[key] for key in ("source_index", "side", "receive_ns")}
            row.update(probe)
            row["in_recorded_interval"] = event.get("in_recorded_interval")
            write_row(stream, row)
            a.append(row)
    return a


def masked_source(tick):
    if not in_replay(tick) or not tick["nominal_pair_eligible"]:
        return [None, None]
    return [token if valid else None for token, valid in zip(tick["source_index"], tick["pose_valid"], strict=True)]


def selected_native(native, ticks, by_id, *, partition_resets):
    result = []
    previous_partition = None
    previous_tokens = [None, None]
    previous_outputs = [None, None]
    earliest = min(0, clock_of(ticks[0]))
    native.reset()
    for tick in ticks:
        partition = tick["partition"]
        if partition_resets and in_replay(tick) and previous_partition is not None and partition != previous_partition:
            native.reset()
            previous_tokens = [None, None]
            previous_outputs = [None, None]
        if in_replay(tick):
            previous_partition = partition
        tokens = masked_source(tick)
        for side, token in enumerate(tokens):
            if token is not None and token != previous_tokens[side]:
                previous_outputs[side] = native.step(by_id[token], clock_of(tick) - earliest)
                previous_tokens[side] = token
        result.append([previous_outputs[i] if token is not None else None for i, token in enumerate(tokens)])
    return result


class HandTap:
    def __init__(self, hand):
        self.hand = hand
        self.control = None
        self.points = [None, None]

    def reset(self):
        self.points = [None, None]
        self.hand.reset()

    def step(self, side, points):
        self.points[side] = points.copy()
        if self.control is None:
            return self.hand.step(side, points)
        injected = self.control[side]
        if injected is None:
            from dexgym.tasks.direct.wholebody_s05.v1.rmb.nominal import SDKFailure
            raise SDKFailure("Original callback did not reach the Wuji SDK seam")
        return self.hand.step(side, np.ascontiguousarray(injected, dtype=np.float32))


def selected_python(hand, ticks, *, control=None):
    from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineTeleopNominal

    tap = HandTap(hand)
    # The real refresh depends only on hand state and the clock after reset.
    # Identity frames and a minimal Home stub keep arm IK outside this probe.
    robot = SimpleNamespace(bounds=np.tile([-1.0, 1.0], (54, 1)))
    kernel = SimpleNamespace(home_reference=lambda q: np.zeros(2))
    nominal = OfflineTeleopNominal(robot, kernel, tap, np.repeat(np.eye(4)[None], 7, axis=0))
    nominal.reset(np.zeros(54))
    result = []
    for index, tick in enumerate(ticks):
        accessible = in_replay(tick) and tick["nominal_pair_eligible"]
        if accessible:
            tokens = masked_source(tick)
            common = [np.asarray(x) if x is not None else None for x in tick["common"]]
            order = [np.asarray(x) if x is not None else None for x in tick["order"]]
            if control is not None:
                tap.control = [row["sdk_keypoints"] if row is not None else None for row in control[index]]
            nominal.refresh(None, common, order, tokens, tick_ns=clock_of(tick))
        result.append([
            {"valid": bool(nominal.hand_valid[i] and accessible),
             "raw_rad": nominal.raw_hand[i].tolist() if nominal.hand_valid[i] and accessible else None,
             "filtered_rad": nominal.filtered_hand[i].tolist() if nominal.hand_valid[i] and accessible else None,
             "sdk_keypoints": tap.points[i].tolist() if tap.points[i] is not None and accessible else None,
             "failure_reason": nominal.hand_failure_reason[i] if accessible else None}
            for i in range(2)
        ])
    return result


def row_error(left, right):
    if left is None or right is None:
        return None
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def summarize(values):
    finite = np.asarray([x for x in values if x is not None])
    if not len(finite):
        return {"n": 0}
    return {"n": len(finite), "exact": int(np.count_nonzero(finite == 0)), "max": float(finite.max()),
            "p50": float(np.quantile(finite, 0.50)), "p95": float(np.quantile(finite, 0.95)),
            "p99": float(np.quantile(finite, 0.99))}


def rates(native, events, reference, factors):
    result = {}
    origin = events[0]["receive_ns"]
    for factor in factors:
        native.reset(recreate=True)
        errors = {kind: [[], []] for kind in ("raw_rad", "filtered_rad")}
        for event, a in zip(events, reference, strict=True):
            probe = native.step(event, int(round((event["receive_ns"] - origin) / factor)))
            side = SIDES.index(event["side"])
            for field in errors:
                errors[field][side].append(row_error(probe[field], a[field]))
        result[str(factor)] = {key: [summarize(values) for values in pair] for key, pair in errors.items()}
    return result


def pair_output(path, by_id, a, ticks, b, b_partition, c, control):
    latest = [None, None]
    event_index = 0
    errors = {key: [[], []] for key in (
        "pre_rotation_m", "sdk_input_m", "same_sdk_input_raw_rad", "selected_native_vs_offline_raw_rad",
        "selected_native_vs_offline_filtered_rad", "full_vs_selected_root_raw_rad",
        "full_vs_selected_root_filtered_rad", "selected_root_vs_partition_raw_rad",
        "selected_root_vs_partition_filtered_rad", "same_token_full_vs_selected_root_raw_rad",
        "same_token_full_vs_selected_root_filtered_rad",
    )}
    with path.open("x", encoding="utf-8") as stream:
        for tick, selected, partitioned, offline, exact in zip(ticks, b, b_partition, c, control, strict=True):
            cutoff = clock_of(tick)
            while event_index < len(a) and a[event_index]["receive_ns"] <= cutoff:
                entry = a[event_index]
                if entry["published"]:
                    latest[SIDES.index(entry["side"])] = entry
                event_index += 1
            row = {key: tick[key] for key in ("partition", "partition_tick", "tick_ns", "source_index", "pose_valid",
                         "joint_valid", "safety_permission", "nominal_pair_eligible", "history_eligible",
                         "in_start_window", "arrival_ns") if key in tick}
            row["control_cutoff_ns"] = cutoff
            row["a_source_index"] = [value["source_index"] if value is not None else None for value in latest]
            row["a_receive_ns"] = [value["receive_ns"] if value is not None else None for value in latest]
            row["a_valid"] = [value is not None for value in latest]
            row["b_valid"] = [x is not None and x["published"] for x in selected]
            row["c_valid"] = [x["valid"] for x in offline]
            row["c_failure_reason"] = [x["failure_reason"] for x in offline]
            for prefix, values in (("a", latest), ("b", selected), ("b_partition", partitioned),
                                   ("c", offline), ("c_exact", exact)):
                for field in ("raw_rad", "filtered_rad", "sdk_keypoints"):
                    row[f"{prefix}_{field}"] = [x[field] if x is not None else None for x in values]
            for side, token in enumerate(masked_source(tick)):
                if token is None:
                    continue
                for name, left, right in (
                    ("sdk_input_m", row["b_sdk_keypoints"], row["c_sdk_keypoints"]),
                    ("same_sdk_input_raw_rad", row["b_raw_rad"], row["c_exact_raw_rad"]),
                    ("selected_native_vs_offline_raw_rad", row["b_raw_rad"], row["c_raw_rad"]),
                    ("selected_native_vs_offline_filtered_rad", row["b_filtered_rad"], row["c_filtered_rad"]),
                    ("full_vs_selected_root_raw_rad", row["a_raw_rad"], row["b_raw_rad"]),
                    ("full_vs_selected_root_filtered_rad", row["a_filtered_rad"], row["b_filtered_rad"]),
                    ("selected_root_vs_partition_raw_rad", row["b_raw_rad"], row["b_partition_raw_rad"]),
                    ("selected_root_vs_partition_filtered_rad", row["b_filtered_rad"], row["b_partition_filtered_rad"]),
                ):
                    errors[name][side].append(row_error(left[side], right[side]))
                if selected[side] is not None and selected[side]["node_ids"] is not None:
                    nodes = {node["node_id"]: node["xyz"] for node in by_id[token]["raw_nodes"]}
                    exact_xyz = [nodes[node_id] for node_id in selected[side]["node_ids"]]
                    signed_xyz = np.asarray(tick["common"][side])[tick["order"][side]]
                    errors["pre_rotation_m"][side].append(row_error(exact_xyz, signed_xyz))
                if latest[side] is not None and latest[side]["source_index"] == token:
                    for field in ("raw_rad", "filtered_rad"):
                        error = row_error(row[f"a_{field}"][side], row[f"b_{field}"][side])
                        errors[f"same_token_full_vs_selected_root_{field}"][side].append(error)
            write_row(stream, row)
    return {key: [summarize(values) for values in pair] for key, pair in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--ticks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wuji", type=Path)
    parser.add_argument("--effective-params", type=Path)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--rate", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument("--actual-callback-clocks", action="store_true",
                        help="Require actual before/after ROS clocks and original headers on every raw event")
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.rate):
        parser.error("Rate multipliers must be positive and finite")
    root = args.teleop.resolve()
    out = args.out.resolve()
    sdk = (args.wuji or root / VENDOR / "lib/libwuji_sdk_c.so").resolve()
    events, ticks = read_jsonl(args.events), read_jsonl(args.ticks)
    bind_control_grid(ticks)
    by_id = check_inputs(events, ticks)
    configs, profile_hashes = profile(root, args.effective_params)
    out.mkdir(parents=True)
    library, provenance = native_build(root, out / "build", args.cxx, sdk)

    with ExitStack() as stack:
        native = NativePair(library, configs)
        stack.callback(native.close)
        a = scheduled_a(native, events, out / "wuji-a-updates.jsonl", actual_clocks=args.actual_callback_clocks)
        native.reset(recreate=True)
        b = selected_native(native, ticks, by_id, partition_resets=False)
        native.reset(recreate=True)
        b_partition = selected_native(native, ticks, by_id, partition_resets=True)
        rate_summary = rates(native, events, a, args.rate)

    from dexgym.tasks.direct.wholebody_s05.v1.rmb.nominal import Wuji

    with ExitStack() as stack:
        hand = Wuji(sdk)
        stack.callback(hand.close)
        c = selected_python(hand, ticks)
        control = selected_python(hand, ticks, control=b)

    summary = {
        "provenance": provenance, "profile_by_side": configs, "profile_sha256": profile_hashes,
        "source_sha256": {"event_jsonl": digest(args.events), "selection_jsonl": digest(args.ticks)},
        "raw_event_clock": "SUPPLIED_CALLBACK_CLOCKS" if args.actual_callback_clocks else "RECONSTRUCTED_COLLECTOR_ORDER",
        "raw_event_gate": "SUPPLIED_CALLBACK_GATE" if args.actual_callback_clocks else "DIAGNOSTIC_OPEN",
        "selected_event_clock": "SIGNED_CONTROL_CUTOFF_WITH_IMMEDIATE_RESTAMP_AND_OPEN_GATE",
        "selected_session_lifecycle": "ONE_RESET_AT_START_ANCHOR_PER_ROOT; B_PARTITION IS RESET SENSITIVITY",
        "raw_event_callback_status": Counter((row["side"] + "/" + row["stage"]) for row in a),
        "signed_tick_count": len(ticks), "signed_pair_eligible_ticks": sum(t["nominal_pair_eligible"] for t in ticks),
        "inputs_in_scope": {"raw_events": len(events), "signed_ticks": len(ticks)},
        "order": "Hand2 joint names, left then right in paired rows",
        "summary_axis": "maximum absolute coordinate or joint error per paired hand/tick",
        "repeat_cold_start_rate_multiplier_errors": rate_summary,
        "comparisons": pair_output(out / "wuji-bc-ticks.jsonl", by_id, a, ticks, b, b_partition, c, control),
    }
    with (out / "wuji-summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
