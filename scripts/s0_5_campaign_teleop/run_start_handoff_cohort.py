#!/usr/bin/env python3
"""Run the strict measured START handoff across the signed 32-root simulation cohort."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import signal
import subprocess
import sys
from threading import Event
import time

from hmb_cohort import EXPECTED_SPLITS, cohort as signed_experiment
from run_nominal_cohort import terminate


DOMAINS = tuple(range(206, 214))
FRAMES_SHA = "33cbc93c4a2db925986f596b4b1730c7eaca026a9facfdf26aa4aaa8032c8f94"
MODEL_SHA = "be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d"
HISTORICAL = "historical_human_inputs"
SCRIPTS = Path(__file__).resolve().parent


def digest(path, opener=open):
    with opener(path, "rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def save(path, data):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verified_roots(path, docs):
    receipt = json.loads(path.read_bytes())
    signed = signed_experiment(docs)
    local_keys = {"selected_roots", "registry_checked", "native_assets_sha256_checked"}
    if (any(receipt.get(key) != value for key, value in signed.items() if key not in local_keys)
            or receipt.get("registry_checked") is not True or receipt.get("native_assets_sha256_checked") is not True):
        raise ValueError("Expected the original full byte-verified, signed HMB cohort receipt")
    actual = {root["root"]: root for root in receipt["selected_roots"]}
    if len(actual) != len(receipt["selected_roots"]) or len(actual) != sum(split[0] for split in EXPECTED_SPLITS.values()):
        raise ValueError("Expected 32 unique signed roots")
    canonical_uris = set()
    for expected in signed["selected_roots"]:
        root = actual[expected["root"]]
        if (any(root.get(key) != value for key, value in expected.items() if key not in {"inputs", "canonical_episode_uri"})
                or not Path(root["canonical_episode_uri"]).is_absolute() or len(root["inputs"]) != 4):
            raise ValueError(f"Signed root differs: {expected['root']}")
        for asset, signed_asset in zip(root["inputs"], expected["inputs"], strict=True):
            uri = asset["canonical_uri"]
            if (any(asset.get(key) != value for key, value in signed_asset.items() if key not in {"canonical_uri", "asset_check"})
                    or asset["asset_check"] != "SHA256_AND_SIZE_MATCH" or not Path(uri).is_absolute()
                    or uri in canonical_uris):
                raise ValueError(f"Signed native input differs: {root['root']} {signed_asset['role']}")
            canonical_uris.add(uri)
    return actual


def check_anchor(path, root, cohort_sha):
    data = path.read_bytes()
    anchor = json.loads(data)
    partition = next((part for part in root["hmb_partitions"] if part["key"] == anchor["partition"]), None)
    tick = anchor["partition_tick"]
    if (anchor["schema"] != "s05.teleop.preposition_start_handoff.v1"
            or anchor["scope"] != "SIGNED_HUMAN_INPUT_ANCHOR_SIMULATION_DIAGNOSTIC"
            or anchor["cohort_sha256"] != cohort_sha or anchor["root"] != root["root"]
            or anchor["split"] != root["split"] or anchor["source_manifest_sha256"] != root["manifest_sha256"]
            or partition is None or type(tick) is not int or not 0 <= tick < partition["planned_ticks"]
            or type(anchor["source_cutoff_ns"]) is not int or type(anchor["hmb_cutoff_ns"]) is not int
            or not 0 <= anchor["hmb_cutoff_ns"] < anchor["source_cutoff_ns"]
            or set(anchor["hmb_geometry"]) != {asset["role"] for asset in root["inputs"]}
            or set(anchor["inputs"]) != {asset["topic"] for asset in root["inputs"]}):
        raise ValueError("START anchor identity or signed partition differs")
    for value in anchor["inputs"].values():
        if (type(value["record_index"]) is not int or value["record_index"] < 0
                or value["record_index"] % 2 != 0 or not re.fullmatch(r"[0-9a-f]{64}", value["cdr_sha256"])):
            raise ValueError("START anchor has an invalid native physical record or CDR digest")
    return data, anchor


def clean_env(domain, attempt, frames):
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOG_DIR=str(attempt / "ros"),
               ROS_LOCALHOST_ONLY="1", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
               S05_SHARED_RIGID_MAP=str(frames), PYTHONDONTWRITEBYTECODE="1")
    for key in ("ROS_STATIC_PEERS", "CYCLONEDDS_URI", "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE"):
        env.pop(key, None)
    return env


def graph_empty(env, log, attempts=1):
    command = ["ros2", "node", "list", "--no-daemon", "--spin-time", "1.5"]
    with log.open("a", encoding="utf-8") as stream:
        for _ in range(attempts):
            check = subprocess.run(command, env=env, capture_output=True, text=True, timeout=8, check=False)
            stream.write(f"exit={check.returncode}\n{check.stdout}{check.stderr}\n")
            if check.returncode == 0 and not check.stdout.strip():
                return True
    return False


def group_exists(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_owned_group(process):
    if not group_exists(process.pid):
        process.poll()
        return True
    try:
        terminate(process)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not group_exists(process.pid):
            return True
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        deadline = time.monotonic() + 3
        while group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(.1)
    return not group_exists(process.pid)


def wait_owned(process, timeout, cancelled):
    deadline = time.monotonic() + timeout
    while not cancelled.is_set():
        now = time.monotonic()
        if now >= deadline:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(.5, deadline - now))
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("Campaign interrupted; terminating owned processes")


def trace_receipt(path):
    history, historical_timebases, starts, stops = 0, 0, 0, 0
    handoff, model, seal = None, None, None
    before = -1
    with path.open("rb") as source:
        for record in source:
            if not record.endswith(b"\n"):
                raise ValueError("Incomplete JSONL tail")
            row = json.loads(record)
            if row["mono_ns"] < before:
                raise ValueError("Native observer receipt time regressed")
            before = row["mono_ns"]
            if row["kind"] == "model":
                if model is not None or row["sha256"] != MODEL_SHA:
                    raise ValueError("Live trace model differs from the verified 03 B MuJoCo model")
                model = row["sha256"]
            if row["kind"] == "start_handoff":
                if (handoff is not None or row["stage"] != "preposition_start" or row["continue_after_miss"] is not False
                        or row["status"] not in {"START_REACHED", "START_NOT_REACHED"} or model is None or starts != 1):
                    raise ValueError("Invalid native START handoff decision or missing model/START provenance")
                handoff = row
            if row["stage"] == HISTORICAL:
                if handoff is None or handoff["status"] != "START_REACHED":
                    raise ValueError("Primary replay entered history before the strict native START decision")
                if row["kind"] == "input" and seal is not None:
                    raise ValueError("Historical input arrived after its final seal")
                history += row["kind"] == "input"
                historical_timebases += row["kind"] == "timebase"
            if row["kind"] == "historical_input_seal":
                expected = row["expected_cdr_sha256"]
                if (seal is not None or row["stage"] != HISTORICAL or type(row["messages"]) is not int
                        or not 0 < row["messages"] == history or not re.fullmatch(r"[0-9a-f]{64}", expected)
                        or expected != row["published_cdr_sha256"]):
                    raise ValueError("Historical input seal differs from the recorded replay count or CDR digest")
                seal = row
            if row["kind"] == "service":
                response = row["response"]
                accepted = response.get("success", response.get("accepted", False)) is True
                starts += accepted and row.get("service") == "/whole_body_teleop/start"
                stops += accepted and row.get("service") == "/whole_body_teleop/stop"
    status = handoff["status"] if handoff else "NO_HANDOFF_RECEIPT"
    return {"start_status": status, "start_handoff": handoff, "historical_input_count": history,
            "historical_timebase_count": historical_timebases, "accepted_start_count": starts,
            "accepted_stop_count": stops, "loaded_model_sha256": model, "historical_input_seal": seal}


def archive(trace):
    compressed = trace.with_suffix(".jsonl.gz")
    provisional = compressed.with_suffix(".gz.partial")
    with trace.open("rb") as source, gzip.open(provisional, "wb", compresslevel=1) as output:
        shutil.copyfileobj(source, output)
    raw_hash = digest(trace)
    if digest(provisional, gzip.open) != raw_hash:
        raise ValueError("Compressed trace did not round-trip; original retained")
    provisional.replace(compressed)
    trace.unlink()
    return {"trace": str(compressed), "trace_sha256": digest(compressed), "trace_uncompressed_sha256": raw_hash}


def run_attempt(args, root, domain, i, output, cohort_sha, cancelled):
    attempt = output / f"attempt-{i:02d}"
    attempt.mkdir()
    result_path = attempt / "result.json"
    result = {"root": root["root"], "split": root["split"], "planned_ticks": root["planned_ticks"],
              "domain": domain, "attempt": i, "directory": str(attempt), "start_status": "NO_HANDOFF_RECEIPT",
              "result": "FAILED_BEFORE_START_DECISION", "started_unix_ns": time.time_ns(), "domain_reusable": True}
    save(result_path, result)
    env = clean_env(domain, attempt, args.shared_frames)
    anchor = attempt / "anchor.json"
    prep = sim = replay = None
    try:
        if args.anchors is not None:
            prior = args.anchors / f"{root['root'].split(':')[-1]}.json"
            result["precomputed_anchor"] = str(prior)
            if not prior.is_file():
                result["result"] = "MISSING_START_ANCHOR"
                raise ValueError("No verified precomputed START anchor for this planned root")
            anchor_bytes, evidence = check_anchor(prior, root, cohort_sha)
            anchor.write_bytes(anchor_bytes)
        else:
            command = [str(args.anchor_python), "-B", str(SCRIPTS / "prepare_start_handoff.py"),
                       "--cohort", str(args.cohort), "--root", root["root"], "--out", str(anchor), "--docs", str(args.docs)]
            result["anchor_command"] = command
            save(result_path, result)
            with (attempt / "anchor.log").open("w") as log:
                prep = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                anchor_exit = wait_owned(prep, 180, cancelled)
                if anchor_exit:
                    raise subprocess.CalledProcessError(anchor_exit, command)
            anchor_bytes, evidence = check_anchor(anchor, root, cohort_sha)
        result.update(anchor_sha256=sha256(anchor_bytes).hexdigest(), anchor_partition=evidence["partition"],
                      anchor_partition_tick=evidence["partition_tick"], anchor_source_cutoff_ns=evidence["source_cutoff_ns"])
        result["graph_empty_before"] = graph_empty(env, attempt / "preflight.log")
        if not result["graph_empty_before"]:
            result["domain_reusable"] = False
            raise RuntimeError("Cannot establish a clean isolated ROS domain before launch")
        if cancelled.is_set():
            raise RuntimeError("Campaign interrupted before simulation launch")
        with (attempt / "sim.log").open("w") as sim_log, (attempt / "replay.log").open("w") as replay_log:
            launch = ["ros2", "launch", "whole_body_bringup", "sim.launch.py"]
            sim = subprocess.Popen(launch, env=env, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True)
            result.update(sim_pid=sim.pid, sim_command=launch)
            save(result_path, result)
            if cancelled.wait(3):
                raise RuntimeError("Campaign interrupted during simulation startup")
            command = [str(args.ros_python), "-B", str(SCRIPTS / "replay_start_handoff.py"),
                       "--cohort", str(args.cohort), "--anchor", str(anchor), "--condition", "preposition",
                       "--output", str(attempt / "evidence")]
            replay = subprocess.Popen(command, env=env, stdout=replay_log, stderr=subprocess.STDOUT, start_new_session=True)
            result.update(replay_pid=replay.pid, replay_command=command)
            save(result_path, result)
            duration = (root["interval_ns"][1] - root["interval_ns"][0]) / 1e9
            result["replay_exit"] = wait_owned(replay, duration + 115, cancelled)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    finally:
        if prep is not None:
            result["anchor_process_group_gone"] = stop_owned_group(prep)
        if sim is not None:
            result["replay_process_group_gone"] = replay is None or stop_owned_group(replay)
            result["sim_process_group_gone"] = stop_owned_group(sim)
            result["sim_exit"] = sim.poll()
            try:
                result["graph_empty_after"] = graph_empty(env, attempt / "cleanup.log", attempts=3)
            except (OSError, subprocess.SubprocessError) as error:
                result["cleanup_error"] = str(error)
                result["graph_empty_after"] = False
            result["domain_reusable"] = all(result[key] for key in
                                             ("replay_process_group_gone", "sim_process_group_gone", "graph_empty_after"))
        result["domain_reusable"] &= result.get("anchor_process_group_gone", True)
    trace = attempt / "evidence" / "trace.jsonl"
    if trace.is_file():
        try:
            result.update(trace_receipt(trace))
            inputs = json.loads((trace.parent / "input.json").read_bytes())
            if (inputs["root"] != root or inputs["cohort_receipt_sha256"] != cohort_sha
                    or inputs["anchor_sha256"] != result["anchor_sha256"]
                    or inputs["shared_frames_sha256"] != FRAMES_SHA or inputs["condition"] != "preposition"):
                raise ValueError("Native replay input receipt differs from this root and strict route")
            if result["start_status"] == "START_NOT_REACHED":
                result["result"] = "START_NOT_REACHED"
            elif result["start_status"] == "START_REACHED":
                complete = (result.get("replay_exit") == 0 and result["historical_input_seal"] is not None
                            and result["historical_timebase_count"] == result["accepted_start_count"]
                            == result["accepted_stop_count"] == 1)
                result["result"] = "START_REACHED_REPLAY_COMPLETE" if complete else "REPLAY_FAILED_AFTER_START_REACHED"
        except (OSError, KeyError, TypeError, ValueError) as error:
            result.update(result="INVALID_NATIVE_TRACE", trace_error=str(error))
        try:
            result.update(archive(trace))
        except (OSError, ValueError) as error:
            result.update(trace=str(trace), archive_error=str(error), result="TRACE_ARCHIVE_FAILED")
    if not result["domain_reusable"]:
        result["result_before_domain_quarantine"] = result["result"]
        result["result"] = "DOMAIN_QUARANTINED"
    elif cancelled.is_set() and result["result"] == "FAILED_BEFORE_START_DECISION":
        result["result"] = "CANCELLED"
    result["finished_unix_ns"] = time.time_ns()
    save(result_path, result)
    print(root["root"], f"attempt-{i:02d}", result["result"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    anchor = parser.add_mutually_exclusive_group(required=True)
    anchor.add_argument("--anchors", type=Path, help="Precomputed <capture hex>.json from prepare_start_handoff.py")
    anchor.add_argument("--anchor-python", type=Path, help="ROS-aware HMB/Python virtual environment for anchor preparation")
    parser.add_argument("--ros-python", "--replay-python", dest="ros_python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument("--shared-frames", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    parser.add_argument("--domains", type=int, nargs="+", default=DOMAINS)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--early-retries", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if os.name != "posix" or len(set(args.domains)) != len(args.domains) or not set(args.domains) <= set(DOMAINS):
        parser.error("Use Ubuntu and distinct isolated ROS domains 206–213")
    for key in ("cohort", "output", "docs", "anchors", "anchor_python", "ros_python", "shared_frames"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.absolute())
    frame_bytes = args.shared_frames.read_bytes()
    if sha256(frame_bytes).hexdigest() != FRAMES_SHA:
        parser.error("Shared rigid transforms differ from the prior verified 03 B run")
    roots = verified_roots(args.cohort, args.docs)
    selected = set(args.root) or roots.keys()
    if not selected <= roots.keys():
        parser.error("Select only signed roots")
    args.output.mkdir(parents=True, exist_ok=False)
    args.shared_frames = args.output / "shared_rigid_transforms.txt"
    args.shared_frames.write_bytes(frame_bytes)
    cohort_sha = digest(args.cohort)
    plan = {"schema": "s05.teleop.strict_start_handoff_cohort.v1", "cohort_sha256": cohort_sha,
            "shared_frames_sha256": FRAMES_SHA, "domains": list(args.domains), "early_retries": args.early_retries,
            "historical_replay_after_start_miss": False, "planned": [
                {"root": r["root"], "split": r["split"], "planned_ticks": r["planned_ticks"],
                 "selected": r["root"] in selected} for r in roots.values()]}
    save(args.output / "plan.json", plan)
    cancelled = Event()
    for termination in (signal.SIGINT, signal.SIGTERM):
        signal.signal(termination, lambda signum, frame: cancelled.set())
    queue = Queue()
    for root_id in sorted(selected, key=lambda key: -(roots[key]["interval_ns"][1] - roots[key]["interval_ns"][0])):
        queue.put(roots[root_id])

    def worker(domain):
        completed = {}
        while not cancelled.is_set():
            try:
                root = queue.get_nowait()
            except Empty:
                return completed
            directory = args.output / f"root-{root['root'].split(':')[-1]}"
            directory.mkdir()
            attempts = []
            for attempt in range(1, args.early_retries + 2):
                result = run_attempt(args, root, domain, attempt, directory, cohort_sha, cancelled)
                attempts.append(result)
                if result["result"] != "FAILED_BEFORE_START_DECISION":
                    break
            completed[root["root"]] = attempts
            if not result["domain_reusable"]:
                return completed
        return completed

    attempts = {}
    with ThreadPoolExecutor(max_workers=len(args.domains)) as pool:
        for completed in pool.map(worker, args.domains):
            attempts.update(completed)
    rows = []
    for item in plan["planned"]:
        tries = attempts.get(item["root"], [])
        status = "NOT_SELECTED"
        if item["selected"]:
            status = "CANCELLED" if cancelled.is_set() else "NO_CLEAN_DOMAIN_AVAILABLE"
        if tries:
            status = tries[-1]["result"]
        rows.append(dict(item, result=status, attempts=tries))
    save(args.output / "cohort_run.json", dict(plan, roots=rows, result_counts=dict(Counter(r["result"] for r in rows))))
    nonterminal = {"FAILED_BEFORE_START_DECISION", "REPLAY_FAILED_AFTER_START_REACHED", "INVALID_NATIVE_TRACE",
                   "TRACE_ARCHIVE_FAILED", "DOMAIN_QUARANTINED", "NO_CLEAN_DOMAIN_AVAILABLE", "MISSING_START_ANCHOR",
                   "CANCELLED"}
    sys.exit(1 if any(row["result"] in nonterminal for row in rows) else 0)


if __name__ == "__main__":
    main()
