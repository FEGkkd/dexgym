#!/usr/bin/env python3
"""Bounded launcher for the explicitly authorized, isolated DCR-009 MuJoCo Pilot."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from threading import Event

from run_start_handoff_cohort import DOMAINS, clean_env, digest, graph_empty, save, stop_owned_group, verified_roots, wait_owned


SCRIPTS = Path(__file__).resolve().parent
RUN_ID = "HOME-ANCHORED-PILOT-DCR009-01"
MAX_PER_ATTEMPT_SECONDS = 180
MAX_ROOT_TICKS = 6000
MAX_TOTAL_TICKS = 72000


def load_launch_inputs(args):
    authority = json.loads(args.authority.read_bytes())
    if (digest(args.authority) != args.authority_sha256
            or authority.get("schema") != "s05.teleop.home_anchored_pilot_authority.v1"
            or authority.get("run_id") != RUN_ID or authority.get("scope") != "MUJOCO_ONLY"
            or authority.get("pilot_execution_authorized") is not True
            or any(authority.get(key) is not False for key in
                   ("full_campaign_authorized", "training_authorized", "real_robot_authorized"))):
        raise ValueError("The isolated Pilot requires an explicit, byte-bound Stage II authorization")
    identities = authority["paths_sha256"]
    if digest(args.calibration) != identities["calibration_receipt"]:
        raise ValueError("Authorized calibration receipt changed")
    calibration = json.loads(args.calibration.read_bytes())
    if (calibration.get("schema_version") != 1 or calibration.get("scope") != "MUJOCO_ONLY"
            or calibration.get("status") != "QUALIFIED"
            or calibration.get("qualification", {}).get("recorded_mcap_pilot") != "QUALIFIED"):
        raise ValueError("Recorded-input MuJoCo CAL is not qualified")
    if authority.get("qualifications", {}).get("independent_whole_body_publish_guard") != "VERIFIED":
        raise ValueError("No independently verified signed 54-axis swept-path publish guard")
    inputs = {
        "cohort": args.cohort, "pilot_plan": args.plan,
        "home_reference": args.home_reference, "live_model": args.model, "shared_frames": args.shared_frames,
        "runtime_files": args.runtime_files,
    }
    for name, path in inputs.items():
        if digest(path) != identities[name]:
            raise ValueError(f"Authorized {name} changed")
    roots = verified_roots(args.cohort, args.docs)
    plan = json.loads(args.plan.read_bytes())
    if (plan.get("selection_status") != "PILOT_PLAN_READY_FOR_REVIEW"
            or plan.get("calibration_receipt_sha256") != authority["paths_sha256"]["calibration_receipt"]
            or plan.get("cohort_sha256") != authority["paths_sha256"]["cohort"]
            or plan.get("shared_frames_sha256") != authority["paths_sha256"]["shared_frames"]):
        raise ValueError("Six-root plan is not bound to the exact cohort and qualified calibration")
    selected = plan["selected_roots"]
    ids = [row["root"] for row in selected]
    repeats = plan["repeat_roots"]
    if (len(ids) != 6 or len(set(ids)) != 6 or not set(ids) <= roots.keys()
            or len(repeats) != 2 or len(set(repeats)) != 2 or not set(repeats) <= set(ids)):
        raise ValueError("Expected exactly six signed roots and two distinct locked repeats")
    applicable_roots = calibration.get("qualification", {}).get("qualified_recorded_roots", [])
    if not set(ids) <= set(applicable_roots):
        raise ValueError("CAL does not cover every selected recording")
    anchors = {}
    for row in selected:
        root = roots[row["root"]]
        root_suffix = root["root"].split(":")[-1]
        if re.fullmatch(r"[0-9a-f]{24}", root_suffix) is None or row["anchor_path"] != f"anchors/{root_suffix}.json":
            raise ValueError("Anchor path does not identify the selected signed source")
        anchor = args.plan.parent / row["anchor_path"]
        replay_ticks = row["replay_planned_ticks"]
        if (root["split"] != "Train" or row["split"] != "Train" or type(replay_ticks) is not int
                or not 0 < replay_ticks <= min(MAX_ROOT_TICKS, root["planned_ticks"])
                or digest(anchor) != row["anchor_sha256"]):
            raise ValueError(f"Unqualified root, budget or anchor: {root['root']}")
        anchors[root["root"]] = anchor, replay_ticks
    sequence = ids + repeats
    if sum(anchors[root][1] for root in sequence) > MAX_TOTAL_TICKS:
        raise ValueError("Pilot tick budget exceeded")
    return authority, roots, anchors, sequence


def resolved_executable(package, binary, environment):
    prefix = subprocess.check_output(
        ["ros2", "pkg", "prefix", package], env=environment, timeout=10, text=True
    ).strip()
    return Path(prefix) / "lib" / package / binary


def attempt_once(args, authority, root, anchor, index, cancelled):
    attempt = args.output / f"attempt-{index:02d}"
    attempt.mkdir()
    state_path = attempt / "result.json"
    state = {"root": root["root"], "split": "Train", "attempt": index,
             "primary": index <= 6, "domain": args.domain, "anchor_sha256": digest(anchor),
             "planned_ticks": root["planned_ticks"], "started_unix_ns": time.time_ns(),
             "status": "PREFLIGHT", "safe_to_continue": False}
    save(state_path, state)
    deadline = time.monotonic() + MAX_PER_ATTEMPT_SECONDS
    session_map = attempt / "session-map.txt"
    env = clean_env(args.domain, attempt, args.shared_frames)
    env.pop("S05_SHARED_RIGID_MAP", None)
    env["S05_HOME_ANCHORED_SESSION_MAP"] = str(session_map)
    children = []
    try:
        if not graph_empty(env, attempt / "preflight.log") or cancelled.is_set():
            raise RuntimeError("Cannot establish an empty isolated simulation graph")
        for key, package, node in (
            ("controller_executable", "whole_body_teleop", "whole_body_teleop_node"),
            ("simulator_executable", "whole_body_mujoco_driver", "whole_body_mujoco_driver_node"),
        ):
            executable = resolved_executable(package, node, env)
            if digest(executable) != authority["paths_sha256"][key]:
                raise ValueError(f"Resolved {package} binary differs from authority")
            state[key] = str(executable)
        launch = ["ros2", "launch", "whole_body_bringup", "sim.launch.py"]
        command = [str(args.ros_python), "-B", str(SCRIPTS / "replay_home_anchored_pilot.py")]
        arguments = {
            "authority": args.authority, "authority-sha256": args.authority_sha256,
            "calibration": args.calibration, "plan": args.plan, "anchor": anchor, "cohort": args.cohort,
            "shared-frames": args.shared_frames, "home-reference": args.home_reference,
            "model": args.model, "session-map": session_map, "output": attempt / "evidence",
            "runtime-files": args.runtime_files, "controller-executable": state["controller_executable"],
            "simulator-executable": state["simulator_executable"], "attempt": "primary" if index <= 6 else "repeat",
        }
        for name, value in arguments.items():
            command.extend((f"--{name}", str(value)))
        with (attempt / "sim.log").open("w") as sim_log, (attempt / "replay.log").open("w") as replay_log:
            sim = subprocess.Popen(launch, env=env, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(sim)
            state.update(sim_command=launch, sim_pid=sim.pid)
            save(state_path, state)
            if cancelled.wait(3) or sim.poll() is not None:
                raise RuntimeError("Simulator did not reach startup")
            command.extend(("--launch-pid", str(sim.pid)))
            replay = subprocess.Popen(command, env=env, stdout=replay_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(replay)
            state.update(replay_command=command, replay_pid=replay.pid)
            save(state_path, state)
            state["replay_exit"] = wait_owned(replay, deadline - time.monotonic(), cancelled)
            state["status"] = "EXITED_TO_EVIDENCE_REVIEW"
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        state.update(status="INTERRUPTED_OR_INVALID", error=str(error))
    finally:
        cleanup = [stop_owned_group(process) for process in reversed(children)]
        try:
            state["graph_empty_after"] = graph_empty(env, attempt / "cleanup.log", attempts=3)
        except (OSError, subprocess.SubprocessError) as error:
            state["cleanup_error"] = str(error)
        state["owned_process_groups_gone"] = all(cleanup)
        trace = attempt / "evidence" / "trace.jsonl"
        if trace.is_file():
            state["trace_sha256"] = digest(trace)
        # Strict replay exits zero for a measured miss with a proper STOP. Unexpected failures halt the queue.
        state["safe_to_continue"] = (state.get("replay_exit") == 0 and trace.is_file()
                                     and all(cleanup) and state.get("graph_empty_after") is True)
        state["finished_unix_ns"] = time.time_ns()
        save(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("authority", "calibration", "plan", "cohort", "shared-frames", "home-reference", "model",
                "runtime-files", "output"):
        parser.add_argument(f"--{key}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--ros-python", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=SCRIPTS.parents[1] / "docs/s0-5")
    parser.add_argument("--domain", type=int, choices=DOMAINS, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("The existing isolated ROS2 MuJoCo runtime requires Ubuntu")
    authority, roots, anchors, sequence = load_launch_inputs(args)
    args.output.mkdir(exist_ok=False)
    cancelled = Event()
    for termination in (signal.SIGINT, signal.SIGTERM):
        signal.signal(termination, lambda signum, frame: cancelled.set())
    records = []
    queue_path = args.output / "campaign_receipt.json"
    ledger = {"run_id": RUN_ID, "plan_sha256": digest(args.plan), "authority_sha256": args.authority_sha256,
              "ordered_attempt_roots": sequence, "attempt_cap": 8, "max_attempt_seconds": MAX_PER_ATTEMPT_SECONDS,
              "max_total_ticks": MAX_TOTAL_TICKS, "attempts": records, "status": "IN_PROGRESS"}
    save(queue_path, ledger)
    for index, root in enumerate(sequence, start=1):
        result = attempt_once(args, authority, roots[root], anchors[root][0], index, cancelled)
        records.append(result)
        save(queue_path, ledger)
        print(index, root, result["status"], result.get("replay_exit"), flush=True)
        if not result["safe_to_continue"]:
            break
    ledger["status"] = "READY_FOR_EVIDENCE_REVIEW" if len(records) == len(sequence) and records[-1][
        "safe_to_continue"] else "STOPPED_WITH_UNFINISHED_ATTEMPTS"
    ledger["unused_attempt_roots"] = sequence[len(records):]
    save(queue_path, ledger)
    return 0 if ledger["status"] == "READY_FOR_EVIDENCE_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
