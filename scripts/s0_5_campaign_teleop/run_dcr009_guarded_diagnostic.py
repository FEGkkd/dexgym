#!/usr/bin/env python3
"""Run at most six preregistered simulation diagnostics through the native guarded controller."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
import time

from replay_dcr009_guarded_diagnostic import (
    CENTER_OFFSET_SEQUENCE_MM, COHORT_SHA, LEGACY_BRIDGE_SHA, LIVE_SHA, ROOT_FIVE, ROOT_FIVE_PLAN_SHA,
    RUN_ID, SCOPE, SIGNED_SHA, VIRTUAL_SHA, check_scope,
)
from run_home_anchored_pilot import resolved_executable
from run_start_handoff_cohort import (
    DOMAINS, clean_env, digest, graph_empty, save, stop_owned_group, verified_roots, wait_owned,
)


SCRIPTS = Path(__file__).resolve().parent
YAW_BASELINE_CONTROLLER_SHA256 = "9f95f3f6fb38f2db225f9f69fa57a24492daada252570f813ca7ac243655b9ba"
YAW_BASELINE_DRIVER_SHA256 = "b226d4cbf4bbe7897a93c212c2e494430cbfe62da2aa97661c51e1fdd27fce59"
YAW_BASELINE_GUARD_SHA256 = "fd785b5cc88c567862c32854d717dab307bed6263130d7cb51ae0c5506760909"
STAGED_START_CONTROLLER_SHA256 = "13ed4843a57fdf1b58ac7ac78202b0d2ef0f686b1b468e78a5aa73d38f9adcc9"
HISTORY_CONTROLLER_SHA256 = "9a4a318b11984a1659284cb3c38e8f382ff3de8539f14f495f6ca335e8016376"
SWEEP_V2_SCOPE = "DIAGNOSTIC_ONLY_MUJOCO_SWEEP_V2"


def verify_sweep_v2_review(path):
    report = json.loads(path.read_bytes())
    required = {
        "status": "TECHNICAL_PASS",
        "scope": SWEEP_V2_SCOPE,
        "root": ROOT_FIVE,
        "formal_coverage_credit": False,
        "implementation_sha256": digest(SCRIPTS / "swept_live_guard_v2.py"),
        "strict_guard_sha256": YAW_BASELINE_GUARD_SHA256,
        "observed_models": {"live": LIVE_SHA, "signed": SIGNED_SHA},
        "budget_ms": 25,
        "max_box_queries": 63,
        "max_box_depth": 7,
    }
    if any(report.get(key) != expected for key, expected in required.items()):
        raise ValueError("The independent V2 qualification does not bind the isolated runtime")
    return report


def evidence_summary(path, audited_stop=False):
    counts = {"historical_packets": 0, "historical_seals": 0, "preposition_packets": 0}
    reached = "NOT_OBSERVED"
    stop = False
    protective_stop = None
    start_only_done = False
    history_intent = None
    history_fault = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Incomplete diagnostic trace; retained")
            row = json.loads(line)
            if row["kind"] == "start_handoff":
                if reached != "NOT_OBSERVED":
                    raise ValueError("Repeated START outcome")
                reached = row["status"]
            if row["kind"] == "pilot_stop":
                stop = (audited_stop if row.get("native_barrier_audit_required") is True
                        else row["late_observed_command_count"] == 0)
            if row["kind"] == "historical_window_intent":
                if history_intent is not None:
                    raise ValueError("Repeated historical window intent")
                history_intent = row["expected_packets"]
            if row["kind"] == "start_only_completed":
                if (start_only_done or reached != "START_REACHED"
                        or row.get("original_history_started") is not False
                        or row.get("profile") not in {"original", "segmented"}):
                    raise ValueError("Invalid bounded START-only completion")
                start_only_done = True
            if row["kind"] == "pilot_lifecycle" and row.get("status") == "FAULT_STOP":
                failure = row.get("failure", {})
                if row["stage"] == "historical_human_inputs":
                    history_fault = failure
                if row["stage"] != "preposition_start":
                    continue
                if (
                    failure.get("source") == "/tianji/robot_state"
                    and failure.get("code") == "DRIVER_NOT_COMMANDABLE"
                    and failure.get("reason") == "position held; home required before next enable"
                ):
                    protective_stop = "NATIVE_PROTECTIVE_HOLD_NO_HISTORY"
                elif (
                    failure.get("source") == "/tianji/safety" and failure.get("code") == "JOINT_LIMIT"
                    and failure.get("reason") == "both arms held: joint command violates uniform joint limit margin"
                ):
                    protective_stop = "NATIVE_JOINT_LIMIT_REJECTION_NO_HISTORY"
            if row["kind"] == "input" and row["stage"] == "preposition_start":
                counts["preposition_packets"] += 1
            if row["stage"] == "historical_human_inputs":
                if reached != "START_REACHED":
                    raise ValueError("Historical input before a measured START reached")
                counts["historical_packets"] += row["kind"] == "input"
                if row["kind"] == "historical_input_seal":
                    if (row["messages"] != counts["historical_packets"]
                            or (history_intent is not None and row["messages"] != history_intent)
                            or row["expected_cdr_sha256"] != row["published_cdr_sha256"]):
                        raise ValueError("Historical stream does not match its source seal")
                    counts["historical_seals"] += 1
    if stop and reached == "START_REACHED" and history_intent is not None and history_fault:
        phase = "HISTORICAL_REPLAY_GUARDED_STOP"
    elif stop and reached == "START_REACHED" and counts["historical_packets"] and counts["historical_seals"] == 1:
        phase = "HISTORICAL_REPLAY_COMPLETE"
    elif stop and reached == "START_REACHED" and start_only_done and not counts["historical_packets"]:
        phase = "START_REACHED_NO_HISTORY_BY_SCOPE"
    elif stop and reached == "START_NOT_REACHED" and not counts["historical_packets"]:
        phase = "START_NOT_REACHED_NO_HISTORY"
    elif (stop and protective_stop and reached == "NOT_OBSERVED" and counts["preposition_packets"]
          and not counts["historical_packets"]):
        phase = protective_stop
    else:
        phase = "INCOMPLETE_TRACE"
    result = {"observed_phase": phase, "start_status": reached, **counts, "formal_coverage_credit": False}
    if history_intent is not None:
        result.update(historical_expected_packets=history_intent, historical_fault=history_fault)
    return result


def attempt(args, index, selection, cancelled):
    path = args.output / f"attempt-{index:02d}"
    path.mkdir()
    result_file = path / "result.json"
    result = {"root": selection["root"], "scope": SCOPE, "attempt": index,
              "phase": "PREFLIGHT", "started_unix_ns": time.time_ns(), "formal_coverage_credit": False}
    if "center_offset_mm" in selection:
        result["center_offset_mm"] = selection["center_offset_mm"]
    profile = selection.get("start_only_profile")
    if profile is not None:
        result["start_only_profile"] = profile
    if args.align_start_yaw_to_home:
        result["align_start_yaw_to_home"] = True
    if args.wrist_first_fingers:
        result["wrist_first_fingers"] = True
    if args.replay_root_five_history:
        result["replay_root_five_history"] = True
    if args.allow_one_unproven_arm:
        result["simulation_guard_exception"] = "SIMULATION_ONLY_ALLOW_ONE_UNPROVEN_ARM_SWEEP"
    if args.sweep_v2_review is not None:
        result["sweep_guard_variant"] = SWEEP_V2_SCOPE
        result["sweep_v2_review_sha256"] = digest(args.sweep_v2_review)
    save(result_file, result)
    children = []
    deadline = time.monotonic() + 180
    ready = path / "sweep-ready.json"
    with TemporaryDirectory(prefix="dcr9-") as private_socket_dir:
        endpoint = Path(private_socket_dir) / "guard.sock"
        session_map = path / "session-map.txt"
        env = clean_env(args.domain, path, args.frames)
        env.pop("S05_SHARED_RIGID_MAP", None)
        env["S05_HOME_ANCHORED_SESSION_MAP"] = str(session_map)
        env["S05_DCR009_GUARD_SOCKET"] = str(endpoint)
        env["S05_DCR009_GUARD_TIMEOUT_MS"] = "35"
        env.pop("S05_DCR009_SEGMENTED_START_STATE", None)
        env.pop("S05_DCR009_FINGER_STAGE", None)
        env.pop("S05_DCR009_HISTORY_AUDIT", None)
        if args.replay_root_five_history:
            env["S05_DCR009_HISTORY_AUDIT"] = "1"
        if profile == "segmented":
            stage_marker = Path(private_socket_dir) / "stage"
            stage_marker.write_bytes(b"PREPOSITION\n")
            stage_marker.chmod(0o600)
            env["S05_DCR009_SEGMENTED_START_STATE"] = str(stage_marker)
        if args.wrist_first_fingers:
            finger_marker = Path(private_socket_dir) / "finger"
            finger_marker.write_bytes(b"0 0\n")
            finger_marker.chmod(0o600)
            env["S05_DCR009_FINGER_STAGE"] = str(finger_marker)
        try:
            if not graph_empty(env, path / "preflight-ros.log") or cancelled.is_set():
                raise RuntimeError("Could not prove an unused local-only ROS domain")
            controller = resolved_executable("whole_body_teleop", "whole_body_teleop_node", env)
            simulator = resolved_executable("whole_body_mujoco_driver", "whole_body_mujoco_driver_node", env)
            if digest(controller) != args.controller_sha256 or digest(simulator) != args.simulator_sha256:
                raise ValueError("Resolved native executable identity differs from the locked run record")
            guard_command = [
                str(args.python), "-B", str(SCRIPTS / "swept_live_guard.py"),
                "--live-model", str(args.model), "--signed-model", str(args.signed_model),
                "--model-receipt", str(args.model_receipt), "--socket", str(endpoint), "--ready", str(ready),
                "--trace", str(path / "sweep-trace.jsonl"), "--budget-ms", "25",
            ]
            if args.allow_one_unproven_arm:
                guard_command[2] = str(SCRIPTS / "simulate_one_unproven_sweep.py")
                guard_command.extend(("--stage", str(stage_marker), "--exception-log", str(path / "guard-exception.jsonl"),
                                      "--mode-receipt", str(path / "guard-mode.json"), "--root", selection["root"]))
            if args.sweep_v2_review is not None:
                guard_command[2] = str(SCRIPTS / "swept_live_guard_v2.py")
                guard_command.extend(("--metrics", str(path / "sweep-v2-metrics.jsonl"),
                                      "--v2-receipt", str(path / "sweep-v2-ready.json")))
            with (path / "guard.log").open("w", encoding="utf-8") as guard_log:
                guard = subprocess.Popen(guard_command, env=env, stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(guard)
            result.update(guard_command=guard_command, guard_pid=guard.pid)
            save(result_file, result)
            ready_deadline = time.monotonic() + 30
            configuration = None
            while configuration is None:
                if endpoint.is_socket() and ready.is_file():
                    try:
                        configuration = json.loads(ready.read_bytes())
                    except json.JSONDecodeError:
                        pass
                if guard.poll() is not None or time.monotonic() >= ready_deadline or cancelled.wait(.1):
                    raise RuntimeError("Whole-body sweep guard did not verify assets and become ready")
            if (configuration["pid"] != guard.pid or configuration["socket"] != str(endpoint)
                    or configuration["scope"] != "DIAGNOSTIC_ONLY_MUJOCO_COMMAND_ENVELOPE"
                    or configuration["live_model_sha256"] != LIVE_SHA or configuration["signed_model_sha256"] != SIGNED_SHA
                    or configuration["guard_script_sha256"] != digest(SCRIPTS / "swept_live_guard.py")):
                raise RuntimeError("Sweep guard readiness names an unexpected process, socket, or model")
            result["guard_ready_sha256"] = digest(ready)
            if args.sweep_v2_review is not None:
                verified = verify_sweep_v2_review(args.sweep_v2_review)
                v2_ready = path / "sweep-v2-ready.json"
                runtime = json.loads(v2_ready.read_bytes())
                keys = ("scope", "implementation_sha256", "strict_guard_sha256", "budget_ms", "max_box_queries",
                        "max_box_depth", "formal_coverage_credit")
                if any(runtime.get(key) != verified[key] for key in keys):
                    raise ValueError("V2 runtime differs from the independently qualified envelope profile")
                result["sweep_v2_ready_sha256"] = digest(v2_ready)
            if args.allow_one_unproven_arm:
                guard_mode = path / "guard-mode.json"
                exception_mode = json.loads(guard_mode.read_bytes())
                if (exception_mode["mode"] != result["simulation_guard_exception"]
                        or exception_mode["root"] != ROOT_FIVE or exception_mode["max_exceptions"] != 1
                        or exception_mode["wrapper_sha256"] != digest(SCRIPTS / "simulate_one_unproven_sweep.py")
                        or exception_mode["strict_guard_sha256"] != configuration["guard_script_sha256"]):
                    raise ValueError("The isolated single-exception guard mode differs")
                result["guard_mode_sha256"] = digest(guard_mode)
            launch = ["ros2", "launch", "whole_body_bringup", "sim.launch.py"]
            with (path / "sim.log").open("w", encoding="utf-8") as sim_log, (path / "replay.log").open("w", encoding="utf-8") as replay_log:
                sim = subprocess.Popen(launch, env=env, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(sim)
                result.update(sim_pid=sim.pid, simulator_command=launch)
                save(result_file, result)
                if cancelled.wait(3) or sim.poll() is not None or guard.poll() is not None:
                    raise RuntimeError("Simulation or guard did not reach startup")
                program = [str(args.python), "-B", str(SCRIPTS / "replay_dcr009_guarded_diagnostic.py")]
                arguments = {
                    "plan": args.plan, "plan-sha256": args.plan_sha256, "cohort": args.cohort, "root": selection["root"],
                    "virtual-axes": args.virtual_axes, "legacy-frames": args.legacy_frames, "frames": args.frames,
                    "home-reference": args.home_reference, "model": args.model, "session-map": session_map,
                    "signed-model": args.signed_model, "model-receipt": args.model_receipt,
                    "guard-socket": endpoint, "output": path / "evidence", "controller-executable": controller,
                    "simulator-executable": simulator, "launch-pid": sim.pid,
                }
                if "center_offset_mm" in selection:
                    arguments["center-offset-mm"] = selection["center_offset_mm"]
                if profile is not None:
                    arguments["start-only-profile"] = profile
                for key, value in arguments.items():
                    program.extend((f"--{key}", str(value)))
                if args.align_start_yaw_to_home:
                    program.append("--align-start-yaw-to-home")
                if args.wrist_first_fingers:
                    program.append("--wrist-first-fingers")
                if args.replay_root_five_history:
                    program.append("--replay-root-five-history")
                replay = subprocess.Popen(program, env=env, stdout=replay_log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(replay)
                result.update(replay_pid=replay.pid, replay_command=program, phase="RUNNING")
                save(result_file, result)
                result["replay_exit"] = wait_owned(replay, deadline - time.monotonic(), cancelled)
                result["phase"] = "COMPLETED" if result["replay_exit"] == 0 else "ABORTED"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result.update(phase="ABORTED", error=str(error))
        finally:
            result["owned_groups_stopped"] = all([stop_owned_group(p) for p in reversed(children)])
            metrics = path / "sweep-v2-metrics.jsonl"
            if args.sweep_v2_review is not None and metrics.is_file():
                result["sweep_v2_metrics_sha256"] = digest(metrics)
            if args.allow_one_unproven_arm and (path / "guard-exception.jsonl").is_file():
                exception_trace = path / "guard-exception.jsonl"
                records = exception_trace.read_bytes().splitlines()
                result["guard_exception_trace_sha256"] = digest(exception_trace)
                result["guard_exception_count"] = len(records)
            try:
                result["same_domain_empty_after"] = graph_empty(env, path / "cleanup-ros.log", attempts=3)
            except (OSError, subprocess.SubprocessError) as error:
                result["cleanup_error"] = str(error)
            trace = path / "evidence" / "trace.jsonl"
            if trace.is_file():
                result["trace_sha256"] = digest(trace)
                try:
                    audited_stop = False
                    if args.replay_root_five_history:
                        from dcr009_stop_audit import PASS, audit_attempt

                        save(result_file, result)
                        save(path / "stop-audit-input-receipt.json", result)
                        audit = audit_attempt(path)
                        audit_receipt = path / "stop-audit.json"
                        save(audit_receipt, audit)
                        audited_stop = audit["status"] == PASS
                        result.update(stop_audit_status=audit["status"], stop_audit_sha256=digest(audit_receipt))
                    result.update(evidence_summary(trace, audited_stop=audited_stop))
                    if args.replay_root_five_history and not audited_stop:
                        result["phase"] = "STOP_AUDIT_FAILED"
                except (OSError, ValueError, KeyError) as error:
                    result.update(phase="INVALID_TRACE", trace_error=str(error))
            if result["phase"] == "COMPLETED":
                result["phase"] = result.get("observed_phase", "INCOMPLETE_TRACE")
            elif result.get("observed_phase") in {
                "NATIVE_PROTECTIVE_HOLD_NO_HISTORY", "NATIVE_JOINT_LIMIT_REJECTION_NO_HISTORY",
                "HISTORICAL_REPLAY_GUARDED_STOP",
            }:
                result["phase"] = result["observed_phase"]
            result["finished_unix_ns"] = time.time_ns()
            save(result_file, result)
    return result


def closed(row):
    return (
        row["phase"] in {"HISTORICAL_REPLAY_COMPLETE", "HISTORICAL_REPLAY_GUARDED_STOP",
                         "START_REACHED_NO_HISTORY_BY_SCOPE", "START_NOT_REACHED_NO_HISTORY",
                         "NATIVE_PROTECTIVE_HOLD_NO_HISTORY", "NATIVE_JOINT_LIMIT_REJECTION_NO_HISTORY"}
        and row["owned_groups_stopped"] and row.get("same_domain_empty_after") is True
    )


def completed_prefix(previous_runs, plan_sha, roster):
    index = 0
    previous_receipt_sha = None
    for previous_run in previous_runs:
        receipt_path = previous_run / "run_receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        if (receipt["scope"] != SCOPE or receipt["plan_sha256"] != plan_sha
                or receipt.get("prior_closed_attempt_count", 0) != index
                or receipt.get("prior_run_receipt_sha256") != previous_receipt_sha):
            raise ValueError("Earlier work does not belong to the same locked run chain")
        for row in receipt["attempts"]:
            index += 1
            if index >= len(roster):
                raise ValueError("No incomplete original roster to continue")
            trace = previous_run / f"attempt-{index:02d}" / "evidence" / "trace.jsonl"
            planned = roster[index - 1]
            if (row["attempt"] != index or row["root"] != planned["root"]
                    or row.get("center_offset_mm") != planned.get("center_offset_mm")
                    or row.get("start_only_profile") != planned.get("start_only_profile")
                    or row["trace_sha256"] != digest(trace)):
                raise ValueError("Earlier protected attempt does not match the original trace or root")
            actual = evidence_summary(trace)
            if not closed(dict(row, phase=actual["observed_phase"])):
                raise ValueError("Previous session was not a verified independent closed diagnostic")
        previous_receipt_sha = digest(receipt_path)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "cohort", "virtual-axes", "legacy-frames", "frames", "home-reference", "model", "signed-model",
                "model-receipt", "output", "python"):
        parser.add_argument(f"--{key}", type=lambda path: Path(path).resolve(), required=True)
    for key in ("plan-sha256", "controller-sha256", "simulator-sha256"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--resume-from", action="append", default=[], type=lambda path: Path(path).resolve())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare-root-five-centers", action="store_true")
    mode.add_argument("--root-five-start-only", choices=("original", "segmented"))
    parser.add_argument("--align-start-yaw-to-home", action="store_true")
    parser.add_argument("--wrist-first-fingers", action="store_true")
    parser.add_argument("--replay-root-five-history", action="store_true")
    parser.add_argument("--allow-one-unproven-arm", action="store_true")
    parser.add_argument("--sweep-v2-review", type=lambda item: Path(item).resolve())
    parser.add_argument("--domain", type=int, choices=DOMAINS, required=True)
    args = parser.parse_args()
    if os.name != "posix" or not args.output.parent.is_dir():
        parser.error("Use the project Ubuntu host and a dedicated run directory")
    identities = ((args.plan, args.plan_sha256), (args.cohort, COHORT_SHA),
                  (args.virtual_axes, VIRTUAL_SHA), (args.legacy_frames, LEGACY_BRIDGE_SHA),
                  (args.model, LIVE_SHA), (args.signed_model, SIGNED_SHA))
    if any(digest(path) != expected for path, expected in identities):
        raise ValueError("Requested diagnostic inputs differ from pinned assets")
    if args.align_start_yaw_to_home and (
        args.root_five_start_only != "segmented" or args.resume_from
        or args.wrist_first_fingers == (args.controller_sha256 == YAW_BASELINE_CONTROLLER_SHA256)
        or args.simulator_sha256 != YAW_BASELINE_DRIVER_SHA256
        or digest(SCRIPTS / "swept_live_guard.py") != YAW_BASELINE_GUARD_SHA256
    ):
        raise ValueError("Yaw diagnostic must reuse the previous segmented controller, driver and sweep guard")
    if args.wrist_first_fingers and not args.align_start_yaw_to_home:
        raise ValueError("Finger staging requires the single prior yaw-aligned diagnostic")
    if args.replay_root_five_history and (
        not args.wrist_first_fingers or args.resume_from or args.controller_sha256 == STAGED_START_CONTROLLER_SHA256
    ):
        raise ValueError("A single wrist-first root-five history requires the dedicated STOP-audited native controller")
    if args.allow_one_unproven_arm and (
        not args.replay_root_five_history or args.domain != 213 or args.controller_sha256 != HISTORY_CONTROLLER_SHA256
    ):
        raise ValueError("The one-step exception requires the exact prior isolated root-five history controller")
    if args.sweep_v2_review is not None:
        if (not args.replay_root_five_history or args.allow_one_unproven_arm
                or args.controller_sha256 != HISTORY_CONTROLLER_SHA256 or args.domain != 213):
            raise ValueError("V2 applies only to the single isolated prior-controller root-five history")
        verify_sweep_v2_review(args.sweep_v2_review)
    plan = json.loads(args.plan.read_bytes())
    roster = check_scope(plan)
    roots = verified_roots(args.cohort, SCRIPTS.parents[1] / "docs/s0-5")
    if (plan["cohort_sha256"] != COHORT_SHA or plan["shared_frames_sha256"] != digest(args.frames)
            or any(roots[row["root"]]["split"] != "Train" or not 0 < row["replay_planned_ticks"] <= 6000 for row in roster)):
        raise ValueError("Plan is outside the verified Train cohort or budget")
    if args.compare_root_five_centers or args.root_five_start_only:
        if args.plan_sha256 != ROOT_FIVE_PLAN_SHA or roster[4]["root"] != ROOT_FIVE:
            raise ValueError("This diagnostic requires root five in the frozen plan")
        if args.root_five_start_only:
            roster = [dict(roster[4], center_offset_mm=0, start_only_profile=args.root_five_start_only)]
        else:
            roster = [dict(roster[4], center_offset_mm=value) for value in CENTER_OFFSET_SEQUENCE_MM]
    prior = completed_prefix(args.resume_from, args.plan_sha256, roster) if args.resume_from else 0
    args.output.mkdir(exist_ok=False)
    cancelled = Event()
    for stop in (signal.SIGINT, signal.SIGTERM):
        signal.signal(stop, lambda signum, frame: cancelled.set())
    ledger = {"run_id": RUN_ID, "scope": SCOPE, "formal_coverage_credit": False, "plan_sha256": args.plan_sha256,
              "preregistered_roots": [row["root"] for row in roster], "attempts": [], "status": "IN_PROGRESS"}
    if args.compare_root_five_centers:
        ledger["comparison"] = "ROOT_FIVE_COMMON_CENTER_OFFSETS"
        ledger["preregistered_center_offsets_mm"] = list(CENTER_OFFSET_SEQUENCE_MM)
    if args.root_five_start_only:
        ledger["experiment"] = "ROOT_FIVE_ORIGINAL_CENTER_START_ONLY"
        ledger["profile"] = args.root_five_start_only
    if args.align_start_yaw_to_home:
        ledger["align_start_yaw_to_home"] = True
    if args.wrist_first_fingers:
        ledger["wrist_first_fingers"] = True
    if args.replay_root_five_history:
        ledger["experiment"] = "ROOT_FIVE_WRIST_FIRST_GUARDED_HISTORY"
        ledger["replay_root_five_history"] = True
    if args.allow_one_unproven_arm:
        ledger["experiment"] = "ROOT_FIVE_ONE_SWEEP_UNPROVEN_PASS"
    if args.sweep_v2_review is not None:
        ledger["experiment"] = "ROOT_FIVE_PROVEN_SWEEP_V2"
        ledger["sweep_v2_review_sha256"] = digest(args.sweep_v2_review)
    if prior:
        ledger["prior_closed_attempt_count"] = prior
        ledger["prior_run_receipt_sha256"] = digest(args.resume_from[-1] / "run_receipt.json")
    receipt = args.output / "run_receipt.json"
    save(receipt, ledger)
    for index, root in enumerate(roster[prior:], start=prior + 1):
        verdict = attempt(args, index, root, cancelled)
        ledger["attempts"].append(verdict)
        save(receipt, ledger)
        print(index, root["root"], verdict["phase"], verdict.get("start_status"), flush=True)
        if not closed(verdict) or cancelled.is_set():
            break
    complete = len(ledger["attempts"]) + prior == len(roster) and all(closed(row) for row in ledger["attempts"])
    if complete:
        if args.replay_root_five_history:
            ledger["status"] = "ONE_ROOT_HISTORY_ATTEMPT_CLOSED"
        elif args.root_five_start_only:
            ledger["status"] = "ONE_ROOT_START_ONLY_CLOSED"
        else:
            ledger["status"] = "ALL_FIVE_CENTER_OFFSETS_CLOSED" if args.compare_root_five_centers else "ALL_SIX_DIAGNOSTICS_CLOSED"
    else:
        ledger["status"] = "STOPPED_WITH_INCOMPLETE_DIAGNOSTICS"
    save(receipt, ledger)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
