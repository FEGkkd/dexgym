#!/usr/bin/env python3
"""Run verified HMB roots through the actual nominal Teleop simulation, one START per root."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def terminate(sim):
    if sim.poll() is not None:
        return
    sim.send_signal(signal.SIGINT)
    try:
        sim.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(sim.pid, signal.SIGTERM)
        sim.wait(timeout=8)


def run_root(cohort, root, domain, output):
    run = output / f"root-{root['root'].split(':')[-1]}"
    run.mkdir()
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOG_DIR=str(run / "ros"))
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    for key in ("ROS_STATIC_PEERS", "CYCLONEDDS_URI", "FASTRTPS_DEFAULT_PROFILES_FILE",
                "FASTDDS_DEFAULT_PROFILES_FILE"):
        env.pop(key, None)
    start = time.time_ns()
    receipt = {"root": root["root"], "split": root["split"], "domain": domain, "start_ns": start}
    result = run / "result.json"
    try:
        check = subprocess.run(["ros2", "node", "list", "--no-daemon", "--spin-time", "1.5"],
                               env=env, capture_output=True, text=True, timeout=8, check=True)
        if check.stdout.strip():
            raise RuntimeError(f"ROS domain {domain} is in use: {check.stdout}")
        with (run / "sim.log").open("w") as sim_log:
            launch = ["ros2", "launch", "whole_body_bringup", "sim.launch.py"]
            sim = subprocess.Popen(launch, env=env, stdout=sim_log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
            receipt.update(sim_pid=sim.pid, sim_command=launch)
            result.write_text(json.dumps(receipt, indent=2))
            try:
                time.sleep(3)
                command = [
                    sys.executable, str(Path(__file__).with_name("nominal_live_replay.py")),
                    "--cohort", str(cohort), "--root", root["root"], "--output", str(run / "evidence")
                ]
                elapsed = (root["interval_ns"][1] - root["interval_ns"][0]) / 1_000_000_000
                with (run / "replay.log").open("w") as replay_log:
                    completed = subprocess.run(command, env=env, stdout=replay_log, stderr=subprocess.STDOUT,
                                               timeout=elapsed + 85, check=False)
                receipt.update(replay_command=command, replay_exit=completed.returncode)
            finally:
                terminate(sim)
                receipt["sim_exit"] = sim.poll()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
    trace = run / "evidence" / "trace.jsonl"
    if trace.is_file():
        # Retain uncompressed trace until a faithful compressed copy exists.
        compressed = trace.with_suffix(".jsonl.gz")
        with trace.open("rb") as source, gzip.open(compressed, "wb", compresslevel=3) as dest:
            shutil.copyfileobj(source, dest)
        trace.unlink()
        receipt["trace"] = str(compressed)
    receipt["finish_ns"] = time.time_ns()
    receipt["result"] = "RECORDED" if receipt.get("replay_exit") == 0 else "FAILED"
    result.write_text(json.dumps(receipt, indent=2) + "\n")
    print(root["root"], receipt["result"], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--domains", type=int, nargs="+", required=True)
    parser.add_argument("--skip-root", action="append", default=[])
    args = parser.parse_args()
    if len(set(args.domains)) != len(args.domains) or not all(1 <= domain <= 232 for domain in args.domains):
        parser.error("Specify distinct nonzero local ROS domains")
    if len(args.domains) > 8:
        parser.error("Limit simultaneous isolated simulators to eight")
    cohort = json.loads(args.cohort.read_text())
    if cohort.get("schema") != "s05.teleop.hmb_source_cohort.v1" or not cohort["native_assets_sha256_checked"]:
        parser.error("A verified HMB cohort is required")
    roots = [row for row in cohort["selected_roots"] if row["root"] not in args.skip_root]
    if not roots:
        parser.error("Select at least one root")
    roots.sort(key=lambda row: -(row["interval_ns"][1] - row["interval_ns"][0]))
    args.output.mkdir(parents=True, exist_ok=False)
    count = len(args.domains)

    def worker(index, domain):
        return [run_root(args.cohort, root, domain, args.output) for root in roots[index::count]]

    with ThreadPoolExecutor(max_workers=count) as workers:
        results = [workers.submit(worker, index, domain) for index, domain in enumerate(args.domains)]
        summary = [row for result in results for row in result.result()]
    (args.output / "cohort_run.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(row["result"] != "RECORDED" for row in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
