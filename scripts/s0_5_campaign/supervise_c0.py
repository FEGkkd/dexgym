"""Reserve and supervise one authorized C0 learner process on Ubuntu.

Run IDs are permanent. A failed run charges its physical reservation because
buffered metrics cannot count the final in-flight work. 'inspect' never starts
a process. 'recover' reconciles an existing terminal receipt under the lock.
"""
import argparse
from datetime import datetime, timezone
import json
from math import ceil
import os
from pathlib import Path
import re
import signal
import subprocess
from time import monotonic

from check_execution import ATTACHMENT_SHA, BASELINE, FROZEN_SHA, REPO, read, sha

STAGE = "V1-C0"
B_STAGE = "V1-B"
B_SMOKE_LIMIT = 25_000
B_SMOKE_MODE = "b_smoke"
LEARNER = "dexgym.tasks.direct.wholebody_s05.v1.learning.train"
PHYSICAL_ACCOUNTS = ("train_precursor", "train_static_bucket", "train_primary_total", "train_total")
ACTIVE = {"STARTING", "RUNNING"}
GRACE_SECONDS = 5
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def exact_nonnegative(value):
    return type(value) is int and value >= 0


def process_identity(pid):
    try:
        record = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = record.rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        return {"pid": pid, "boot_id": boot, "start_ticks": int(fields[19])}
    except FileNotFoundError:
        return None


def alive(identity):
    return identity is not None and process_identity(identity["pid"]) == identity


def physical_reservation(environments, warmup, iterations):
    if environments < 2 or warmup < 1 or iterations < 1:
        raise ValueError("positive warmup and iterations with at least two source lanes required")
    return environments * (warmup + 2 * iterations)


def check_reservation(state, reservation):
    if state["execution_authorized"] is not True or state["dcr_review"] != "TECHNICAL_PASS":
        raise ValueError("independent DCR acceptance is required")
    if any(run["status"] in ACTIVE for run in state.get("c0_supervised_runs", {}).values()):
        raise ValueError("an earlier supervised learner run must be reconciled first")
    for account, units in reservation.items():
        used = state["resource_usage"][account]
        cap = state["available_budgets"][account]
        if not exact_nonnegative(used) or not exact_nonnegative(units) or used + units > cap:
            raise ValueError(f"insufficient {account} reservation")


def check_b_smoke_budget(state, config_sha, physical):
    account = state.get("b_physical_smoke_budget", {})
    previous = account.get("previous_physical_transitions")
    if account.get("execution_config_sha256") != config_sha or not exact_nonnegative(previous):
        raise ValueError("B smoke requires a bound ledger of previous physical smoke")
    used = previous
    for run in state.get("c0_supervised_runs", {}).values():
        if run["mode"] in {B_SMOKE_MODE, "smoke"}:
            charge = run["charged_physical_transitions"]
            if not exact_nonnegative(charge):
                raise ValueError("invalid recorded B smoke charge")
            used += charge
    if any(used > state["resource_usage"][key] for key in PHYSICAL_ACCOUNTS):
        raise ValueError("B smoke consumption is missing from campaign totals")
    if used + physical > B_SMOKE_LIMIT:
        raise ValueError("B smoke reservation exceeds the cumulative 25000-transition limit")


def check_stage_scope(state, config, config_sha, args, physical):
    stage = B_STAGE if args.smoke else STAGE
    if (state.get("authorization_sha256") != ATTACHMENT_SHA
            or state.get("design_sha256") != FROZEN_SHA
            or stage not in config["authorization"]["scope"]):
        raise ValueError(f"{stage} requires its frozen campaign authorization")
    if args.smoke:
        if args.environments not in {8, 16} or args.iterations > 4:
            raise ValueError("B learner smoke is limited to 8 or 16 lanes and at most four learner iterations")
        check_b_smoke_budget(state, config_sha, physical)
        return {}

    receipts = {}
    for prerequisite, option in (("V1-A", "g_a"), (B_STAGE, "g_b")):
        if state["stages"].get(prerequisite) != "TECHNICAL_PASS":
            raise ValueError(f"formal C0 requires a {prerequisite} technical pass")
        path, declared = getattr(args, option), getattr(args, option + "_sha")
        registered = state.get("stage_gate_evidence", {}).get(prerequisite, {}).get("receipt_sha256")
        if (not isinstance(registered, str) or not SHA.fullmatch(registered)
                or declared != registered or path is None or not path.is_file()
                or sha(path.read_bytes()) != registered):
            raise ValueError(f"formal C0 requires the exact registered {prerequisite} receipt")
        receipts[option + "_receipt_sha256"] = registered
    return receipts


def elapsed_since_start(state):
    started = datetime.fromisoformat(state["started_at"])
    return max(0, ceil((datetime.now(timezone.utc) - started).total_seconds()))


def prior_progress(log):
    """Read the last complete, monotone actual physical count."""
    previous = 0
    if not log.exists():
        return previous
    with log.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                break
            sample = json.loads(raw)
            count = sample.get("physical_transitions")
            if not exact_nonnegative(count) or count < previous:
                raise ValueError("nonmonotone or invalid physical counter in learner log")
            previous = count
    return previous


def read_outcome(output, returncode, physical_cap, expected_iterations, environments, *, mode, gate_receipts):
    """Distinguish reported physical work from conservative accounting."""
    observed = None
    try:
        observed = prior_progress(output / "metrics.jsonl")
        if returncode == 0:
            report = json.loads((output / "fit_report.json").read_bytes())
            final = report["physical_transitions"]
            smoke = mode == B_SMOKE_MODE
            expected_scope = "PHYSICAL_SYNTHETIC_PREACTOR_SMOKE" if smoke else "C0_WORLD_MODEL_FITTING_ONLY"
            expected_state = "SMOKE_ONLY" if smoke else "FIT_COMPLETE_REQUIRES_INDEPENDENT_QUALIFICATION"
            valid_report = (
                exact_nonnegative(final) and observed <= final <= physical_cap
                and report["design_freeze_sha256"] == FROZEN_SHA
                and report["expected_hard_upper_bound"] == physical_cap
                and report["scope"] == expected_scope and report["state"] == expected_state
                and report["actor_locked"] is True and report["critic_locked"] is True
                and report["actor_updates"] == report["critic_updates"] == 0
                and all(report.get(key) == gate_receipts.get(key)
                        for key in ("g_a_receipt_sha256", "g_b_receipt_sha256"))
                and report["learner_iterations"] == expected_iterations
                and report["environments"] == environments
                and report["wm_gate"] == "NOT_QUALIFIED"
                and report["resume_class"] == "RESUME_REFILL"
                and SHA.fullmatch(report["checkpoint_sha256"])
            )
            if valid_report:
                return {
                    "status": "EXITED",
                    "logged_physical_transitions": observed,
                    "final_physical_transitions": final,
                    "charged_physical_transitions": final,
                    "unresolved_physical_transitions": 0,
                    "checkpoint_sha256_reported": report["checkpoint_sha256"],
                    "resume_class": report["resume_class"],
                }
            raise ValueError("final fit report did not bind this run")
        reason = "actual learner did not exit cleanly"
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason = str(error)
    return {
        "status": "FAILED",
        "logged_physical_transitions": observed,
        "final_physical_transitions": None,
        "charged_physical_transitions": physical_cap,
        "unresolved_physical_transitions": max(0, physical_cap - (observed or 0)),
        "reason": reason,
    }


def make_command(args, physical):
    interpreter = args.python.absolute()
    if not interpreter.is_file():
        raise ValueError("learner Python entrypoint is unavailable")
    inputs = ("g_a", "g_b", "bank", "root_plan")
    if args.smoke and any(getattr(args, name) is not None or getattr(args, name + "_sha") is not None
                          for name in inputs):
        raise ValueError("synthetic B smoke rejects formal reference and gate inputs")
    if not args.smoke:
        for name in inputs:
            path, digest = getattr(args, name), getattr(args, name + "_sha")
            if path is None or digest is None or not SHA.fullmatch(digest):
                raise ValueError(f"C0 production requires {name} and its SHA-256")
            if name != "bank" and not path.is_file():
                raise ValueError(f"C0 production file is missing: {name}")
            if name == "bank" and not path.exists():
                raise ValueError("C0 production reference bank is missing")
    command = [
        str(interpreter), "-u", "-m", LEARNER, "--model", str(args.model.absolute()),
        "--output", str(args.output), "--device", args.device,
        "--environments", str(args.environments), "--warmup", str(args.warmup),
        "--iterations", str(args.iterations), "--buffer-steps", str(args.buffer_steps),
        "--reserved-physical-transitions", str(physical),
    ]
    if args.smoke:
        command.append("--smoke")
    else:
        for name in inputs:
            option = "--" + name.replace("_", "-")
            command.extend([option, str(getattr(args, name).absolute()),
                            option + "-sha", getattr(args, name + "_sha")])
    return command


def get_state(ledger, root):
    events = ledger.event_history(root / "events.jsonl")
    if not events:
        raise ValueError("campaign ledger must already exist")
    return events[-1]["state"]


def record_run(ledger, root, state, run, event):
    state.setdefault("c0_supervised_runs", {})[run["run_id"]] = run
    stage = B_STAGE if run["mode"] in {B_SMOKE_MODE, "smoke"} else STAGE
    state["runs"] = [row for row in state["runs"] if row["run_id"] != run["run_id"]] + [{
        "run_id": run["run_id"], "stage": stage, "status": run["status"], "scope": run["mode"],
        "pid": run.get("child_pid"), "evidence_path": run["evidence_path"],
        "exit_code": run.get("exit_code"), "elapsed_seconds": run.get("elapsed_seconds"),
    }]
    pending = run["reservation"] if run["status"] in ACTIVE else dict.fromkeys(run["reservation"], 0)
    state["c0_reserved_budget"] = pending
    for account, units in pending.items():
        state["remaining_budget"][account] = (
            state["available_budgets"][account] - state["resource_usage"][account] - units
        )
    state["stage"] = stage
    state["run_status"] = run["status"]
    if run["mode"] == "preactor" and state["stages"][STAGE] != "TECHNICAL_PASS":
        state["stages"][STAGE] = {
            "STARTING": "IN_PROGRESS", "RUNNING": "IN_PROGRESS",
            "EXITED": "READY_FOR_REVIEW", "FAILED": "FAILED",
        }[run["status"]]
    state["updated_at"] = ledger.utc()
    ledger.append(root, state, event, run)


def finalize(ledger, root, state, run, outcome):
    if run["status"] not in ACTIVE:
        return run
    if outcome["run_id"] != run["run_id"] or outcome["receipt_nonce"] != run["receipt_nonce"]:
        raise ValueError("terminal receipt does not bind the reserved run")
    physical = outcome["charged_physical_transitions"]
    gpu = outcome["gpu_seconds"]
    reservation = run["reservation"]
    if not exact_nonnegative(physical) or not exact_nonnegative(gpu):
        raise ValueError("invalid terminal budget")
    charges = dict.fromkeys(PHYSICAL_ACCOUNTS, physical) | {"gpu_seconds": gpu}
    for account, units in charges.items():
        if units > reservation.get(account, 0):
            raise ValueError(f"terminal charge exceeds {account} reservation")
        used = state["resource_usage"][account] + units
        if used > state["available_budgets"][account]:
            raise ValueError(f"terminal charge exceeds campaign {account} limit")
        state["resource_usage"][account] = used
        state["remaining_budget"][account] = state["available_budgets"][account] - used
    return dict(run, **outcome)


def stop_owned_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return child.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return child.wait()


def launch(ledger, root, config, config_sha, args):
    if args.timeout_seconds < 1:
        raise ValueError("a positive process timeout is required")
    physical = physical_reservation(args.environments, args.warmup, args.iterations)
    args.output = root / "runs" / args.run_id / "learner"
    command = make_command(args, physical)
    gpu = args.device == "cuda:0"
    timeout = args.timeout_seconds
    reserve = dict.fromkeys(PHYSICAL_ACCOUNTS, physical)
    if gpu:
        reserve["gpu_seconds"] = timeout + GRACE_SECONDS
    run_dir = args.output.parent
    run = {
        "run_id": args.run_id, "status": "STARTING", "evidence_path": str(run_dir),
        "supervisor": process_identity(os.getpid()), "child": None,
        "receipt_nonce": os.urandom(16).hex(), "created_at": ledger.utc(),
        "reservation": reserve, "command": command,
        "execution_config_sha256": config_sha, "design_commit": BASELINE,
        "git_sha": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "mode": B_SMOKE_MODE if args.smoke else "preactor",
        "timeout_seconds": timeout,
    }
    with ledger.lock(root):
        state = get_state(ledger, root)
        if state["execution_config_sha256"] != config_sha:
            raise ValueError("ledger configuration changed")
        check_reservation(state, reserve)
        run["gate_receipts"] = check_stage_scope(state, config, config_sha, args, physical)
        if elapsed_since_start(state) + timeout + GRACE_SECONDS > config["budget"]["elapsed_seconds"]:
            raise ValueError("time reservation exceeds the campaign deadline")
        if any(row["run_id"] == args.run_id for row in state["runs"]):
            raise ValueError("run ID has already been used")
        if gpu:
            query = subprocess.check_output([
                "nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ], text=True).strip()
            if query:
                raise ValueError("GPU 0 already has a compute process")
        run_dir.mkdir(parents=True, exist_ok=False)
        record_run(ledger, root, state, run, "C0_RUN_RESERVED")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO / "source"), env.get("PYTHONPATH"))))
    env["CUDA_VISIBLE_DEVICES"] = "0" if gpu else ""
    started = monotonic()
    stop_reason = None
    try:
        with (run_dir / "process.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            run = dict(run, child=process_identity(child.pid), child_pid=child.pid, status="RUNNING")
            with ledger.lock(root):
                record_run(ledger, root, get_state(ledger, root), run, "C0_PROCESS_STARTED")
            returncode = child.wait(timeout=max(0.0, timeout - (monotonic() - started)))
    except subprocess.TimeoutExpired:
        returncode = stop_owned_child(child)
        stop_reason = "deadline exceeded"
    except KeyboardInterrupt:
        if "child" not in locals():
            raise
        returncode = stop_owned_child(child)
        stop_reason = "supervisor interrupted"
    except OSError as error:
        # If no Popen result exists, the learner could not execute.
        if "child" in locals():
            raise
        accounting = {"status": "FAILED", "logged_physical_transitions": 0, "final_physical_transitions": 0,
                      "charged_physical_transitions": 0, "unresolved_physical_transitions": 0,
                      "reason": str(error)}
        returncode = None
    if returncode is not None:
        accounting = read_outcome(args.output, returncode if stop_reason is None else -1,
                                  physical, args.iterations, args.environments, mode=run["mode"],
                                  gate_receipts=run["gate_receipts"])
        if stop_reason:
            accounting["reason"] = stop_reason

    seconds = ceil(monotonic() - started)
    exit_code = (None if returncode is None else returncode if returncode >= 0 else 128 - returncode)
    outcome = dict(accounting, run_id=args.run_id, receipt_nonce=run["receipt_nonce"], closed_at=ledger.utc(),
                   returncode=returncode, exit_code=exit_code, stop_reason=stop_reason,
                   elapsed_seconds=seconds, gpu_seconds=seconds if gpu and returncode is not None else 0)
    ledger.atomic(run_dir / "runner.exit.json", ledger.encoded(outcome) + b"\n")
    return reconcile(ledger, root, args.run_id, record=True)


def reconcile(ledger, root, run_id, *, record):
    with ledger.lock(root):
        state = get_state(ledger, root)
        run = state.get("c0_supervised_runs", {}).get(run_id)
        if run is None:
            raise ValueError("unknown supervised run")
        if run["status"] not in ACTIVE:
            return run
        receipt = Path(run["evidence_path"]) / "runner.exit.json"
        if receipt.is_file():
            outcome = ledger.read_json(receipt)
            if record:
                run = finalize(ledger, root, state, run, outcome)
                record_run(ledger, root, state, run, "C0_PROCESS_RECONCILED")
                return run
            return dict(run, observation="TERMINAL_RECEIPT_PENDING_RECONCILIATION")
        if alive(run["supervisor"]):
            return dict(run, observation="SUPERVISOR_ALIVE")
        if alive(run.get("child")):
            return dict(run, observation="WORKER_ALIVE_EXIT_OWNER_UNAVAILABLE")
        return dict(run, observation="EXIT_UNKNOWN_RESERVATION_RETAINED")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    tools = cli.add_subparsers(dest="action", required=True)
    for action in ("inspect", "recover"):
        tools.add_parser(action).add_argument("--run-id", required=True)
    run = tools.add_parser("launch")
    run.add_argument("--run-id", required=True)
    run.add_argument("--python", type=Path)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    run.add_argument("--envs", dest="environments", type=int, required=True)
    run.add_argument("--warmup", type=int, default=66)
    run.add_argument("--iterations", type=int, required=True)
    run.add_argument("--buffer-steps", type=int, choices=(128, 256, 512, 1024, 1800), default=128)
    run.add_argument("--timeout-seconds", type=int, required=True)
    run.add_argument("--b-smoke", dest="smoke", action="store_true",
                     help="run only disposable, synthetic V1-B physics and WM parameter smoke")
    for option in ("g-a", "g-b", "bank", "root-plan"):
        run.add_argument("--" + option, type=Path)
        run.add_argument("--" + option + "-sha")
    return cli


def interrupted(_signal, _frame):
    raise KeyboardInterrupt


def main():
    import ledger

    args = parser().parse_args()
    if not RUN_NAME.fullmatch(args.run_id):
        raise ValueError("invalid supervised run ID")
    config, checksum = read()
    root = Path(config["execution"]["evidence_root"])
    if args.action == "launch":
        args.python = args.python or Path(config["execution"]["python"])
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            result = launch(ledger, root, config, checksum, args)
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        result = reconcile(ledger, root, args.run_id, record=args.action == "recover")
    print(json.dumps(result, sort_keys=True))
    if args.action == "launch" and result["status"] != "EXITED":
        raise SystemExit(result.get("exit_code") or 1)


if __name__ == "__main__":
    main()
