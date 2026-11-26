"""Persistent Campaign ledger. Store evidence separately; progress never implies a gate pass."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile

from check_execution import ATTACHMENT_SHA, read, sha


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encoded(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_json(path):
    return json.loads(path.read_bytes())


def atomic(path, contents):
    directory = path.parent
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".ledger-", delete=False) as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)


@contextmanager
def lock(root):
    with (root / ".ledger.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def event_history(path):
    events = []
    if not path.exists():
        return events
    for raw in path.read_bytes().splitlines():
        event = json.loads(raw)
        receipt = {key: value for key, value in event.items() if key != "event_sha256"}
        predecessor = events[-1]["event_sha256"] if events else None
        if (event["sequence"] != len(events) + 1 or event["previous_sha256"] != predecessor
                or event["event_sha256"] != sha(encoded(receipt))):
            raise ValueError("event chain integrity failure")
        events.append(event)
    return events


def append(root, state, event_type, facts):
    event_path = root / "events.jsonl"
    history = event_history(event_path)
    event = {
        "sequence": len(history) + 1, "previous_sha256": history[-1]["event_sha256"] if history else None,
        "at": utc(), "event_type": event_type, "facts": facts, "state": state,
    }
    event["event_sha256"] = sha(encoded(event))
    with event_path.open("ab") as out:
        out.write(encoded(event) + b"\n")
        out.flush()
        os.fsync(out.fileno())
    atomic(root / "ledger.json", encoded(state) + b"\n")


def register(root):
    config, config_sha = read()
    if root.resolve() != Path(config["execution"]["evidence_root"]):
        raise ValueError("wrong campaign evidence root")
    root.mkdir(parents=True, exist_ok=True)
    with lock(root):
        path = root / "ledger.json"
        history = event_history(root / "events.jsonl")
        if history:
            result = history[-1]["state"]
            if result["execution_config_sha256"] != config_sha:
                raise ValueError("immutable execution config differs from ledger")
            if not path.exists() or read_json(path) != result:
                atomic(path, encoded(result) + b"\n")
            return result
        if path.exists():
            raise ValueError("orphaned ledger without append-only log")
        at = utc()
        state = {
            "campaign_id": config["campaign_id"], "git_sha": config["design_commit"],
            "design_sha256": config["design_freeze_sha256"], "started_at": at, "updated_at": at,
            "execution_config_sha256": config_sha, "authorization_sha256": ATTACHMENT_SHA,
            "execution_authorized": True, "dcr_review": "REVIEW_PENDING",
            "stage": "PREFLIGHT", "run_status": "PREFLIGHT", "progress": {},
            "stages": dict.fromkeys(config["authorization"]["scope"], "NOT_STARTED"),
            "available_budgets": config["budget"], "resource_usage": dict.fromkeys(config["budget"], 0),
            "remaining_budget": config["budget"], "stage_gate_evidence": {},
            "agent_work_packages": {}, "runs": [], "checkpoint": None, "best_checkpoint": None,
            "training": {"target_iterations": 30000, "target_physical_envs": 4096},
            "hardware_permission": "BLOCKED", "final_test": "BLOCKED",
        }
        append(root, state, "USER_AUTHORIZATION_RECORDED", {"design": config["design_commit"]})
        return state


def record_physics_smoke(state, facts):
    """Reserve conservative B-only physical work without advancing a scientific stage."""
    if state["execution_authorized"] is not True or state["dcr_review"] != "TECHNICAL_PASS":
        raise ValueError("physics smoke requires the reviewed campaign")
    cap = 25000
    accounts = ("train_precursor", "train_static_bucket", "train_primary_total", "train_total")
    book = state.setdefault("b_physical_smoke_budget", {
        "execution_config_sha256": state["execution_config_sha256"],
        "previous_physical_transitions": state["physical_evidence"]["synthetic_smoke_actual_transitions"],
        "external_runs": {},
    })
    if book["execution_config_sha256"] != state["execution_config_sha256"]:
        raise ValueError("physics smoke budget bound to different execution configuration")
    runs = book.setdefault("external_runs", {})
    run_id = facts.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith("V1-B-"):
        raise ValueError("an explicit V1-B smoke Run ID is required")
    if facts["action"] == "complete":
        run = runs[run_id]
        code, actual = facts["exit"], facts["actual_physical_transitions"]
        if run["outcome"] is not None or type(code) is not int or (
            actual is not None and (type(actual) is not int or not 0 <= actual <= run["reserved"])
        ):
            raise ValueError("invalid smoke outcome")
        run["outcome"] = facts
        return
    if facts["action"] != "reserve" or run_id in runs:
        raise ValueError("invalid smoke reservation")
    units = facts["reserved"]
    if type(units) is not int or units < 1:
        raise ValueError("a positive physical upper bound is required")
    supervised = [run for run in state.get("c0_supervised_runs", {}).values() if run.get("mode") == "b_smoke"]
    if any(run["status"] in {"STARTING", "RUNNING"} for run in supervised):
        raise ValueError("finish the reserved learner smoke before adding a physical smoke")
    charged = sum(run["outcome"]["charged_physical_transitions"] for run in supervised)
    if book["previous_physical_transitions"] + charged + units > cap or any(
        state["remaining_budget"][key] < units for key in accounts
    ):
        raise ValueError("insufficient remaining physics smoke budget")
    runs[run_id] = {"reserved": units, "reservation": facts, "outcome": None}
    book["previous_physical_transitions"] += units
    for key in accounts:
        state["resource_usage"][key] += units
        state["remaining_budget"][key] -= units


def record(root, kind, data):
    with lock(root):
        history = event_history(root / "events.jsonl")
        if not history:
            raise ValueError("campaign has not been initialized")
        state = history[-1]["state"]
        facts = read_json(data)
        if kind == "PREFLIGHT":
            state["preflight"] = facts
        elif kind == "DCR_REVIEW":
            independent = (facts.get("verdict") == "TECHNICAL_PASS" and bool(facts.get("reviewer"))
                           and facts.get("scope") == "DCR-007 only"
                           and facts.get("evidence_sha256") == sha(encoded(facts["evidence"])))
            if not independent or facts.get("execution_config_sha256") != state["execution_config_sha256"]:
                raise ValueError("independent DCR review must bind the exact execution config")
            state["dcr_review"] = "TECHNICAL_PASS"
        elif kind == "RUN_EVIDENCE":
            if state["dcr_review"] != "TECHNICAL_PASS":
                raise ValueError("scientific run requires reviewed DCR")
            if facts.get("stage") not in state["stages"] or not facts.get("run_id"):
                raise ValueError("run requires explicit stage and unique identity")
            state["runs"] = [row for row in state["runs"] if row["run_id"] != facts["run_id"]] + [facts]
            state["stage"] = facts["stage"]
            state["run_status"] = facts["status"]
            status = facts["status"]
            state["stages"][facts["stage"]] = {
                "RUNNING": "IN_PROGRESS", "EXITED": "READY_FOR_REVIEW",
                "FAILED": "FAILED", "BLOCKED": "BLOCKED",
            }[status]
        elif kind == "WORK_PACKAGES":
            state["agent_work_packages"].update(facts)
        elif kind == "DASHBOARD":
            state["dashboard"] = facts
        elif kind == "B_SMOKE_ACCOUNTING":
            record_physics_smoke(state, facts)
        elif kind == "INFRASTRUCTURE_BLOCKED":
            if not facts.get("mount_is_only_persistent_device") or not facts.get("blocking_evidence"):
                raise ValueError("blocking record requires device and observed kernel evidence")
            state["blocked_reason"] = facts["blocking_evidence"]
            state["run_status"] = "BLOCKED"
            state["stage"] = "V1-A/V1-B"
            for stage in facts["affected_stages"]:
                if state["stages"].get(stage) != "TECHNICAL_PASS":
                    state["stages"][stage] = "BLOCKED"
            state["physical_evidence"] = facts["physical_evidence"]
            for name, value in facts["consumed"].items():
                if (name not in state["resource_usage"] or type(value) is not int
                        or not state["resource_usage"][name] <= value <= state["available_budgets"][name]):
                    raise ValueError("invalid verified consumption")
                state["resource_usage"][name] = value
                state["remaining_budget"][name] = state["available_budgets"][name] - value
        else:
            raise ValueError("unsupported event kind")
        state["updated_at"] = utc()
        append(root, state, kind, facts)


def main():
    config, _ = read()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "record", "verify"))
    parser.add_argument("--kind")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    root = Path(config["execution"]["evidence_root"])
    if args.action == "init":
        state = register(root)
        print(json.dumps({"campaign_id": state["campaign_id"], "stage": state["stage"],
                          "execution_config_sha256": state["execution_config_sha256"]}))
    elif args.action == "record":
        if not args.kind or args.evidence is None:
            parser.error("record requires --kind and --evidence")
        record(root, args.kind, args.evidence)
    else:
        with lock(root):
            events = event_history(root / "events.jsonl")
            if not events or events[-1]["state"] != read_json(root / "ledger.json"):
                raise ValueError("ledger snapshot differs from the last event")
            print(f"{len(events)} linked events")


if __name__ == "__main__":
    main()
