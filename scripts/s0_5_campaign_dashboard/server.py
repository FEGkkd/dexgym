"""Read-only loopback monitor for one S0-5 campaign."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
MAX_RECORDS = 450
FRESH_SECONDS = 60
RESOURCE_REFRESH_SECONDS = 8
ACTUAL_EVALUATION = {"ACTUAL_MUJOCO", "ACTUAL_TELEOP_MUJOCO"}
INPUTS = {
    "ledger": "ledger.json",
    "status": "dashboard/status.json",
    "agents": "dashboard/agent_status.json",
    "data": "dashboard/data.json",
    "world_model": "dashboard/world_model.json",
    "deployment": "dashboard/deployment.json",
    "events": "events.jsonl",
    "training": "dashboard/training_metrics.jsonl",
    "evaluation": "dashboard/evaluation_metrics.jsonl",
    "reviews": "dashboard/iteration_history.jsonl",
}


def iso_utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def timestamp(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if math.isfinite(value) and 0 <= value <= 253402300799 else None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp() if dt.tzinfo is not None else None
        except (ValueError, OSError, OverflowError):
            return None
    return None


def parse_json(source):
    def invalid_constant(token):
        raise ValueError(f"Non-finite JSON value: {token}")

    return json.loads(source, parse_constant=invalid_constant)


def clean_json(item):
    """Turn non-finite floats from oversized numeric exponents into JSON null."""
    if isinstance(item, float):
        return item if math.isfinite(item) else None
    if isinstance(item, list):
        return [clean_json(value) for value in item]
    if isinstance(item, dict):
        return {key: clean_json(value) for key, value in item.items()}
    return item


def first_value(mappings, keys):
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is not None and value != "":
                return value
    return None


class CampaignReader:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Campaign root must be a directory.")
        self.resource_lock = threading.Lock()
        self.last_resources = -math.inf
        self.resources = {}

    def read(self, name, now):
        default = [] if INPUTS[name].endswith(".jsonl") else {}
        info = {"path": INPUTS[name], "state": "NOT_RUN"}
        try:
            path = self.root / INPUTS[name]
            if not path.exists():
                return default, info
            path = path.resolve(strict=True)
            if not path.is_relative_to(self.root) or not path.is_file():
                info["state"] = "INVALID_PATH"
                return default, info
            with path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                info.update(mtime=iso_utc(stat.st_mtime), age_seconds=max(0, now - stat.st_mtime))
                if isinstance(default, list):
                    offset = max(0, stat.st_size - MAX_LOG_BYTES)
                    handle.seek(offset)
                    payload = handle.read(MAX_LOG_BYTES)
                    if offset:
                        # The first slice can begin inside a record.
                        payload = payload.split(b"\n", 1)[-1] if b"\n" in payload else b""
                    rows = []
                    invalid = 0
                    for row in payload.splitlines():
                        if not row.strip():
                            continue
                        try:
                            entry = parse_json(row)
                            if not isinstance(entry, dict):
                                raise ValueError("Log records must be objects.")
                            rows.append(entry)
                        except (ValueError, UnicodeError, RecursionError):
                            invalid += 1
                    info.update(state="PARTIAL" if invalid else "AVAILABLE",
                                invalid_records=invalid, bounded_tail=bool(offset or len(rows) > MAX_RECORDS))
                    return rows[-MAX_RECORDS:], info
                if stat.st_size > MAX_JSON_BYTES:
                    info["state"] = "TOO_LARGE"
                    return default, info
                raw = parse_json(handle.read(MAX_JSON_BYTES + 1))
                if not isinstance(raw, dict):
                    raise ValueError("Snapshot must contain an object.")
                info["state"] = "AVAILABLE"
                return raw, info
        except (OSError, ValueError, UnicodeError, RecursionError):
            info["state"] = "UNREADABLE_OR_PARTIAL"
            return default, info

    def snapshot(self):
        now = time.time()
        result = {}
        metadata = {}
        for name in INPUTS:
            result[name], metadata[name] = self.read(name, now)
        evaluation = result["evaluation"]
        result["evaluation"] = [entry for entry in evaluation if entry.get("evidence_kind") in ACTUAL_EVALUATION]
        metadata["evaluation"]["excluded_nonphysical_records"] = len(evaluation) - len(result["evaluation"])
        result.update(
            inputs=metadata,
            refreshed_at=iso_utc(now),
            runtime=runtime_state(result["ledger"], result["status"], result["training"], now),
            resources=self.sample_resources(now),
        )
        return clean_json(result)

    def sample_resources(self, now):
        with self.resource_lock:
            if now - self.last_resources >= RESOURCE_REFRESH_SECONDS:
                self.resources = read_resources(self.root, now)
                self.last_resources = now
            return self.resources


def linux_process(pid, expected_start):
    """Observe the local PID without signaling it, checking the identity when available."""
    if type(pid) is not int or not 0 < pid <= 2**31 - 1 or not Path("/proc").is_dir():
        return "UNVERIFIED"
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if expected_start is not None and str(expected_start) != stat[19]:
            return "PID_REUSED"
        return "EXITED" if stat[0] == "Z" else "OBSERVED"
    except FileNotFoundError:
        return "EXITED"
    except (OSError, IndexError):
        return "UNVERIFIED"


def runtime_state(ledger, status, training, now):
    runs = ledger.get("runs")
    latest = runs[-1] if isinstance(runs, list) and runs and isinstance(runs[-1], dict) else {}
    run = status.get("active_run") or ledger.get("active_run") or latest
    if not isinstance(run, dict):
        run = {"run_id": run}
    input_maps = (run, status, ledger)
    pid = first_value(input_maps, ("pid", "active_pid"))
    start_ticks = first_value(input_maps, ("pid_start_ticks", "process_start_ticks"))
    heartbeat_raw = first_value(input_maps, ("heartbeat_at", "last_heartbeat_at", "heartbeat"))
    heartbeat = timestamp(heartbeat_raw)
    if heartbeat is not None and heartbeat > now + 10:
        heartbeat = None
    state = first_value(input_maps, ("process_state", "run_state", "run_status", "state", "status"))
    normalized = str(state or "UNKNOWN").upper()
    active_states = {"RUNNING", "TRAINING", "VALIDATING", "EVALUATING", "IN_PROGRESS"}
    declared_active = normalized in active_states
    identity = linux_process(pid, start_ticks)
    age = max(0, now - heartbeat) if heartbeat is not None else None
    if declared_active:
        if identity in {"EXITED", "PID_REUSED"}:
            normalized = identity
        elif age is None or age > FRESH_SECONDS:
            normalized = "STALE"
    last_log = None
    for entry in reversed(training):
        last_log = timestamp(first_value((entry,), ("recorded_at", "timestamp", "ts", "time")))
        if last_log is not None:
            break
    return {"state": normalized, "reported_state": state, "pid": pid, "pid_observation": identity,
            "run_id": first_value(input_maps, ("run_id", "active_run_id")),
            "heartbeat_at": iso_utc(heartbeat) if heartbeat is not None else None,
            "heartbeat_age_seconds": age, "stale_after_seconds": FRESH_SECONDS,
            "last_training_log_at": iso_utc(last_log) if last_log is not None else None}


def read_resources(root, now):
    disk = shutil.disk_usage(root)
    result = {"observed_at": iso_utc(now), "cpu_count": os.cpu_count(),
              "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
              "gpu": [], "gpu_processes": [], "gpu_state": "UNAVAILABLE"}
    try:
        result["load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        result["load_average"] = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            field, _, value = line.partition(":")
            if field in {"MemTotal", "MemAvailable"}:
                result["ram_total_bytes" if field == "MemTotal" else "ram_available_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        fields = ("index", "name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu")
        result["gpu"], result["gpu_state"] = nvidia_rows(executable, "--query-gpu", fields)
        result["gpu_processes"], _ = nvidia_rows(executable, "--query-compute-apps",
                                                 ("pid", "process_name", "used_gpu_memory"))
    return result


def nvidia_rows(executable, query, fields):
    try:
        output = subprocess.run(
            [executable, f"{query}={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=2,
        ).stdout
        rows = []
        for row in output.splitlines():
            values = [value.strip() for value in row.split(",")]
            if len(values) == len(fields):
                rows.append(dict(zip(fields, values)))
        return rows, "OBSERVED"
    except (OSError, subprocess.SubprocessError):
        return [], "UNAVAILABLE"


POLICY = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
          "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8")}


def handler_for(reader, allowed_hosts):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Explicit local Host checks also prevent DNS rebinding to this service.
            if self.headers.get("Host", "").lower() not in allowed_hosts:
                self.reply(403, b"Local access only.", "text/plain; charset=utf-8")
                return
            route = urlsplit(self.path).path
            if route == "/api/snapshot":
                payload = json.dumps(reader.snapshot(), ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.reply(200, payload, "application/json; charset=utf-8")
            elif route in ASSETS:
                filename, content_type = ASSETS[route]
                self.reply(200, (HERE / filename).read_bytes(), content_type)
            else:
                self.reply(404, b"Not found.", "text/plain; charset=utf-8")

        def do_POST(self):
            self.reply(405, b"Read-only service.", "text/plain; charset=utf-8")

        def reply(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, pattern, *args):
            # Successful polling should not grow a second metrics log.
            if len(args) > 1 and str(args[1]) != "200":
                super().log_message(pattern, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--tunnel-port", action="append", default=[], type=int,
                        help="Additional browser-side loopback port accepted through SSH forwarding.")
    args = parser.parse_args()
    ports = {args.port, *args.tunnel_port}
    if not all(1 <= port <= 65535 for port in ports):
        parser.error("Ports must be between 1 and 65535.")
    allowed_hosts = {"localhost", "127.0.0.1"}
    allowed_hosts |= {f"{host}:{port}" for host in allowed_hosts for port in ports}
    reader = CampaignReader(args.campaign_root)
    address = ipaddress.IPv4Address("127.0.0.1")
    with ThreadingHTTPServer((str(address), args.port), handler_for(reader, allowed_hosts)) as server:
        print(f"Dashboard http://{address}:{args.port} using {reader.root}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
