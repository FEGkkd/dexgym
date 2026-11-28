#!/usr/bin/env python3
"""Fail-closed dual-model, 54-axis swept guard for isolated MuJoCo diagnostics."""

import argparse
from enum import IntEnum
import json
import os
from pathlib import Path
import signal
import socket
import struct
import time

import numpy as np

from check_dcr009_frames import LIVE_SHA, SIGNED_SHA
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, asset_files, digest
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep


MAGIC = b"S05SWP01"
REQUEST = struct.Struct("<8sQI162d")
RESPONSE = struct.Struct("<8sQII")
PEER = struct.Struct("3i")
JOINTS = 54
MAX_BOX_QUERIES = 63
MAX_BOX_DEPTH = 7
EXACT_MODEL_RECEIPT_SHA = "1d67a6dad1831574af10a1fdf6349638097af6457e9d8527c9bd77e150e876fe"
GROUPS = ((0, 14), (14, 34), (34, 54))
POSITION_TOLERANCE_RAD = 1e-9
DISTANCE_ROUNDING_MARGIN_M = 2e-6


class Reason(IntEnum):
    PERMIT = 0
    BAD_REQUEST = 1
    SEQUENCE = 2
    NONFINITE = 3
    GROUP_MISMATCH = 4
    LIMIT_OR_COUPLING = 5
    COLLISION = 6
    SWEEP_UNPROVEN = 7
    TIME_BUDGET = 8
    INTERNAL = 9
    START_UNBOUND = 10


def verified_robot(path, expected, identity):
    path = path.resolve(strict=True)
    if identity["xml_sha256"] != expected or digest(path) != expected:
        raise ValueError("MuJoCo XML identity differs from the audited model")
    sources = identity["dependencies"]
    original_xml, = [Path(raw) for raw, sha in sources.items() if sha == expected]
    original = {Path(raw).relative_to(original_xml.parent).as_posix(): sha for raw, sha in sources.items()}
    actual = {Path(raw).relative_to(path.parent).as_posix(): sha for raw, sha in asset_files(path).items()}
    if actual != original:
        raise ValueError("MuJoCo include/mesh inventory differs from the audited model")
    return RobotModel(path)


class SweptEnvelope:
    """Prove private exact-model collision clearance over an axis-aligned box."""

    def __init__(self, robot, *, max_queries=MAX_BOX_QUERIES):
        self.native = NativeSweep(robot)
        self.max_queries = max_queries
        self.anchor = None
        self.distances = None
        self.sensitivity = self.native.weights.max(axis=0)

    def _variation(self, half_width, subset=None):
        weights = self.native.weights if subset is None else self.native.weights[subset]
        return np.einsum("ij,j->i", weights, half_width)

    def prove(self, low, high, deadline_ns):
        if time.monotonic_ns() >= deadline_ns:
            return Reason.TIME_BUDGET
        sweep = self.native
        clearance = sweep.clearance + DISTANCE_ROUNDING_MARGIN_M
        if self.anchor is not None:
            radius = np.maximum(np.abs(low - self.anchor), np.abs(high - self.anchor))
            if np.all(self.distances - self._variation(radius) >= clearance):
                return Reason.PERMIT

        center, half = (high + low) * .5, (high - low) * .5
        distances = sweep.pair_distances(center)
        if np.any(distances < sweep.clearance):
            return Reason.COLLISION
        self.anchor, self.distances = center, distances
        active = np.flatnonzero(distances - self._variation(half) < clearance)
        if not active.size:
            return Reason.PERMIT

        stack = [(center, half, active, 0)]
        queries = 0
        while stack:
            if time.monotonic_ns() >= deadline_ns:
                return Reason.TIME_BUDGET
            if queries >= self.max_queries:
                return Reason.SWEEP_UNPROVEN
            center, half, subset, depth = stack.pop()
            required = clearance + self._variation(half, subset)
            remaining = sweep._probe(center, subset, required)
            queries += 1
            if remaining is None:
                return Reason.COLLISION
            if not remaining.size:
                continue
            axis = int(np.argmax(self.sensitivity * half))
            if depth >= MAX_BOX_DEPTH or half[axis] <= 1e-12:
                return Reason.SWEEP_UNPROVEN
            smaller = half.copy()
            smaller[axis] *= .5
            offset = np.zeros(JOINTS)
            offset[axis] = smaller[axis]
            left = subset[remaining]
            stack.append((center - offset, smaller, left, depth + 1))
            stack.append((center + offset, smaller, left, depth + 1))
        return Reason.PERMIT


class Guard:
    """Serial admission with a whole-body baseline across all producer groups."""

    def __init__(self, robots):
        live, signed = robots
        same_axes = live.names == signed.names and all(
            a.name == b.name for a, b in zip(live.actuators, signed.actuators, strict=True)
        )
        if not same_axes:
            raise ValueError("Live and signed canonical 54-axis wire layouts differ")
        self.names = live.names
        self.actuators = tuple(a.name for a in live.actuators)
        self.limits = np.asarray([live.bounds, signed.bounds])
        self.couplings = (live.coupling, signed.coupling)
        self.envelopes = tuple(SweptEnvelope(robot) for robot in robots)
        self.sequence = 0
        self.issued_baselines = ()

    def _legal_box(self, low, high):
        if np.any(low < self.limits[:, :, 0] - POSITION_TOLERANCE_RAD):
            return False
        if np.any(high > self.limits[:, :, 1] + POSITION_TOLERANCE_RAD):
            return False
        from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT

        for coefficients in self.couplings:
            worst = np.maximum(coefficients * low, coefficients * high).sum(axis=1)
            if np.any(worst > COUPLING_LIMIT + POSITION_TOLERANCE_RAD):
                return False
        return True

    def evaluate(self, packet, deadline_ns):
        if len(packet) != REQUEST.size:
            return 0, Reason.BAD_REQUEST, None, None
        magic, seq, group, *values = REQUEST.unpack(packet)
        if magic != MAGIC or not 0 <= group < len(GROUPS):
            return seq, Reason.BAD_REQUEST, group, None
        if seq <= self.sequence:
            return seq, Reason.SEQUENCE, group, None
        self.sequence = seq
        states = np.asarray(values).reshape(3, JOINTS)
        issued, measured, proposed = states
        if not np.isfinite(states).all():
            return seq, Reason.NONFINITE, group, None
        begin, end = GROUPS[group]
        if np.any(issued[:begin] != proposed[:begin]) or np.any(issued[end:] != proposed[end:]):
            return seq, Reason.GROUP_MISMATCH, group, states
        baselines = self.issued_baselines or (measured,)
        if not any(np.array_equal(issued, prior) for prior in baselines):
            return seq, Reason.START_UNBOUND, group, states
        low, high = states.min(axis=0), states.max(axis=0)
        if not self._legal_box(low, high):
            return seq, Reason.LIMIT_OR_COUPLING, group, states
        for envelope in self.envelopes:
            result = envelope.prove(low, high, deadline_ns)
            if result != Reason.PERMIT:
                return seq, result, group, states
        if time.monotonic_ns() >= deadline_ns:
            return seq, Reason.TIME_BUDGET, group, states
        return seq, Reason.PERMIT, group, states

    def publish_decision(self, reason, states):
        if reason == Reason.PERMIT:
            # Either delivered candidate or still-committed predecessor is valid
            # next time. Keeping both covers a reply lost before native publish.
            self.issued_baselines = (states[0].copy(), states[2].copy())


def _read_exact(client, deadline_ns):
    message = bytearray()
    while len(message) < REQUEST.size:
        time_left = (deadline_ns - time.monotonic_ns()) / 1e9
        if time_left <= 0:
            raise TimeoutError
        client.settimeout(time_left)
        part = client.recv(REQUEST.size - len(message))
        if not part:
            break
        message.extend(part)
    return message


def serve(args, guard):
    addr = args.socket.resolve()
    if len(os.fsencode(addr)) > 103 or args.socket.exists() or args.socket.is_symlink():
        raise ValueError("Choose a new short private AF_UNIX socket path")
    parent = addr.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("The AF_UNIX parent directory must be private and owner-controlled")
    work_ns = args.budget_ms * 1_000_000
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server, args.trace.open(
        "x", encoding="utf-8", buffering=1
    ) as trace:
        server.bind(str(addr))
        try:
            addr.chmod(0o600)
            server.listen(2)
            with args.ready.open("x", encoding="utf-8") as ready:
                receipt = {
                    "scope": "DIAGNOSTIC_ONLY_MUJOCO_COMMAND_ENVELOPE",
                    "pid": os.getpid(), "socket": str(addr), "magic": MAGIC.decode("ascii"),
                    "request_bytes": REQUEST.size, "response_bytes": RESPONSE.size,
                    "live_model_sha256": LIVE_SHA, "signed_model_sha256": SIGNED_SHA,
                    "model_receipt_sha256": EXACT_MODEL_RECEIPT_SHA, "guard_script_sha256": digest(__file__),
                    "joint_names": list(guard.names), "actuator_names": list(guard.actuators),
                    "budget_ms": args.budget_ms, "distance_rounding_margin_m": DISTANCE_ROUNDING_MARGIN_M,
                    "max_box_depth": MAX_BOX_DEPTH, "max_box_queries": MAX_BOX_QUERIES,
                }
                json.dump(receipt, ready, indent=2, allow_nan=False)
                ready.write("\n")
            while True:
                client, _ = server.accept()
                with client:
                    pid, uid, _ = PEER.unpack(client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, PEER.size))
                    if uid != os.getuid():
                        continue
                    clock_start = time.monotonic_ns()
                    deadline_ns = clock_start + work_ns
                    seq, group, states = 0, None, None
                    try:
                        request = _read_exact(client, deadline_ns)
                        seq, reason, group, states = guard.evaluate(request, deadline_ns)
                    except TimeoutError:
                        reason = Reason.TIME_BUDGET
                    except (OSError, RuntimeError, ValueError, ArithmeticError):
                        reason = Reason.INTERNAL
                    proof_end = time.monotonic_ns()
                    response = RESPONSE.pack(MAGIC, seq, reason == Reason.PERMIT, int(reason))
                    trace.write(json.dumps({
                        "sequence": seq, "group": group, "peer_pid": pid, "reason": reason.name,
                        "receipt_ns": clock_start, "proof_complete_ns": proof_end,
                        "issued_measured_proposed_rad": states.tolist() if states is not None else None,
                    }, separators=(",", ":"), allow_nan=False) + "\n")
                    trace.flush()
                    guard.publish_decision(reason, states)
                    time_left = (deadline_ns - time.monotonic_ns()) / 1e9
                    if time_left > 0:
                        client.settimeout(time_left)
                        try:
                            client.sendall(response)
                        except OSError:
                            pass
        finally:
            addr.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("live-model", "signed-model", "model-receipt", "socket", "ready", "trace"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--budget-ms", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.budget_ms <= 100:
        parser.error("The isolated IPC guard budget must be in [1, 100] ms")
    if digest(args.model_receipt) != EXACT_MODEL_RECEIPT_SHA:
        parser.error("The audited model manifest SHA differs")
    if args.ready.exists() or args.trace.exists():
        parser.error("The receipt and trace paths must be new")
    expected = json.loads(args.model_receipt.read_bytes())["model_identity"]
    live = verified_robot(args.live_model, LIVE_SHA, expected["live"])
    signed = verified_robot(args.signed_model, SIGNED_SHA, expected["signed"])
    guard = Guard((live, signed))
    signal.signal(signal.SIGTERM, lambda _signal, _frame: signal.raise_signal(signal.SIGINT))
    try:
        serve(args, guard)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
