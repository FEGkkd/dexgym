#!/usr/bin/env python3
"""Single-root MuJoCo counterfactual for one unproven historical arm sweep."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

import swept_live_guard as base

ROOT_FIVE = "capture:b63bc72d2e8b2ee395e966f6"
MODE = "SIMULATION_ONLY_ALLOW_ONE_UNPROVEN_ARM_SWEEP"


class SinglePassGuard(base.Guard):
    def __init__(self, robots, stage, archive):
        super().__init__(robots)
        self.stage, self.archive, self.used = stage, archive, False

    def evaluate(self, packet, deadline_ns):
        if (self.used or len(packet) != base.REQUEST.size
                or base.REQUEST.unpack_from(packet)[2] != 0 or self.stage.read_bytes() != b"HISTORY\n"):
            return super().evaluate(packet, deadline_ns)
        observed = {}
        originals = [envelope.prove for envelope in self.envelopes]

        def recorded_proof(model, prove):
            def check(low, high, deadline):
                status = prove(low, high, deadline)
                observed[model] = status.name
                # Evaluate the other model under the original global deadline
                # rather than returning its untested result as a permit.
                return base.Reason.PERMIT if status == base.Reason.SWEEP_UNPROVEN else status
            return check

        try:
            for model, envelope, proof in zip(("live", "signed"), self.envelopes, originals, strict=True):
                envelope.prove = recorded_proof(model, proof)
            outcome = super().evaluate(packet, deadline_ns)
        finally:
            for envelope, proof in zip(self.envelopes, originals, strict=True):
                envelope.prove = proof
        sequence, reason, group, states = outcome
        if reason == base.Reason.PERMIT and "SWEEP_UNPROVEN" in observed.values():
            self.used = True
            self.archive.write(json.dumps({"mode": MODE, "root": ROOT_FIVE, "sequence": sequence, "group": group,
                                           "model_checks": observed, "decision_mono_ns": time.monotonic_ns(),
                                           "issued_measured_proposed_rad": states.tolist()}, allow_nan=False) + "\n")
            self.archive.flush()
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("live-model", "signed-model", "model-receipt", "socket", "ready", "trace", "stage", "exception-log", "mode-receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--budget-ms", type=int, choices=(25,), required=True)
    parser.add_argument("--root", choices=(ROOT_FIVE,), required=True)
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    if (stage != args.socket.resolve().parent / "stage" or args.stage.is_symlink()
            or stage.stat().st_uid != os.getuid() or stage.stat().st_mode & 0o077
            or stage.read_bytes() != b"PREPOSITION\n" or os.environ.get("ROS_DOMAIN_ID") != "213"
            or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
            or base.digest(args.model_receipt) != base.EXACT_MODEL_RECEIPT_SHA):
        raise ValueError("Require the isolated root-five MuJoCo profile and its owner-only START marker")
    models = json.loads(args.model_receipt.read_bytes())["model_identity"]
    robots = (base.verified_robot(args.live_model, base.LIVE_SHA, models["live"]),
              base.verified_robot(args.signed_model, base.SIGNED_SHA, models["signed"]))
    with args.exception_log.open("x", encoding="utf-8", buffering=1) as log:
        with args.mode_receipt.open("x", encoding="utf-8") as receipt:
            json.dump({"mode": MODE, "root": ROOT_FIVE, "max_exceptions": 1, "stage": "HISTORY",
                       "wrapper_sha256": base.digest(__file__), "strict_guard_sha256": base.digest(base.__file__),
                       "formal_coverage_credit": False}, receipt, indent=2)
            receipt.write("\n")
        signal.signal(signal.SIGTERM, lambda _signal, _frame: signal.raise_signal(signal.SIGINT))
        try:
            base.serve(args, SinglePassGuard(robots, stage, log))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
