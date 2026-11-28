"""Read and verify exact command vectors for escalation and terminal proof events."""

import argparse
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


FIELDS = (
    "proof_inputs", "proof_inputs_mask", "combined_trial", "combined_trial_mask",
    "q_issued", "q_issued_mask", "physical_interval_mask", "physical_state_mask",
    "applied_substeps", "physical_event",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(run, compact, out):
    receipt_path = run / "receipt.json"
    summary = json.loads((compact / "step-status-summary.json").read_bytes())
    ledger = compact / "step-status.jsonl.gz"
    if (sha(ledger) != summary["compressed_step_status_sha256"]
            or sha(receipt_path) != summary["source_receipt_sha256"]):
        raise ValueError("Compact trace or original run receipt hash differs")
    receipt = json.loads(receipt_path.read_bytes())
    if receipt["diagnostic_events_sha256"] != summary["source_diagnostic_events_sha256"]:
        raise ValueError("Compact trace references a different original event log")
    partitions = {part["partition"]: part for part in receipt["partitions"]}
    source_hashes = {}

    @lru_cache(maxsize=1)
    def read_inputs(partition):
        part = partitions[partition]
        arrays, hashes = {}, {}
        for field in FIELDS:
            path = run / part["directory"] / f"{field}.npy"
            field_sha = sha(path)
            if field_sha != part["arrays"][field]["sha256"]:
                raise ValueError(f"Archive field identity differs: {path}")
            arrays[field] = np.load(path, mmap_mode="r", allow_pickle=False)
            hashes[field] = field_sha
        source_hashes[partition] = hashes
        return arrays

    def witness(row, step):
        arrays = read_inputs(row["partition"])
        tick, sub = row["tick"], step["arm_subcycle"]
        target = tick, sub
        masks = {key: bool(arrays[key][target]) for key in (
            "proof_inputs_mask", "combined_trial_mask", "q_issued_mask", "physical_interval_mask",
        )}
        proof_inputs = arrays["proof_inputs"][target] if masks["proof_inputs_mask"] else None
        combined = arrays["combined_trial"][target] if masks["combined_trial_mask"] else None
        actual = arrays["q_issued"][target] if masks["q_issued_mask"] else None
        exact_hold = (None if actual is None or proof_inputs is None else
                      proof_inputs[0].astype(actual.dtype).tobytes() == actual.tobytes())
        escalated = "original_hold_proof" in step["hold_metrics"]
        if escalated and step["hold_proof"] == "PERMIT" and exact_hold is not True:
            raise ValueError("An escalated HOLD issued something other than the previous command")
        if step["fault"] and step["fault"].startswith("HOLD_INFEASIBLE:"):
            if masks["q_issued_mask"] or masks["physical_interval_mask"]:
                raise ValueError("An infeasible HOLD issued a command or advanced physics")
        return {
            **{key: row[key] for key in ("source_row", "partition", "tick", "time_ns", "control_cutoff_ns", "status")},
            "arm_subcycle": sub,
            "proof": step["proof"],
            "proof_metrics": step["metrics"],
            "hold_proof": step["hold_proof"],
            "hold_metrics": step["hold_metrics"],
            "fault": step["fault"],
            "masks": masks,
            "proof_inputs": None if proof_inputs is None else dict(zip(
                ("previous_issued", "previous_filtered", "precycle_measured", "nominal_proposal"),
                proof_inputs.tolist(), strict=True,
            )),
            "combined_trial": None if combined is None else combined.tolist(),
            "actual_issued": None if actual is None else actual.tolist(),
            "actual_is_bit_exact_previous_issued": exact_hold,
            "physical_event": int(arrays["physical_event"][target]),
            "elapsed_substeps_this_20ms_row": int(arrays["applied_substeps"][tick]),
            "physical_state_mask_this_20ms_row": arrays["physical_state_mask"][tick].tolist(),
        }

    escalations, first_fault = [], None
    with gzip.open(ledger, "rt", encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            for step in row["safety"]:
                relevant_fault = first_fault is None and step["fault"] is not None
                escalated = "original_hold_proof" in step["hold_metrics"]
                if relevant_fault or escalated:
                    exact = witness(row, step)
                    if relevant_fault:
                        first_fault = exact
                    if escalated:
                        escalations.append(exact)
    if len(escalations) != sum(summary["unchanged_hold_escalations_by_result"].values()):
        raise ValueError("Exact witness count differs from compact trace summary")
    with out.open("x", encoding="utf-8") as stream:
        json.dump({
            "scope": "DIAGNOSTIC_ONLY_READ_ONLY",
            "source_receipt_sha256": summary["source_receipt_sha256"],
            "source_diagnostic_events_sha256": summary["source_diagnostic_events_sha256"],
            "source_array_sha256": source_hashes,
            "joint_names_in_vector_order": receipt["robot_model_servo"]["joints"],
            "unchanged_hold_escalations": escalations,
            "first_4ms_fault": first_fault,
        }, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"WITNESSES {len(escalations)} escalations, first 4 ms fault present: {first_fault is not None}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("compact", type=Path)
    parser.add_argument("out", type=Path)
    options = parser.parse_args()
    extract(options.run, options.compact, options.out)
