"""Execute bounded paired Servo checks on pre-traced DCR-008 static witnesses."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ...p02.model import RobotModel
from .candidate_servo_check import PairedStaticServo, TICKS
from .evaluate_teleop_candidate import GEOMETRY_SHAPES
from .schema import file_sha, write_json


def trace_path(run, partition):
    path = (run / partition["trace"]).resolve()
    if not path.is_relative_to(run) or file_sha(path) != partition["trace_sha256"]:
        raise ValueError("Candidate trace path or SHA differs from the frozen run")
    return path


def select_points(run, decision, limit):
    """Take at most one proved kinematic point per original source."""
    ranked, source_counts = {}, Counter()
    for split, evidence in decision["split_results"].items():
        for partition in evidence["partitions"]:
            key = split, partition["root"]
            path = trace_path(run, partition)
            with np.load(path, allow_pickle=False) as trace:
                ticks = trace["tick_ns"]
                candidates = np.flatnonzero(trace["witness_ok"])
                source_counts[key] += len(candidates)
                for i in candidates:
                    stamp = int(ticks[i])
                    priority = sha256(f"49:{split}:{partition['root']}:{partition['partition']}:{stamp}".encode()).hexdigest()
                    if key not in ranked or priority < ranked[key]["priority"]:
                        ranked[key] = dict(priority=priority, split=split, root=partition["root"],
                                           partition=partition["partition"], trace=partition["trace"],
                                           trace_sha256=partition["trace_sha256"], index=int(i), tick_ns=stamp)
    rows = sorted(ranked.values(), key=lambda item: item["priority"])[:limit]
    return rows, [{"split": key[0], "root": key[1], "kinematic_witness_count": count}
                  for key, count in sorted(source_counts.items())]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--candidate-run", type=Path, required=True)
    cli.add_argument("--comparison-sha256", required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--physical-model-signature-sha256", required=True)
    cli.add_argument("--max-pairs", type=int, default=32, choices=range(1, 33))
    cli.add_argument("--allow-bounded-train", action="store_true")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    candidate_run = args.candidate_run.resolve()
    manifest = candidate_run / "candidate_comparison.json"
    if file_sha(manifest) != args.comparison_sha256:
        raise ValueError("Candidate comparison SHA does not match")
    decision = json.loads(manifest.read_bytes())
    if decision["official_admission"] or decision["eligible_for_training"]:
        raise ValueError("Expected a DCR-008 candidate-only trace")
    if not decision["full_comparison"] and not (
        args.allow_bounded_train and set(decision["split_results"]) == {"Train"}
    ):
        raise ValueError("Require either a full comparison or an explicitly bounded Train smoke")
    args.out.mkdir(parents=True, exist_ok=False)
    rows, sources = select_points(candidate_run, decision, args.max_pairs)
    write_json(args.out / "selection.json", {
        "rule": "Seed 49 hash rank, one pre-proved kinematic witness at most per signed source",
        "candidate_comparison_sha256": args.comparison_sha256, "selected": rows, "sources": sources,
        "maximum_authorized_control_transitions": args.max_pairs * 2 * TICKS,
        "planned_control_transitions": len(rows) * 2 * TICKS,
    })
    robot = RobotModel(args.model)
    check = PairedStaticServo(robot, expected_model_signature_sha256=args.physical_model_signature_sha256)
    cases, counts = [], Counter()
    for ordinal, item in enumerate(rows):
        with np.load(trace_path(candidate_run, item), allow_pickle=False) as trace:
            index = item["index"]
            if not trace["witness_ok"][index] or int(trace["tick_ns"][index]) != item["tick_ns"]:
                raise ValueError("Selected witness differs from the signed selection list")
            geometry = {name: trace["mapped_" + name][index] for name in GEOMETRY_SHAPES}
            result = check.evaluate(trace["q_nominal"][index], trace["witness_q"][index], geometry,
                                    trace["pinch_active"][index], source_tick_ns=item["tick_ns"])
        output = args.out / f"case-{ordinal:02d}.json"
        write_json(output, {"selection": item, "observed": result})
        counts[result["status"]] += 1
        counts["actual_physical_transitions"] += result["actual_physical_transitions"]
        counts["actual_physics_substeps"] += result["actual_physics_substeps"]
        cases.append({"split": item["split"], "root": item["root"], "partition": item["partition"],
                      "tick_ns": item["tick_ns"], "status": result["status"], "evidence": output.name,
                      "sha256": file_sha(output), "score": result.get("score")})
        print(f"case={ordinal} split={item['split']} status={result['status']} "
              f"actual={result['actual_physical_transitions']}", flush=True)
    write_json(args.out / "summary.json", {
        "scope": "DCR008_ACTUAL_STATIC_MUJOCO_WITNESS_ONLY", "formal_contract_frozen": False,
        "candidate_comparison_sha256": args.comparison_sha256, "selection_sha256": file_sha(args.out / "selection.json"),
        "scores": dict(counts), "planned_pairs": len(rows), "planned_control_transitions": len(rows) * 2 * TICKS,
        "observed_source_counts": Counter(case["split"] for case in cases), "cases": cases, "actual_d10": "NOT_RUN",
    })


if __name__ == "__main__":
    main()
