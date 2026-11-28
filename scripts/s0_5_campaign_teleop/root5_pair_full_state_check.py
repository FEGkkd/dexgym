#!/usr/bin/env python3
"""Check all named-pair proposal gaps again without common-frame reduction."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main(run, model, proof_ledger, out):
    expected = json.loads((proof_ledger.parent / "summary.json").read_text())
    assert digest(proof_ledger) == expected["request_ledger_sha256"]
    assert digest(model) == expected["model_sha256"]
    assert mj.__version__ == expected["mujoco"]
    signed_receipt = run / "receipt.json"
    assert digest(signed_receipt) == expected["receipt_sha256"]
    rep = json.loads(signed_receipt.read_bytes())
    partitions = {p["partition"]: p for p in rep["partitions"]}
    robot = RobotModel(model)
    m, d = robot.model, robot.data
    a, b = robot.collision_pairs[2425]
    classifications = Counter()
    recent = None
    worst = {"gap_m": float("inf")}
    mismatches = 0
    absolute_gap_delta_m = 0.
    with proof_ledger.open() as rows:
        for row in map(json.loads, rows):
            if row["partition"] != recent:
                recent = row["partition"]
                part = partitions[recent]
                commands = run / part["directory"] / "nominal_proposal.npy"
                assert digest(commands) == part["arrays"]["nominal_proposal"]["sha256"]
                values = np.load(commands, allow_pickle=False, mmap_mode="r")
            d.qpos[robot.q_indices] = values[row["tick"], row["subcycle"]]
            mj.mj_kinematics(m, d)
            gap = float(mj.mj_geomDistance(m, d, a, b, .1, None))
            if not np.isfinite(gap):
                raise ValueError("Non-finite exact full-state named-pair gap")
            projected = row["independent_right_pair_gaps_m"][2]
            absolute_gap_delta_m = max(absolute_gap_delta_m, abs(projected-gap))
            classify = lambda x: "PENETRATION" if x < 0 else "CLEARANCE_VIOLATION" if x < .0005 else "MEETS_CLEARANCE"
            label = classify(gap)
            classifications[label] += 1
            mismatches += classify(projected) != label
            if gap < 0.:
                mj.mj_fwdPosition(m, d)
                depths = [float(contact.dist) for contact in d.contact
                          if {int(contact.geom[0]),int(contact.geom[1])} == {a,b}]
                classifications["PENETRATION_STATIC_CONTACT_CONFIRMED" if depths and min(depths) < 0.
                                else "PENETRATION_WITHOUT_STATIC_CONTACT_WITNESS"] += 1
            if gap < worst["gap_m"]:
                worst = {key: row[key] for key in ("line", "partition", "tick", "subcycle", "time_ns")}
                worst["gap_m"] = gap
                worst["static_pair_contacts_m"] = depths if gap < 0 else None
    result = {"scope": "DIAGNOSTIC_ONLY_NO_MJ_STEP", "source_ledger_sha256": digest(proof_ledger),
              "script_sha256": digest(Path(__file__)), "runtime_assets_sha256": expected["model_sha256"],
              "pair_ids": [a, b], "threshold_classifications": dict(classifications),
              "difference_in_classification_vs_common_frame": mismatches,
              "max_abs_gap_delta_m_vs_common_frame": absolute_gap_delta_m, "worst": worst}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(*(Path(value).resolve() for value in sys.argv[1:]))
