from itertools import product
from pathlib import Path
from time import monotonic_ns
import hashlib
import json
import sys
import mujoco as mj
import numpy as np
from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep
from replay_dcr009_guarded_diagnostic import SIGNED_SHA
from swept_live_guard import DISTANCE_ROUNDING_MARGIN_M, verified_robot
from swept_live_guard_v2 import SweptEnvelopeV2

run, terminal_path, model, model_receipt, out = map(Path, sys.argv[1:])
terminal, receipt = (json.loads(p.read_bytes()) for p in (terminal_path, run / "receipt.json"))
first = next(row for row in receipt["partitions"] if row["partition"] == terminal["partition"])
input_path = run / first["directory"] / "proof_inputs.npy"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(input_path) == terminal["partition_array_sha256"]["proof_inputs"]
t, s = terminal["partition_tick"], terminal["rejected_subcycle_zero_based"]
snapshot = np.load(input_path, allow_pickle=False)[t, s]
issued, filtered, measured, requested = snapshot
states = np.stack([issued, filtered, measured, issued])
low, high = states.min(axis=0), states.max(axis=0)
robot = verified_robot(model, SIGNED_SHA, json.loads(model_receipt.read_bytes())["model_identity"]["signed"])
assert np.isfinite(states).all()
hard_valid_box = bool((low >= robot.bounds[:, 0] - 1e-9).all() and (high <= robot.bounds[:, 1] + 1e-9).all() and
                      (np.maximum(robot.coupling * low, robot.coupling * high).sum(axis=1) <= COUPLING_LIMIT + 1e-9).all())
assert hard_valid_box
sweep = NativeSweep(robot, max_queries=8191, max_depth=18)
status = terminal["terminal_archive_events"][-1]["safety"][-1]
failing_pairs = status["hold_metrics"]["unresolved_pair_indices"]
assert failing_pairs == [2425]
pair = failing_pairs[0]
a, b = robot.collision_pairs[pair]
pair_names = [robot.model.geom(x).name for x in (a, b)]
axes = np.flatnonzero(sweep.weights[pair])
assert len(axes) == 8
d = mj.MjData(robot.model)
def gap(q):
    d.qpos[robot.q_indices] = q
    mj.mj_kinematics(robot.model, d)
    return float(mj.mj_geomDistance(robot.model, d, a, b, .1, None))
def named(q):
    return dict(zip(robot.names, map(float, q), strict=True))
def proof(depth):
    g = SweptEnvelopeV2(robot, max_queries=8191, max_depth=depth)
    start = monotonic_ns()
    result = g.prove(low, high, start + 2_000_000_000)
    return {"verdict": result.name, "max_queries":8191,"depth_limit":depth,
            "wall_ms":(monotonic_ns() - start) / 1e6, **g.last_metrics}
corners = []
for corner in product((0, 1), repeat=len(axes)):
    point = .5 * (low + high)
    point[axes] = np.where(corner, high[axes], low[axes])
    corners.append((gap(point), corner, point))
min_gap, corner, witness = min(corners, key=lambda x: x[0])
minimum_endpoint = min(gap(q) for q in states)
assert (witness >= low).all() and (witness <= high).all()
d.qpos[robot.q_indices] = witness
mj.mj_fwdPosition(robot.model, d)
witness_contacts = [{"geoms":[robot.model.geom(int(m)).name for m in contact.geom],
                     "distance_m":float(contact.dist)} for contact in d.contact[:d.ncon]]
proof18, proof24, proof26 = [proof(depth) for depth in (18, 24, 26)]
result = {
  "scope":"DIAGNOSTIC_ONLY_STATIC_KINEMATICS_NO_MJ_STEP","model_sha256":sha(model),
  "exact_proof_inputs_sha256":sha(input_path),"original_run_receipt_sha256":sha(run/"receipt.json"),
  "terminal_archive_summary_sha256":sha(terminal_path),"same_issued_endpoint_verified":True,
  "recorded_proof":status,"hard_bounds_and_coupling_for_full_hold_box":hard_valid_box,
  "pair_index":pair,"pair_names":pair_names,"relative_axes":[robot.names[i] for i in axes],
  "endpoint_gaps_m":dict(zip(("issued","filtered","measured","new_request"),[gap(q) for q in snapshot])),
  "box_min_endpoint_m":minimum_endpoint,"production_base_clearance_m":robot.clearance_m,
  "rounding_margin_m":DISTANCE_ROUNDING_MARGIN_M,
  "box_lower_full54":named(low),"box_upper_full54":named(high),
  "sampled_pair_relative_corners":len(corners),
  "minimum_corner_gap_m":min_gap,"minimum_corner_index":corner,"minimum_corner_full54":named(witness),
  "corner_valid_base_clearance_counterexample":bool(min_gap < robot.clearance_m),
  "minimum_corner_static_contacts":witness_contacts,
  "all_corner_gaps_m":[x[0] for x in corners],"reproof": [proof18, proof24, proof26],
}
with out.open("x") as file:
    json.dump(result,file,allow_nan=False,indent=2)
    file.write("\n")
print(json.dumps({k: result[k] for k in ("pair_names","endpoint_gaps_m","minimum_corner_gap_m",
         "corner_valid_base_clearance_counterexample","minimum_corner_static_contacts","reproof")}))
print("READ_ONLY_TERMINAL_REPROOF_SHA256",sha(out))
