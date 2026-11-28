from pathlib import Path
import collections
import hashlib
import json
import sys
import numpy as np

run, out = map(Path, sys.argv[1:])
receipt = json.loads((run / "receipt.json").read_bytes())
events = [json.loads(line) for line in (run / "diagnostic-events.jsonl").read_bytes().splitlines()]
first_fault = next(i for i, event in enumerate(events) if any(x["fault"] for x in event["safety"]))
terminal = events[first_fault]
safety_index = next(i for i, x in enumerate(terminal["safety"]) if x["fault"])
directory = next(row["directory"] for row in receipt["partitions"] if row["partition"] == terminal["partition"])
prefix = run / directory
array = lambda name: np.load(prefix / (name + ".npy"), allow_pickle=False)
i = terminal["tick"]
done = int(array("applied_substeps")[i])
sequence = np.asarray([True] * done + [False] * (10 - done))
assert done == 2 * safety_index and np.array_equal(array("physical_state_mask")[i], sequence)
p = done - 1
frame = {name: array(name)[i, p] for name in ("physical_q", "physical_qd", "q_applied", "actuator_ctrl", "actuator_torque")}
m = receipt["robot_model_servo"]
names = m["joints"]
limits = np.r_[np.full(14, .027 / .02), np.full(40, .10 / .02)]
effort_limit = np.array([m["drive"][name]["effort"] for name in names])
in_tolerance = abs(frame["physical_qd"]) <= limits + 2e-5
assert in_tolerance.all()
assert (abs(frame["actuator_torque"]) <= effort_limit + 1e-4).all()
proof_input = array("proof_inputs")[i, safety_index]
assert np.array_equal(proof_input[2], frame["physical_q"])
assert not bool(array("q_issued_mask")[i, safety_index])
assert (array("substep_event")[i, :done] == 0).all()
assert not array("substep_contact")[i, :done].any()
past_qd = array("physical_qd")[array("physical_state_mask")]
value = abs(past_qd).max(0)
group_ranges = ((0, 7), (7, 14), (14, 34), (34, 54))
def group_peaks(sample, limit):
    rows = []
    for a, b in group_ranges:
        j = a + int(np.argmax(abs(sample[a:b]) / limit[a:b]))
        rows.append({"axis": names[j], "value": float(sample[j]), "limit": float(limit[j]),
                     "absolute_ratio": float(abs(sample[j]) / limit[j])})
    return rows
def last_rows(arrays):
    return {k: dict(zip(names, map(float, val), strict=True)) for k, val in arrays.items()}
partitions = [run / row["directory"] for row in receipt["partitions"]]
hashes = {}
prefix_gap, prefix_depth = [], []
prefix_events = collections.Counter()
for part in partitions:
    a = lambda name: np.load(part / (name + ".npy"), allow_pickle=False)
    mask = a("physical_state_mask")
    if mask.any():
        prefix_gap.extend(a("substep_clearance_m")[mask].tolist())
        prefix_depth.extend(a("substep_penetration_m")[mask].tolist())
        prefix_events.update(a("substep_event")[mask].tolist())
for name in ("physical_q", "physical_qd", "proof_inputs", "q_issued", "q_issued_mask",
             "q_applied", "actuator_ctrl", "actuator_torque", "substep_clearance_m",
             "substep_event", "substep_penetration_m", "substep_contact", "physical_state_mask"):
    hashes[name] = hashlib.sha256((prefix / (name + ".npy")).read_bytes()).hexdigest()
out.write_text(json.dumps({
  "schema": "s05.dcr010.root5.grouped_terminal_archive_audit.v1",
  "no_simulation_step_executed": True,
  "partition": terminal["partition"], "partition_tick": i,
  "recorded_time_ns": terminal["time_ns"], "historical_event_index": first_fault,
  "rejected_subcycle_zero_based": safety_index, "terminal_archive_events": events[first_fault-3:first_fault+1],
  "last_completed_substeps_on_terminal_tick": done,
  "last_applied_command": last_rows(frame | {"q_issued": proof_input[0], "q_filtered_f64": proof_input[1],
                                          "q_proposed": proof_input[3]}),
  "terminal_motion_witness": {"last_velocity_peaks": group_peaks(frame["physical_qd"], limits),
      "last_torque_peaks": group_peaks(frame["actuator_torque"], effort_limit),
      "prefix_velocity_peaks": group_peaks(value, limits),
      "last_gap_m": float(array("substep_clearance_m")[i, p]),
      "last_depth_m": float(array("substep_penetration_m")[i, p]),
      "physical_event_codes": dict(prefix_events), "prefix_minimum_gap_m": min(prefix_gap),
      "prefix_maximum_penetration_m": max(prefix_depth)},
  "partition_array_sha256": hashes,
}, allow_nan=False, indent=2) + "\n")
print(json.dumps(json.loads(out.read_text())["terminal_motion_witness"]))
print("CHECK_TERMINAL_ARCHIVE_PASS", len(prefix_gap), hashlib.sha256(out.read_bytes()).hexdigest())
