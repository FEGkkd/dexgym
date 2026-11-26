"""Run a bounded, signed-model, synthetic Warp physics preflight."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from ...p02.model import RobotModel
from . import JOINTS, SUBSTEPS
from .physics import ServoProfile, signature
from .safety import Stop
from .smoke import make_task, target_frame


MODEL_MANIFEST_SHA256 = "92c44cabe65b278ab8261c2659c2cfa8c670fad07fd7d3a640e8629bc5c06a0d"
MODEL_MANIFEST = Path("docs/s0-5/evidence/P02/refinement/model_assets_sha256.json")
MAX_ENVIRONMENTS = 64
MAX_TRANSITIONS = 25_000
AMPLITUDE = 0.015


def validate_workload(environments, steps, transition_budget):
    if not 1 <= environments <= MAX_ENVIRONMENTS or steps < 1:
        raise ValueError(f"Expected 1–{MAX_ENVIRONMENTS} environments and at least one control tick")
    if not 1 <= transition_budget <= MAX_TRANSITIONS or environments * steps > transition_budget:
        raise ValueError(f"The planned control transitions must fit the supplied limit, at most {MAX_TRANSITIONS}")


def fixture_actions(home, tick):
    """Small signed pulses, one joint per arm and hand per lane."""
    action = torch.zeros_like(home)
    lane = torch.arange(len(home), device=home.device)
    for group, (start, size) in enumerate(((0, 7), (7, 7), (14, 20), (34, 20))):
        offset = lane + 3 * tick + 2 * group
        axis = start + offset % size
        sign = 1 - 2 * ((offset // size + tick + group) % 2)
        action[lane, axis] = AMPLITUDE * sign
    return action


def validate_assets(physics_signature, manifest):
    raw = manifest.read_bytes()
    if sha256(raw).hexdigest() != MODEL_MANIFEST_SHA256:
        raise ValueError("Model manifest is not the signed P02 asset manifest")
    expected = {name.split("/generated/")[-1]: digest for name, digest in json.loads(raw).items()}
    if physics_signature["assets"] != expected:
        raise ValueError("Model XML or dependencies differ from the signed P02 assets")


def tick_evidence(result, tick, elapsed):
    event_bits = result.events.tolist()
    finite = (torch.isfinite(result.x1_pre_reset).all(-1)
              & torch.isfinite(result.applied).all(dim=(1, 2))
              & torch.isfinite(result.ctrl).all(dim=(1, 2))
              & torch.isfinite(result.torque).all(dim=(1, 2)))
    return {
        "tick": tick, "elapsed_s": elapsed,
        "actual_transitions": int((result.substeps > 0).sum()),
        "actual_substeps": int(result.substeps.sum()),
        "full_step_transitions": int(result.full_step.sum()),
        "valid_transitions": int(result.valid.sum()),
        "fallbacks": int(result.fallback.sum()),
        "hard_safety_failures": int(result.safety_failure.sum()),
        "terminated": int(result.terminated.sum()),
        "nonfinite_lanes": int((~finite).sum()),
        "event_bits": event_bits,
        "event_counts": {name.name: sum(bool(bits & int(name)) for bits in event_bits) for name in Stop},
    }


@torch.inference_mode()
def run(robot, *, environments, steps, transition_budget, device, manifest, on_tick=None):
    validate_workload(environments, steps, transition_budget)
    profile = ServoProfile()
    physics_signature = signature(robot, profile)
    validate_assets(physics_signature, manifest)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    setup_started = perf_counter()
    task, home, _ = make_task(robot, environments, profile, "B54", device)
    if cuda:
        torch.cuda.synchronize(device)
    setup_s = perf_counter() - setup_started
    if task.num_envs != environments or task.physics.num_envs != environments:
        raise ValueError("Constructed physics population differs from the requested size")
    rows, total_events = [], Counter()
    axis_requests = torch.zeros((2, JOINTS), dtype=torch.int64, device=device)
    started = perf_counter()
    for tick in range(steps):
        tick_started = perf_counter()
        action = fixture_actions(home, tick)
        axis_requests[0] += (action < 0).sum(0)
        axis_requests[1] += (action > 0).sum(0)
        task.observe(target_frame(task, home, tick))
        transition = task.step(action)
        if cuda:
            torch.cuda.synchronize(device)
        evidence = tick_evidence(transition, tick, perf_counter() - tick_started)
        rows.append(evidence)
        total_events.update(evidence["event_counts"])
        if on_tick is not None:
            on_tick(evidence)
        if (evidence["valid_transitions"] != environments or evidence["nonfinite_lanes"]
                or any(evidence["event_bits"])):
            break
    loop_s = perf_counter() - started
    sum_keys = ("actual_transitions", "actual_substeps", "full_step_transitions", "valid_transitions", "fallbacks",
                "hard_safety_failures", "terminated", "nonfinite_lanes")
    totals = {name: sum(row[name] for row in rows) for name in sum_keys}
    plans = environments * steps
    return {
        "scope": "SYNTHETIC_ONLY", "qualification": False, "model": physics_signature,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256, "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "backend": type(task.physics).__name__, "torch_version": str(torch.__version__),
        "requested_envs": environments, "constructed_envs": task.physics.num_envs,
        "requested_ticks": steps, "executed_tick_calls": len(rows),
        "planned_transitions": plans, "planned_substeps": plans * SUBSTEPS, "transition_budget": transition_budget,
        "setup_s": setup_s, "control_loop_s": loop_s,
        "actual_transitions_per_s": totals["actual_transitions"] / loop_s,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
        "memory_scope": "PyTorch CUDA allocator peaks, excluding direct Warp/CUDA allocations" if cuda else None,
        "fixture": {"normalized_pulse_amplitude": AMPLITUDE, "axes_per_lane": 4,
                    "negative_requests_per_joint": axis_requests[0].tolist(),
                    "positive_requests_per_joint": axis_requests[1].tolist()},
        **totals, "event_counts": dict(total_events), "ticks": rows,
        "run_checks_pass": (totals["valid_transitions"] == plans and totals["actual_substeps"] == plans * SUBSTEPS
                            and not any(total_events.values()) and totals["nonfinite_lanes"] == 0),
    }


def log_tick(evidence):
    print(f"tick={evidence['tick']} actual_transitions={evidence['actual_transitions']} "
          f"actual_substeps={evidence['actual_substeps']} valid={evidence['valid_transitions']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, default=MODEL_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-transitions", type=int, default=MAX_TRANSITIONS)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    args = parser.parse_args()
    try:
        validate_workload(args.envs, args.steps, args.max_transitions)
    except ValueError as error:
        parser.error(str(error))
    with args.out.open("x", encoding="utf-8") as stream:
        result = run(RobotModel(args.model), environments=args.envs, steps=args.steps,
                     transition_budget=args.max_transitions, device=args.device, manifest=args.model_manifest,
                     on_tick=log_tick)
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"run_checks_pass={result['run_checks_pass']} scope={result['scope']} "
          f"sha256={sha256(args.out.read_bytes()).hexdigest()}", flush=True)
    return 0 if result["run_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
