"""Measure open-loop World Model forecasts on recorded S0 simulator actions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from dexgym.algorithms.supermimic.dex_reference.networks import FailureHead, WorldModel

from ..marvin54_tracking.model import sha256
from ..marvin54_tracking.task import DT, JOINTS, JointCodec, TrackingGeometry, TrackingTask
from .io import write_json
from .physics import AUDIT_DEPTH_M, COLLISION, JOINT_LIMIT, NONFINITE, OVERFLOW
from .task import TASK_ID, VelocityIntegratedJointCodec


HORIZONS = (1, 8, 16, 24, 32)


def summary(values):
    values = np.concatenate(values)
    return dict(windows=len(values), mean=float(values.mean()), p95=float(np.percentile(values, 95)))


def classification(probabilities, labels):
    probability, label = np.concatenate(probabilities), np.concatenate(labels)
    positive = probability >= 0.5
    hits = int((positive & label).sum())
    return dict(
        transitions=len(label), positives=int(label.sum()), brier=float(np.square(probability - label).mean()),
        precision_at_half=hits / int(positive.sum()) if positive.any() else None,
        recall_at_half=hits / int(label.sum()) if label.any() else None,
    )


@torch.inference_mode()
def measure(wm, failure, geometry, traces, device):
    errors = defaultdict(list)
    contact_probability, contact_label = [], []
    failure_probability, failure_label = [], []
    depth_errors = []
    for path in traces:
        with np.load(path, allow_pickle=False) as trace:
            short = np.flatnonzero(~np.isclose(np.diff(trace["time_s"]), DT, atol=1e-8))
            depth = torch.as_tensor(trace["depth_m"], dtype=torch.float32, device=device)
            steps = int(short[0]) if len(short) else len(depth)
            states = torch.as_tensor(np.concatenate((trace["qpos"], trace["qvel"]), -1),
                                     dtype=torch.float32, device=device)
            targets = torch.as_tensor(trace["target"], dtype=torch.float32, device=device)
            flags = torch.as_tensor(trace["failure_bits"], device=device) if "failure_bits" in trace else None
        valid = torch.isfinite(states[:-1]).all(-1) & torch.isfinite(targets).all(-1) & torch.isfinite(depth)
        if flags is not None:
            valid &= (flags & (NONFINITE | OVERFLOW)) == 0
        x, u = states[:-1][valid], targets[valid]
        if len(x):
            _, logits = wm(x, u, x)
            contact_probability.append(logits[:, 1].sigmoid().cpu().numpy())
            contact_label.append((depth[valid] > AUDIT_DEPTH_M).cpu().numpy())
            if logits.shape[-1] > 2:
                depth_errors.append((torch.nn.functional.softplus(logits[:, 2]) - 1000 * depth[valid]).abs().cpu().numpy())
            if flags is not None:
                failure_probability.append(failure(failure.features(x, u)).sigmoid().cpu().numpy())
                failure_label.append(((flags[valid] & (COLLISION | JOINT_LIMIT)) != 0).cpu().numpy())
        bad = torch.where(~valid[:steps])[0]
        steps = int(bad[0]) if len(bad) else steps
        if not steps:
            continue
        state, target, depth = states[:steps + 1], targets[:steps], depth[:steps]
        _, _, _, true_tips, true_rot = geometry.pose(state[:, :JOINTS])
        predicted, peak = state, torch.zeros_like(state[:, 0])
        for horizon in range(1, min(steps, max(HORIZONS)) + 1):
            prior = predicted[:-1]
            predicted, _ = wm(prior, target[horizon - 1:], prior)
            peak = torch.maximum(peak[:-1], depth[horizon - 1:])
            if horizon in HORIZONS:
                _, _, _, tips, rot = geometry.pose(predicted[:, :JOINTS])
                tip_mm = (tips - true_tips[horizon:]).norm(dim=-1).mean(-1) * 1000
                trace_r = (rot * true_rot[horizon:]).sum(dim=(-1, -2))
                tip_deg = ((trace_r - 1) * 0.5).clamp(-1, 1).acos().rad2deg().mean(-1)
                joint = (predicted[:, :JOINTS] - state[horizon:, :JOINTS]).abs().rad2deg().mean(-1)
                mask = (peak > AUDIT_DEPTH_M).cpu().numpy()
                for key, metric in (("tip_mm", tip_mm), ("tip_deg", tip_deg), ("joint_deg", joint)):
                    metric = metric.cpu().numpy()
                    if not np.isfinite(metric).all():
                        raise FloatingPointError(f"Nonfinite forecast in {path} at horizon {horizon}")
                    errors[horizon, "all", key].append(metric)
                    if mask.any():
                        errors[horizon, "deep_contact", key].append(metric[mask])
    if not errors:
        raise ValueError("No full control intervals found in the traces")
    report = {}
    for (horizon, group, metric), values in errors.items():
        report.setdefault(str(horizon), {}).setdefault(group, {})[metric] = summary(values)
    return dict(
        horizons=report,
        contact=classification(contact_probability, contact_label),
        failure=classification(failure_probability, failure_label) if failure_label else None,
        contact_depth_mm=summary(depth_errors) if depth_errors else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True, help="A mode directory from evaluate --trace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    provenance = json.loads((args.checkpoint.parent / "provenance.json").read_text())
    if sha256(args.model) != provenance["model_sha256"]:
        raise ValueError("MJCF hash differs from the training model")
    traces = sorted(args.traces.glob("*.npz"))
    model = mujoco.MjModel.from_xml_path(str(args.model))
    geometry = TrackingGeometry(model, SimpleNamespace(names=provenance["joint_names"]), args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["task_id"] != TASK_ID:
        raise ValueError("Expected an S0 training checkpoint")
    config = payload["config"]
    state_update = provenance.get("world_model", {}).get("state", "delta")
    codec = VelocityIntegratedJointCodec if state_update == "integrated" else JointCodec
    wm = WorldModel(108, 2, codec, JOINTS, TrackingTask.error_floor, config["wm_hidden"], amp=False,
                    control_error_features=config["control_error_features"],
                    predict_contact_depth=config.get("contact_depth_weight", 0.0) > 0).to(args.device).eval()
    wm.load_state_dict(payload["modules"]["wm"])
    failure = FailureHead(JOINTS, 2 * JOINTS, control_error_features=config["control_error_features"])
    failure.load_state_dict(payload["modules"]["fail_head"])
    failure.to(args.device).eval()
    del payload
    result = measure(wm, failure, geometry, traces, args.device)
    result.update(checkpoint=str(args.checkpoint), trace_directory=str(args.traces), state_update=state_update,
                  clips=len(traces), control_dt_s=DT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
