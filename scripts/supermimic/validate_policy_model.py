"""Measure open-loop world-model error on recorded, real Newton policy transitions."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from dexgym.algorithms.supermimic.action import ArmActionMap
from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK
from dexgym.algorithms.supermimic.networks import DeltaWorldModel
from dexgym.algorithms.supermimic.rewards import so3_geodesic
from dexgym.motion.motion_manager import MotionManager
from preflight import REFERENCE, REFERENCE_SHA256, SOURCE, verify


def _stats(array):
    return {"mean": float(array.mean()), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def validate(checkpoint: Path, states_path: Path, output: Path, stride: int = 4):
    verify()
    if stride < 1:
        raise ValueError("stride must be positive")
    record = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if record["reference_sha256"] != REFERENCE_SHA256:
        raise ValueError("Reference hash mismatch")
    hidden = int(record["training"]["configuration"]["world_hidden"])
    world = DeltaWorldModel(hidden).cuda().eval()
    world.load_state_dict(record["world_model"])
    motion = MotionManager(REFERENCE, 1, "cuda")
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    arm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in motion.joint_names_arm]
    limits = torch.as_tensor(model.jnt_range[arm_ids].copy(), device="cuda", dtype=torch.float32)
    origin = motion.data["q_arm"][0]
    scale = (motion.data["q_arm"] - origin).abs().amax(0) + 0.15
    mapping = ArmActionMap(origin, limits[:, 0] + 0.005, limits[:, 1] - 0.005, scale, 0.04)
    fk = DifferentiableMarvinFK(SOURCE, device="cuda")
    with np.load(states_path, allow_pickle=False) as states:
        q = states["q"][:, 0, arm_ids]
        qd = states["qd"][:, 0, arm_ids]
        actions = states["action"][:, 0]
        commands = states["q_command"][:, 0]
    if len(q) != len(actions) + 1 or commands.shape != actions.shape:
        raise ValueError("Newton state and action trace lengths differ")
    command_state = np.concatenate((origin.cpu().numpy()[None], commands), axis=0)
    state = torch.as_tensor(np.concatenate((q, qd, command_state), axis=-1), device="cuda")
    action = torch.as_tensor(actions, device="cuda")
    result = {
        "checkpoint": str(checkpoint),
        "recorded_newton_states": str(states_path),
        "reference_sha256": REFERENCE_SHA256,
        "start_stride": stride,
        "horizons": {},
    }
    with torch.no_grad():
        for horizon in (1, 8, 32):
            if len(q) <= horizon:
                continue
            starts = torch.arange(0, len(q) - horizon, stride, device="cuda")
            predicted = state[starts]
            for step in range(horizon):
                predicted = world(predicted, action[starts + step], mapping)
            truth = state[starts + horizon]
            predicted_pos, predicted_rot = fk(predicted[:, :14])
            actual_pos, actual_rot = fk(truth[:, :14])
            position = (predicted_pos[:, :, 2] - actual_pos[:, :, 2]).norm(dim=-1).cpu().numpy() * 1000
            rotation = so3_geodesic(predicted_rot[:, :, 2], actual_rot[:, :, 2]).cpu().numpy() * 180 / np.pi
            result["horizons"][str(horizon)] = {
                "num_starts": len(starts),
                "joint_position_rmse_rad": float((predicted[:, :14] - truth[:, :14]).square().mean().sqrt()),
                "joint_velocity_rmse_rad_s": float((predicted[:, 14:28] - truth[:, 14:28]).square().mean().sqrt()),
                "left_tcp_position_mm": _stats(position[:, 0]),
                "right_tcp_position_mm": _stats(position[:, 1]),
                "left_tcp_orientation_deg": _stats(rotation[:, 0]),
                "right_tcp_orientation_deg": _stats(rotation[:, 1]),
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(validate(args.checkpoint, args.states, args.output, args.stride), indent=2))
