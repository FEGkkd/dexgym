"""Offline native FK-to-IK roundtrip and original/projected target diagnostics."""
import argparse
from itertools import islice
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...p02.model import RobotModel
from ..task.native_sweep import NativeSweep
from .nominal import GROUPS, Nominal, SDKFailure, matrix, sdk_context
from .reader import RobotReferenceBank
from .schema import file_sha, write_json


PHASE1_REVISION = "6d0b0b5b2601618f1b0052d93ef2403eea9b064a"
# Pinned phase1 sim.yaml and TeleopCore::process_arm. This is geometry-only:
# the bridge supplies neither the live START-latched NSP nor production guards.
PHASE1_DT_S = 1 / 250
PHASE1_TRANSLATION_RATE_M_S = 0.2592
PHASE1_ROTATION_RATE_RAD_S = 1.0857344210806326
PHASE1_BACKOFF_SCALES = (1.0, 0.5, 0.25, 0.1)


def probe(marvin, which, target, seed):
    try:
        joints = marvin.ik(which, target, seed)
        returned = marvin.fk(which, joints)
        return {
            "succeeded": True, "q_rad": joints.tolist(),
            "translation_error_m": float(np.linalg.norm(target[:3, 3] - returned[:3, 3])),
            "rotation_error_rad": float(Rotation.from_matrix(target[:3, :3].T @ returned[:3, :3]).magnitude()),
        }
    except SDKFailure as error:
        return {"succeeded": False, "reason": str(error)}


def phase1_cartesian_trials(origin, projected):
    """Reproduce phase1's nominal-tick Cartesian clip and backoff geometry."""
    delta = projected[:3, 3] - origin[:3, 3]
    angle = Rotation.from_matrix(projected[:3, :3] @ origin[:3, :3].T).as_rotvec()
    distance, radians = np.linalg.norm(delta), np.linalg.norm(angle)
    translation_cap = PHASE1_TRANSLATION_RATE_M_S * PHASE1_DT_S
    rotation_cap = PHASE1_ROTATION_RATE_RAD_S * PHASE1_DT_S
    if distance > translation_cap:
        delta *= translation_cap / distance
    if radians > rotation_cap:
        angle *= rotation_cap / radians
    for scale in PHASE1_BACKOFF_SCALES:
        rotation = Rotation.from_rotvec(scale * angle).as_matrix() @ origin[:3, :3]
        yield scale, matrix(origin[:3, 3] + scale * delta, rotation)


def probe_partition(part, nominal, marvin, home, count, *, phase1_geometry):
    arrays = part.arrays
    candidates = np.flatnonzero(arrays["domain_mapped_valid"])
    selection = candidates[np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)]
    for tick in selection:
        positions = arrays["mapped_wrist_xyz"][tick]
        rotations = Rotation.from_quat(arrays["mapped_wrist_xyzw"][tick]).as_matrix()
        for side in range(2):
            seed = home[GROUPS[side]]
            world = matrix(positions[side], rotations[side])
            wanted = np.linalg.solve(nominal.bases[side], world @ nominal.wrist_to_native[side])
            row = {
                "partition": part.key, "root": part.root, "tick": int(tick), "side": side,
                "wrist_W_xyz_m": positions[side].tolist(), "direct": probe(marvin, side, wanted, seed),
            }
            workspace = marvin.workspace_probe(side, wanted)
            if workspace:
                projected = workspace["projected_pose"]
                row["workspace"] = {field: value for field, value in workspace.items() if field != "projected_pose"}
                row["projected_diagnostic"] = probe(marvin, side, projected, seed)
                if phase1_geometry:
                    origin = marvin.fk(side, seed)
                    row["phase1_single_step_diagnostic"] = [
                        {
                            "scale": scale, "sdk": probe(marvin, side, attempt, seed),
                            "request_offset_m": float(np.linalg.norm(wanted[:3, 3] - attempt[:3, 3])),
                            "request_offset_rad": float(
                                Rotation.from_matrix(wanted[:3, :3].T @ attempt[:3, :3]).magnitude()
                            ),
                        }
                        for scale, attempt in phase1_cartesian_trials(origin, projected)
                    ]
            yield row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--marvin-bridge", type=Path, required=True)
    cli.add_argument("--marvin-config", type=Path, required=True)
    cli.add_argument("--wuji-sdk", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--rmb", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--max-partitions", type=int, default=1)
    cli.add_argument("--ticks-per-partition", type=int, default=8)
    cli.add_argument("--phase1-geometry", action="store_true",
                     help="probe one nominal 250 Hz tick from simulated home without command admission")
    args = cli.parse_args()
    if args.max_partitions < 1 or args.ticks_per_partition < 1:
        cli.error("max partitions and ticks per partition must be positive")
    model = RobotModel(args.model)
    bank = RobotReferenceBank(args.rmb, verify=True, require_qualified=False)
    stack, marvin, hand = sdk_context(args.marvin_bridge, args.marvin_config, args.wuji_sdk)
    results = {
        "scope": "DIAGNOSTIC_ONLY", "split": "Train", "home_and_pulses": [], "mapped_targets": [],
        "manifest_sha256": file_sha(bank.directory / "rmb_manifest.json"),
        "partition_limit": args.max_partitions, "ticks_per_partition": args.ticks_per_partition,
        "sampled_partitions": [],
    }
    if args.phase1_geometry:
        results["phase1_profile"] = {
            "revision": PHASE1_REVISION, "dt_s": PHASE1_DT_S, "origin": "SIGNED_SIMULATION_HOME_FK",
            "translation_rate_m_s": PHASE1_TRANSLATION_RATE_M_S, "rotation_rate_rad_s": PHASE1_ROTATION_RATE_RAD_S,
            "scales": PHASE1_BACKOFF_SCALES, "eligibility_credit": False,
            "method": "CARTESIAN_CLIP_AND_BACKOFF_GEOMETRY_WITH_OFFLINE_SDK",
            "reference_blobs": {
                "src/whole_body_teleop/src/teleop_core.cpp": "da49daa718b8169f78074a6c9508540584045c9e",
                "src/whole_body_teleop/src/kinematics_adapter.cpp": "ad224f27881fb343ae812e27ee8063b05337b2db",
                "src/whole_body_teleop/config/real.yaml": "3ad6456c550992716e9c6d7f94395dd468c3ad0d",
                "src/whole_body_teleop/config/sim.yaml": "14b7537d8d18ab198385dbce810abe7f29a75597",
            },
            "excluded": ["live_nsp_and_start", "axis_projection", "retryability_and_command_admission"],
        }
    with stack:
        nominal = Nominal(model, marvin, hand, np.repeat([.04, .10], [14, 40]), NativeSweep(model))
        for side in range(2):
            home = model.home[GROUPS[side]]
            for axis in (-1, *range(7)):
                q = home.copy()
                if axis >= 0:
                    q[axis] += 0.04
                wanted = marvin.fk(side, q)
                results["home_and_pulses"].append({"side": side, "axis": axis,
                                                   **probe(marvin, side, wanted, home)})
        for part in islice(bank.partitions("Train"), args.max_partitions):
            arrays = part.arrays
            results["sampled_partitions"].append({
                "partition": part.key, "root": part.root, "planned_ticks": len(arrays["tick_ns"]),
                "hmb_joint_valid_ticks": int(arrays["inputs_pose_joint_valid"].sum()),
                "mapped_valid_ticks": int(arrays["domain_mapped_valid"].sum()),
                "recorded_raw_ik_valid_ticks": arrays["nominal_raw_ik_valid"].sum(axis=0).tolist(),
                "recorded_raw_ik_both_valid_ticks": int(arrays["nominal_raw_ik_valid"].all(axis=1).sum()),
                "recorded_nominal_legal_ticks": int(arrays["domain_nominal_legal"].sum()),
            })
            results["mapped_targets"].extend(
                probe_partition(part, nominal, marvin, model.home, args.ticks_per_partition,
                                phase1_geometry=args.phase1_geometry)
            )
    write_json(args.out, results)


if __name__ == "__main__":
    main()
