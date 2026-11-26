"""Offline native FK-to-IK roundtrip and original/projected target diagnostics."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...p02.model import RobotModel
from ..task.native_sweep import NativeSweep
from .nominal import GROUPS, Nominal, SDKFailure, matrix, sdk_context
from .reader import RobotReferenceBank
from .schema import write_json


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


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--marvin-bridge", type=Path, required=True)
    cli.add_argument("--marvin-config", type=Path, required=True)
    cli.add_argument("--wuji-sdk", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--rmb", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    model = RobotModel(args.model)
    bank = RobotReferenceBank(args.rmb, verify=True, require_qualified=False)
    part = next(bank.partitions("Train"))
    arr = part.arrays
    stack, marvin, hand = sdk_context(args.marvin_bridge, args.marvin_config, args.wuji_sdk)
    results = {"scope": "DIAGNOSTIC_ONLY", "home_and_pulses": [], "mapped_targets": []}
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
        candidates = np.flatnonzero(arr["domain_mapped_valid"])
        for tick in candidates[np.linspace(0, len(candidates) - 1, min(8, len(candidates)), dtype=int)]:
            positions = arr["mapped_wrist_xyz"][tick]
            rotations = Rotation.from_quat(arr["mapped_wrist_xyzw"][tick]).as_matrix()
            for side in range(2):
                home = model.home[GROUPS[side]]
                world = matrix(positions[side], rotations[side])
                wanted = np.linalg.solve(nominal.bases[side], world @ nominal.wrist_to_native[side])
                row = {"tick": int(tick), "side": side, "wrist_W_xyz_m": positions[side].tolist(),
                       "direct": probe(marvin, side, wanted, home)}
                workspace = marvin.workspace_probe(side, wanted)
                if workspace:
                    projected = workspace.pop("projected_pose")
                    row["workspace"] = workspace
                    row["projected_diagnostic"] = probe(marvin, side, projected, home)
                results["mapped_targets"].append(row)
    write_json(args.out, results)


if __name__ == "__main__":
    main()
