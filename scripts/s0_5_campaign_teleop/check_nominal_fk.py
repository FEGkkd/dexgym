#!/usr/bin/env python3
"""Compare the live Teleop and signed S0-5 FK at identical named joint values."""

import argparse
import json
from pathlib import Path

import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, digest


def compare(live_path, reference_path):
    live, reference = RobotModel(live_path), RobotModel(reference_path)
    if live.names != reference.names:
        raise ValueError("Expected identical named joint coordinates")
    fields = ("wrists", "rotations", "palm", "palm_rotation", "landmarks", "local")
    maxima = dict.fromkeys(fields, 0.0)
    low = np.maximum(live.bounds[:, 0], reference.bounds[:, 0])
    high = np.minimum(live.bounds[:, 1], reference.bounds[:, 1])
    if np.any(low >= high):
        raise ValueError("No common nonempty domain for one or more joints")
    rng = np.random.default_rng(20261127)
    probes = [live.home, reference.home, (low + high) / 2]
    probes.extend(np.clip(reference.home + delta, low + 1e-8, high - 1e-8)
                  for delta in 0.02 * np.eye(54))
    probes.extend(rng.uniform(low, high) for _ in range(48))
    for q in probes:
        actual, expected = live.geometry(q), reference.geometry(q)
        for field in fields:
            difference = np.max(np.abs(getattr(actual, field) - getattr(expected, field)))
            maxima[field] = max(maxima[field], float(difference))
    return {
        "scope": "NAMED_KINEMATICS_ONLY",
        "actual_xml_sha256": digest(live_path),
        "reference_xml_sha256": digest(reference_path),
        "max_abs_component_difference": maxima,
        "max_home_qdiff_rad": float(np.max(np.abs(live.home - reference.home))),
        "max_joint_bound_diff_rad": float(np.max(np.abs(live.bounds - reference.bounds))),
        "tested_q_count": len(probes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live_model", type=Path)
    parser.add_argument("signed_model", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.live_model, args.signed_model), indent=2))


if __name__ == "__main__":
    main()
