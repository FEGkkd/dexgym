"""Compare 500-Hz contact points with the authored external cap collision proxy.

The PhysX manifold's signed separation is a pairwise collision result. A
contact-point distance to one cap shape is a different measurement; this
audit reports both without equating either to exact material penetration.
Run with ``scripts/run_isaaclab.sh`` so OpenUSD's ``pxr`` module is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def box_sdf(point: np.ndarray, center: np.ndarray, half: np.ndarray,
            yaw_rad: float) -> float:
    """Signed Euclidean distance to a z-aligned, yaw-rotated cuboid."""
    delta = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    local = np.array((c * delta[0] + s * delta[1],
                      -s * delta[0] + c * delta[1], delta[2]))
    q = np.abs(local) - np.asarray(half, dtype=float)
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))


def cylinder_sdf(point: np.ndarray, center: np.ndarray,
                 radius: float, half_height: float) -> float:
    """Signed Euclidean distance to a z-aligned finite cylinder."""
    local = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    q = np.array((np.linalg.norm(local[:2]) - radius, abs(local[2]) - half_height))
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))


def world_to_cap(point: np.ndarray, pose_xyzw: np.ndarray) -> np.ndarray:
    """Transform a world contact point to the cap rigid-body frame."""
    pose = np.asarray(pose_xyzw, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("cap pose must be finite xyz+xyzw")
    q = pose[3:] / np.linalg.norm(pose[3:])
    xyz = np.asarray(point, dtype=float) - pose[:3]
    v = q[:3]
    # Inverse unit-quaternion rotation, avoiding an optional scipy dependency.
    return xyz - 2.0 * q[3] * np.cross(v, xyz) + 2.0 * np.cross(v, np.cross(v, xyz))


def read_external_proxy(cap_usd: Path) -> tuple[list[tuple], dict]:
    """Read the actual 12 box walls and top cylinder from the USD asset."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(cap_usd))
    if stage is None:
        raise ValueError(f"Cannot open cap USD: {cap_usd}")
    cap = stage.GetPrimAtPath("/Cap")
    if not cap or UsdGeom.Xformable(cap).GetOrderedXformOps():
        raise ValueError("Expected an identity /Cap parent transform")
    walls = []
    offset_attrs = {}
    for i in range(12):
        prim = stage.GetPrimAtPath(f"/Cap/wall_{i:02d}")
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(f"Missing collision wall {i}")
        ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        if set(ops) != {"xformOp:translate", "xformOp:rotateZ", "xformOp:scale"}:
            raise ValueError(f"Unexpected wall transforms: {prim.GetPath()}")
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        walls.append((np.array(ops["xformOp:translate"], dtype=float),
                      np.array(ops["xformOp:scale"], dtype=float) * size / 2.0,
                      math.radians(float(ops["xformOp:rotateZ"]))))
        offset_attrs[str(prim.GetPath())] = [a.GetName() for a in prim.GetAttributes()
                                             if "contactOffset" in a.GetName()
                                             or "restOffset" in a.GetName()]
    top = stage.GetPrimAtPath("/Cap/top")
    if not top or not top.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("Missing top cylinder collision")
    top_ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(top).GetOrderedXformOps()}
    if set(top_ops) != {"xformOp:translate"}:
        raise ValueError("Unexpected top cylinder transform")
    cylinder = (np.array(top_ops["xformOp:translate"], dtype=float),
                float(UsdGeom.Cylinder(top).GetRadiusAttr().Get()),
                float(UsdGeom.Cylinder(top).GetHeightAttr().Get()) / 2.0)
    offset_attrs[str(top.GetPath())] = [a.GetName() for a in top.GetAttributes()
                                      if "contactOffset" in a.GetName()
                                      or "restOffset" in a.GetName()]
    return walls, {"top": cylinder, "offset_attrs": offset_attrs}


def audit(trace: Path, cap_usd: Path) -> dict:
    data = json.loads(trace.read_text(encoding="utf-8"))
    walls, extra = read_external_proxy(cap_usd)
    top_center, top_radius, top_half_height = extra["top"]
    pairs = []
    for step in data["rows"]:
        cap_pose = np.asarray(step["cap_pose_world_xyzw"], dtype=float)
        for contact in step["right_cap_contact_points"]:
            local = world_to_cap(contact["point_world_m"], cap_pose)
            cap_sdf = min(*(box_sdf(local, center, half, yaw)
                            for center, half, yaw in walls),
                          cylinder_sdf(local, top_center, top_radius, top_half_height))
            pairs.append((cap_sdf, float(contact["signed_separation_m"])))
    if not pairs:
        raise ValueError("Trace has no right-cap contact points")
    pairs = np.asarray(pairs)
    quantiles = (0, 10, 25, 50, 75, 90, 100)
    worst = pairs[int(np.argmin(pairs[:, 1]))]
    return {
        "trace": str(trace), "cap_usd": str(cap_usd),
        "cap_usd_sha256": hashlib.sha256(cap_usd.read_bytes()).hexdigest(),
        "right_cap_contact_points": len(pairs),
        "external_proxy": "12 USD collision cubes plus top collision cylinder",
        "authored_contact_or_rest_offset_shapes": sum(bool(x) for x in extra["offset_attrs"].values()),
        "cap_proxy_sdf_mm_percentiles": dict(zip(map(str, quantiles),
                                              np.percentile(pairs[:, 0] * 1000, quantiles).tolist())),
        "physx_signed_separation_mm_percentiles": dict(zip(map(str, quantiles),
                                                         np.percentile(pairs[:, 1] * 1000, quantiles).tolist())),
        "cap_sdf_vs_physx_separation_pearson_r": float(np.corrcoef(pairs.T)[0, 1]),
        "worst_physx_separation_mm": float(worst[1] * 1000),
        "cap_proxy_sdf_at_worst_mm": float(worst[0] * 1000),
        "interpretation": "Contact-point cap-proxy SDF is not pairwise geometric penetration or effective PhysX contact offset.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cap-usd", type=Path, default=Path(
        "source/dexteroushands_lab/assets/bottle/generated/cap.usda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trace, args.cap_usd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
