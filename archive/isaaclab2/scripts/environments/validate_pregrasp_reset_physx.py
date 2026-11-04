"""Native USD one-step geometry probe for randomized bimanual pregrasps.

This probe checks real PhysX hand/object and hand/hand contacts, plus optional
self-contact views. It does not require or claim a one-second suspended hold.
The certified states are written to a separate geometric Reset bank.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import os
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidates", type=Path, required=True)
parser.add_argument("--bank", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--max", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict  # noqa: E402
from dexteroushands_lab.scenes.pregrasp_reset import (  # noqa: E402
    PregraspResetBank, validate_pregrasp_state,
)
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402
from isaaclab_physx.physics import PhysxManager  # noqa: E402


class SelfContactProbe:
    """Exact native contact views for nonadjacent bodies in each same hand."""

    def __init__(self, specs):
        self.views = []
        sim_view = PhysxManager.get_physics_sim_view()
        for side, spec in zip(("LeftHand", "RightHand"), specs):
            names = spec.body_names
            root = f"/World/envs/env_0/{side}"
            for ix, name in enumerate(names):
                source = f"{root}/{name}"
                filters = [f"{root}/{later}" for later in names[ix+1:]]
                if not filters:
                    continue
                view = sim_view.create_rigid_contact_view(
                    source, filter_patterns=filters, max_contact_data_count=128)
                if view is None or list(view.sensor_paths) != [source]:
                    raise RuntimeError(f"Cannot inspect native self-contact at {source}")
                if list(view.filter_paths[0]) != filters:
                    raise RuntimeError(f"Self-contact filter order mismatch: {source}")
                self.views.append((source, filters, view))

    def points(self, dt):
        import warp as wp
        result = []
        for source, filters, view in self.views:
            forces, _, _, distance, count, start = view.get_contact_data(dt=dt)
            forces, distance, count, start = map(wp.to_torch,
                                                  (forces, distance, count, start))
            count, start = count.reshape(-1), start.reshape(-1)
            for ix, partner in enumerate(filters):
                begin = int(start[ix].item())
                end = begin + int(count[ix].item())
                for j in range(begin, end):
                    result.append({"source_path": source, "partner_path": partner,
                                   "separation_m": float(distance.reshape(-1)[j]),
                                   "normal_force_n": float(forces[j].norm())})
        return result


def probe(fixture, self_probe, sample):
    fixture.reset_manager.reset([0], [sample])
    fixture.step()  # refresh native contact data once; never a hold test
    contacts = fixture.monitor.raw_contact_points(0, fixture.dt)
    self_contacts = self_probe.points(fixture.dt)
    all_points = contacts + self_contacts
    if any(not math.isfinite(row["separation_m"]) or
           not math.isfinite(row["normal_force_n"]) for row in all_points):
        raise RuntimeError("Nonfinite native PhysX contact value")
    min_gap = min((row["separation_m"] for row in all_points), default=None)
    peak = max((row["normal_force_n"] for row in all_points), default=0.0)
    poses = {name: fixture.scene[name].data.root_pose_w.torch[0].detach().cpu().tolist()
             for name in ("left_hand", "right_hand", "bottle_body", "bottle_cap")}
    original = {"left_hand": sample.left.root_pose,
                "right_hand": sample.right.root_pose,
                "bottle_body": sample.bottle.root_pose,
                "bottle_cap": sample.cap.root_pose}
    drift = {name: math.dist(pose[:3], original[name][:3])
             for name, pose in poses.items()}
    # MJCF proxy gives a positive margin; native PhysX verifies there is no
    # overlapping signed contact after the Reset boundary. A 0.1 mm epsilon
    # accommodates PhysX contact-distance float rounding, not deep overlap.
    qualified = ((min_gap is None or min_gap >= -1e-4)
                 and peak < 50.0 and max(drift.values()) < .003)
    return {"qualified": qualified,
            "backend": "official_native_USD_PhysX_one_step_contact",
            "physics_dt_s": fixture.dt, "minimum_reported_separation_m": min_gap,
            "peak_normal_force_n": peak, "root_translation_after_one_step_m": drift,
            "contacts": contacts, "same_hand_contacts": self_contacts,
            "no_one_second_hold_required_or_performed": True}


def main():
    print("PREGRASP_PHYSX_START", flush=True)
    source = json.loads(args.candidates.read_text(encoding="utf-8"))
    if source.get("status") != "MJCF_SCREEN_ONLY_NOT_NATIVE_USD_QUALIFIED":
        raise ValueError("Input must be an official-MJCF screen result")
    rows = source["accepted"][:args.max]
    if not rows:
        raise ValueError("No MJCF-screened candidates to test in native USD")
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device, config=BottleConfig(),
        control_mode="pose_command_wrist", hand_self_collisions=True,
        include_hand_hand=True,
    )
    print("PREGRASP_PHYSX_FIXTURE_READY", flush=True)
    self_probe = SelfContactProbe(fixture.specs)
    print("PREGRASP_PHYSX_SELF_VIEWS_READY", flush=True)
    # Sensor negative control: move the four-finger side through the bottle.
    # If native contact data cannot see this known overlap, absence of points
    # on a candidate is not evidence that its native USD is collision free.
    reference = grasp_sample_from_dict(rows[0]["sample"])
    bad_pose = (reference.left.root_pose[0] - .012,
                *reference.left.root_pose[1:])
    overlap = replace(reference, left=replace(
        reference.left, root_pose=bad_pose, wrist_target_pose=bad_pose))
    negative_control = probe(fixture, self_probe, overlap)
    observed_overlap = (negative_control["minimum_reported_separation_m"] is not None
                        and negative_control["minimum_reported_separation_m"] < -.001)
    print("PREGRASP_PHYSX_NEGATIVE_CONTROL " + json.dumps({
        "observed": observed_overlap,
        "minimum_separation_m": negative_control["minimum_reported_separation_m"]}),
        flush=True)
    if not observed_overlap:
        raise RuntimeError("Native PhysX contact view missed the deliberate bottle overlap")
    # Separate self-collision negative control: cross the right index and
    # middle fingers within their official joint limits. The official MJCF
    # source meshes put this pair about 9.5 mm into overlap.
    crossed = list(reference.right.joint_position)
    crossing = {
        "r_index_finger_mcp_flex": 1.0255672317759654,
        "r_index_finger_mcp_abd": .6336653649756663,
        "r_index_finger_pip": -.3156454883045692,
        "r_index_finger_dip": .37992430752485773,
        "r_middle_finger_mcp_flex": .8670142301387904,
        "r_middle_finger_mcp_abd": .42085803554525714,
        "r_middle_finger_pip": .17886981483208841,
        "r_middle_finger_dip": -.6844316539586298,
    }
    for name, value in crossing.items():
        crossed[fixture.specs[1].joint_names.index(name)] = value
    self_overlap = replace(reference, right=replace(
        reference.right, joint_position=tuple(crossed), joint_target=tuple(crossed)))
    self_outcome = probe(fixture, self_probe, self_overlap)
    self_min = min((point["separation_m"] for point in
                    self_outcome["same_hand_contacts"]), default=None)
    observed_self_overlap = self_min is not None and self_min < -.001
    print("PREGRASP_PHYSX_SELF_NEGATIVE_CONTROL " + json.dumps({
        "observed": observed_self_overlap, "minimum_separation_m": self_min}),
        flush=True)
    if not observed_self_overlap:
        raise RuntimeError("Native PhysX self-contact view missed the crossed-finger overlap")
    args.bank.parent.mkdir(parents=True, exist_ok=True)
    handle, path = tempfile.mkstemp(prefix="pregrasp-bank-", suffix=".json",
                                    dir=args.bank.parent)
    os.close(handle)
    Path(path).unlink()
    bank = PregraspResetBank(path)
    results = []
    for row in rows:
        sample = grasp_sample_from_dict(row["sample"])
        validate_pregrasp_state(sample, fixture.config, *fixture.specs)
        outcome = probe(fixture, self_probe, sample)
        results.append({"seed": sample.seed, "physx": outcome})
        print("PREGRASP_PHYSX " + json.dumps({
            "seed": sample.seed, "qualified": outcome["qualified"],
            "min_separation_m": outcome["minimum_reported_separation_m"],
            "peak_force_n": outcome["peak_normal_force_n"],
            "root_drift_m": outcome["root_translation_after_one_step_m"]}), flush=True)
        if outcome["qualified"]:
            qualified = replace(sample, metadata={
                **sample.metadata, "geometry_qualified": True,
                "geometry_evidence": {"official_mjcf": row["mjcf_geometry"],
                                      "native_physx": outcome},
            })
            bank.add(qualified)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"status": "geometry_only_no_hold_claim",
        "negative_control": negative_control,
        "self_collision_negative_control": self_outcome,
        "attempted": len(results), "qualified": sum(r["physx"]["qualified"] for r in results),
        "results": results}, indent=2) + "\n", encoding="utf-8")
    if not bank.load():
        raise RuntimeError("No native USD geometry-qualified pregrasp samples")
    bank.path.replace(args.bank)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
