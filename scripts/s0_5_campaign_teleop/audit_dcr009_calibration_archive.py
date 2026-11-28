#!/usr/bin/env python3
"""Verify the archived CAL provenance and seal a bounded replay disposition."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


HEADING = ".validation/s0-policy/t-heading-20261119.DfY0t7G4/results.json"
PALM = ".validation/s0-policy/palm-wrist-joint-20261119.xoQMOMg3/"
PALM_REPORT = PALM + "joint_calibration_report.json"
PALM_REVIEW = PALM + "joint_calibration_review.json"
HEADING_WINDOWS = "data/{}/calibration_windows.json"
ARCHIVE_DIGESTS = {
    HEADING: "6ef610e12b36165adbea995e2b118b341fe5569b2f7a54d3b4bd06941cf0cb50",
    PALM_REPORT: "6f6ee03c890d3bbdfcb7fb747b0f9845962a4e7d2ac6c983382e9b77050ee337",
    PALM_REVIEW: "44b8333614b97521a32514de93c28aaea11e0f668ec00e4071ace4f9c27dab1b",
    PALM + "policy_s02_sim.joint_calibration_candidate.yaml":
        "7e98b5af6beeef917ef5c08ee36338ac0bd5d26cd133282ee73d02940195fd80",
    ".validation/s0-policy/palm-calibration-20261117-173506/live_calibration_verified.json":
        "ff8d60a83dd37b4e726e6dcfc9e774c721729a7ead49bde6cdf746b2055b78e9",
    HEADING_WINDOWS.format("s0_heading_reference_20261119_165342"):
        "2ac38e1618e69d0a6440ebee2c3fb694c488c9a398f7455ab03db6e0da634d10",
    HEADING_WINDOWS.format("s0_heading_reference_20261119_165505"):
        "3630a09fbdbf17038f02cd78aa18e92b2a94a953366848c6f9fbc697d7599489",
    HEADING_WINDOWS.format("s0_heading_reference_20261119_165606"):
        "cadd0c75a1b225447c76eb0dfad1027400dd25b2ea8bc10d724531eb88d0a84c",
    "src/vive_tracker_ros/config/tracker.yaml":
        "c626978236190d61d6b95fd6d03d92fd1cd8d9ef9528a3c6386a3ad3e431dcda",
    "src/vive_tracker_ros/config/tracker_calibration.yaml":
        "9637f10c1bf8419c72ed8001190621e4a9bfb395a28a67c6f3c608984c503c84",
}


def archive_payloads(path):
    buffers = {}
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            expected = ARCHIVE_DIGESTS.get(entry.name)
            if expected is None or entry.name in buffers or not entry.isfile() or entry.size > 1_000_000:
                raise ValueError(f"Unexpected CAL archive member {entry.name}")
            with archive.extractfile(entry) as stream:
                data = stream.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"CAL archive member digest mismatch: {entry.name}")
            buffers[entry.name] = data
    if buffers.keys() != ARCHIVE_DIGESTS.keys():
        raise ValueError("Missing CAL archive members")
    return buffers


def audited_evidence(archive_path):
    source = archive_payloads(archive_path)
    heading = json.loads(source[HEADING])
    palm = json.loads(source[PALM_REPORT])
    review = json.loads(source[PALM_REVIEW])
    references = []
    for row in heading["rows"]:
        window = json.loads(source[HEADING_WINDOWS.format(row["bag"])])
        values = row["heading"]
        references.append({
            "bag": row["bag"], "role": window["role"], "direction": window["directed_motion"],
            "topic": window["forward_topic"], "fit_yaw_deg": values["yaw_deg"],
            "directed_holdout_angle_deg": values["heldout_angle_deg"],
            "vertical_excursion_mm": values["z_peak_to_peak_mm"],
        })
    per_profile = {}
    keys = ("contact_p95_mm", "contact_inferred_relative_angle_deg_p95", "contact_inferred_relative_wrist_gap_mm_p95")
    for profile, sessions in palm["independent_holdouts"].items():
        per_profile[profile] = {bag: {key: session[key] for key in keys} for bag, session in sessions.items()}
    return {
        "archive_member_sha256": ARCHIVE_DIGESTS,
        "status": "ARCHIVE_NOT_QUALIFIED_FOR_CAPTURE_CAL",
        "heading": {"common_yaw_deg": heading["common_yaw_deg"], "old_rotation": heading["old_matrix"],
                    "historical_candidate_rotation": heading["new_matrix"], "references": references,
                    "scope": "Historical manually directed capture. External common-frame, fixture and recording "
                             "generation correspondence are unavailable."},
        "tracker_attachment": {
            "candidate_status": review["status"], "holdouts": per_profile,
            "scope": "Historical labeled contacts, wearer tracker pivots and relative pose proxies; absolute "
                     "side-specific anatomical mount and cohort recording-epoch equivalence remain unestablished.",
            "expected_replay_topics": ["/vive/tracker/left_wrist", "/vive/tracker/right_wrist"],
            "replay_topic_semantics": "Post-software-registration wrist child poses. Keep the raw marker as a "
                                      "separate upstream frame and do not reapply the raw-to-child fixed edge.",
        },
    }


def bind_json(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def disposition(archive, model, axes):
    evidence = audited_evidence(archive)
    models, model_sha = bind_json(model)
    virtual, virtual_sha = bind_json(axes)
    live = models["model_identity"]["live"]["xml_sha256"]
    signed = models["model_identity"]["signed"]["xml_sha256"]
    scoped_axis = (virtual["scope"] == "VIRTUAL_SCENE_COORDINATE_CONVENTION_ONLY"
                   and virtual["status"] == "PASS"
                   and virtual.get("exact_model_support", {}).get("model_frame_receipt_sha256") == model_sha
                   and virtual["exact_model_support"]["live_model_sha256"] == live)
    return {
        "schema_version": 1, "scope": "MUJOCO_ONLY", "status": "BLOCKED",
        "live_model_sha256": live, "signed_model_sha256": signed,
        "qualification": {
            "virtual_scene_axis_convention": "PASS" if scoped_axis else "UNQUALIFIED",
            "physical_shared_registration": "UNQUALIFIED",
            "recorded_capture_heading": "UNQUALIFIED",
            "recorded_tracker_to_anatomical_attachment": "UNQUALIFIED",
            "recorded_mcap_pilot": "BLOCKED",
        },
        "blocking_conditions": [
            "No independently qualified link from the as-recorded registered Tracker wrist frames to the matching "
            "anatomical palms across the selected recordings.",
            "Virtual co-oriented scene axes validate a separate coordinate standard. The archived physical "
            "directed-heading references do not validate transfer of that standard to the recorded room.",
        ],
        "evidence": {
            "model_frame_receipt": {"path": model.name, "sha256": model_sha},
            "virtual_axis_receipt": {"path": axes.name, "sha256": virtual_sha},
            "tracker_archive": evidence,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--model-frame-receipt", required=True, type=Path)
    parser.add_argument("--virtual-axis-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = disposition(args.archive, args.model_frame_receipt, args.virtual_axis_receipt)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
