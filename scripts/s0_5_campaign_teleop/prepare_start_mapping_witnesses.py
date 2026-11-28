"""Build same-cutoff human, M1, shared, and MuJoCo visual witnesses from the sealed full rerun."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from compare_start_handoff_layers import RAW_PARITY_TOLERANCE_M_RAD, se3
from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, SIMULATED, checked_json
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.rmb.build import CAL_SHA, FIT_SHA, MODEL_SHA as SIGNED_MODEL_SHA
from dexgym.tasks.direct.wholebody_s05.v1.rmb.mapping import IntentMapper, compiled_topologies
from dexgym.tasks.direct.wholebody_s05.v1.rmb.reader import RobotReferenceBank
from hmb_cohort import IDENTITIES
from pair_nominal_hmb import DCR_COMPARISON_SHA, digest
from prepare_shared_rigid_frames import palm_axes
from run_start_handoff_cohort import FRAMES_SHA, MODEL_SHA


INDEX_SHA = "2bc6a63922e9f7790b4366690401c1d3934bd405b1a135cab1266c068fb1acb4"
CALIBRATION_SHA = "a35f26c089e75f105c0bab78aa9442c7870698c442211bfe34ad033000912363"
CALIBRATION_BASIS_SHA = "d720547d649929c5697f8094ef7e543356ff3ace766d890695ae2295fea7f590"
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def read(path):
    return json.loads(path.read_bytes())


def so3_degrees(first, second):
    return float(np.degrees(Rotation.from_matrix(first.T @ second).magnitude()))


def degrees(first, second):
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1., 1.))))


def panel(origins, fingers):
    knuckles = fingers[:, 1:, 0]
    basis = np.stack([palm_axes(origin, points) for origin, points in zip(origins, knuckles, strict=True)])
    normals = basis[:, :, 2] * np.array([1., -1.])[:, None]
    tip_distances = np.linalg.norm(fingers[1, None, :, -1] - fingers[0, :, None, -1], axis=-1)
    near = np.unravel_index(int(tip_distances.argmin()), tip_distances.shape)
    return {
        "wrist_xyz_m": origins.tolist(), "palm_origins_xyz_m": origins.tolist(),
        "finger_landmarks_xyz_m": fingers.tolist(), "palm_centres_xyz_m": ((origins + knuckles.sum(1)) / 5).tolist(),
        "palm_normals_world": normals.tolist(), "palm_basis_world": basis.tolist(),
        "midpoint_xyz_m": origins.mean(0).tolist(), "wrist_gap_mm": float(1e3 * np.linalg.norm(origins[1] - origins[0])),
        "directed_palm_angle_deg": degrees(*normals),
        "relative_wrist_in_left_palm_mm": (1e3 * basis[0].T @ (origins[1] - origins[0])).tolist(),
        "relative_palm_rotation": (basis[0].T @ basis[1]).tolist(),
        "matched_cross_tip_distances_mm": (tip_distances * 1e3).tolist(),
        "nearest_cross_tip_mm": float(tip_distances[near] * 1e3),
        "nearest_cross_tip_pair": [FINGERS[i] for i in near],
    }


def relative_error(human, other):
    vector = np.asarray(other["relative_wrist_in_left_palm_mm"]) - np.asarray(human["relative_wrist_in_left_palm_mm"])
    tips = np.asarray(other["matched_cross_tip_distances_mm"]) - np.asarray(human["matched_cross_tip_distances_mm"])
    return {"wrist_gap_delta_mm": other["wrist_gap_mm"] - human["wrist_gap_mm"],
            "left_palm_relative_vector_error_mm": float(np.linalg.norm(vector)),
            "relative_palm_so3_error_deg": so3_degrees(np.asarray(human["relative_palm_rotation"]),
                                                       np.asarray(other["relative_palm_rotation"])),
            "directed_palm_angle_delta_deg": other["directed_palm_angle_deg"] - human["directed_palm_angle_deg"],
            "cross_tip_distance_mae_mm": float(np.abs(tips).mean())}


def physical_human(raw_tracker, glove, basis, stored_bases, calibration, display):
    origins, world = [], []
    for k, side in enumerate(SIDES):
        fixture = calibration[side]
        tracker = raw_tracker[k]
        rotation = tracker[:3, :3]
        mapped = rotation @ Rotation.from_quat(fixture["rotation_xyzw"]).as_matrix()
        origin = tracker[:3, 3] + rotation @ np.asarray(fixture["position_offset_m"])
        world_r = display[:3, :3] @ mapped @ stored_bases[k] @ basis[k].T
        p = display[:3, :3] @ origin + display[:3, 3]
        origins.append(p)
        world.append(glove[k] @ world_r.T + p)
    return np.stack(origins), np.stack(world)


def choose_roots(index, layers_dir, n):
    keys = [index["layers"].index(name) for name in ("m1", "shared_hmb_raw", "native_shared_raw", "measured_fk")]
    picks = []
    for item in index["by_root"]:
        path = layers_dir / item["array_path"]
        if digest(path) != item["array_sha256"]:
            raise ValueError(f"Sealed array differs for {item['root']}")
        with np.load(path, allow_pickle=False) as z:
            good = z["hmb_joint_valid"] & z["wrist_valid"][:, keys].all(axis=(1, 2)) & z["q_measured_valid"].all(axis=1)
            parity = z["raw_shared_reconstruction_error_m_rad"]
            good &= (np.isfinite(parity) & (parity <= RAW_PARITY_TOLERANCE_M_RAD)).all(axis=(1, 2))
            rows = np.flatnonzero(good)
            if not len(rows):
                continue
            poses = z["wrist_pose_world"][rows][:, keys]
            scores = 1e3 * np.linalg.norm(poses[:, 0, :, :3] - poses[:, -1, :, :3], axis=-1).max(axis=1)
            p95 = float(np.percentile(scores, 95))
            selected = int(np.argmin(np.abs(scores - p95)))
            i = rows[selected]
            picks.append({"root": item["root"], "p95_mm": p95, "eligible_rows": len(rows),
                          "cutoff_ns": int(z["cutoff_ns"][i]), "tick": int(z["partition_tick"][i]),
                          "partition_index": int(z["partition_index"][i]), "score_mm": float(scores[selected]),
                          "wrist_poses": poses[selected].tolist(), "measured_q_rad": z["q_measured_rad"][i].tolist(),
                          "measured_source_age_ms": (z["replay_cutoff_header_ns"][i] - z["measured_header_ns"][i]) / 1e6})
    if len(picks) < n:
        raise ValueError("Insufficient separate roots with a valid four-panel witness")
    ranked = sorted(picks, key=lambda item: (-item["p95_mm"], item["root"]))
    return ranked[:n], [{k: row[k] for k in ("root", "eligible_rows", "p95_mm")} for row in ranked]


def source_landmarks(raw, meta, tick):
    anatomical, sdk_orders = compiled_topologies(meta, raw["topology_id"][tick, :2])
    if not raw["pose_valid"][tick, :4].all():
        raise ValueError("Visual witness lacks all four signed source poses")
    fingers, bases = [], []
    for side, ((root, indices), sdk_order) in enumerate(zip(anatomical, sdk_orders, strict=True)):
        mask = raw["node_mask"][tick, side]
        common = raw["sdk_common_pose"][tick, side, :, :3]
        if not (mask[root] and mask[indices].all() and mask[sdk_order].all()):
            raise ValueError("Visual witness lacks one or more named raw MANUS landmarks")
        sdk = common[sdk_order]
        bases.append(palm_axes(sdk[0], sdk[1:].reshape(5, 4, 3)[1:, 0]))
        fingers.append(common[indices] - common[root])
    return np.stack(fingers), np.stack(bases), anatomical


def signed_m1(exact, tick, tracker, mapper, shape, order):
    fields = ("mapped_wrist_xyz", "mapped_wrist_xyzw", "mapped_local_landmark_xyz", "mapped_world_tip_xyz")
    if any(not exact[f"{key}_mask"][tick].all() for key in fields):
        raise ValueError("Frozen M1 witness is incomplete")
    wrist, quat, local, tip = [exact[key][tick] for key in fields]
    wrist_r = Rotation.from_quat(quat).as_matrix()
    palm_p = wrist + np.einsum("sij,sj->si", wrist_r, mapper.mount_p)
    palm_r = wrist_r @ mapper.mount_r
    landmarks = local @ palm_r.transpose(0, 2, 1)[:, None] + palm_p[:, None, None]
    np.testing.assert_allclose(landmarks[:, :, -1], tip, rtol=0, atol=2e-7)
    reproduced = mapper.map(tracker, shape, order)
    np.testing.assert_allclose(reproduced["mapped"]["local_landmark_xyz"], local, rtol=0, atol=2e-7)
    return palm_p, landmarks, wrist, reproduced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "docs", "frames", "calibration", "basis", "direct-bank", "dcr-comparison", "model", "signed-model", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--alternate-calibration", type=Path)
    args = parser.parse_args()
    identities = {args.run / "layer_index.json": INDEX_SHA, args.frames: FRAMES_SHA, args.calibration: CALIBRATION_SHA,
                  args.basis: CALIBRATION_BASIS_SHA, args.dcr_comparison: DCR_COMPARISON_SHA, args.model: MODEL_SHA,
                  args.signed_model: SIGNED_MODEL_SHA}
    if any(digest(path) != wanted for path, wanted in identities.items()):
        raise ValueError("A sealed input or calibration artifact changed")
    index = read(args.run / "layer_index.json")
    cohort_path, run_path = args.run / "cohort.json", args.run / "native/cohort_run.json"
    if digest(cohort_path) != index["cohort_sha256"] or digest(run_path) != index["cohort_run_sha256"]:
        raise ValueError("Native and layer identities differ")
    cohort = read(cohort_path)
    roots = {item["root"]: item for item in cohort["selected_roots"]}
    attempts = {item["root"]: item["attempts"][-1] for item in read(run_path)["roots"]}
    samples, ranking = choose_roots(index, args.run / "layers-verified", 3)
    frames = np.repeat(np.eye(4)[None], 7, axis=0)
    frames[:, :3] = np.loadtxt(args.frames).reshape(7, 3, 4)
    with np.load(args.basis, allow_pickle=False) as archive:
        historical_bases = archive["bases"]
    cal = yaml.safe_load(args.calibration.read_bytes())["wrist_calibration"]
    alternate = yaml.safe_load(args.alternate_calibration.read_bytes())["wrist_calibration"] if args.alternate_calibration else None
    evidence = args.docs / "evidence"
    calibrated = checked_json(evidence / "P02/mapping_contract.json", CAL_SHA)
    fitted = checked_json(evidence / "P02/refinement/fitted_candidates.json", FIT_SHA)
    mapper = IntentMapper(RobotModel(args.signed_model).geometry(), calibrated, fitted)
    m1_world = fitted["configs"]["M1"]
    bank = RobotReferenceBank(args.direct_bank, expected_sha=read(args.dcr_comparison)["identity_sha256"]["direct_bank"],
                              verify=True, require_qualified=False)
    robot = RobotModel(args.model)
    home = robot.geometry(robot.home)
    wuji = np.stack([home.palm_rotation[s].T @ palm_axes(home.palm[s], home.landmarks[s, 1:, 0]) for s in range(2)])
    site_in_palm = np.einsum("sji,sj->si", home.palm_rotation, home.wrists - home.palm)
    for rank, sample in enumerate(samples, 1):
        root = roots[sample["root"]]
        sample["split"] = root["split"]
        sample["task"] = root["task"]
        key, = [x["key"] for x in root["hmb_partitions"] if x["index"] == sample["partition_index"]]
        tick = sample["tick"]
        reader = HMBReader(*(evidence / IDENTITIES[key_][0] for key_ in ("hmb_manifest", "plan", "config")),
                           cohort["signed_hmb_ubuntu_products"], root["split"], SIMULATED)
        raw, meta, sources, _ = reader.read(key)
        direct = bank.read(key, root["split"]).arrays
        if int(raw["cutoff_ns"][tick]) != sample["cutoff_ns"] or int(direct["tick_ns"][tick]) != sample["cutoff_ns"]:
            raise ValueError("Panel source cutoffs differ")
        glove, basis, order = source_landmarks(raw, meta, tick)
        tracker = raw["pose"][tick, 2:4, 0]
        raw_tracker = np.stack([se3(item[:3], item[3:]) for item in tracker])
        attempt = attempts[sample["root"]]
        shared = np.asarray(attempt["start_handoff"]["measurement"]["shared_transform"])
        np.testing.assert_allclose(shared[:3, :3], frames[0, :3, :3], atol=2e-10, rtol=0)
        mapped_palm = shared @ raw_tracker @ frames[3:5]
        shared_points = (glove @ (basis @ wuji.transpose(0, 2, 1))[:, None]
                         @ mapped_palm[:, None, :3, :3].transpose(0, 1, 3, 2)) + mapped_palm[:, None, None, :3, 3]
        shared_site = np.einsum("sij,sj->si", mapped_palm[:, :3, :3], site_in_palm) + mapped_palm[:, :3, 3]
        m1_palm, m1_points, m1_site, reproduction = signed_m1(
            direct, tick, tracker, mapper, raw["pose"][tick, :2], order,
        )
        rob = robot.geometry(sample.pop("measured_q_rad"))
        archived_poses = np.asarray(sample.pop("wrist_poses"))
        for pose, i in ((m1_site, 0), (shared_site, 1), (rob.wrists, 3)):
            np.testing.assert_allclose(pose, archived_poses[i, :, :3], rtol=0, atol=2e-7)
        human_palm, human_points = physical_human(raw_tracker, glove, basis, historical_bases, cal, shared)
        np.testing.assert_allclose(human_palm, mapped_palm[:, :3, 3], atol=2e-8, rtol=0)
        np.testing.assert_allclose(human_points, shared_points, atol=2e-8, rtol=0)
        panels = {"original_human": panel(human_palm, human_points), "m1": panel(m1_palm, m1_points),
                  "shared": panel(mapped_palm[:, :3, 3], shared_points), "mujoco": panel(rob.palm, rob.landmarks)}
        raw_gap = float(1e3 * np.linalg.norm(tracker[1, :3] - tracker[0, :3]))
        m1_trackers = tracker[:, :3] @ np.asarray(m1_world["rotation"]).T + np.asarray(m1_world["translation"])
        shared_trackers = tracker[:, :3] @ shared[:3, :3].T + shared[:3, 3]
        np.testing.assert_allclose(m1_trackers, m1_site, atol=2e-7, rtol=0)
        sample.update(rank=rank, partition=key, panels=panels,
                      selection=f"Root P95 rank {rank}; paired P95 witness",
                      relative_errors={name: relative_error(panels["original_human"], value) for name, value in panels.items()},
                      source_records=[sources[int(ref)]["record"] for ref in raw["source_index"][tick, :4, 0]],
                      raw_tracker=tracker.tolist(),
                      stages={"raw_tracker_gap_mm": raw_gap,
                              "m1_before_palm_offset_gap_mm": float(1e3 * np.linalg.norm(m1_site[1] - m1_site[0])),
                              "shared_before_palm_offset_gap_mm": float(1e3 * np.linalg.norm(shared_trackers[1] - shared_trackers[0])),
                              "global_registration_rotation_difference_deg": so3_degrees(np.asarray(m1_world["rotation"]), shared[:3, :3]),
                              "first_global_placement_midpoint_difference_mm": float(1e3 * np.linalg.norm((m1_trackers - shared_trackers).mean(0))),
                              "hard_pinch_active": reproduction["pinch_active"].tolist()})
        if alternate is not None:
            alt = panel(*physical_human(raw_tracker, glove, basis, historical_bases, alternate, shared))
            sample["alternate_calibration_relative_errors"] = {name: relative_error(alt, value) for name, value in panels.items()}
    result = {"schema": "s05.teleop.start_mapping_witnesses.v1", "layer_index_sha256": INDEX_SHA,
              "calibration_sha256": CALIBRATION_SHA, "historical_basis_sha256": CALIBRATION_BASIS_SHA,
              "alternate_calibration_sha256": digest(args.alternate_calibration) if alternate is not None else None,
              "m1_mapping_sha256": FIT_SHA, "m1_palm_contract_sha256": CAL_SHA,
              "calibrated_recorded_up_expressed_in_world": {
                  "m1": np.asarray(m1_world["rotation"])[:, 1].tolist(), "shared": frames[0, :3, 1].tolist(),
              },
              "human_scope": "Historical tracker-to-palm calibration, unverified for each individual recording",
              "visual_wrist_definition": "Anatomical wrist or palm root, corresponding across four panels; ranked with the sealed TCP-site position error",
              "display_registration": "Original human uses the per-root shared START world placement for display; shared preserves the same calibrated landmarks by construction",
              "ranking": ranking, "samples": samples}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
