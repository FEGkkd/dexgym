#!/usr/bin/env python3
"""Read-only geometry/command attribution for the DCR010 archived root05 execution."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.intent import hand_basis
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.rmb.home_anchored_intent import SignedHumanPartition, SimulationIntent
from dexgym.tasks.direct.wholebody_s05.v1.rmb.teleop_candidate import HAND_FILTER_ALPHA
from home_anchored_start_anchor import role_identity, train_reader


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def stats(samples):
    a = np.asarray(samples)
    if not len(a):
        return {"n": 0}
    p = np.quantile(a, [.5, .95, .99, 1.], axis=0)
    return {"n": len(a), **dict(zip(("p50", "p95", "p99", "max"), p.tolist()))}


def quat_error(a, b):
    return np.rad2deg((Rotation.from_quat(a).inv() * Rotation.from_quat(b)).magnitude())


def rotation_error(a, b):
    return np.rad2deg(Rotation.from_matrix(np.swapaxes(a, -1, -2) @ b).magnitude())


def shape_metrics(finger_world, finger_local, target_world, target_local):
    local = finger_local[:, :, -1]
    target = target_local[:, :, -1]
    delta = lambda a, b: 1000 * np.linalg.norm(a - b, axis=-1)
    pinch = local[:, 1:] - local[:, :1]
    target_pinch = target[:, 1:] - target[:, :1]
    cross = finger_world[1, None] - finger_world[0, :, None]
    target_cross = target_world[1, None] - target_world[0, :, None]
    combos = list(combinations(range(5), 2))
    pairs = np.asarray(combos).T
    pair_dist = np.linalg.norm(local[:, pairs[1]] - local[:, pairs[0]], axis=-1)
    target_pair_dist = np.linalg.norm(target[:, pairs[1]] - target[:, pairs[0]], axis=-1)
    return {
        "local_tip_mm_by_side_and_finger": delta(local, target),
        "world_tip_mm_by_side_and_finger": delta(finger_world, target_world),
        "local_max_tip_mm_by_side": delta(local, target).max(axis=1),
        "world_max_tip_mm_by_side": delta(finger_world, target_world).max(axis=1),
        "world_max_ten_tip_mm": delta(finger_world, target_world).max(),
        "local_landmarks_mm_by_side_finger_landmark": delta(finger_local, target_local),
        "pinch_vector_mm_by_side_and_finger": delta(pinch, target_pinch),
        "thumb_finger_distance_abs_error_mm": 1000 * abs(np.linalg.norm(pinch, axis=-1)
                                                      - np.linalg.norm(target_pinch, axis=-1)),
        "interfinger_distance_abs_error_mm": 1000 * abs(pair_dist - target_pair_dist),
        "cross_vector_mm_by_left_and_right_finger": delta(cross, target_cross),
        "cross_vector_max_mm": delta(cross, target_cross).max(),
    }


def pair_names(model, gids):
    result = []
    for geom_id in gids:
        geom = model.geom(geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        chain = []
        b = body_id
        while b:
            body = model.body(b)
            count = int(model.body_jntnum[b])
            ids = range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + count)
            chain.append({"body": body.name, "joints": [model.joint(j).name for j in ids]})
            b = int(model.body_parentid[b])
        peers = []
        for i in range(model.ngeom):
            if i != geom_id and model.geom_bodyid[i] == body_id:
                visual = model.geom(i)
                peers.append({"name": visual.name, "mesh": model.mesh(int(visual.dataid[0])).name,
                              "same_mesh_and_pose": bool(geom.dataid[0] == visual.dataid[0]
                              and np.array_equal(geom.pos, visual.pos) and np.array_equal(geom.quat, visual.quat)),
                              "contype": int(visual.contype[0]), "conaffinity": int(visual.conaffinity[0])})
        result.append({"geom": geom.name, "geom_id": geom_id, "body_joint_ancestry": chain,
                       "rbound_mm": float(model.geom_rbound[geom_id]) * 1000,
                       "mesh": model.mesh(int(geom.dataid[0])).name, "visual_peers": peers,
                       "contype": int(geom.contype[0]), "conaffinity": int(geom.conaffinity[0])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "repo", "cohort", "xml", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt_file = args.run / "receipt.json"
    r = json.loads(receipt_file.read_bytes())
    assert r["scope"] == "DIAGNOSTIC_ONLY" and r["historical"]["plan_ticks"] == 4804
    assert digest(args.cohort) == r["input_cohort_sha256"]
    starts = args.run / "start.json"
    assert digest(starts) == r["start_sha256"]
    lock = np.asarray(json.loads(starts.read_bytes())["locked_transform"])
    manifest = r["robot_model_servo"]["assets"]
    expected_model = manifest.get("dual_arm_marvin.xml")
    assert expected_model and digest(args.xml) == expected_model
    robot = RobotModel(args.xml)
    assert list(robot.names) == r["robot_model_servo"]["joints"]
    initial = robot.geometry(robot.home)
    aligner = SimulationIntent(lock, initial)
    _, reader, cohort_sha = train_reader(args.cohort, args.repo / "docs/s0-5", None)
    assert cohort_sha == r["input_cohort_sha256"]
    depths = np.zeros((2, 20))
    tokens = [None, None]
    side_counts = [Counter(), Counter()]
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    limits = defaultdict(list)
    compared = Counter()
    geo_rows = []
    stage_pairs = None
    geom_ids = tuple(int(g) for g in robot.collision_pairs[2425])
    bounds = robot.bounds
    fields = (
        "tick_ns", "hmb_joint_valid", "hmb_source_index", "raw_tracker", "raw_tracker_mask", "registered",
        "registered_mask", "mapped", "mapped_mask", "raw_manus_local", "raw_manus_local_mask",
        "target_local_landmark_xyz", "target_local_landmark_xyz_mask", "target_world_tip_xyz", "target_wrist_xyz",
        "target_wrist_xyzw", "actual_local_landmark_xyz", "actual_world_tip_xyz", "actual_wrist_xyz",
        "actual_wrist_xyzw", "q_nominal_mask", "q_nominal", "nominal_proposal", "nominal_status",
        "raw_wuji", "raw_wuji_mask", "q_issued", "q_issued_mask", "q_applied", "q_measured", "q_measured_mask",
        "applied_substeps", "control_cutoff_ns"
    )
    for j, part in enumerate(r["partitions"]):
        a = {}
        for field in fields:
            sig = part["arrays"][field]
            path = args.run / part["directory"] / (field + ".npy")
            assert digest(path) == sig["sha256"], path
            a[field] = np.load(path, allow_pickle=False)
            assert list(a[field].shape) == sig["shape"]
        data, meta, sources, _ = reader.read(part["partition"])
        assert np.array_equal(a["tick_ns"], data["tick_ns"])
        assert np.array_equal(a["hmb_joint_valid"], data["joint_valid"])
        assert np.array_equal(a["hmb_source_index"], data["source_index"][:, :4, 0])
        human = SignedHumanPartition(data, meta, sources)
        for i, t in enumerate(a["tick_ns"]):
            h = human[i]
            for label, value in (("raw_tracker", h.raw_tracker),("registered", h.registered_wrist)):
                assert bool(a[label+"_mask"][i]) == (value is not None), (j,i,label)
                if value is not None:
                    assert np.array_equal(a[label][i], value.astype(np.float32)), (j,i,label)
            for side, points in enumerate(h.manus_local):
                assert bool(a["raw_manus_local_mask"][i,side]) == (points is not None)
                if points is not None:
                    assert np.array_equal(a["raw_manus_local"][i,side], points.astype(np.float32))
            compared["full_rows"] += 1
            if not (r["window"]["hmb_cutoff_start_ns"] <= t < r["window"]["hmb_cutoff_stop_exclusive_ns"]):
                continue
            compared["history_rows"] += 1
            active = bool(a["q_nominal_mask"][i])
            output_states = ("OBSERVED", "ACTIVE") if active else ("OBSERVED",)
            placement = bool(a["target_local_landmark_xyz_mask"][i].all())
            if h.registered_wrist is not None and h.raw_tracker is not None:
                raw, reg = h.raw_tracker, h.registered_wrist
                world = a["mapped"][i].astype(float)
                expected = lock @ reg
                assert np.allclose(world, expected, atol=1e-7, rtol=1e-7)
                actual_rel = np.linalg.inv(world[0]) @ world[1]
                expected_rel = np.linalg.inv(reg[0]) @ reg[1]
                for state in output_states:
                    sink = summary["tracker_and_shared_map"][state]
                    sink["raw_to_registered_position_mm_by_side"].append(1000*np.linalg.norm(raw[:,:3,3]-reg[:,:3,3],axis=1))
                    sink["raw_to_registered_attitude_deg_by_side"].append(rotation_error(raw[:,:3,:3], reg[:,:3,:3]))
                    sink["raw_registered_wrist_separation_difference_mm"].append(abs(np.linalg.norm(raw[1,:3,3]-raw[0,:3,3])-np.linalg.norm(reg[1,:3,3]-reg[0,:3,3])) * 1000)
                    sink["shared_map_pair_relative_translation_numeric_mm"].append(1000*np.linalg.norm(actual_rel[:3,3]-expected_rel[:3,3]))
                    sink["shared_map_pair_relative_rotation_numeric_deg"].append(rotation_error(actual_rel[None,:3,:3],expected_rel[None,:3,:3])[0])
            valid_input = bool(data["joint_valid"][i] and data["safety_permission"][i]
                               and data["pose_valid"][i,:4].all()
                               and (data["source_index"][i,:4,0] >= 0).all()
                               and (data["capture_bounds_ns"][i,:4,1] <= t).all()
                               and (data["arrival_bounds_ns"][i,:4,1] <= t).all())
            raw_wuji = a["raw_wuji"][i, -1].astype(float)
            wuji_ok = a["raw_wuji_mask"][i,-1].all() and valid_input
            if valid_input:
                for side in range(2):
                    source = role_identity(sources[int(data["source_index"][i,side,0])])
                    side_counts[side]["valid_input_rows"] += 1
                    if source == tokens[side]:
                        continue
                    side_counts[side]["distinct_source_updates"] += 1
                    if a["raw_wuji_mask"][i,-1,side]:
                        if tokens[side] is None:
                            depths[side] = raw_wuji[side]
                        else:
                            depths[side] = (1.-HAND_FILTER_ALPHA)*depths[side]+HAND_FILTER_ALPHA*raw_wuji[side]
                    else:
                        side_counts[side]["invalid_sdk_updates"] += 1
                    tokens[side] = source
            if not placement:
                compared["missing_morphology_target"] += 1
                continue
            assert all(point is not None for point in h.manus_local)
            target_local = a["target_local_landmark_xyz"][i].astype(float)
            target_world = a["target_world_tip_xyz"][i].astype(float)
            normalized_raw = []
            for side in range(2):
                raw = h.manus_local[side].astype(float)
                to_model = aligner.basis[side] @ hand_basis(raw[1:,0]).T
                normalized_raw.append(raw @ to_model.T)
                np.testing.assert_allclose(aligner._local_shape(side, raw),target_local[side],atol=1e-8,rtol=1e-7)
            raw = np.stack(normalized_raw)
            raw_links = np.linalg.norm(np.diff(raw,axis=2),axis=-1)
            target_links = np.linalg.norm(np.diff(target_local,axis=2),axis=-1)
            bone_delta = abs(raw_links-target_links)*1000
            raw_tips, target_tips = raw[:,:, -1], target_local[:,:, -1]
            pinch_raw = raw_tips[:,1:] - raw_tips[:,:1]
            pinch_target = target_tips[:,1:] - target_tips[:,:1]
            for state in output_states:
                sink = summary["raw_human_to_morphology"][state]
                sink["bone_length_delta_mm_by_side_finger_segment"].append(bone_delta)
                sink["bone_length_delta_max_mm"].append(bone_delta.max())
                sink["pinch_vector_mm_by_side_and_finger"].append(1000*np.linalg.norm(pinch_raw-pinch_target,axis=-1))
                sink["thumb_finger_distance_abs_error_mm"].append(1000*abs(np.linalg.norm(pinch_raw,axis=-1)-np.linalg.norm(pinch_target,axis=-1)))
                sink["thumb_index_raw_gap_mm"].append(1000*np.linalg.norm(pinch_raw[:,0],axis=-1))
                sink["thumb_index_robot_morphology_gap_mm"].append(1000*np.linalg.norm(pinch_target[:,0],axis=-1))
            if not (a["q_measured_mask"][i] and a["q_issued_mask"][i,-1] and a["applied_substeps"][i] == 10):
                compared["missing_execution_feedback"] += 1
                continue
            compared["scorable_execution_rows"] += 1
            q = {
                "NOMINAL_PROPOSAL": a["nominal_proposal"][i,-1].astype(float),
                "ISSUED": a["q_issued"][i,-1].astype(float),
                "APPLIED": a["q_applied"][i,9].astype(float),
                "MEASURED": a["q_measured"][i].astype(float),
            }
            if active:
                assert np.array_equal(a["q_nominal"][i],a["nominal_proposal"][i,-1])
            if wuji_ok:
                compared["scorable_raw_Wuji_rows"] += 1
                q["RAW_WUJI_HYBRID"] = np.r_[q["NOMINAL_PROPOSAL"][:14],raw_wuji.reshape(-1)]
                q["FILTERED_WUJI_RECONSTRUCTED_HYBRID"] = np.r_[q["NOMINAL_PROPOSAL"][:14],depths.reshape(-1)]
            samples = {}
            pair_gap = {}
            target_rot = Rotation.from_quat(a["target_wrist_xyzw"][i]).as_matrix()
            tcp_from_palm = np.linalg.inv(aligner.model_palm_from_scoring_tcp)
            goal_palm_rot = target_rot @ tcp_from_palm[:,:3,:3]
            goal_palm_xyz = a["target_wrist_xyz"][i] + np.einsum("sij,sj->si",target_rot,tcp_from_palm[:,:3,3])
            for name, arm_hand in q.items():
                values = robot.geometry(arm_hand)
                q_ok = bool(np.all(arm_hand >= bounds[:,0]-1e-7) and np.all(arm_hand <= bounds[:,1]+1e-7))
                limits[name].append(q_ok)
                sample = shape_metrics(values.tips,values.local,target_world,target_local)
                sample["palm_local_attitude_difference_deg_by_side"] = rotation_error(goal_palm_rot,values.palm_rotation)
                sample["palm_origin_difference_mm_by_side"] = 1000*np.linalg.norm(values.palm-goal_palm_xyz,axis=-1)
                sample["wrist_position_mm_by_side"] = 1000*np.linalg.norm(values.wrists-a["target_wrist_xyz"][i],axis=-1)
                sample["wrist_attitude_deg_by_side"] = rotation_error(target_rot,values.rotations)
                pair_gap[name] = mujoco.mj_geomDistance(robot.model,robot.data,*geom_ids,.30,None)*1000
                sample["right_thumb_index_collision_mesh_gap_mm"] = pair_gap[name]
                sample["right_thumb_index_site_gap_mm"] = 1000*np.linalg.norm(values.tips[1,0]-values.tips[1,1])
                if name.startswith(("RAW_WUJI", "FILTERED_WUJI")):
                    sample = {key:x for key,x in sample.items() if key.startswith(("local","pinch","right_thumb","thumb_finger","interfinger"))}
                samples[name] = values
                for state in output_states:
                    for metric, value in sample.items():
                        summary[f"{name}_vs_MORPHOLOGY"][state][metric].append(value)
            measured = samples["MEASURED"]
            np.testing.assert_allclose(measured.tips,a["actual_world_tip_xyz"][i],atol=4e-7,rtol=0)
            np.testing.assert_allclose(measured.local,a["actual_local_landmark_xyz"][i],atol=4e-7,rtol=0)
            for before, after in (("NOMINAL_PROPOSAL","ISSUED"),("ISSUED","APPLIED"),("APPLIED","MEASURED")):
                first, second = samples[before],samples[after]
                diff = shape_metrics(second.tips,second.local,first.tips,first.local)
                diff["joint_max_abs_delta_deg_by_group"] = np.asarray([max(abs(q[after][index]-q[before][index])) for index in (
                      slice(0,7), slice(7,14), slice(14,34), slice(34,54))]) * (180/np.pi)
                for state in output_states:
                    for metric, value in diff.items():
                        summary[f"{before}_to_{after}"][state][metric].append(value)
            if wuji_ok:
                delta = 180/np.pi*np.max(abs(raw_wuji-depths),axis=1)
                for state in output_states:
                    summary["RAW_to_FILTERED_WUJI"][state]["max_abs_joint_delta_deg_by_side"].append(delta)
            geo_rows.append({"partition": part["partition"],"row":i,"clock_ns":int(t),"active":active,
                             "gap_mm":pair_gap, "qh_legality":{name:limits[name][-1] for name in q},
                             "target_right_thumb_index_tip_gap_mm":float(1000*np.linalg.norm(target_world[1,0]-target_world[1,1]))})
        print(f"verified {part['partition']} rows={len(a['tick_ns'])}",file=sys.stderr,flush=True)
    assert compared["full_rows"] == r["full"]["plan_ticks"] and compared["history_rows"] == r["historical"]["plan_ticks"]
    assert compared["scorable_execution_rows"] == 4801
    assert all(v["invalid_sdk_updates"] == 0 for v in side_counts)
    assert args.out.parent.is_dir()
    output={
        "scope": "DIAGNOSTIC_ONLY", "formal_coverage_credit": False,
        "run_receipt_sha256":digest(receipt_file), "script_sha256":digest(Path(__file__)),
        "model_sha256":digest(args.xml), "cohort_sha256":digest(args.cohort),
        "paths": {"run":str(args.run), "repo":str(args.repo), "model":str(args.xml)},
        "counts":dict(compared), "Wuji_source_updates_left_right":[dict(v) for v in side_counts],
        "denominators":{"whole_root_planned":r["full"]["plan_ticks"],"historical_planned":r["historical"]["plan_ticks"],
                         "historical_joint_valid":r["historical"]["hmb_joint_valid_ticks"],
                         "historical_active":r["historical"]["legal_active_nominal_ticks"]},
        "joint_bound_sample_counts":{k: {"within_limits":sum(v),"tested":len(v)} for k,v in limits.items()},
        "pair_index": 2425, "collision_pair_geoms":pair_names(robot.model,geom_ids),
        "sampled_pair_endpoint_categories_by_layer":{
             k:dict(Counter("PENETRATION" if v<0 else "BELOW_0P5_MM" if v<.5 else "ABOVE_0P5_MM" for row in geo_rows if (v:=row["gap_mm"].get(k)) is not None)) for k in limits
        },
        "input_and_clock_notes":[
             "Historical visible signed HMB 50 Hz rows matched byte-for-byte to post-export float32 fields and source indices.",
             "Wuji filter reconstructed from recorded float32 raw SDK output, original valid HMB source tokens, the isolated preposition initial call, and alpha=.30 at distinct source updates. It is not direct logging.",
             "NOMINAL_PROPOSAL is final staged request each 20 ms interval. It can include retained native group values and is not authorized or presumed feasible. Q_NOMINAL exists only for ACTIVE rows; exact equality verified on ACTIVE.",
             "Wuji raw/filter hybrid FK replaces hand axes at the same nominal arm pose. Only local hand and tip pair measures are attributed to Wuji; these positions are not executed.",
             "Observed-domain geometric comparison includes holds, on rows with full actual 10 MuJoCo substeps and known target. ACTIVE is separate.",
             "Robot palm axes are model/bridge axes, anatomical palm remains UNKNOWN. World normalization of original MANUS digits is not available without anatomical calibration.",
             "Signed pair clearance is a static MuJoCo point distance. It is not evidence of continuous path clearance or physical contact."
        ],
        "statistics":{name:{domain:{key:stats(samples) for key,samples in metrics.items()}
                            for domain,metrics in domains.items()} for name,domains in summary.items()}
    }
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(output,f,indent=2,allow_nan=False,sort_keys=True)
        f.write("\n")
    pair_path = args.out.with_suffix(".pairs.jsonl")
    with pair_path.open("x",encoding="utf-8") as f:
        for row in geo_rows:
            f.write(json.dumps(row,separators=(",", ":"),allow_nan=False)+"\n")
    print(f"metadata={digest(args.out)} pair_rows={digest(pair_path)} script={digest(Path(__file__))}")
    print(json.dumps({"counts":dict(compared),"pair_endpoint":output["sampled_pair_endpoint_categories_by_layer"]}))


if __name__ == "__main__":
    main()
