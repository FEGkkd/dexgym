#!/usr/bin/env python3
"""Bounded static hand-only geometry witnesses for the untouched ROOT05 archive."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep

SCOPE = "DIAGNOSTIC_ONLY_STATIC_NOT_EXECUTED"
SYMMETRIC_RAD = np.deg2rad(5.)
SCALING = (1., .5, .25, .125, .0625)
PER_STRATUM = 12
CLEARANCE_RESERVE_M = 2e-6
FUNCTION_TOL_MM = 2.
SOLVER_EVALUATIONS = 30


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def error(g, local, world):
    gap = lambda a, b: np.linalg.norm(a - b, axis=-1) * 1000.
    lp = gap(g.local[:, :, -1], local[:, :, -1])
    wp = gap(g.tips, world)
    pinch_actual = g.local[:, 1:3, -1] - g.local[:, :1, -1]
    pinch_target = local[:, 1:3, -1] - local[:, :1, -1]
    cross = g.tips[1, None] - g.tips[0, :, None]
    tcross = world[1, None] - world[0, :, None]
    return dict(local_tip_mm=lp.tolist(), world_tip_mm=wp.tolist(),
                local_max_by_side_mm=lp.max(axis=1).tolist(), max_world_tip_mm=float(wp.max()),
                max_local_tip_mm=float(lp.max()),
                pinch_index_middle_vector_mm=gap(pinch_actual, pinch_target).tolist(),
                pinch_index_middle_distance_mm=(abs(np.linalg.norm(pinch_actual,axis=-1)
                                                   - np.linalg.norm(pinch_target,axis=-1))*1000).tolist(),
                world_cross_max_mm=float(gap(cross, tcross).max()))


def objective(robot, q0, local, x, *, want_jac):
    g = robot.geometry(np.r_[q0[:14], x])
    loc = g.local[:, :, -1]
    p = loc - local[:, :, -1]
    pinch = p[:, 1:3] - p[:, :1]
    if not want_jac:
        return np.r_[p.ravel(), .35 * pinch.ravel(), .005 * (x-q0[14:])]
    jac = np.stack([robot.point_jac("site", robot.model.site(i).name)[0][:,14:]
                    for i in robot.tip_ids]).reshape(2,5,3,40)
    jlocal = np.einsum("sji,sfjd->sfid", g.palm_rotation, jac)
    jpinch = jlocal[:, 1:3] - jlocal[:, :1]
    return np.concatenate([jlocal.reshape(-1,40), .35 * jpinch.reshape(-1,40), .005 * np.eye(40)])


def admissible_error(before, after):
    return all(np.all(np.asarray(after[k]) <= np.asarray(before[k]) + FUNCTION_TOL_MM)
               for k in ("local_max_by_side_mm", "max_world_tip_mm", "pinch_index_middle_vector_mm",
                         "world_cross_max_mm"))


def fit(robot, sweep, q0, local, world):
    baseline = error(robot.geometry(q0), local, world)
    legal_start = robot.legality(q0)
    if not legal_start["feasible"] or not sweep.permits(q0,q0):
        return {"status":"BASELINE_NOT_CERTIFIED", "before":baseline, "after":baseline, "issued_q_rad":q0.tolist(),
                "issued_legality":legal_start,"witness_q_rad":None}
    low = np.maximum(robot.bounds[14:,0], q0[14:] - SYMMETRIC_RAD)
    high = np.minimum(robot.bounds[14:,1], q0[14:] + SYMMETRIC_RAD)
    center = np.clip(q0[14:], low, high)
    request = least_squares(lambda x: objective(robot,q0,local,x,want_jac=False),
        center,jac=lambda x: objective(robot,q0,local,x,want_jac=True),bounds=(low,high),
        max_nfev=SOLVER_EVALUATIONS, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    attempts = []
    for scale in SCALING:
        trial = np.r_[q0[:14], np.clip(q0[14:] + scale * (request.x-q0[14:]),low,high)]
        assert np.array_equal(trial[:14],q0[:14])
        change = float(abs(trial-q0).max())
        assert change <= SYMMETRIC_RAD + 1e-12
        after = error(robot.geometry(trial),local,world)
        if change < 1e-8 or after["max_local_tip_mm"] >= baseline["max_local_tip_mm"] - 1e-4:
            attempts.append({"scale":scale,"reason":"NO_LOCAL_MAX_TIP_IMPROVEMENT"})
            continue
        if not admissible_error(baseline,after):
            attempts.append({"scale":scale,"reason":"NONDEGENERATE_FUNCTIONAL_CHECK"})
            continue
        legality = robot.legality(trial)
        if not legality["feasible"]:
            attempts.append({"scale":scale,"reason":"STATIC_POINT_ILLEGAL"})
            continue
        if not sweep.permits(q0, trial):
            attempts.append({"scale":scale,"reason":"STRAIGHT_PATH_UNPROVEN"})
            continue
        return {"status":"CERTIFIED_STATIC_IMPROVEMENT","accepted_scale":scale,"path_proof":"PERMIT",
                "solver_nfev":request.nfev,"tested_prior_attempts":attempts,"issued_q_rad":q0.tolist(),
                "witness_q_rad":trial.tolist(),"max_delta_deg":float(np.rad2deg(change)),
                "before":baseline,"after":after,"issued_legality":legal_start,"witness_legality":legality}
    return {"status":"NO_WITNESS_FOUND_WITHIN_DECLARED_SEARCH","solver_nfev":request.nfev,
            "tested_prior_attempts":attempts,"issued_q_rad":q0.tolist(),"witness_q_rad":None,
            "before":baseline,"after":baseline,"issued_legality":legal_start}


def describe(rows,key):
    scopes = {}
    for label, subset in (("ALL_SELECTED",rows),
                          ("CERTIFIED_WITNESS_ONLY",[r for r in rows if r["status"]=="CERTIFIED_STATIC_IMPROVEMENT"])):
        if not subset:
            scopes[label] = {"n":0}
            continue
        scopes[label] = {"n": len(subset)}
        for stage in ("before","after"):
            q = {metric:np.quantile(np.asarray([row[stage][metric] for row in subset]), [.5,.95,.99,1], axis=0).tolist()
                 for metric in rows[0][stage]}
            scopes[label][stage] = q
    return {"stratum":key,"status_counts":dict(Counter(row["status"] for row in rows)),"stats":scopes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("run","xml","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    receipt=a.run/"receipt.json"
    r=json.loads(receipt.read_bytes())
    assert r["scope"]=="DIAGNOSTIC_ONLY" and r["root"]=="capture:b63bc72d2e8b2ee395e966f6"
    assert digest(a.xml)==r["robot_model_servo"]["assets"]["dual_arm_marvin.xml"]
    rob=RobotModel(a.xml)
    assert list(rob.names)==r["robot_model_servo"]["joints"]
    swept=NativeSweep(rob,clearance_m=rob.clearance_m+CLEARANCE_RESERVE_M,max_depth=18,max_queries=8191)
    support={"OBSERVED":[],"ACTIVE":[]}
    rows_for_key={}
    for part in r["partitions"]:
        vals={}
        wanted=("tick_ns","target_local_landmark_xyz_mask","target_local_landmark_xyz","target_world_tip_xyz",
                "q_issued","q_issued_mask","q_measured_mask","applied_substeps","q_nominal_mask","control_cutoff_ns")
        for k in wanted:
            p=a.run/part["directory"]/(k+".npy")
            assert digest(p)==part["arrays"][k]["sha256"]
            vals[k]=np.load(p,allow_pickle=False)
        for i,t in enumerate(vals["tick_ns"]):
            if not (r["window"]["hmb_cutoff_start_ns"] <= t < r["window"]["hmb_cutoff_stop_exclusive_ns"]):
                continue
            if not (vals["target_local_landmark_xyz_mask"][i].all() and vals["q_measured_mask"][i]
                    and vals["applied_substeps"][i]==10 and vals["q_issued_mask"][i,-1]):
                continue
            key=part["partition"],int(i)
            support["OBSERVED"].append(key)
            if vals["q_nominal_mask"][i]:
                support["ACTIVE"].append(key)
            rows_for_key[key]=(int(t),int(vals["control_cutoff_ns"][i]),vals["q_issued"][i,-1].astype(float),
                                vals["target_local_landmark_xyz"][i].astype(float),vals["target_world_tip_xyz"][i].astype(float))
    assert [len(support[k]) for k in ("OBSERVED","ACTIVE")]==[4801,146]
    selected={}
    for label, rows in support.items():
        marks=np.rint(np.linspace(0,len(rows)-1,PER_STRATUM)).astype(int)
        selected[label]=[rows[j] for j in marks]
    results={}
    for key in dict.fromkeys(selected["ACTIVE"] + selected["OBSERVED"]):
        t,cutoff,q,local,world=rows_for_key[key]
        row=fit(rob,swept,q,local,world)
        results[key]={"partition":key[0],"partition_row":key[1],"original_clock_ns":t,
                      "control_cutoff_ns":cutoff,**row}
        print(f"{key} {row['status']} {row['before']['max_local_tip_mm']:.2f}->{row['after']['max_local_tip_mm']:.2f}",flush=True)
    output={"scope":SCOPE,"formal_coverage_credit":False,"executed_physical_steps":0,
            "run_receipt_sha256":digest(receipt),"model_sha256":digest(a.xml),"script_sha256":digest(__file__),
            "selection":"12 evenly spaced nearest ordinal samples per ACTIVE and OBSERVED support, inclusive bounds. See overlaps; no outcome-based selection.",
            "support_counts":{k:len(v) for k,v in support.items()},"distinct_tested":len(results),
            "proof":{"method":"whole-body straight configuration line NativeSweep","clearance_m":swept.clearance,
                     "max_depth":swept.max_depth,"max_queries":swept.max_queries},
            "candidate":{"arms_frozen":True,"max_abs_shift_each_hand_axis_deg":5,"fixed_scoring_layers":True,
                         "bounded_solver_evaluations":SOLVER_EVALUATIONS,"line_search_scales":SCALING,
                         "max_tolerated_degradation_mm":FUNCTION_TOL_MM},
            "quantile_order":["p50","p95","p99","max"],
            "results_by_strata":{k:describe([results[x] for x in rows],k) for k,rows in selected.items()},
            "selected_per_stratum":selected,"row_witnesses":list(results.values()),
            "qualification":"Witnesses establish a static feasible geometry and proven straight configuration path, not time-dependent Servo execution or a new training/reference target. Absence of witness means unknown."}
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(output,f,indent=2,allow_nan=False)
        f.write("\n")
    print(f"sha256={digest(a.out)}")

if __name__=="__main__":
    main()
