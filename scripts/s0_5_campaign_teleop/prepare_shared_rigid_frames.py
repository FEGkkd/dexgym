#!/usr/bin/env python3
"""Build the fixed frame bridges for the isolated shared-START diagnostic."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.rmb.nominal import Marvin, matrix


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def palm_axes(wrist, finger_roots):
    def unit(value):
        return value / np.linalg.norm(value)
    longitudinal = unit(finger_roots.mean(axis=0)-wrist)
    lateral = finger_roots[0]-finger_roots[-1]
    lateral = unit(lateral-longitudinal*(lateral@longitudinal))
    return np.column_stack((longitudinal,lateral,np.cross(longitudinal,lateral)))


def verify_rigid(transform):
    rotation = transform[:3,:3]
    if not (np.isfinite(transform).all() and np.allclose(rotation.T@rotation, np.eye(3), atol=1e-8)
            and np.isclose(np.linalg.det(rotation),1.,atol=1e-8)):
        raise ValueError("A fixed frame bridge was not a proper rigid transform")


def pose_delta(one, two):
    return (float(np.linalg.norm(one[:3,3]-two[:3,3])),
            float(Rotation.from_matrix(one[:3,:3].T@two[:3,:3]).magnitude()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for item in ("model", "marvin-bridge", "marvin-config", "hand-calibration-report", "out"):
        p.add_argument(f"--{item}", type=Path, required=True)
    args=p.parse_args()
    robot=RobotModel(args.model)
    model_sha=sha(args.model)
    if model_sha!="be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d":
        raise ValueError("Require the exact original runtime MJCF")
    record=json.loads(args.hand_calibration_report.read_text())
    hand=record["saved"]["contact_and_flat_orientation"]
    hand_file=Path(hand["path"])
    if sha(hand_file)!=hand["sha256"]:
        raise ValueError("Historical physical calibration file drift")
    cfg=yaml.safe_load(hand_file.read_text())
    base_rotation=np.asarray(cfg["tracker_to_robot_world"]).reshape(3,3)
    world_from_tracker=matrix(np.zeros(3),base_rotation)
    if not np.allclose(base_rotation[:,1],[0,0,1],atol=1e-8):
        raise ValueError("The pre-registered world rotation must preserve vertical")
    basis_path=args.hand_calibration_report.parent.parent/"calibration_samples.npz"
    with np.load(basis_path,allow_pickle=False) as samples:
        historical_basis=samples["bases"].copy()
    home=robot.geometry(robot.home)
    installed=Marvin(args.marvin_bridge,args.marvin_config)
    bases, mounts, tools, bridge_errors=[],[],[],[]
    try:
        for side, label in enumerate(("left","right")):
            if cfg["tracker_topics"][label]!=f"/vive/tracker/{label}_wrist":
                raise ValueError("Historical calibration does not consume the original registered stream")
            body=robot.data.body("Base_L" if side==0 else "Base_R")
            world_from_base=matrix(body.xpos,body.xmat.reshape(3,3))
            model_palm=matrix(home.palm[side],home.palm_rotation[side])
            section=slice(side*7,side*7+7)
            native_at_home=installed.fk(side,robot.home[section])
            wuji_from_native_tcp=np.linalg.solve(model_palm,world_from_base@native_at_home)
            anatomical=palm_axes(home.palm[side],home.landmarks[side,1:,0])
            wuji_basis=model_palm[:3,:3].T@anatomical
            fitted=cfg["wrist_calibration"][label]
            fitted_orientation=Rotation.from_quat(fitted["rotation_xyzw"]).as_matrix()
            tracker_from_wuji=matrix(fitted["position_offset_m"],
                                     fitted_orientation@historical_basis[side]@wuji_basis.T)
            errors=[]
            for axis in range(section.start,section.stop):
                for delta in (-.04,.04):
                    q=robot.home.copy()
                    q[axis]+=delta
                    perturbed=robot.geometry(q)
                    palm=matrix(perturbed.palm[side],perturbed.palm_rotation[side])
                    native=world_from_base@installed.fk(side,q[section])
                    errors.append(pose_delta(palm@wuji_from_native_tcp,native))
            max_pos,max_angle=np.max(errors,axis=0)
            if max_pos>.0015 or max_angle>np.deg2rad(.5):
                raise ValueError(f"Native-to-live TCP bridge is unqualified: {label} {max_pos}, {max_angle}")
            bridge_errors.append({"side":label,"probe_count":len(errors),
                                  "max_position_m":float(max_pos),"max_angle_rad":float(max_angle)})
            bases.append(world_from_base)
            mounts.append(tracker_from_wuji)
            tools.append(wuji_from_native_tcp)
            # Keep the fixed body query at the named model Home before the next side.
            robot.set(robot.home)
    finally:
        installed.close()
    ordered=[world_from_tracker,*bases,*mounts,*tools]
    for transform in ordered:
        verify_rigid(transform)
    args.out.mkdir(exist_ok=False,parents=True)
    config=args.out/"shared_rigid_transforms.txt"
    np.savetxt(config,np.vstack([t[:3] for t in ordered]),fmt="%.17g")
    receipt={
        "scope":"ISOLATED_MUJOCO_MAPPING_DIAGNOSTIC",
        "loaded_model_sha256":model_sha,"historical_hand_calibration_sha256":hand["sha256"],
        "historical_basis_source_sha256":sha(basis_path),"marvin_bridge_sha256":sha(args.marvin_bridge),
        "marvin_config_sha256":sha(args.marvin_config),"transform_file_sha256":sha(config),
        "config_order":["world_from_tracker","world_from_base_left","world_from_base_right",
                        "tracker_from_wuji_left","tracker_from_wuji_right",
                        "wuji_from_native_tcp_left","wuji_from_native_tcp_right"],
        "sides":bridge_errors,"runtime_start_translation":"Align the two wrist midpoints",
    }
    (args.out/"coordinate_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    main()
