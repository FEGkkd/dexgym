"""Check composed-hand physics and arm/palm-relative FK against independent sources."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from dexgym.assets.marvin_wuji.build_hand2_2 import ROOT
from dexgym.assets.marvin_wuji.robot_cfg import LEGACY_MJCF, PACKAGED_MJCF
from dexgym.assets.wuji_hand2_cube import ROOT as CUBE_ROOT
from dexgym.assets.wuji_hand2_cube.source_model import WUJI_HAND2_TASK_NAMES


def validate():
    dual = mujoco.MjModel.from_xml_path(str(PACKAGED_MJCF))
    old = mujoco.MjModel.from_xml_path(str(LEGACY_MJCF))
    cube = mujoco.MjModel.from_xml_path(str(CUBE_ROOT / "generated/hand2_cube_rig.xml"))
    left = mujoco.MjModel.from_xml_path(str(ROOT / "hand2_2/official/mjcf/left.xml"))
    errors = {}
    checks = 0

    def check(label, a, b, tol=2e-5):
        nonlocal checks
        err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        errors[label] = max(errors.get(label, 0.0), err)
        checks += 1
        if err > tol:
            raise AssertionError(f"{label}: {err} > {tol}")

    for oldname, cubename in WUJI_HAND2_TASK_NAMES["joint"].items():
        cj = cube.joint(cubename).id
        ca = int(np.flatnonzero(cube.actuator_trnid[:, 0] == cj)[0])
        for prefix in ("r", "l"):
            name = prefix + oldname[1:]
            j = dual.joint(name).id
            d = dual.jnt_dofadr[j]
            a = int(np.flatnonzero(dual.actuator_trnid[:, 0] == j)[0])
            check("hand_range", dual.jnt_range[j], cube.jnt_range[cj])
            check("hand_kp_kd", dual.actuator_biasprm[a, 1:3], cube.actuator_biasprm[ca, 1:3])
            check("hand_effort", dual.actuator_forcerange[a], cube.actuator_forcerange[ca])
            check("hand_total_effort", dual.jnt_actfrcrange[j], cube.jnt_actfrcrange[cj])
            check("hand_armature", dual.dof_armature[d], cube.dof_armature[cube.jnt_dofadr[cj]])
    for oldname, cubename in WUJI_HAND2_TASK_NAMES["body"].items():
        for model, name, label in [(cube, cubename, "right"), (left, "l" + oldname[1:], "left")]:
            target = ("r" if label == "right" else "l") + oldname[1:]
            b = dual.body(target).id
            s = model.body(name).id
            for attr in ("body_mass", "body_inertia", "body_ipos"):
                check(label + "_" + attr, getattr(dual, attr)[b], getattr(model, attr)[s], 2e-6)
    for g in range(cube.ngeom):
        if not (cube.geom_contype[g] or cube.geom_conaffinity[g]):
            continue
        name = cube.geom(g).name
        assert name.startswith("right_")
        target = "r_" + name[6:]
        dg = dual.geom(target).id
        for attr in (
            "geom_type",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "geom_condim",
            "geom_priority",
            "geom_contype",
            "geom_conaffinity",
        ):
            check("right_" + attr, getattr(dual, attr)[dg], getattr(cube, attr)[g], 2e-6)
    # Arm kinematic/physical properties must not be silently changed.
    for j in range(14):
        name = f"Joint{j % 7 + 1}_{'L' if j < 7 else 'R'}"
        a = dual.joint(name).id
        b = old.joint(name).id
        for attr in ("jnt_pos", "jnt_axis", "jnt_range"):
            check("arm_" + attr, getattr(dual, attr)[a], getattr(old, attr)[b], 1e-12)
        for attr in ("dof_armature", "dof_damping", "dof_frictionloss"):
            check("arm_" + attr, getattr(dual, attr)[dual.jnt_dofadr[a]], getattr(old, attr)[old.jnt_dofadr[b]], 1e-12)
    ds = {k: mujoco.MjData(v) for k, v in [("dual", dual), ("cube", cube), ("left", left), ("old", old)]}
    rng = np.random.default_rng(20261111)
    for _ in range(100):
        ds["dual"].qpos[:] = rng.uniform(dual.jnt_range[:, 0] * 0.8, dual.jnt_range[:, 1] * 0.8)
        for model, key in [(cube, "cube"), (left, "left"), (old, "old")]:
            for j in range(model.njnt):
                name = model.joint(j).name
                if key == "cube":
                    name = {v: k for k, v in WUJI_HAND2_TASK_NAMES["joint"].items()}[name]
                ds[key].qpos[model.jnt_qposadr[j]] = ds["dual"].qpos[dual.jnt_qposadr[dual.joint(name).id]]
            mujoco.mj_forward(model, ds[key])
        mujoco.mj_forward(dual, ds["dual"])
        for side, model, key, palm in [("r", cube, "cube", "right_palm_link"), ("l", left, "left", "l_wrist")]:
            dp = dual.body(side + "_wrist").id
            sp = model.body(palm).id
            dr = ds["dual"].xmat[dp].reshape(3, 3)
            sr = ds[key].xmat[sp].reshape(3, 3)
            for oldname, cn in WUJI_HAND2_TASK_NAMES["body"].items():
                name = side + oldname[1:]
                b = dual.body(name).id
                s = model.body(cn if side == "r" else name).id
                check(
                    side + "_palm_relative_position_m",
                    (ds["dual"].xpos[b] - ds["dual"].xpos[dp]) @ dr,
                    (ds[key].xpos[s] - ds[key].xpos[sp]) @ sr,
                    3e-6,
                )
                check(
                    side + "_palm_relative_rotation",
                    dr.T @ ds["dual"].xmat[b].reshape(3, 3),
                    sr.T @ ds[key].xmat[s].reshape(3, 3),
                    5e-6,
                )
        for name in ("TCP_Link_L", "TCP_Link_R"):
            check("unchanged_arm_tcp_m", ds["dual"].xpos[dual.body(name).id], ds["old"].xpos[old.body(name).id], 1e-10)
    return {"pass": True, "poses": 100, "checks": checks, "max_errors": errors}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    r = validate()
    if a.output:
        a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
