"""Resolve the pinned upstream scene in its own MJLab runtime for comparison.

This optional audit needs the reference repository and its dependencies. Native
training does not import MJLab or this reference checkout.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if sha != "2becb7380fed86171056f9bf4338b2a6a0c14b01":
        raise ValueError("Reference source revision mismatch")
    sys.path.insert(0, str(args.reference / "src"))
    sys.path.insert(0, str(args.reference / "packages/wuji-description/src"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json

    import mujoco
    import numpy as np
    import torch
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_env_cfg(4, 50)
    cfg.seed = 11
    cfg.auto_reset = False
    e = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        e.reset()
        e.step(torch.zeros(4, 20, device="cuda:0"))
        m = e.sim.mj_model
        floors = []
        for i in range(m.ngeom):
            if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                floors.append(
                    {
                        "name": m.geom(i).name,
                        "contype": int(m.geom_contype[i]),
                        "conaffinity": int(m.geom_conaffinity[i]),
                        "friction": m.geom_friction[i].tolist(),
                        "solref": m.geom_solref[i].tolist(),
                        "solimp": m.geom_solimp[i].tolist(),
                        "condim": int(m.geom_condim[i]),
                        "pos": m.geom_pos[i].tolist(),
                        "quat": m.geom_quat[i].tolist(),
                    }
                )
        result = {
            "ngeom": m.ngeom,
            "nbody": m.nbody,
            "floors": floors,
            "options": {
                k: (
                    getattr(m.opt, k).tolist()
                    if hasattr(getattr(m.opt, k), "tolist")
                    else int(getattr(m.opt, k))
                    if k
                    in (
                        "disableflags",
                        "enableflags",
                        "integrator",
                        "solver",
                        "iterations",
                        "ls_iterations",
                        "ccd_iterations",
                    )
                    else float(getattr(m.opt, k))
                )
                for k in (
                    "disableflags",
                    "enableflags",
                    "integrator",
                    "solver",
                    "iterations",
                    "ls_iterations",
                    "ccd_iterations",
                    "timestep",
                    "tolerance",
                    "gravity",
                )
            },
        }
        live = {}
        for k in result["options"]:
            v = getattr(e.sim.wp_model.opt, k)
            a = (
                v.detach().cpu().numpy()
                if isinstance(v, torch.Tensor)
                else np.asarray(v.numpy() if hasattr(v, "numpy") else v)
            )
            live[k] = a.tolist()
        result["live_options"] = live
        result["reference_sha"] = sha
        result["mujoco_version"] = mujoco.__version__
        args.output.write_text(json.dumps(result, indent=2))
        print("ORIGINAL_SCENE", result, flush=True)
    finally:
        e.close()


if __name__ == "__main__":
    main()
