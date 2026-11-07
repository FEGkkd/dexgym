"""Author an isolated USD layer with MuJoCo-equivalent Newton angular drives."""

import math
import os

import mujoco
import numpy as np
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract
from dexgym.assets.marvin_wuji.robot_cfg import NO_HAND_CONTACT_USD, PACKAGED_MJCF, PACKAGED_USD
from pxr import Sdf, Usd

OUTPUT = PACKAGED_USD.with_name("hand_only_drive_corrected.usda")
NO_HAND_CONTACT_OUTPUT = PACKAGED_USD.with_name("hand_only_drive_corrected_no_hand_contacts.usda")
TUNED_OUTPUT = PACKAGED_USD.with_name("hand_only_drive_corrected_hand_x2.usda")
TUNED_NO_HAND_CONTACT_OUTPUT = PACKAGED_USD.with_name("hand_only_drive_corrected_hand_x2_no_hand_contacts.usda")
USD_ANGULAR_GAIN_TO_SI = 180.0 / math.pi


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(PACKAGED_MJCF))
    contract = inspect_joint_contract(model)
    source = Usd.Stage.Open(str(PACKAGED_USD))
    if not source or source.GetDefaultPrim().GetVariantSet("Physics").GetVariantSelection() != "physics":
        raise ValueError("Pinned source USD must compose the physics variant")
    drives = {}
    for prim in source.Traverse():
        stiffness = prim.GetAttribute("drive:angular:physics:stiffness")
        damping = prim.GetAttribute("drive:angular:physics:damping")
        if stiffness and damping and stiffness.HasAuthoredValueOpinion() and damping.HasAuthoredValueOpinion():
            name = prim.GetName()
            if name in drives or name not in contract.ctrl_by_name:
                raise ValueError(f"Unexpected or duplicate angular drive: {name}")
            aid = contract.ctrl_by_name[name]
            source_kp = float(model.actuator_gainprm[aid, 0])
            source_kd = float(-model.actuator_biasprm[aid, 2])
            if not np.allclose([stiffness.Get(), damping.Get()], [source_kp, source_kd], rtol=1e-5, atol=1e-6):
                raise ValueError(f"USD and source MJCF drive values differ for {name}")
            drives[name] = (prim.GetPath(), source_kp, source_kd)
    if set(drives) != set(contract.names):
        raise ValueError(f"Expected all 54 independent drives; found {len(drives)}")

    profiles = (
        (PACKAGED_USD, OUTPUT, 1.0),
        (NO_HAND_CONTACT_USD, NO_HAND_CONTACT_OUTPUT, 1.0),
        (PACKAGED_USD, TUNED_OUTPUT, 2.0),
        (NO_HAND_CONTACT_USD, TUNED_NO_HAND_CONTACT_OUTPUT, 2.0),
    )
    for base, output, right_hand_scale in profiles:
        if output.exists():
            output.unlink()
        layer = Sdf.Layer.CreateNew(str(output))
        layer.subLayerPaths.append(os.path.relpath(base, output.parent))
        layer.defaultPrim = source.GetDefaultPrim().GetName()
        layer.customLayerData = {"purpose": "isolated hand-only drive diagnostic; source USD unchanged"}
        stage = Usd.Stage.Open(layer.identifier)
        if not stage:
            raise RuntimeError(f"Could not compose hand-only USD overlay {output}")
        for name, (path, kp, kd) in drives.items():
            prim = stage.GetPrimAtPath(path)
            scale = right_hand_scale if name.startswith("r_") else 1.0
            prim.GetAttribute("drive:angular:physics:stiffness").Set(kp * scale / USD_ANGULAR_GAIN_TO_SI)
            prim.GetAttribute("drive:angular:physics:damping").Set(kd * scale / USD_ANGULAR_GAIN_TO_SI)
        layer.Save()

        check = Usd.Stage.Open(str(output))
        if not check or not check.GetDefaultPrim().IsValid():
            raise RuntimeError(f"Hand-only USD overlay did not compose: {output}")
        for name, (path, kp, kd) in drives.items():
            prim = check.GetPrimAtPath(path)
            values = [
                prim.GetAttribute("drive:angular:physics:stiffness").Get(),
                prim.GetAttribute("drive:angular:physics:damping").Get(),
            ]
            scale = right_hand_scale if name.startswith("r_") else 1.0
            expected = [kp * scale / USD_ANGULAR_GAIN_TO_SI, kd * scale / USD_ANGULAR_GAIN_TO_SI]
            if not np.allclose(values, expected, rtol=1e-6, atol=1e-8):
                raise AssertionError(f"Incorrect composed drive in {prim.GetPath()}")
        print(f"Wrote {len(drives)} corrected angular drives (right-hand scale {right_hand_scale:g}) to {output}")


if __name__ == "__main__":
    main()
