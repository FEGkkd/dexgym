"""Pair the real-aligned Marvin arm servos with the packaged Hand2.2 model."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ARM_JOINTS = tuple(f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drive_parameters(model, names):
    """Read compiled SI drive parameters in policy joint order."""
    result = {}
    for name in names:
        joint = model.joint(name).id
        ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(ids) != 1:
            raise ValueError(f"Expected a single position actuator for {name}")
        a = int(ids[0])
        kp = float(model.actuator_gainprm[a, 0])
        kd = float(-model.actuator_biasprm[a, 2])
        lo, hi = model.actuator_forcerange[a]
        if (not model.actuator_forcelimited[a] or not model.actuator_ctrllimited[a]
                or model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE
                or not np.allclose(model.actuator_gear[a], [1, 0, 0, 0, 0, 0])
                or not np.allclose(model.actuator_biasprm[a, :3], [0, -kp, -kd])
                or kp <= 0 or kd < 0 or hi <= 0 or not np.isclose(lo, -hi)):
            raise ValueError(f"Unsupported position servo for {name}")
        result[name] = dict(actuator_id=a, kp=kp, kd=kd, effort=float(hi))
    return result


def validate_drive_only_model(base_xml, aligned_xml):
    """An alternate MJCF may change only the 14 arm position servos and mesh path."""
    original, aligned = [ET.parse(path).getroot() for path in (base_xml, aligned_xml)]
    for tree in (original, aligned):
        tree.find("compiler").attrib.pop("meshdir", None)
        for actuator in tree.findall("./actuator/position"):
            if actuator.get("joint") in ARM_JOINTS:
                for field in ("kp", "kv", "forcerange", "ctrlrange"):
                    actuator.attrib.pop(field, None)
    if ET.canonicalize(ET.tostring(original), strip_text=True) != ET.canonicalize(
        ET.tostring(aligned), strip_text=True
    ):
        raise ValueError("Aligned model must preserve the packaged non-drive asset")


def compose_model(base_xml, arm_xml, output):
    base_xml, arm_xml, output = (Path(p).resolve() for p in (base_xml, arm_xml, output))
    if output in (base_xml, arm_xml):
        raise ValueError("Use a separate output XML")
    original = ET.parse(base_xml)
    supplied = ET.parse(arm_xml)
    for name in ARM_JOINTS:
        for query in (f'.//joint[@name="{name}"]', f'.//body[@name="Link{name[5:]}"]/inertial'):
            old, new = (tree.find(query) for tree in (original, supplied))
            if old is None or new is None or old.attrib != new.attrib:
                raise ValueError(f"Arm geometry or passive parameters differ for {query}")
        old, new = (tree.find(f'./actuator/position[@joint="{name}"]') for tree in (original, supplied))
        if old is None or new is None:
            raise ValueError(f"Missing arm position servo for {name}")
        for field in ("kp", "kv", "forcerange", "ctrlrange"):
            old.set(field, new.attrib[field])
    for name in (f"Link{i}_{side}" for side in ("L", "R") for i in range(1, 8)):
        if original.find(f'.//body[@name="{name}"]').attrib != supplied.find(f'.//body[@name="{name}"]').attrib:
            raise ValueError(f"Arm link transform differs for {name}")
    compiler = original.find("compiler")
    compiler.set("meshdir", str((base_xml.parent / compiler.attrib["meshdir"]).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    original.write(output, encoding="utf-8")
    validate_drive_only_model(base_xml, output)
    model = mujoco.MjModel.from_xml_path(str(output))
    if model.nq != 54 or model.nu != 54:
        raise ValueError("Expected the complete 54-joint system")
    params = drive_parameters(model, (model.joint(i).name for i in range(model.njnt)))
    return {
        "arm_source": str(arm_xml), "arm_source_sha256": sha256(arm_xml),
        "base_model_sha256": sha256(base_xml), "model_sha256": sha256(output),
        "arm_drives": {name: params[name] for name in ARM_JOINTS},
        "hand_drives": len(params) - len(ARM_JOINTS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--arms", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compose_model(args.base, args.arms, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
