"""Pin the M4 bimanual motion and independently verify its physics provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from dexgym.assets.supermimic import WAVE_REFERENCE, WAVE_VALIDATION
from scipy.spatial.transform import Rotation

REFERENCE = WAVE_REFERENCE
REPORT = WAVE_VALIDATION
SOURCE = Path(__file__).resolve().parents[2] / "source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml"
USD = Path(__file__).resolve().parents[2] / "source/dexgym/assets/marvin_wuji/usd/dual_arm_marvin_no_hand_contacts.usda"
REFERENCE_SHA256 = "b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f"
SOURCE_SHA256 = "6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce"
USD_SHA256 = "8ed2cb9a14a813022149aaa8a101159af86dab676aec7dfdde4a00a8d9698fa6"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    if digest(REFERENCE) != REFERENCE_SHA256 or digest(SOURCE) != SOURCE_SHA256 or digest(USD) != USD_SHA256:
        raise ValueError("Pinned reference or asset SHA256 mismatch")
    report = json.loads(REPORT.read_text())
    if report["status"] != "validated" or not all(report[f"gate_{k}"] for k in "abcd"):
        raise ValueError("M3 bimanual validation did not pass all gates")
    with np.load(REFERENCE, allow_pickle=False) as ref:
        required = {
            "time": (1357,),
            "q_arm": (1357, 14),
            "qd_arm": (1357, 14),
            "qdd_arm": (1357, 14),
            "q_full": (1357, 54),
            "tcp_pos": (1357, 2, 3),
            "tcp_quat_xyzw": (1357, 2, 4),
            "link_pos": (1357, 2, 3, 3),
            "link_quat_xyzw": (1357, 2, 3, 4),
            "link_linear_vel": (1357, 2, 3, 3),
            "link_angular_vel": (1357, 2, 3, 3),
        }
        for name, shape in required.items():
            if ref[name].shape != shape or not np.isfinite(ref[name]).all():
                raise ValueError(f"Invalid {name} shape or finite state")
        if not np.array_equal(ref["time"], np.arange(1357, dtype=np.float64) * 0.02) or int(ref["fps"]) != 50:
            raise ValueError("Reference clock differs from exact 50 Hz")
        if np.max(abs(np.linalg.norm(ref["tcp_quat_xyzw"], axis=-1) - 1)) > 1e-6:
            raise ValueError("TCP quaternions are not unit XYZW")
        if np.any(np.ptp(ref["q_arm"][:, :7], axis=0) < 0.1) or np.any(np.ptp(ref["q_arm"][:, 7:], axis=0) < 0.1):
            raise ValueError("Both seven-joint arms must move")
        q_hash = hashlib.sha256(ref["q_full"].tobytes()).hexdigest()
        if q_hash != report["reference_q_full_sha256"]:
            raise ValueError("Reference joint array differs from validated replay")
        model = mujoco.MjModel.from_xml_path(str(SOURCE))
        data = mujoco.MjData(model)
        names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt))
        if names != tuple(ref["joint_names_full"]) or tuple(ref["joint_names_arm"]) != tuple(
            f"Joint{j}_{side}" for side in ("L", "R") for j in range(1, 8)
        ):
            raise ValueError("Reference joint name order changed")
        link_names = tuple(tuple(x) for x in ref["link_names"])
        if link_names != (("Link3_L", "Link4_L", "tcp_L"), ("Link3_R", "Link4_R", "tcp_R")):
            raise ValueError("Tracked link order changed")
        pos_error, rot_error = [], []
        for frame in (0, 20, 60, 200, 500, 900, 1356):
            data.qpos[:] = ref["q_full"][frame]
            mujoco.mj_forward(model, data)
            for side, suffix in enumerate(("L", "R")):
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{suffix}")
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{suffix}")
                if model.site_bodyid[site] != body:
                    raise ValueError("TCP site/body attachment changed")
                pos_error.append(np.linalg.norm(data.site_xpos[site] - ref["tcp_pos"][frame, side]) * 1000)
                q = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))
                rot_error.append(
                    np.rad2deg((q * Rotation.from_quat(ref["tcp_quat_xyzw"][frame, side]).inv()).magnitude())
                )
                for j, name in enumerate(link_names[side]):
                    if name.startswith("tcp_"):
                        p, r = data.site_xpos[site], q
                    else:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                        p = data.xpos[body_id]
                        r = Rotation.from_matrix(data.xmat[body_id].reshape(3, 3))
                    if (
                        np.linalg.norm(p - ref["link_pos"][frame, side, j]) > 1e-6
                        or (r * Rotation.from_quat(ref["link_quat_xyzw"][frame, side, j]).inv()).magnitude() > 1e-6
                    ):
                        raise ValueError("Tracked link FK mismatch")
        if max(pos_error) > 0.001 or max(rot_error) > 0.001:
            raise ValueError("TCP FK differs from pinned reference")
    physical = {}
    for name in report["physical_reports"]:
        path = WAVE_VALIDATION.parent / "reports" / f"{name}.json"
        item = json.loads(path.read_text())
        if (
            not item["complete"]
            or item["reference_q_full_sha256"] != q_hash
            or item["source_mjcf_sha256"] != SOURCE_SHA256
        ):
            raise ValueError(f"{name} replay provenance/completion mismatch")
        if name.startswith("newton") and (
            item["usd_sha256"] != USD_SHA256 or item["source_geometry_contact_physics_step_env_count"] != 0
        ):
            raise ValueError(f"{name} USD/contact mismatch")
        if name == "newton_16env" and item["num_envs"] != 16:
            raise ValueError("Newton 16 environment replay absent")
        physical[name] = {"complete": item["complete"], "frames": item["frames"], "num_envs": item.get("num_envs", 1)}
    return {
        "reference_sha256": REFERENCE_SHA256,
        "source_mjcf_sha256": SOURCE_SHA256,
        "newton_usd_sha256": USD_SHA256,
        "reference_q_full_sha256": q_hash,
        "frames": 1357,
        "fps": 50,
        "fk_max_position_error_mm": max(pos_error),
        "fk_max_rotation_error_deg": max(rot_error),
        "physics": physical,
        "status": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
