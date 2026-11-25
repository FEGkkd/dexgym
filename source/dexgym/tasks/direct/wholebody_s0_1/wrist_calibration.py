"""TCP to physical Wuji wrist calibration and isolated simulation asset."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ..wholebody_s0.bag import digest
from ..wholebody_s0.prepare import write_json

WRIST_BODIES = ("l_wrist", "r_wrist")
TCP_SITES = ("tcp_L", "tcp_R")


@dataclass(frozen=True)
class WristCalibration:
    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self):
        if (self.translation.shape != (2, 3) or self.quaternion_xyzw.shape != (2, 4)
                or not np.isfinite(self.translation).all()
                or not np.allclose(np.linalg.norm(self.quaternion_xyzw, axis=-1), 1., rtol=0., atol=1e-6)):
            raise ValueError("Expected finite bilateral TCP translations and unit XYZW quaternions")

    @property
    def rotation(self):
        return Rotation.from_quat(self.quaternion_xyzw).as_matrix()

    @classmethod
    def load(cls, path):
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["whole_body_teleop"]["ros__parameters"]
        keys = [f"arm.{side}_tcp_to_wuji_wrist_" for side in ("left", "right")]
        return cls(*(np.asarray([values[key + field] for key in keys], dtype=float)
                     for field in ("translation_m", "quaternion_xyzw")))

    @classmethod
    def from_metadata(cls, metadata):
        values = metadata["wrist_calibration"]
        return cls(np.asarray(values["translation_m"]), np.asarray(values["quaternion_xyzw"]))

    def compose(self, position, rotation):
        return position + np.einsum("...sij,sj->...si", rotation, self.translation), rotation @ self.rotation

    def manifest(self):
        return {"translation_m": self.translation.tolist(), "quaternion_xyzw": self.quaternion_xyzw.tolist(),
                "direction": "marvin_native_tcp_from_wuji_wrist", "wrist_bodies": list(WRIST_BODIES),
                "tcp_sites": list(TCP_SITES)}


def wrist_mount(model, q=None):
    """Measure the compiled TCP-to-wrist transform without assuming a parent body."""
    data = mujoco.MjData(model)
    if q is not None:
        data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    sites = [model.site(name).id for name in TCP_SITES]
    bodies = [model.body(name).id for name in WRIST_BODIES]
    inv = data.site_xmat[sites].reshape(2, 3, 3).swapaxes(-1, -2)
    p = (inv @ (data.xpos[bodies] - data.site_xpos[sites])[..., None])[..., 0]
    r = inv @ data.xmat[bodies].reshape(2, 3, 3)
    return WristCalibration(p, Rotation.from_matrix(r).as_quat(canonical=True))


def audit_model(source, profile, output):
    """Bind simulation data to its own compiled wrist bodies and record the hardware transform."""
    model = mujoco.MjModel.from_xml_path(str(source))
    mount = wrist_mount(model)
    calibration = WristCalibration.load(profile)
    hardware = {"file": str(profile.resolve()), "sha256": digest(profile), **calibration.manifest()}
    model_path, model_sha = str(source.resolve()), digest(source)
    simulated = {"file": model_path, "sha256": model_sha, "source": "native_mjcf_wrist_bodies", **mount.manifest()}
    report = {"original_model": model_path, "original_model_sha256": model_sha,
              "calibrated_model": model_path, "calibrated_model_sha256": model_sha,
              "wrist_calibration": simulated, "original_model_mount": mount.manifest(),
              "model_mount": mount.manifest(), "hardware_wrist_calibration": hardware,
              "hardware_minus_simulation_m": (calibration.translation - mount.translation).tolist(),
              "hardware_vs_simulation_deg":
                  (Rotation.from_matrix(calibration.rotation @ mount.rotation.swapaxes(-1, -2)).magnitude()
                   * 180. / np.pi).tolist()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "wrist_calibration.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "profile", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    print(audit_model(args.model, args.profile, args.output))


if __name__ == "__main__":
    main()
