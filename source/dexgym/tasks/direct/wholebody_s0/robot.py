"""Name-based 54-DoF joint and MuJoCo geometry contract for wholebody_s0."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract

from . import SIDES

_MODEL_DIR = Path(__file__).resolve().parents[3] / "assets/marvin_wuji/hand2_2/generated"
DEFAULT_MODEL = _MODEL_DIR / "dual_arm_marvin_wholebody_s0.xml"
NEWTON_USD = _MODEL_DIR / "usd/dual_arm_marvin_wholebody_s0.usda"

FIRMWARE_FINGERS = ("THJ", "FFJ", "MFJ", "RFJ", "LFJ")
FINGER_SITES = ("thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip")


class Robot:
    def __init__(self, path):
        self.path = path
        self.model = m = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(m)
        contract = inspect_joint_contract(m)
        arms = (*contract.groups["arm_left"], *contract.groups["arm_right"])
        self.firmware_names = (*arms, *(
            f"{side[0]}_{finger}{joint}" for side in SIDES for finger in FIRMWARE_FINGERS for joint in range(4)
        ))
        hand_joints = tuple(
            m.joint(int(m.actuator(name).trnid[0])).name for name in self.firmware_names[14:]
        )
        self.names = (*arms, *hand_joints)
        if len(set(self.names)) != 54 or set(self.names) != set(contract.names):
            raise ValueError("Firmware/model joint mapping does not cover 54 unique joints")
        self.qpos = contract.indices_for(self.names)
        self.ctrl = [contract.ctrl_by_name[name] for name in self.names]
        self.bounds = np.asarray([contract.limits_by_name[name] for name in self.names])
        mujoco.mj_resetDataKeyframe(m, self.data, m.key("home").id)
        mujoco.mj_forward(m, self.data)
        self.home = self.data.qpos[self.qpos].copy()
        base = [m.body("Base_" + suffix).id for suffix in "LR"]
        self.base_position = self.data.xpos[base].copy()
        self.base_rotation = self.data.xmat[base].reshape(2, 3, 3).copy()
        self.tcp = [m.site("tcp_" + suffix).id for suffix in "LR"]
        self.tips = [m.site(f"{side[0]}_{tip}").id for side in SIDES for tip in FINGER_SITES]
        self.nodes, self.links = self.skeleton_contract()

    def skeleton_contract(self):
        nodes = [("body", self.model.body("center_mount").id)]
        links = []
        for side, suffix in zip(SIDES, "LR"):
            arm = [f"Base_{suffix}", *(f"Link{i}_{suffix}" for i in range(1, 8)), f"TCP_Link_{suffix}", f"{side[0]}_wrist"]
            previous = 0
            for name in arm:
                nodes.append(("body", self.model.body(name).id))
                links.append([previous, len(nodes) - 1, side, "arm"])
                previous = len(nodes) - 1
            palm = previous
            for tip in FINGER_SITES:
                site = self.model.site(f"{side[0]}_{tip}")
                body = int(site.bodyid[0])
                chain = []
                while body != nodes[palm][1]:
                    if body <= 0:
                        raise ValueError(f"Fingertip is outside the expected hand hierarchy: {site.name}")
                    chain.append(body)
                    body = int(self.model.body_parentid[body])
                previous = palm
                for body in reversed(chain):
                    nodes.append(("body", body))
                    links.append([previous, len(nodes) - 1, side, "finger"])
                    previous = len(nodes) - 1
                nodes.append(("site", site.id))
                links.append([previous, len(nodes) - 1, side, "finger"])
        return nodes, links

    def world_tcp(self, local):
        world = np.full_like(local, np.nan)
        for i, (rotation, translation) in enumerate(zip(self.base_rotation, self.base_position)):
            good = np.isfinite(local[:, i]).all(axis=1)
            world[good, i, :3] = local[good, i, :3] @ rotation.T + translation
            world[good, i, 3:] = (Rotation.from_matrix(rotation) * Rotation.from_quat(local[good, i, 3:])).as_quat()
        return world

    def geometry(self, q):
        data = self.data
        frames = len(q)
        points = np.empty((frames, len(self.nodes), 3), dtype=np.float32)
        wrists = np.empty((frames, 2, 7), dtype=np.float32)
        tips = np.empty((frames, 2, 5, 7), dtype=np.float32)
        penetration = np.zeros(frames, dtype=np.float32)
        bodies = np.array([i for i, (kind, _) in enumerate(self.nodes) if kind == "body"])
        sites = np.array([i for i, (kind, _) in enumerate(self.nodes) if kind == "site"])
        body_ids = [self.nodes[i][1] for i in bodies]
        site_ids = [self.nodes[i][1] for i in sites]
        rotations = np.empty((frames, 12, 3, 3), dtype=float)
        for index, pose in enumerate(q):
            data.qpos[self.qpos] = pose
            mujoco.mj_fwdPosition(self.model, data)
            points[index, bodies] = data.xpos[body_ids]
            points[index, sites] = data.site_xpos[site_ids]
            wrists[index, :, :3] = data.site_xpos[self.tcp]
            tips[index, ..., :3] = data.site_xpos[self.tips].reshape(2, 5, 3)
            rotations[index] = data.site_xmat[self.tcp + self.tips].reshape(12, 3, 3)
            if data.ncon:
                penetration[index] = max(0, -float(np.min(data.contact.dist)))
        quat = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(frames, 12, 4)
        wrists[..., 3:] = quat[:, :2]
        tips[..., 3:] = quat[:, 2:].reshape(frames, 2, 5, 4)
        return {
            "robot_skeleton": points, "robot_wrist_pose": wrists,
            "robot_fingertip_pose": tips, "static_penetration_m": penetration,
        }
