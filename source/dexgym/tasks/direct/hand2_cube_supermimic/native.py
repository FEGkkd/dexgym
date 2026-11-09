"""Newton backend boundary. Task/model code uses WXYZ; Isaac Lab uses XYZW.

Contacts and forces are privileged training labels, never actor inputs.
"""

import re

import mujoco
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from dexgym.assets.wuji_hand2_cube import ROOT

from .kinematics import HandKinematics
from .mdp import DexMDP
from .rotations import mv, quat_to_mat
from .state import pack


def rotation_xyzw(q):
    return quat_to_mat(q[..., [3, 0, 1, 2]])


class NativePhysics:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.obj = env.scene["cube"]
        self.device = env.device
        # Backend-specific access is isolated here (Isaac Lab 3 Newton).
        self.solver = NewtonManager._solver
        self.reference = mujoco.MjModel.from_xml_path(str(ROOT / "generated/hand2_cube_rig.xml"))
        ref = self.reference

        def tensor(a):
            return torch.as_tensor(np.array(a), dtype=torch.float32, device=self.device)

        self.joint_names = [ref.joint(i).name for i in range(ref.njnt)]
        assert self.robot.joint_names == self.joint_names
        bodies = list(range(1, ref.nbody))
        tip_sites = [ref.site(f"right_finger{i}_tip").id for i in range(1, 6)]
        self.kin = HandKinematics(ref, bodies, range(20), tip_sites, self.device)
        data = mujoco.MjData(ref)
        mujoco.mj_forward(ref, data)
        tag = ref.site("right_wrist_tag").id
        self.mdp = DexMDP(
            self.kin,
            self.robot.data.default_joint_pos.torch[0].clone(),
            self.robot.data.soft_joint_pos_limits.torch[0, :, 0].clone(),
            self.robot.data.soft_joint_pos_limits.torch[0, :, 1].clone(),
            tensor(data.site_xpos[tag]),
            tensor(data.site_xmat[tag].reshape(3, 3)),
            50,
            tensor(ref.actuator_forcerange[:, 1]),
            cage_up_axis=1,
        )
        self.mdp.kp_default = tensor(ref.actuator_gainprm[:, 0])
        self.mdp.kd_default = tensor(ref.actuator_biasprm[:, 2])
        self.mdp.force_limit = tensor(ref.actuator_forcerange[:, 1])
        self.mdp.gear = tensor(ref.actuator_gear[:, 0])
        self.native_model = self.solver.mj_model
        # MJLab resolves the scene options after attaching assets. Its parent
        # scene keeps EULERDAMP enabled (disableflags=0), ignoring the child flag.
        model = self.native_model
        self.joint_ids = [
            next(i for i in range(model.njnt) if model.joint(i).name.endswith("_" + name)) for name in self.joint_names
        ]
        self.dof_ids = torch.tensor(model.jnt_dofadr[self.joint_ids], device=self.device, dtype=torch.long)
        self.body_ids = [
            next(i for i in range(model.nbody) if model.body(i).name.endswith("_" + ref.body(b).name)) for b in bodies
        ]
        self.native_to_ref_body = {native: b for native, b in zip(self.body_ids, bodies)}
        self.geom_ids = {}
        for i in range(model.ngeom):
            b = self.native_to_ref_body.get(int(model.geom_bodyid[i]))
            if b is None:
                continue
            candidates = [
                j
                for j in range(ref.ngeom)
                if ref.geom_bodyid[j] == b and (ref.geom_contype[j] or ref.geom_conaffinity[j])
            ]
            if len(candidates) != 1:
                raise ValueError(f"Expected one active collider on {ref.body(b).name}, got {candidates}")
            self.geom_ids[ref.geom(candidates[0]).name] = i
        self.cube_geom = next(i for i in range(model.ngeom) if "/Cube/" in model.geom(i).name)
        self.contact_slots = [
            ref.geom(i).name for i in range(ref.ngeom) if re.fullmatch(r".*finger.*_col", ref.geom(i).name)
        ]
        self.contact_dim = 7 + len(self.contact_slots)
        self._contact_table = torch.zeros(model.ngeom, model.ngeom, self.contact_dim, device=self.device)
        for name, g in self.geom_ids.items():
            body_name = ref.body(int(ref.geom_bodyid[ref.geom(name).id])).name
            if "link4" in body_name or "tip_sensor" in body_name:
                digit = int(re.search(r"finger([1-5])", body_name).group(1)) - 1
                self._contact_table[g, self.cube_geom, digit] = 1
            if "palm_link" in body_name or re.search(r"link[12]$", body_name):
                self._contact_table[g, self.cube_geom, 5] = 1
            if re.search(r"link[34]$|tip_sensor$", body_name):
                self._contact_table[g, self.cube_geom, 6] = 1
            if name in self.contact_slots:
                self._contact_table[g, list(self.geom_ids.values()), 7 + self.contact_slots.index(name)] = 1
        self._contact_table = torch.maximum(self._contact_table, self._contact_table.transpose(0, 1)).contiguous()
        self.root_joint_ids = wp.to_torch(self.solver.mjc_mocap_to_newton_jnt).long()[:, 0]
        root_child = wp.to_torch(self.solver.model.joint_X_c)[self.root_joint_ids]
        identity = root_child.new_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if not torch.allclose(root_child, identity.expand_as(root_child), atol=1e-6):
            raise ValueError("Cube task expects an identity fixed-root child frame")
        self.palm_body_id = self.robot.body_names.index("right_palm_link")
        self.refresh_frame()

    def write_hand_root(self, pose, env_ids):
        # Isaac Lab's fixed-root ProxyArray can be a gathered view. Write the
        # authoritative Newton joint transform as well, then sync the mocap root.
        from newton import ModelFlags

        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        wp.to_torch(self.solver.model.joint_X_p)[self.root_joint_ids[env_ids]] = pose
        self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
        self.robot.data._reset_pose(env_ids=self.robot._resolve_env_ids(env_ids))

    def refresh_frame(self):
        root = self.robot.data.body_link_pose_w.torch[:, self.palm_body_id]
        self.palm_pos = root[:, :3].clone()
        self.palm_rot = rotation_xyzw(root[:, 3:])

    def state(self):
        self.refresh_frame()
        pose = self.obj.data.root_link_pose_w.torch
        velocity = self.obj.data.root_link_vel_w.torch
        inv = self.palm_rot.transpose(-1, -2)
        return pack(
            self.robot.data.joint_pos.torch.clone(),
            self.robot.data.joint_vel.torch.clone(),
            mv(inv, pose[:, :3] - self.palm_pos),
            inv @ rotation_xyzw(pose[:, 3:]),
            mv(inv, velocity[:, :3]),
            mv(inv, velocity[:, 3:]),
        )

    def torque(self):
        return wp.to_torch(self.solver.mjw_data.qfrc_actuator)[:, self.dof_ids].clone()

    def contact(self):
        d = self.solver.mjw_data
        pair = wp.to_torch(d.contact.geom).long()
        world = wp.to_torch(d.contact.worldid).long()
        active = (
            (torch.arange(len(pair), device=self.device) < wp.to_torch(d.nacon)[0])
            & (world >= 0)
            & (world < self.env.num_envs)
        )
        labels = self._contact_table[
            pair[:, 0].clamp(0, self.native_model.ngeom - 1), pair[:, 1].clamp(0, self.native_model.ngeom - 1)
        ]
        labels = labels * active[:, None]
        found = labels.new_zeros(self.env.num_envs, self.contact_dim)
        found.scatter_add_(0, world.clamp(0, self.env.num_envs - 1)[:, None].expand_as(labels), labels)
        return found.clamp_max(1)
