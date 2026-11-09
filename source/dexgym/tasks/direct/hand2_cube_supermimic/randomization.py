"""Reference startup distributions, authored into Newton's persistent model.

Newton owns the authoritative parameters; notify_model_changed propagates them
into MuJoCo Warp. Thus resets or a later model sync cannot erase randomization.
"""

import math
import re

import torch
import warp as wp
from newton import ModelFlags

from .rotations import quat_to_mat


def apply_startup_randomization(env):
    p = env.physics
    s = p.solver
    model = s.model
    ref = p.reference
    n = env.num_envs
    device = env.device

    def tensor(a):
        return torch.as_tensor(a, device=device, dtype=torch.float32)

    def uniform(shape, lo, hi):
        return torch.empty(shape, device=device).uniform_(lo, hi)

    shapes = wp.to_torch(s.mjc_geom_to_newton_shape).long()
    bodies = wp.to_torch(s.mjc_body_to_newton).long()
    dofs = wp.to_torch(s.mjc_dof_to_newton_dof).long()[:, p.dof_ids]
    cube_body = int(p.native_model.geom_bodyid[p.cube_geom])
    cube_ids = bodies[:, cube_body]
    hand_ids = bodies[:, p.body_ids]
    # Uniform COM offset, then independent link inertia and mass scales.
    wp.to_torch(model.body_com)[cube_ids] += uniform((n, 3), -0.003, 0.003)
    cube_mass_scale = uniform((n, 1), 0.4, 1.6)
    mass = wp.to_torch(model.body_mass)
    inertia = wp.to_torch(model.body_inertia)
    mass[cube_ids] *= cube_mass_scale[:, 0]
    inertia[cube_ids] *= cube_mass_scale[:, :, None]
    hand_mass_scale = uniform((n, len(p.body_ids)), 0.4, 1.5)
    mass[hand_ids] *= hand_mass_scale
    hand_inertia_scale = uniform((n, len(p.body_ids), 3), 0.4, 1.5)
    mjr = quat_to_mat(tensor(p.native_model.body_iquat[p.body_ids]))
    diagonal = tensor(p.native_model.body_inertia[p.body_ids])[None] * hand_inertia_scale
    inertia[hand_ids] = mjr[None] @ torch.diag_embed(diagonal) @ mjr[None].transpose(-1, -2)
    wp.to_torch(model.body_inv_mass).copy_(torch.where(mass > 0, mass.reciprocal(), 0))
    inverse = wp.to_torch(model.body_inv_inertia)
    movable = mass > 0
    inverse[movable] = torch.linalg.inv(inertia[movable])
    # Exact per-joint PD ratios are part of the WM context.
    kp = uniform((n, 20), math.log(0.75), math.log(1.5)).exp()
    kd = uniform((n, 20), math.log(0.5), math.log(2)).exp()
    wp.to_torch(model.joint_target_ke)[dofs] = env.mdp.kp_default[None] * kp
    wp.to_torch(model.joint_target_kd)[dofs] = -env.mdp.kd_default[None] * kd
    wp.to_torch(model.joint_armature)[dofs] *= uniform((n, 20), 0.75, 1.3)
    env.encoder_bias.copy_(uniform((n, 20), -0.01, 0.01))
    # Match the source selection, including disabled L1 geoms in the DR summary.
    friction_names = [
        ref.geom(i).name for i in range(ref.ngeom) if re.fullmatch(r".*palm_.*|.*finger.*_col", ref.geom(i).name)
    ]
    friction_scale = uniform((n, len(friction_names), 3), 0.7, 1.3)
    for name, g in p.geom_ids.items():
        if name not in friction_names:
            continue
        ratios = friction_scale[:, friction_names.index(name)]
        shape_ids = shapes[:, g]
        for col, field in enumerate(("shape_material_mu", "shape_material_mu_torsional", "shape_material_mu_rolling")):
            wp.to_torch(getattr(model, field))[shape_ids] *= ratios[:, col]
    # One size multiplier per environment and asset, shared across selected primitives.
    robot_size_scale = uniform((n, 1), 0.97, 1.03)
    cube_size_scale = uniform((n, 1), 0.85, 1.15)
    primitive_geoms = []
    scale = wp.to_torch(model.shape_scale)
    for name, g in p.geom_ids.items():
        if re.fullmatch(r".*finger.*_link[123]_col", name):
            scale[shapes[:, g]] *= robot_size_scale
            primitive_geoms.append(g)
    scale[shapes[:, p.cube_geom]] *= cube_size_scale
    # MuJoCo raw contact parameters survive Newton model synchronization.
    solref = wp.to_torch(model.mujoco.solref)
    solimp = wp.to_torch(model.mujoco.geom_solimp)
    groups = [
        ([g for name, g in p.geom_ids.items() if "palm" in name or "finger1_" in name], (2.0, 5.0)),
        ([g for name, g in p.geom_ids.items() if re.search(r"finger[2-5]_", name)], (1.0, 2.0)),
    ]
    for geoms, width_range in groups:
        ids = shapes[:, geoms]
        solref[ids, 0] *= uniform((n, 1), 1, 2)
        solref[ids, 1] *= uniform((n, 1), 0.8, 1.2)
        solimp[ids, 2] *= uniform((n, 1), *width_range)
        solimp[ids, 0] *= uniform((n, 1), 0.5, 1)
    flags = ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.SHAPE_PROPERTIES
    s.notify_model_changed(flags)
    update_primitive_bounds(s, primitive_geoms + [p.cube_geom])
    env.dr_params.copy_(
        torch.cat(
            (
                friction_scale[:, :, 0].mean(-1, keepdim=True),
                cube_mass_scale,
                kp.mean(-1, keepdim=True),
                kd.mean(-1, keepdim=True),
                torch.zeros(n, 1, device=device),
                cube_size_scale,
            ),
            -1,
        )
    )
    env.sim_context.copy_(torch.cat((env.dr_params, kp, kd), -1))
    env.randomization_sample = {
        "kp": kp,
        "kd": kd,
        "cube_mass_scale": cube_mass_scale,
        "cube_size_scale": cube_size_scale,
        "robot_size_scale": robot_size_scale,
    }


def update_primitive_bounds(solver, geom_ids):
    """Update MuJoCo broadphase bounds after per-world box/capsule scaling."""
    m = solver.mjw_model
    size = wp.to_torch(m.geom_size)
    aabb = wp.to_torch(m.geom_aabb).expand(size.shape[0], -1, -1, -1).clone()
    radius = wp.to_torch(m.geom_rbound)
    for g in geom_ids:
        typ = int(solver.mj_model.geom_type[g])
        if typ == 3:  # mjGEOM_CAPSULE: local Z is the capsule axis.
            ext = size[:, g, 0, None].expand(-1, 3).clone()
            ext[:, 2] += size[:, g, 1]
            radius[:, g] = size[:, g, 0] + size[:, g, 1]
        elif typ == 6:  # mjGEOM_BOX
            ext = size[:, g]
            radius[:, g] = torch.linalg.vector_norm(ext, dim=-1)
        else:
            raise ValueError(f"Unsupported randomized geom type {typ}")
        aabb[:, g, 0] = 0
        aabb[:, g, 1] = ext
    m.geom_aabb = wp.from_torch(aabb, dtype=m.geom_aabb.dtype)
