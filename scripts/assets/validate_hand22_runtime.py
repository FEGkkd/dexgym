"""Assert SI gains and summed effort bounds in the actual Newton MJWarp solver."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_MJCF, make_robot_cfg
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = make_robot_cfg()


def validate():
    ref = mujoco.MjModel.from_xml_path(str(PACKAGED_MJCF))
    cfg = SimulationCfg(
        dt=0.002,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=1500, nconmax=500, integrator="implicitfast"), use_cuda_graph=False
        ),
    )
    result = {}
    with launch_simulation(cfg):
        sim = SimulationContext(cfg)
        scene = InteractiveScene(SceneCfg(num_envs=4, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        s = NewtonManager._solver
        m = s.mj_model
        expected = {}
        for a in range(ref.nu):
            j = int(ref.actuator_trnid[a, 0])
            expected[ref.joint(j).name] = (
                ref.actuator_gainprm[a, 0],
                -ref.actuator_biasprm[a, 2],
                ref.actuator_forcerange[a, 1],
            )
        table = torch.tensor([expected[n] for n in robot.joint_names], device=robot.device)
        for k, attr in enumerate(("joint_stiffness", "joint_damping", "joint_effort_limits")):
            err = float((getattr(robot.data, attr).torch - table[:, k]).abs().max())
            result[attr + "_max_error"] = err
            assert err < 2e-5, (attr, err)
        joint_ids = [next(j for j in range(m.njnt) if m.joint(j).name.endswith("_" + n)) for n in robot.joint_names]
        dofs = m.jnt_dofadr[joint_ids]
        gain = wp.to_torch(s.mjw_model.actuator_gainprm)
        bias = wp.to_torch(s.mjw_model.actuator_biasprm)
        maxgain = 0.0
        maxforce = 0.0
        for i, j in enumerate(joint_ids):
            aids = np.flatnonzero(m.actuator_trnid[:, 0] == j)
            # Newton represents a position PD drive as separate position/velocity actuators.
            assert len(aids) == 2, (robot.joint_names[i], aids)
            expected_gain = torch.where(bias[:, aids, 1] != 0, -bias[:, aids, 1], -bias[:, aids, 2])
            maxgain = max(maxgain, float((gain[:, aids, 0] - expected_gain).abs().max()))
            kg = -bias[:, aids, 1].sum(-1)
            kd = -bias[:, aids, 2].sum(-1)
            maxgain = max(maxgain, float((kg - table[i, 0]).abs().max()), float((kd - table[i, 1]).abs().max()))
            limit = m.jnt_actfrcrange[j]
            maxforce = max(maxforce, float(np.max(np.abs(limit - np.array([-float(table[i, 2]), float(table[i, 2])])))))
            assert m.jnt_actfrclimited[j], robot.joint_names[i]
        result["solver_gain_max_error"] = maxgain
        result["solver_total_force_limit_max_error"] = maxforce
        assert maxgain < 2e-5 and maxforce < 2e-5, result
        physical_errors = {}
        for rj, nj in enumerate(joint_ids):
            source = ref.joint(robot.joint_names[rj]).id
            for attr in ("dof_armature", "dof_damping", "dof_frictionloss"):
                err = abs(float(getattr(m, attr)[m.jnt_dofadr[nj]]) - float(getattr(ref, attr)[ref.jnt_dofadr[source]]))
                physical_errors[attr] = max(physical_errors.get(attr, 0.0), err)
        for rb in range(1, ref.nbody):
            name = ref.body(rb).name
            nb = next(b for b in range(m.nbody) if m.body(b).name.endswith("_" + name))
            for attr in ("body_mass", "body_ipos"):
                err = float(np.max(np.abs(getattr(m, attr)[nb] - getattr(ref, attr)[rb])))
                physical_errors[attr] = max(physical_errors.get(attr, 0.0), err)

            def inertia(model, body):
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, model.body_iquat[body])
                rot = rot.reshape(3, 3)
                return rot @ np.diag(model.body_inertia[body]) @ rot.T

            err = float(np.max(np.abs(inertia(m, nb) - inertia(ref, rb))))
            physical_errors["body_inertia_tensor"] = max(physical_errors.get("body_inertia_tensor", 0.0), err)
            if not name.startswith(("r_", "l_")):
                continue
            rg = [
                g
                for g in range(ref.ngeom)
                if ref.geom_bodyid[g] == rb and (ref.geom_contype[g] or ref.geom_conaffinity[g])
            ]
            ng = [g for g in range(m.ngeom) if m.geom_bodyid[g] == nb and (m.geom_contype[g] or m.geom_conaffinity[g])]
            assert len(rg) == len(ng), (name, rg, ng)
            for a, b in zip(rg, ng, strict=True):
                for attr in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_priority"):
                    err = float(np.max(np.abs(getattr(ref, attr)[a] - getattr(m, attr)[b])))
                    physical_errors[attr] = max(physical_errors.get(attr, 0.0), err)
        result["physical_parameter_max_errors"] = physical_errors
        assert max(physical_errors.values()) < 2e-5, physical_errors
        home = torch.tensor(ref.key("home").qpos.copy(), dtype=torch.float32, device=robot.device).repeat(4, 1)
        limits = robot.data.joint_pos_limits.torch
        peak = 0.0
        maxreset = 0.0
        for repeat in range(3):
            robot.write_joint_position_to_sim_index(position=home)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(home))
            scene.reset()
            sim.forward()
            scene.update(0.0)
            maxreset = max(maxreset, float((robot.data.joint_pos.torch - home).abs().max()))
            for step in range(300):
                target = home.clone()
                target[::2] = limits[::2, :, 1]
                target[1::2] = limits[1::2, :, 0]
                robot.actuators.target_command.set_position_index(value=target, full_data=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(0.002)
                assert torch.isfinite(robot.data.joint_pos.torch).all()
                force = wp.to_torch(s.mjw_data.qfrc_actuator)[:, dofs]
                ratio = force.abs() / table[:, 2]
                peak = max(peak, float(ratio.max()))
                assert float(ratio.max()) < 1.0001, float(ratio.max())
        result.update(
            pass_runtime=True,
            num_envs=4,
            resets=3,
            steps_per_reset=300,
            max_reset_error_rad=maxreset,
            max_summed_effort_fraction=peak,
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = validate()
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(r)
