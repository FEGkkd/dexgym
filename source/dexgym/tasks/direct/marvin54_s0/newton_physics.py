"""Isaac Lab scene stepping through the installed Newton MJWarp solver."""

from contextlib import contextmanager

from isaaclab_newton.physics import NewtonManager
import mujoco_warp as mjw
import torch
import warp as wp

from ..marvin54_tracking.native import create_tracking_scene, solver_dof_indices
from .physics import AUDIT_DEPTH_M, OVERFLOW, PHYSICS_DT, SUBSTEPS, Physics
from .warp_physics import contact_depth


class NewtonPhysics(Physics):
    def __init__(self, cfg, model, names, num_envs):
        super().__init__(model, names, num_envs, cfg.device)
        self.sim, self.scene, self.robot = create_tracking_scene(cfg, num_envs, model, names)
        self.solver = NewtonManager._solver
        self.dof_ids = solver_dof_indices(self.solver, names, self.device)
        self.mjw_model, self.mjw_data = self.solver.mjw_model, self.solver.mjw_data
        self.stream = wp.get_stream(self.device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        with self.simulation_stream():
            self.depth = wp.zeros(num_envs, dtype=wp.float32)
            self.depth_tensor = wp.to_torch(self.depth)
            self.force = wp.to_torch(self.mjw_data.qfrc_actuator)
            self.overflow = wp.to_torch(self.mjw_data.overflow)
            self.target = self.robot.data.joint_pos.torch.clone()

    @contextmanager
    def simulation_stream(self):
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream), torch.cuda.stream(self.torch_stream):
            yield caller
        caller.wait_stream(self.torch_stream)

    def state(self):
        return torch.cat((self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch), -1)

    def set_targets(self, target):
        self.robot.actuators.target_command.set_position_index(value=target, full_data=True)

    def reset(self, ids, q, qd):
        with self.simulation_stream():
            self.scene.reset(ids)
            self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=ids)
            self.target[ids] = q
            self.set_targets(self.target)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(0.0)

    def contact_depth(self):
        data = self.mjw_data
        # Newton has integrated this state. Refresh only position-dependent collision data.
        mjw.kinematics(self.mjw_model, data)
        mjw.collision(self.mjw_model, data)
        self.depth.zero_()
        wp.launch(contact_depth, dim=data.contact.dist.shape[0],
                  inputs=[data.nacon, data.contact.worldid, data.contact.dist, self.depth])
        return self.depth_tensor

    def interval(self, target, task):
        with self.simulation_stream() as caller:
            self.target.copy_(target)
            self.set_targets(self.target)
            state = self.state()
            torque = torch.zeros_like(target)
            peak = target.new_zeros(self.num_envs)
            first = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            flags = torch.zeros_like(first)
            deep_substeps = torch.zeros_like(first)
            for substep in range(1, SUBSTEPS + 1):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(PHYSICS_DT)
                measured = self.state()
                solver_overflow = self.overflow != 0
                depth = self.contact_depth()
                event = self.flags(task, measured, depth) | (OVERFLOW * (solver_overflow | (self.overflow != 0)).int())
                live = first == 0
                hit = live & (event != 0)
                first = torch.where(hit, substep, first)
                flags = torch.where(hit, event, flags)
                state = torch.where(live[:, None], measured, state)
                torque = torch.where(live[:, None], self.force[:, self.dof_ids], torque)
                peak = torch.where(live, torch.maximum(peak, depth), peak)
                deep_substeps += (live & (depth > AUDIT_DEPTH_M)).int()
            for value in (state, torque, peak, first, flags, deep_substeps):
                value.record_stream(caller)
        return state, torque, peak, first, flags, deep_substeps

    @torch.no_grad()
    def contract(self, env):
        task, lib = env.task, env.task.library
        frames = torch.linspace(0, lib.frames - 1, self.num_envs, device=self.device).long()
        env.reset(frame=frames)
        q, qd = self.state().split(task.action_dim, dim=-1)
        pose = task.geometry.hand_pose(q)
        wrists = [self.robot.body_names.index(f"{side}_wrist") for side in ("l", "r")]
        measured_wrist = self.robot.data.body_pose_w.torch[:, wrists, :3] - self.scene.env_origins[:, None]
        errors = {
            "reset_position_rad": float((q - lib.qpos[frames]).abs().max()),
            "reset_velocity_rad_s": float((qd - lib.qvel[frames]).abs().max()),
            "newton_wrist_m": float((measured_wrist - pose["wrist"]).abs().max()),
            "reference_tip_m": float((lib.ref_tip_pos[frames] - pose["tip"]).abs().max()),
        }
        if max(errors.values()) > 2e-4 or not all(torch.isfinite(torch.tensor(list(errors.values())))):
            raise RuntimeError(f"Isaac Lab Newton reference contract failed: {errors}")
        return {
            "simulation": type(self.sim).__module__ + "." + type(self.sim).__name__,
            "manager": NewtonManager.__module__ + "." + NewtonManager.__name__,
            "solver": type(self.solver).__module__ + "." + type(self.solver).__name__,
            "device": str(self.device), "envs": self.num_envs,
            "physics_dt_s": PHYSICS_DT, "control_substeps": SUBSTEPS, "errors": errors,
        }
