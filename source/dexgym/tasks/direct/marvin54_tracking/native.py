"""Newton MJWarp collector with an explicit pre-reset transition contract."""

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg

from .model import drive_parameters
from .task import DT, HARD_CONTACT_DEPTH_M, JOINT_OVERFLOW_RAD, JOINTS, TrackingTask

PHYSICS_DT = 0.002
SUBSTEPS = round(DT / PHYSICS_DT)
COLLISION, JOINT_LIMIT, NONFINITE, TRACKING = 1, 2, 4, 8
FAILURE_CAUSES = {COLLISION: "collision", JOINT_LIMIT: "joint_limit", NONFINITE: "nonfinite", TRACKING: "tracking"}


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    robot = make_robot_cfg()


def simulation_cfg():
    return SimulationCfg(
        dt=PHYSICS_DT,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=1500, nconmax=500, integrator="implicitfast", iterations=20,
                ls_iterations=50, tolerance=1e-8, use_mujoco_contacts=True,
            ),
            use_cuda_graph=False,
        ),
    )


class NewtonTrackingEnv:
    def __init__(self, cfg, num_envs, model, library, nominal, policy_arch="flat", termination_model="legacy-state",
                 reward_profile="precision", *, first_event=False, tracking_failure_m=0.25,
                 max_episode_steps=None):
        self.sim = SimulationContext(cfg)
        scene_cfg = TrackingSceneCfg(num_envs=num_envs, env_spacing=3.0)
        drives = drive_parameters(model, library.names)
        actuator = scene_cfg.robot.actuators["all"]
        actuator.stiffness = {name: drive["kp"] for name, drive in drives.items()}
        actuator.damping = {name: drive["kd"] for name, drive in drives.items()}
        actuator.joint_effort_limit = {name: drive["effort"] for name, drive in drives.items()}
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.num_envs = num_envs
        self.device = self.robot.device
        if list(self.robot.joint_names) != library.names:
            raise ValueError("Newton/MJCF joint ordering differs")
        self.solver = NewtonManager._solver
        native = self.solver.mj_model
        addresses = {}
        for name in library.names:
            suffix = "_" + name
            matches = [j for j in range(native.njnt) if native.joint(j).name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely map Newton force channel {name}")
            addresses[name] = native.jnt_dofadr[matches[0]]
        self.dof_ids = torch.tensor(list(addresses.values()), device=self.device)
        self.task = TrackingTask(
            self, model, library, nominal, policy_arch, termination_model, reward_profile,
            first_event=first_event, tracking_failure_m=tracking_failure_m,
        )
        for field, desired in (
            ("joint_stiffness", self.task.kp), ("joint_damping", self.task.kd),
            ("joint_effort_limits", self.task.effort),
        ):
            actual = getattr(self.robot.data, field).torch
            if not torch.allclose(actual, desired.expand_as(actual), atol=1e-5, rtol=1e-5):
                raise ValueError(f"Newton {field} does not match the composed MuJoCo model")
        self.max_episode_steps = max_episode_steps
        self.elapsed = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._state = self._physics_state()
        self.memory = {
            "frame": torch.zeros(num_envs, dtype=torch.long, device=self.device),
            "action": torch.zeros(num_envs, JOINTS, device=self.device),
        }
        self.command = library.qpos[self.memory["frame"]].clone()
        self.history = torch.zeros(num_envs, self.task.history_steps, self.task.history_frame_dim, device=self.device)
        self.transition = None
        self.common_step_counter = 0
        # The shared checkpoint format carries these fields. No disturbances are injected.
        self.disturbance = SimpleNamespace(adaptive_scale=0.0, success_scale=0.0)

    def state(self):
        return self._state

    def _physics_state(self):
        return torch.cat((self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch), -1)

    def reset(self, env_ids=None, frame=None, *, position_noise=None, velocity_noise=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        lib = self.task.library
        if frame is None:
            frame = lib.sample(len(env_ids))
        if frame.shape != env_ids.shape or ((frame < 0) | (frame >= lib.frames)).any():
            raise ValueError("One valid reference frame is required per resetting environment")
        self.scene.reset(env_ids)
        q, qd = lib.qpos[frame], lib.qvel[frame]
        if position_noise is not None:
            q = (q + self.task.tensor(position_noise)).clamp(self.task.joint_lo, self.task.joint_hi)
        if velocity_noise is not None:
            qd = qd + self.task.tensor(velocity_noise)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.command[env_ids] = lib.qpos[frame]
        self.robot.actuators.target_command.set_position_index(value=self.command, full_data=True)
        self.memory = {k: v.clone() for k, v in self.memory.items()}
        self.memory["frame"][env_ids] = frame
        self.memory["action"][env_ids] = 0
        self.elapsed[env_ids] = 0
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)
        self._state = self._state.index_copy(0, env_ids, self._physics_state()[env_ids])
        frame_obs = self.task.policy_observation(self.state(), self.memory, None)
        self.history[env_ids] = frame_obs[env_ids, None]
        return self.observation(), {}

    def observation(self):
        return {"policy": self.task.observation_from_history(self.history, self.memory)}

    def contact_depth(self):
        d = self.solver.mjw_data
        world = wp.to_torch(d.contact.worldid).long()
        distance = wp.to_torch(d.contact.dist)
        active = ((torch.arange(len(world), device=self.device) < wp.to_torch(d.nacon)[0])
                  & (world >= 0) & (world < self.num_envs))
        depth = torch.where(active, (-distance).clamp_min(0), 0)
        result = depth.new_zeros(self.num_envs)
        return result.scatter_reduce_(0, world.clamp(0, self.num_envs - 1), depth, reduce="amax")

    def _actuator_effort(self):
        return wp.to_torch(self.solver.mjw_data.qfrc_actuator)[:, self.dof_ids]

    def _safety_flags(self, state, depth):
        invalid = ~torch.isfinite(state).all(-1) | ~torch.isfinite(depth)
        joint = self.task.position_violation(state) > JOINT_OVERFLOW_RAD
        return COLLISION * (depth > HARD_CONTACT_DEPTH_M).int() + JOINT_LIMIT * joint.int() + NONFINITE * invalid.int()

    def _physics_interval(self):
        state = self.state()
        torque = torch.zeros_like(state[:, :JOINTS])
        depth = state.new_zeros(self.num_envs)
        first_substep = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        flags = torch.zeros_like(first_substep)
        for substep in range(1, SUBSTEPS + 1):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
            sampled_depth = self.contact_depth()
            if not self.task.first_event:
                depth = torch.maximum(depth, sampled_depth)
                continue
            # The solver advances the batch together. Keep the first failing sample
            # for this episode while the other environments finish the interval.
            live = first_substep == 0
            measured = self._physics_state()
            detected = self._safety_flags(measured, sampled_depth)
            hit = live & (detected != 0)
            first_substep = torch.where(hit, substep, first_substep)
            flags = torch.where(hit, detected, flags)
            depth = torch.where(live, torch.maximum(depth, sampled_depth), depth)
            state = torch.where(live[:, None], measured, state)
            torque = torch.where(live[:, None], self._actuator_effort(), torque)
        if not self.task.first_event:
            state = self._physics_state()
            torque = self._actuator_effort()
            flags = self._safety_flags(state, depth)
        return state, torque, depth, first_substep, flags

    def step(self, action):
        before = self.memory
        target, u = self.task.control_target(action, before, self.task.encoder_bias)
        self.command = u
        self.robot.actuators.target_command.set_position_index(value=u, full_data=True)
        x, torque, depth, safety_substep, flags = self._physics_interval()
        self._state = x
        safety_failure = flags != 0
        nan = (flags & NONFINITE) != 0
        unsafe = (flags & (COLLISION | JOINT_LIMIT)) != 0
        full_step = (safety_substep == 0) | (safety_substep == SUBSTEPS)
        contact = self.task.contact_labels(depth)
        safe_x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        if self.task.first_event:
            lost = self.task.wrist_tracking_error(safe_x, before) > self.task.tracking_failure_m
            flags = flags | (TRACKING * (lost & ~safety_failure).int())
        failed = flags != 0
        self.memory = self.task.advance_memory(x, before, action, target, None, None, soft=False)
        frame_obs = self.task.policy_observation(x, self.memory, None)
        self.history = torch.cat((self.history[:, 1:], frame_obs[:, None]), dim=1)
        frame = self.memory["frame"]
        complete = (frame == self.task.library.end[frame]) & ~failed
        self.elapsed += 1
        time_limit = torch.zeros_like(complete)
        if self.max_episode_steps is not None:
            time_limit = (self.elapsed >= self.max_episode_steps) & ~(failed | complete)
        terminated = (failed | complete) if self.task.first_event else failed
        truncated = time_limit if self.task.first_event else (complete | time_limit)
        reward, _, _ = self.task.reward(safe_x, before, action, target, contact, torque)
        reward = torch.where(nan, 0.0, reward)
        events = dict(failure_bits=flags, safety_failure=safety_failure, safety_substep=safety_substep,
                      reference_complete=complete, time_limit=time_limit, full_step=full_step)
        self.transition = dict(x1=x, contact=contact, torque=torque, drop=unsafe, nan=nan, c=target, u=u, depth=depth,
                               **events)
        self.common_step_counter += 1
        return self.observation(), reward, terminated, truncated, events
