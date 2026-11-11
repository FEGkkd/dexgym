"""Newton MJWarp collector with an explicit pre-reset transition contract."""

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg

from .task import DT, JOINTS, TrackingTask

PHYSICS_DT = 0.002
SUBSTEPS = round(DT / PHYSICS_DT)


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
    def __init__(self, cfg, num_envs, model, library, nominal):
        self.sim = SimulationContext(cfg)
        self.scene = InteractiveScene(TrackingSceneCfg(num_envs=num_envs, env_spacing=3.0))
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
        self.task = TrackingTask(self, model, library, nominal)
        self.memory = {
            "frame": torch.zeros(num_envs, dtype=torch.long, device=self.device),
            "action": torch.zeros(num_envs, JOINTS, device=self.device),
        }
        self.command = library.qpos[self.memory["frame"]].clone()
        self.transition = None
        self.common_step_counter = 0
        # The shared checkpoint format carries these fields. No disturbances are injected.
        self.disturbance = SimpleNamespace(adaptive_scale=0.0, success_scale=0.0)

    def state(self):
        return torch.cat((self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch), -1)

    def reset(self, env_ids=None, frame=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        lib = self.task.library
        if frame is None:
            frame = lib.sample(len(env_ids))
        if frame.shape != env_ids.shape or ((frame < 0) | (frame >= lib.frames)).any():
            raise ValueError("One valid reference frame is required per resetting environment")
        self.scene.reset(env_ids)
        self.robot.write_joint_position_to_sim_index(position=lib.qpos[frame], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=lib.qvel[frame], env_ids=env_ids)
        self.command[env_ids] = lib.qpos[frame]
        self.robot.actuators.target_command.set_position_index(value=self.command, full_data=True)
        self.memory = {k: v.clone() for k, v in self.memory.items()}
        self.memory["frame"][env_ids] = frame
        self.memory["action"][env_ids] = 0
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)
        return self.observation(), {}

    def observation(self):
        return {"policy": self.task.policy_observation(self.state(), self.memory, None)}

    def contact_depth(self):
        d = self.solver.mjw_data
        world = wp.to_torch(d.contact.worldid).long()
        distance = wp.to_torch(d.contact.dist)
        active = ((torch.arange(len(world), device=self.device) < wp.to_torch(d.nacon)[0])
                  & (world >= 0) & (world < self.num_envs))
        depth = torch.where(active, (-distance).clamp_min(0), 0)
        result = depth.new_zeros(self.num_envs)
        return result.scatter_reduce_(0, world.clamp(0, self.num_envs - 1), depth, reduce="amax")

    def step(self, action):
        before = self.memory
        target, u = self.task.control_target(action, before, self.task.encoder_bias)
        self.command = u
        self.robot.actuators.target_command.set_position_index(value=u, full_data=True)
        depth = torch.zeros(self.num_envs, device=self.device)
        for _ in range(SUBSTEPS):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
            depth = torch.maximum(depth, self.contact_depth())
        x = self.state()
        nan = ~torch.isfinite(x).all(-1) | ~torch.isfinite(depth)
        contact = torch.stack((depth > 0, depth > 0.0005), dim=-1).float()
        torque = wp.to_torch(self.solver.mjw_data.qfrc_actuator)[:, self.dof_ids].clone()
        unsafe = (depth > 0.003) | (self.task.position_violation(x) > 0.05)
        terminated = unsafe | nan
        self.memory = self.task.advance_memory(x, before, action, target, None, None, soft=False)
        frame = self.memory["frame"]
        truncated = (frame == self.task.library.end[frame]) & ~terminated
        reward, _, _ = self.task.reward(x.nan_to_num(), before, action, target, contact, torque)
        reward = torch.where(nan, 0.0, reward)
        self.transition = dict(x1=x, contact=contact, torque=torque, drop=unsafe, nan=nan, c=target, u=u, depth=depth)
        self.common_step_counter += 1
        return self.observation(), reward, terminated, truncated, {}
