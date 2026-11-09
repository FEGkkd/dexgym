"""Wuji Hand 2 cube reorientation with pre-reset transitions for SuperMimic.

Order: physics -> termination -> reward -> command -> disturbance -> observation.
"""

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from dexgym.assets.wuji_hand2_cube.rig import REORIENT_HAND2_MOUNT_POS

from .env_cfg import Hand2CubeSuperMimicCfg
from .native import NativePhysics
from .rotations import quat_to_mat
from .state import MEMORY_DIMS, ema_target


class Hand2CubeSuperMimicEnv(DirectRLEnv):
    cfg: Hand2CubeSuperMimicCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self.physics = NativePhysics(self)
        self.mdp = self.physics.mdp
        self.memory = {k: torch.zeros(self.num_envs, d, device=self.device) for k, d in MEMORY_DIMS.items()}
        self.encoder_bias = torch.zeros(self.num_envs, 20, device=self.device)
        self.dr_params = torch.ones(self.num_envs, 6, device=self.device)
        self.sim_context = torch.ones(self.num_envs, 46, device=self.device)
        self.pert_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.pert_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        self.control_target = self.mdp.default.expand(self.num_envs, -1).clone()
        self.executed_target = self.control_target.clone()
        self.action = torch.zeros_like(self.control_target)
        self.history = torch.zeros(self.num_envs, 3, 69, device=self.device)
        self._history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reward = torch.zeros(self.num_envs, device=self.device)
        self.transition = None
        self.max_common_steps = 20064
        from .disturbance import Disturbance

        self.disturbance = Disturbance(self)
        self.dr_params[:, 4] = 0
        self.sim_context[:, 4] = 0
        if cfg.startup_randomization:
            from .randomization import apply_startup_randomization

            apply_startup_randomization(self)

    def _pre_physics_step(self, actions):
        self.before_state = self.physics.state()
        self.before_memory = {k: v.clone() for k, v in self.memory.items()}
        self.action = actions.clamp(-1, 1)
        self.control_target, self.executed_target = ema_target(
            self.action,
            self.memory["c1"],
            self.mdp.default,
            self.mdp.lower,
            self.mdp.upper,
            self.encoder_bias,
            self.mdp.alpha,
            self.memory["age"].squeeze(-1),
            self.step_dt,
        )

    def _apply_action(self):
        self.robot.actuators.target_command.set_position_index(value=self.executed_target, full_data=True)

    def _get_dones(self):
        x = self.physics.state()
        contact = self.physics.contact()
        torque = self.physics.torque()
        drop = self.before_memory["cage"].squeeze(-1) >= self.mdp.max_outside
        nan = ~torch.isfinite(x).all(-1)
        self.transition = {
            "x": self.before_state,
            "x1": x,
            "contact": contact,
            "torque": torque,
            "drop": drop,
            "nan": nan,
            "action": self.action.clone(),
            "c": self.control_target.clone(),
            "u": self.executed_target.clone(),
            "memory": self.before_memory,
        }
        return drop | nan, self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        cap = self.transition
        m = self.before_memory
        reward, cage, terms = self.mdp.rewards(
            cap["x1"], m, self.action, self.control_target, cap["contact"], cap["torque"]
        )
        cap.update(reward=reward.clone(), terms={k: v.clone() for k, v in terms.items()})
        goal = self.sample_goal(self.num_envs)
        self.memory = self.mdp.command_step(cap["x1"], m, goal, soft=False)
        self.memory.update(
            c1=self.control_target.clone(), c2=m["c1"], a1=self.action.clone(), a2=m["a1"], cage=cage, age=m["age"] + 1
        )
        self.extras["log"] = {
            "goal_count": self.memory["count"].mean(),
            "cage_counter": cage.mean(),
            "drop_rate": cap["drop"].float().mean(),
        }
        self._reward = reward
        if self.cfg.velocity_disturbance:
            self.disturbance.step()
        return reward

    def sample_goal(self, n):
        return quat_to_mat(torch.nn.functional.normalize(torch.randn(n, 4, device=self.device), dim=-1))

    def _observe(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        x = self.physics.state()
        observed = x.clone()
        observed[:, :20] += self.encoder_bias
        frame = self.mdp.observation(observed, self.memory)
        if self.cfg.observation_noise:
            frame = self.mdp.noisy_observation(frame)
        rolled = torch.cat((self.history[ids, 1:], frame[ids, None]), 1)
        self.history[ids] = torch.where(
            self._history_reset[ids, None, None], frame[ids, None].expand(-1, 3, -1), rolled
        )
        self._history_reset[ids] = False
        return {"policy": self.mdp.flatten_history(self.history)}

    def _get_observations(self):
        return self._observe()

    def _reset_envs_from_buffer(self):
        if self.cfg.auto_reset:
            return super()._reset_envs_from_buffer()
        return torch.empty(0, device=self.device, dtype=torch.int32)

    def reset(self, seed=None, options=None, env_ids=None):
        if env_ids is None:
            return super().reset(seed=seed, options=options)
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.obs_buf = self._observe(env_ids)
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        self.disturbance.reset(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        if self.cfg.randomize_reset:
            offset = -0.3 + 0.2 * torch.rand_like(q)
            offset[:, 1::4] = 0
            q = (q + offset).clamp(self.mdp.lower, self.mdp.upper)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        root = self.robot.data.default_root_state.torch[env_ids, :7].clone()
        if self.cfg.randomize_reset:
            pitch = -0.4 + 0.5 * torch.rand(n, device=self.device)
            zero = torch.zeros_like(pitch)
            delta = quat_from_euler_xyz(zero, pitch, zero)
            pivot = torch.tensor(REORIENT_HAND2_MOUNT_POS, device=self.device).expand(n, -1)
            center = root[:, :3] + quat_apply(root[:, 3:], pivot)
            root[:, :3] = center + quat_apply(delta, root[:, :3] - center)
            root[:, 3:] = quat_mul(delta, root[:, 3:])
        root[:, :3] += self.scene.env_origins[env_ids]
        self.physics.write_hand_root(root, env_ids)
        cube = self.cube.data.default_root_state.torch[env_ids, :7].clone()
        cube[:, :3] += self.scene.env_origins[env_ids]
        if self.cfg.randomize_reset:
            cube[:, :3] += (torch.rand(n, 3, device=self.device) * 2 - 1) * 0.01
            cube[:, 3:] = torch.nn.functional.normalize(torch.randn(n, 4, device=self.device), dim=-1)
        self.cube.write_root_pose_to_sim_index(root_pose=cube, env_ids=env_ids)
        self.cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        for v in self.memory.values():
            v[env_ids] = 0
        self.memory["goal"][env_ids] = self.sample_goal(n).flatten(-2)
        self.memory["c1"][env_ids] = self.mdp.default
        self.memory["c2"][env_ids] = self.mdp.default
        self.control_target[env_ids] = self.mdp.default
        self.executed_target[env_ids] = self.mdp.default - self.encoder_bias[env_ids]
        self.pert_direction[env_ids] = 0
        self.pert_velocity[env_ids] = 0
        self._history_reset[env_ids] = True
