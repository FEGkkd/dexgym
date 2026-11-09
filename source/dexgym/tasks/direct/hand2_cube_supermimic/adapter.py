"""Task boundary for the reusable SuperMimic engine.

All task geometry, memories, goal transitions, and exogenous pushes live here.
The optimization engine does not import Isaac Lab or this task package.
"""

import torch
from torch.nn import functional as F

from . import state
from .mdp import angle_error
from .observations import TaskObservations
from .rotations import mv


class CubeAdapter:
    state_dim = 58
    action_dim = 20
    error_floor = [0.003] * 20 + [0.05] * 20 + [0.001] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3
    codec = state
    encode_memory = staticmethod(state.flatten_memory)
    decode_memory = staticmethod(state.unflatten_memory)

    def __init__(self, env):
        self.env = env
        self.reader = env.physics
        self.mdp = env.mdp
        self.device = env.device
        self.observations = TaskObservations(self.reader)
        self.contact_dim = self.reader.contact_dim
        self.context_dim = self.observations.context_dim
        self.dynamics_dim = 58 + 46 + 9
        self.policy_dim = 207
        self.task_id = "Wuji-Hand2-Cube-Reorientation"

    @property
    def simulator_context(self):
        return self.env.sim_context

    @property
    def encoder_bias(self):
        return self.env.encoder_bias

    def context(self, x, m, aux, sim):
        return self.observations.context(x, m, aux, sim)

    def dynamics(self, x, sim, aux):
        return self.observations.dynamics(x, sim, self.observations.disturbance(aux))

    def critic(self, context):
        return self.observations.critic(context)

    def reward(self, *args, **kwargs):
        return self.mdp.rewards(*args, **kwargs)

    def torque(self, x, u, sim):
        return self.mdp.pd_torque(x, u, sim)

    def control_target(self, action, m, bias):
        return state.ema_target(
            action,
            m["c1"],
            self.mdp.default,
            self.mdp.lower,
            self.mdp.upper,
            bias,
            self.mdp.alpha,
            m["age"].squeeze(-1),
            self.mdp.dt,
        )

    @staticmethod
    def current_goal(m):
        return m["goal"].reshape(-1, 3, 3)

    def sample_goal(self, n):
        return self.env.sample_goal(n)

    def advance_memory(self, x1, m, a, target, cage, next_goal, soft=True):
        out = self.mdp.command_step(x1, m, next_goal, soft=soft)
        out.update(c1=target, c2=m["c1"], a1=a, a2=m["a1"], cage=cage, age=m["age"] + 1)
        return out

    def termination(self, x1, m):
        return torch.sigmoid((m["cage"].squeeze(-1) - self.mdp.max_outside + 0.5) / 0.25)

    @staticmethod
    def history_from_obs(obs):
        return torch.cat(
            [v.reshape(*v.shape[:-1], 3, d) for v, d in zip(obs.split((60, 60, 9, 18, 60), -1), (20, 20, 3, 6, 20))], -1
        )

    def flatten_history(self, history):
        return self.mdp.flatten_history(history)

    def policy_observation(self, x, m, aux):
        return self.mdp.noisy_observation(self.mdp.observation(x, m))

    @staticmethod
    def advance_disturbance(aux, event, direction, velocity):
        return torch.cat(
            (
                aux[:, :12],
                torch.where(event, direction, aux[:, 12:15]),
                torch.where(event, torch.cat((velocity, torch.zeros_like(velocity)), -1), aux[:, 15:21]),
                aux[:, 21:],
            ),
            -1,
        )

    def replay_push(self, x1, push, aux):
        # Preserve the pinned reference's replay disturbance reconstruction.
        event = push[..., -6:-3].abs().sum(-1, keepdim=True) > 0
        velocity = push[..., -6:-3]
        direction = torch.where(event, velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(1e-8), velocity)
        return x1 + push, self.advance_disturbance(aux, event, direction, velocity)

    def imagine_push(self, x1, m, aux):
        e = self.env
        scale = min(1, max(0, (e.common_step_counter / e.max_common_steps - 0.05) / 0.75))
        adaptive = e.disturbance.adaptive_scale
        speed = 0.05 + torch.rand(x1.shape[0], 1, device=x1.device) * 0.1 * scale * adaptive
        dv = F.normalize(torch.randn(x1.shape[0], 3, device=x1.device), dim=-1) * speed
        event = torch.rand(x1.shape[0], 1, device=x1.device) < self.mdp.dt / 1.2
        active = event & (m["age"] * self.mdp.dt >= 3)
        push = torch.zeros_like(x1)
        push[..., -6:-3] = dv * active
        rotation = aux[:, 3:12].reshape(-1, 3, 3)
        return x1 + push, self.advance_disturbance(
            aux, event, mv(rotation, F.normalize(dv, dim=-1) * active), mv(rotation, dv * active)
        )

    def read(self):
        return self.reader.state(), {k: v.clone() for k, v in self.env.memory.items()}, self.observations.auxiliary()

    def rotation_error(self, x, m):
        return angle_error(state.unpack(x)[3], self.current_goal(m))
