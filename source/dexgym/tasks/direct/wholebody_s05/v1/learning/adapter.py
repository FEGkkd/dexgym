"""S0-5 SuperMimic adapter; only the independent physical task issues commands."""
from types import SimpleNamespace

import torch
from torch import nn

from ......algorithms.supermimic.dex_reference.common import EmpiricalNorm

from ..task import DT, HISTORY, JOINTS
from ..task.geometry import difference, error6
from ..task.observation import CommandMemory, Feedback, geometry_mask, velocity_mask
from .replay import CausalReplayBuffer


class PhysicalCodec:
    prediction_dim = JOINTS * 2

    @staticmethod
    def difference(target, source):
        # Canonical hinge positions are bounded physical joint coordinates.
        return target - source

    @staticmethod
    def integrate(state, delta):
        return state + delta


class Actor(nn.Module):
    """Stable ABI with four named mechanical blocks."""

    def __init__(self, dim):
        super().__init__()
        self.norm = EmpiricalNorm(dim)
        self.trunk = nn.Sequential(nn.Linear(dim, 512), nn.SiLU(),
                                   nn.Linear(512, 512), nn.SiLU(),
                                   nn.Linear(512, 256), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(256, length) for length in (7, 7, 20, 20))
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, observation):
        features = self.trunk(self.norm(observation))
        return torch.tanh(torch.cat([head(features) for head in self.heads], -1))


class ResidualEnvAdapter:
    """Return step feedback before the engine calls reset on terminal lanes."""

    reset_returns_full_batch = True

    def __init__(self, task, catalog, *, seed=49, holdout_envs=1, max_episode=160):
        self.task, self.catalog = task, catalog
        self.device = torch.device(task.device)
        self.num_envs = task.num_envs
        if not 0 < self.num_envs - holdout_envs or bool(holdout_envs) != bool(catalog.holdout_ids):
            raise ValueError("training and holdout lane counts do not match predeclared source roles")
        self.train_end = self.num_envs - holdout_envs
        self.maximum_episode = max_episode
        shape = (self.num_envs,)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.cursor = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.episode = torch.zeros_like(self.cursor)
        self.episode_remaining = torch.zeros_like(self.cursor)
        self.active = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.common_step_counter = 0
        self.max_common_steps = 0
        self.disturbance = SimpleNamespace(adaptive_scale=0., success_scale=0.)
        self.transition = None
        self.last = None
        self.requested_counts = torch.zeros((2, JOINTS), dtype=torch.long, device=self.device)
        self.executed_counts = torch.zeros_like(self.requested_counts)
        self.residual_square = torch.zeros(JOINTS, device=self.device)
        self.residual_outer = torch.zeros((JOINTS, JOINTS), device=self.device)
        self.exploration_intervals = torch.zeros((), dtype=torch.long, device=self.device)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        for role, pool in ((True, self.catalog.fit_pool), (False, self.catalog.holdout_pool)):
            group = ids[(ids < self.train_end) == role]
            if group.numel():
                draw = torch.randint(pool.numel(), (len(group),), generator=self.generator, device=self.device)
                self.cursor[group] = pool[draw]
        self.episode[ids] += 1
        self.episode_remaining[ids] = self.catalog.remaining[self.cursor[ids]].clamp_max(self.maximum_episode)
        home = self.catalog.target(self.cursor[ids]).nominal
        self.task.reset(ids, home, torch.zeros_like(home))
        self.active[ids] = True
        return {"policy": self.task.observe(self.catalog.target(self.cursor))}, {}

    def step(self, action):
        ref = self.catalog.target(self.cursor)
        self.last = self.task.step(action, truncated=self.episode_remaining == 1)
        transition = self.last
        self.transition = transition.engine_capture()
        self.transition["physical_executed"] = transition.substeps > 0
        # Failed, physically executed terminal steps still supply failure supervision.
        finite = torch.isfinite(transition.x1_pre_reset).all(-1)
        self.transition["valid"] = ref.permitted & finite & (
            transition.valid | (transition.safety_failure & (transition.substeps > 0))
        )
        self.transition["rollout_stats"] = {**transition.reward_terms, **{
            "penalty_" + key: value for key, value in transition.costs.items()}}
        self.transition["rollout_event_flags"] = {
            "projected_hold": transition.fallback, "safety_failure": transition.safety_failure,
        }
        self._count_exploration(ref, transition)
        ended = transition.terminated | transition.truncated
        self.active = ~ended
        self.cursor = self.cursor + self.active.long()
        self.episode_remaining = self.episode_remaining - self.active.long()
        self.common_step_counter += 1
        observation = (transition.obs if ended.any() else self.task.observe(self.catalog.target(self.cursor)))
        return {"policy": observation}, transition.reward, transition.terminated, transition.truncated, {}

    def _count_exploration(self, ref, transition):
        eligible = ((torch.arange(self.num_envs, device=self.device) < self.train_end)
                    & transition.full_step & transition.valid)
        issued = torch.where(eligible[:, None], (transition.c - ref.nominal) / self.task.guard.scale, 0.)
        requested = transition.action
        drive_response = transition.x1_pre_reset[:, :54] - transition.x[:, :54]
        for row, sign in enumerate((1., -1.)):
            self.requested_counts[row] += (eligible[:, None] & (sign * requested >= .05)).sum(0)
            self.executed_counts[row] += ((sign * issued >= .05) & (sign * drive_response >= 1e-4)).sum(0)
        self.residual_square += issued.square().sum(0)
        self.residual_outer += issued.T @ issued
        self.exploration_intervals += eligible.sum()

    def report_exploration(self):
        count = int(self.exploration_intervals)
        eig = torch.linalg.eigvalsh(self.residual_outer)
        threshold = eig[-1].clamp_min(0) * .01
        return {
            "train_complete_exploration_intervals": count,
            "requested_per_axis_plus_minus": self.requested_counts.tolist(),
            "executed_per_axis_plus_minus_with_measured_response": self.executed_counts.tolist(),
            "issued_relative_rms_per_axis": (self.residual_square / max(count, 1)).sqrt().tolist(),
            "issued_covariance_effective_rank": int((eig > threshold).sum()),
        }

    def state_dict(self):
        return {"generator": self.generator.get_state(), "logical_episode": self.episode.cpu()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())
        self.episode.copy_(state["logical_episode"].to(self.device))


class ImaginationAdapter:
    """Integrate the shared engine and deterministic Servo auxiliary dynamics."""

    task_id = "wholebody-s05-v2.0.1"
    state_dim = 2 * JOINTS
    action_dim = command_dim = JOINTS
    contact_dim = 13
    transition_termination = True
    termination_context_dim = JOINTS

    def __init__(self, env, *, replay_device=None, actor_gate=None, smoke=False):
        self.env, self.device, self.task = env, env.device, env.task
        self.reader = self.task.physics
        self.profile = self.reader.profile
        self.catalog = env.catalog
        self.codec = PhysicalCodec()
        self.error_floor = [.001] * JOINTS + [.01] * JOINTS
        self.simulator_context = self.task.sim_context
        self.dynamics_dim = self.state_dim + JOINTS + self.simulator_context.shape[1]
        self.frame_dim = self.task.builder.frame_dim
        self.policy_dim = self.frame_dim * HISTORY
        self.context_dim = self.frame_dim + self.simulator_context.shape[1]
        self.encoder_bias = torch.zeros((env.num_envs, JOINTS), device=self.device)
        self.replay_device, self.gate = replay_device, actor_gate
        self.smoke = smoke
        self.exploration = torch.Generator(device=self.device).manual_seed(49)
        self.held_action = torch.zeros_like(self.encoder_bias)
        self.hold_ticks = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.env.reset_sampler = self

    def make_actor(self):
        return Actor(self.policy_dim)

    def make_replay_buffer(self, capacity):
        return CausalReplayBuffer(capacity, self.env.num_envs, self.device, HISTORY, self.env.train_end,
                                  train_root_ids=self.catalog.fit_ids, holdout_root_ids=self.catalog.holdout_ids,
                                  storage_device=self.replay_device)

    def policy_update_allowed(self, engine):
        return not self.smoke and self.gate is not None and bool(self.gate(engine))

    def read(self):
        state = self.reader.state()
        memory = dict(cursor=self.env.cursor, remaining=self.env.episode_remaining, q=state[:, :54],
                      previous_q=state[:, :54], requested=self.task.previous_requested,
                      command=self.task.previous_issued, residual=self.task.last_residual,
                      earlier=self.task.earlier_residual, servo=self.reader.servo(),
                      previous=self.task.previous_valid, earlier_valid=self.task.earlier_valid,
                      actual_frame=self.task.pending.observation[:, -self.frame_dim:])
        return state, memory, memory["servo"]

    def replay_metadata(self, memory):
        cursor = memory["cursor"]
        return dict(source_id=self.catalog.source[cursor], partition_id=self.catalog.partition[cursor],
                    episode_id=self.env.episode.clone(), generation_id=self.catalog.frames.generation[cursor],
                    clock_tick=self.catalog.frames.tick_ns[cursor],
                    source_train=torch.arange(self.env.num_envs, device=self.device) < self.env.train_end)

    def replay_observation(self, obs):
        return obs[:, -self.frame_dim:]

    def collection_action(self, action, memory):
        if self.gate is not None and self.gate(None):
            return action
        count = action.shape[0]
        generator = self.exploration
        device = action.device
        group = torch.rand(count, 1, device=device, generator=generator)
        index = torch.randint(JOINTS, (count, 1), device=device, generator=generator)
        axes = torch.arange(JOINTS, device=device)[None]
        pair_mask = (axes - index).remainder(JOINTS) < 2
        hand_index = 14 + torch.randint(40, (count, 1), device=device, generator=generator)
        finger_mask = (axes >= 14) & ((axes - 14) // 4 == (hand_index - 14) // 4)
        one_hand = (axes >= 14) & ((axes - 14) // 20 == (hand_index - 14) // 20)
        left = index.remainder(2) == 0
        side = torch.where(left, (axes < 7) | ((axes >= 14) & (axes < 34)),
                           ((axes >= 7) & (axes < 14)) | (axes >= 34))
        mask = ((group >= .20) & (group < .45) & pair_mask
                | (group >= .45) & (group < .70) & torch.where(index.remainder(2) == 0, one_hand, finger_mask)
                | (group >= .70) & (group < .90) & side | (group >= .90))
        shape = action.shape
        signs = torch.rand(shape, device=device, generator=generator).ge(.5).float() * 2 - 1
        proposal = mask * signs * (.1 + .25 * torch.rand(shape, device=device, generator=generator))
        change = (self.hold_ticks <= 0) | ~memory["previous"]
        self.held_action = torch.where(change[:, None], proposal, self.held_action)
        duration = torch.randint(1, 9, (count,), device=device, generator=generator)
        self.hold_ticks = torch.where(change, duration, self.hold_ticks) - 1
        return self.held_action

    # Only compact causal scheduling/control state enters replay. Do not store a
    # second copy of a full observation via the critic's online metadata.
    memory_vectors = ("q", "previous_q", "requested", "command", "residual", "earlier", "servo")

    def encode_memory(self, memory):
        return torch.cat((torch.stack((memory["cursor"], memory["remaining"],
                                       memory["previous"], memory["earlier_valid"]), -1).float(),
                          *(memory[key] for key in self.memory_vectors)), -1)

    def decode_memory(self, encoded):
        cursor, remaining, prev, earlier = encoded[:, :4].unbind(-1)
        return dict(cursor=cursor.long(), remaining=remaining.long(), previous=prev.bool(),
                    earlier_valid=earlier.bool(),
                    **dict(zip(self.memory_vectors, encoded[:, 4:].split(JOINTS, -1), strict=True)))

    @staticmethod
    def dynamics(state, sim, servo):
        return torch.cat((state, servo, sim), -1)

    @staticmethod
    def replay_push(predicted, push, servo):
        # No disturbances are injected by this v1 C0 runner. A reset must not
        # become model input; a change of disturbance profile needs requalification.
        return predicted, servo

    @staticmethod
    def imagine_push(predicted, next_memory, servo):
        return predicted, next_memory["servo"]

    def replay_auxiliary(self, aux, issued):
        return self.profile.applied_interval(aux, issued)[:, -1]

    def current_goal(self, memory):
        return self.catalog.target(memory["cursor"])

    @staticmethod
    def sample_goal(count):
        # The fixed source schedule advances one row, never resamples a goal.
        return None

    @staticmethod
    def imagination_start_mask(memory, horizon):
        return memory["remaining"] > horizon

    def control_target(self, action, memory, bias=None):
        goal = self.current_goal(memory)
        candidate = goal.nominal + action * self.task.guard.scale
        issued = self.task.guard.project(candidate, memory["command"])
        return issued, self.profile.applied_interval(memory["servo"], issued).mean(1)

    def advance_memory(self, x, memory, action, issued, geometry, next_goal, soft=True):
        goal = self.current_goal(memory)
        active = memory["remaining"] > 1
        return dict(cursor=memory["cursor"] + active.long(), remaining=memory["remaining"] - active.long(),
                    requested=goal.nominal + action * self.task.guard.scale, command=issued, q=x[:, :54],
                    previous_q=memory["q"], residual=issued - goal.nominal, earlier=memory["residual"],
                    servo=self.replay_auxiliary(memory["servo"], issued),
                    previous=active, earlier_valid=memory["previous"])

    def termination(self, state, memory):
        q, qd = state.split(JOINTS, -1)
        return (~(self.task.guard.positions_valid(q) & self.task.guard.velocities_valid(qd)) |
                ~self.current_goal(memory).permitted).float()

    @staticmethod
    def torque(state, effective_u, sim):
        q, qd = state.split(JOINTS, -1)
        return ((effective_u - q) * sim[:, 55:109] - qd * sim[:, 109:163]).clamp(
            min=-sim[:, 163:217], max=sim[:, 163:217])

    def reward(self, state, memory, action, issued, contact, torque, soft=False):
        goal = self.current_goal(memory)
        geom = self.task.geometry(state[:, :54])
        vel = difference(geom, self.task.geometry(memory["q"]), DT)
        allowed = self.termination(state, memory).eq(0)
        requested = goal.nominal + action * self.task.guard.scale
        projection = ((issued - requested).abs() / self.task.guard.scale).amax(-1).clamp(0, 1)
        # Only the learned touch probabilities enter this diagnostic soft term.
        # Command legality comes exclusively from the real ResidualTask.
        clearance = self.task.reward.clearance + self.task.reward.margin_band * (
            1 - contact[..., 10:].amax(-1))
        result = self.task.reward(geom, vel, geometry_mask(allowed), velocity_mask(allowed), goal,
                                  issued - goal.nominal, memory["residual"], memory["earlier"],
                                  self.task.guard.scale, memory["previous"], memory["earlier_valid"],
                                  projection, clearance, allowed)
        return result.value, geom, result.terms

    def policy_observation(self, state, memory, aux):
        goal = self.current_goal(memory)
        q, qd = state.split(JOINTS, -1)
        available = self.termination(state, memory).eq(0)
        geom = self.task.geometry(q)
        velocity = difference(geom, self.task.geometry(memory["previous_q"]), DT)
        feedback = Feedback(q, qd, geom, velocity, geometry_mask(available),
                            velocity_mask(available & memory["previous"]),
                            torch.stack((available, available), -1), q.new_zeros(len(q)), available)
        command = CommandMemory(memory["requested"], memory["command"], memory["residual"], aux,
                                memory["previous"] & available, torch.isfinite(aux).all(-1))
        return self.task.builder(feedback, goal, command)

    def context(self, state, memory, servo, sim):
        features = memory.get("actual_frame")
        if features is None:
            features = self.policy_observation(state, memory, servo)
        return torch.cat((features, sim), -1)

    @staticmethod
    def critic(context):
        return context

    def history_from_obs(self, observation):
        return observation.reshape(-1, HISTORY, self.frame_dim)

    @staticmethod
    def flatten_history(history):
        return history.flatten(1)

    def rotation_error(self, state, memory):
        actual = self.task.geometry(state[:, :54])
        goal = self.current_goal(memory).geometry
        return error6(actual.wrist, actual.rotation, goal.wrist, goal.rotation)[..., 3:].norm(dim=-1).amax(-1)

    def state_dict(self):
        return {**self.env.state_dict(), "exploration": self.exploration.get_state(),
                "held_action": self.held_action.cpu(), "hold_ticks": self.hold_ticks.cpu()}

    def load_state_dict(self, snapshot):
        self.env.load_state_dict(snapshot)
        self.exploration.set_state(snapshot["exploration"].cpu())
        self.held_action.copy_(snapshot["held_action"].to(self.device))
        self.hold_ticks.copy_(snapshot["hold_ticks"].to(self.device))
