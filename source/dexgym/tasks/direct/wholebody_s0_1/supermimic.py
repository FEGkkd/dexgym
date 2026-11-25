"""Learn wholebody_s0-1 with DexGym's differentiable SuperMimic engine."""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from torch import nn

from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine

from ..marvin54_s0.physics import open_physics
from ..marvin54_tracking.task import JointCodec
from ..wholebody_s0.bag import digest
from ..wholebody_s0.domain import DR_DIM
from ..wholebody_s0.prepare import write_json, write_npz
from ..wholebody_s0.supermimic_task import collection_steps
from ..wholebody_s0.train_common import atomic_checkpoint
from . import DOF, DT, TASK_ID
from .bank import TaskBank
from .cross_trajectory import add_switch_arguments, switch_config
from .finger_reward import (OPPOSITION_REWARD_VERSION, REWARD_VARIANT, STILL_COMMAND_VERSION, STILL_HUBER_VERSION, STILL_LANDMARK_VERSION,
                            THUMB_IP_REWARD_VERSION, add_finger_reward_arguments, finger_reward_config)
from .control import (ACTOR_ARCHITECTURE, FOUR_POINT_HISTORY_FRAMES, FOUR_POINT_OBSERVATION,
                      HISTORY5_ABSOLUTE_COMMAND_OBSERVATION, HISTORY5_ABSOLUTE_OBSERVATION,
                      HISTORY5_NO_COMMAND_ERROR_OBSERVATION, HISTORY5_OBSERVATION,
                      HISTORY_DELTA_OBSERVATION, OBSERVATION_VERSION,
                      AbsoluteAction, History)
from .kinematics import MAPPED_HAND, REFERENCE_HAND_FK, Geometry, Pose, RightWristKinematics
from .posture import ARM_TEACHER_PROFILES, PRIOR_VERSIONS
from .reference_anchor import (ACTUATION_ANCHOR_VERSION, ARM_MOTION_GATE_VERSION, DELAYED_HAND_ANCHOR_VERSION,
                               ELBOW_ANCHOR_VERSION, ELBOW_NULLSPACE_VERSION,
                               FEEDBACK_ANCHOR_VERSION, HAND_HARDNESS_VERSION,
                               HAND_MOTION_GATE_VERSION,
                               HOLD_ANCHOR_VERSION, LIVE_EXECUTION_ANCHOR_VERSION, LIVE_RECOVERY_ANCHOR_VERSION,
                               LIVE_WRIST_EXECUTION_ANCHOR_VERSION,
                               RECOVERY_ANCHOR_VERSION,
                               RELATIVE_ANCHOR_VERSION, STILL_LIVE_ANCHOR_VERSION,
                               HAND_PRIORITY_ANCHOR_VERSION,
                               ON_POLICY_ANCHOR_VERSION,
                               REFERENCE_ANCHOR_VERSION,
                               REFERENCE_JOINT_PRIOR_VERSION, TEMPORAL_ANCHOR_VERSION, TIP_TAIL_ANCHOR_VERSION,
                               ReferenceAnchor)
from .task import (BRIDGE_STEPS, FAST_HAND_REWARD_VERSION, HAND_EXECUTION_REWARD_VERSION,
                   POSITIVE_TRACKING_REWARD_VERSION,
                   STATIONARY_STEPS,
                   RECOVERY_DISTANCE_M, RECOVERY_ROTATION_CHORD2,
                   SAMPLING_RAMP_UPDATES, WRIST_EXECUTION_REWARD_VERSION, WRIST_ROTATION_REWARD_VERSION,
                   DirectEnv, chord2, squared)
from .thumb_projection import CONTACT_TOLERANCE_M, PRESERVED_FIELDS, VERSION as THUMB_PROJECTION_VERSION
from .targets import joint_names
from .train import check_reset

STANDARD = SuperMimicConfig()


class Actor(nn.Module):
    def __init__(self, scale, features, *, history_delta=False, observation_version=None, action_home=None):
        super().__init__()
        self.net = AbsoluteAction(features, scale, history_delta=history_delta,
                                  observation_version=observation_version, action_home=action_home)

    def forward(self, observation):
        return self.net(observation).tanh()


class ImaginationTask:
    task_id = TASK_ID
    action_dim = DOF
    state_dim = 2 * DOF
    dynamics_dim = state_dim + DR_DIM
    contact_dim = 2
    termination_context_dim = DOF
    transition_termination = True
    freeze_actor_norm = True
    codec = JointCodec
    error_floor = [.0008] * DOF + [.025] * DOF
    max_step_reward = 1.45

    def __init__(self, env):
        self.env = self.reader = env
        self.device = env.device
        self.direct = env.direct.task
        self.policy_dim = self.direct.observe.features
        self.context_dim = self.direct.observe.frame_features + DR_DIM + 2
        self.bank = env.direct.bank
        self.domain = env.direct.domain
        self.simulator_context = self.domain.physical
        self.encoder_bias = self.domain.observation_bias
        self.refine_posture = env.direct.refine_posture
        self.arm_teacher = None
        self.reference_switch = env.direct.reference_switch
        self.switch_generator = torch.Generator(device=self.device).manual_seed(env.direct.switch_generator.initial_seed() + 1)
        self.actor_goal_counts = None

    def make_actor(self):
        version = self.direct.observe.version
        return Actor(self.direct.command.scale, self.policy_dim,
                     history_delta=version == HISTORY_DELTA_OBSERVATION, observation_version=version,
                     action_home=self.direct.command.home)

    def use_arm_teacher(self, actor, model=None, joint_names=None):
        self.arm_teacher = actor
        self.arm_indices = self.direct.command.arm.nonzero().flatten()
        self.right_wrist = None
        self.left_wrist = None
        if self.direct.posture.right_elbow:
            self.right_wrist = RightWristKinematics(model, joint_names, self.direct.observe.physical_wrist).to(self.device)
            self.right_arm = torch.tensor([name.endswith("_R") for name in joint_names if name.startswith("Joint")],
                                          device=self.device)
            self.arm_slack = torch.where(self.right_arm, .045, .012)
            self.arm_weight = torch.where(self.right_arm, .006, .14)
        if self.direct.posture.motion is not None:
            self.left_wrist = RightWristKinematics(model, joint_names, self.direct.observe.physical_wrist,
                                                   side="L").to(self.device)

    def teacher_wrist_pose(self, desired):
        right = self.right_wrist(desired)
        if self.left_wrist is None:
            return right
        left = self.left_wrist(desired)
        return tuple(torch.stack(pair, dim=1) for pair in zip(left, right, strict=True))

    def memory(self):
        env = self.env.direct
        memory = {"frame": env.frame, "bridge": env.bridge, "cmd": env.history.previous_command,
                  "delay": self.domain.delay, "previous_state": env.history.previous_state - self.encoder_bias,
                  "previous_frame": env.previous_frame, "previous_bridge": env.previous_bridge,
                  "last_success": env.last_success, "switch_recovery": env.switch_recovery}
        if self.refine_posture:
            memory.update(history_frame=env.history_frame, history_bridge=env.history_bridge)
        if hasattr(env, "stationary_steps"):
            memory["stationary_steps"] = env.stationary_steps
        if hasattr(env, "recorded_clock"):
            memory["recorded_clock"] = env.recorded_clock
        if hasattr(env, "elbow_hold_anchor"):
            memory["elbow_hold_anchor"] = env.elbow_hold_anchor
        return memory

    def read(self):
        return self.env.state(), self.memory(), self.encoder_bias

    def goals(self, memory):
        goal = self.env.direct._goal(memory["frame"], memory["bridge"])
        previous = self.env.direct._goal(memory["previous_frame"], memory["previous_bridge"])
        return goal, previous

    def current_goal(self, memory):
        return memory["frame"]

    @staticmethod
    def sample_goal(count):
        return None

    def policy_observation(self, x, memory, aux):
        if self.direct.observe.current_only:
            phase = ((memory["previous_frame"], memory["previous_bridge"]) if self.refine_posture
                     else (memory["frame"], memory["bridge"]))
            goal = self.env.direct._goal(*phase)
            return self.direct.observe.forward_current(x + aux, goal, memory["cmd"])
        if self.refine_posture:
            goal = self.env.direct._goal(memory["previous_frame"], memory["previous_bridge"])
            previous_goal = self.env.direct._goal(memory["history_frame"], memory["history_bridge"])
        else:
            goal, previous_goal = self.goals(memory)
        history = History(memory["previous_state"] + aux, memory["cmd"], previous_goal)
        return self.direct.observe(x + aux, goal, history)

    def context(self, x, memory, aux, sim):
        # SuperMimic builds the context at the start of each imagined interval.
        memory["interval_start"] = x
        obs = self.policy_observation(x, memory, torch.zeros_like(aux))
        frame = memory["frame"]
        remaining = self.bank.end[frame] - frame
        if "recorded_clock" in memory:
            remaining = torch.where(memory["recorded_clock"].bool(), self.bank.recorded_remaining[frame], remaining)
        # Home warmup is positive, post-switch recovery negative; keep critic shape.
        clock = memory["bridge"] / BRIDGE_STEPS - memory["switch_recovery"] * DT
        phase = torch.stack((remaining * DT, clock), -1)
        return torch.cat((obs, sim, phase), -1)

    @staticmethod
    def critic(context):
        return context

    def history_from_obs(self, obs):
        return obs.float().reshape(-1, self.direct.observe.history_frames, self.direct.observe.frame_features)

    def observation_from_history(self, history, memory):
        observation = history[:, -1]
        if self.arm_teacher is not None:
            with torch.no_grad():
                prior_action = self.arm_teacher(observation)
                command = self.direct.command
                memory["teacher_target"] = (command.home + command.scale * prior_action).clamp(command.lower, command.upper)
                if self.right_wrist is not None:
                    memory["teacher_wrist"] = self.teacher_wrist_pose(memory["teacher_target"])
        return history.flatten(1)

    def arm_teacher_cost(self, action, memory):
        limits = self.direct.command
        desired = (limits.home + limits.scale * action).clamp(limits.lower, limits.upper)
        delta = (desired - memory["teacher_target"])[:, self.arm_indices].abs()
        if self.right_wrist is None:
            return .14 * ((delta - .012).clamp_min(0.) / .020).square().mean(-1)
        pos, rot = self.teacher_wrist_pose(desired)
        ref_pos, ref_rot = memory["teacher_wrist"]
        if self.left_wrist is not None:
            joint_cost = torch.log1p(((delta - .045).clamp_min(0.) / .035).square()).mean(-1)
            pos_cost = (squared(pos - ref_pos) - .002**2).clamp_min(0.) / .008**2
            rot_cost = (chord2(rot, ref_rot) - .012**2).clamp_min(0.) / .042**2
            return .020 * joint_cost + .080 * pos_cost.mean(-1) + .042 * rot_cost.mean(-1)
        joint_cost = self.arm_weight * ((delta - self.arm_slack).clamp_min(0.) / .020).square()
        # Anchor the right wrist command across redundant arm configurations.
        pos_cost = (squared(pos - ref_pos) - .0015**2).clamp_min(0.) / .007**2
        rot_cost = (chord2(rot, ref_rot) - .008**2).clamp_min(0.) / .035**2
        return joint_cost.mean(-1) + .085 * pos_cost + .045 * rot_cost

    def replay_observation(self, obs):
        # Keep one frame per replay row. Reconstruct causal windows when sampled;
        # storing eight copies would add ~16 GB at 4096 envs x 500 rows.
        return obs[:, -self.direct.observe.frame_features:].to(torch.float16)

    @staticmethod
    def dynamics(x, sim, aux):
        return torch.cat((x, sim), -1)

    def memory_keys(self):
        keys = ("frame", "bridge", "previous_frame", "previous_bridge", "delay", "last_success", "switch_recovery")
        if self.refine_posture:
            keys = (*keys, "history_frame", "history_bridge")
        if hasattr(self.env.direct, "stationary_steps"):
            keys = (*keys, "stationary_steps")
        if hasattr(self.env.direct, "recorded_clock"):
            keys = (*keys, "recorded_clock")
        return keys

    def encode_memory(self, memory):
        indices = torch.stack(tuple(memory[key] for key in self.memory_keys()), -1)
        values = (indices, memory["cmd"], memory["previous_state"])
        if "elbow_hold_anchor" in memory:
            values += (memory["elbow_hold_anchor"].flatten(1),)
        return torch.cat(values, -1)

    def decode_memory(self, encoded):
        keys = self.memory_keys()
        have_anchor = hasattr(self.env.direct, "elbow_hold_anchor")
        sizes = (len(keys), DOF, 2 * DOF, 14) if have_anchor else (len(keys), DOF, 2 * DOF)
        indices, cmd, state, *anchor = encoded.split(sizes, -1)
        memory = dict(zip(keys, indices.long().unbind(-1), strict=True))
        if have_anchor:
            memory["elbow_hold_anchor"] = anchor[0].unflatten(-1, (2, 7))
        return {**memory, "delay": memory["delay"].bool(), "cmd": cmd, "previous_state": state}

    def advance_memory(self, x, memory, action, command, geometry, next_goal, soft=True):
        frame, bridge = memory["frame"], memory["bridge"]
        stationary = memory.get("stationary_steps")
        recorded = memory.get("recorded_clock")
        if recorded is not None:
            recorded = recorded.bool()
        hold = None if stationary is None else stationary > 0
        with torch.no_grad():
            measured = self.direct.observe.fk(x[:, :DOF].detach()).pose
            eligible = ((frame < self.bank.end[frame]) & torch.isfinite(x).all(-1)
                        & (self.direct.position_violation(x) <= .05))
            if getattr(self.direct, "stability", False) and hold is not None:
                eligible &= ~hold
            if "switch_eligible" in memory:
                eligible &= memory["switch_eligible"]
            reference = self.reference_switch.advance(
                measured, frame, bridge, memory["last_success"], memory["switch_recovery"],
                self.switch_generator, eligible=eligible, hold=hold,
                recorded=recorded)
        new = {"frame": reference.frame, "bridge": reference.bridge, "cmd": command,
               "delay": memory["delay"], "previous_state": geometry,
               "previous_frame": reference.previous_frame, "previous_bridge": reference.previous_bridge,
               "last_success": reference.last_success, "switch_recovery": reference.recovery}
        if "recorded_clock" in memory:
            new["recorded_clock"] = memory["recorded_clock"]
        if "elbow_hold_anchor" in memory:
            new["elbow_hold_anchor"] = memory["elbow_hold_anchor"]
        if stationary is not None:
            new["stationary_steps"] = torch.where(reference.switched, 0, (stationary - 1).clamp_min(0))
        if self.refine_posture:
            new.update(history_frame=memory["previous_frame"], history_bridge=memory["previous_bridge"])
        return new

    def imagination_start_mask(self, memory, horizon):
        frame = memory["frame"]
        remaining = self.bank.end[frame] - frame
        if "recorded_clock" in memory:
            remaining = torch.where(memory["recorded_clock"].bool(), self.bank.recorded_remaining[frame], remaining)
        return remaining + memory["bridge"] + memory.get("stationary_steps", 0) >= horizon

    def control_target(self, action, memory, bias=None):
        command = self.direct.command(action, memory["cmd"])
        applied = torch.where(memory["delay"][:, None], memory["cmd"], command)
        return command, applied

    def collection_action(self, action, memory):
        command = self.direct.command
        std = torch.where(command.arm, .032, .045) / command.scale
        return (action + std * torch.randn_like(action)).clamp(-1., 1.)

    @staticmethod
    def torque(x, u, sim):
        # The shared reward uses measured geometry and command smoothness.
        return None

    @staticmethod
    def replay_push(x, push, aux):
        return x + push, aux

    def imagine_push(self, x, memory, aux):
        return x + self.domain.imagine_push(x), aux

    def termination(self, x, memory):
        frame = memory["frame"]
        end = (memory["bridge"] == 0) & (frame == self.bank.end[frame])
        outside = torch.sigmoid((self.direct.position_violation(x) - .05) / .004)
        return torch.where(end, 1., outside)

    def reward(self, x, memory, action, command, contact, torque, soft=False):
        goal, old_goal = self.goals(memory)
        target_jump = self.bank.clip_id[memory["frame"]] != self.bank.clip_id[memory["previous_frame"]]
        if soft and self.actor_goal_counts is not None:
            with torch.no_grad():
                self.actor_goal_counts.scatter_add_(0, memory["frame"], (memory["bridge"] == 0).to(torch.int64))
        # The caller attaches the physical state at the start of this interval.
        prior_x = memory["interval_start"]
        current, previous = x[:, :DOF], prior_x[:, :DOF]
        both = self.direct.observe.fk(torch.cat((current, previous), dim=0))
        measured, old = (Pose(*parts) for parts in zip(*(v.chunk(2, 0) for v in both.pose), strict=True))
        geometry = Geometry(measured, *(v[:len(x)] for v in both[1:]))
        if self.refine_posture:
            old_geometry = Geometry(old, *(v[len(x):] for v in both[1:]))
            phase = memory["frame"], memory["bridge"]
            old_phase = memory["previous_frame"], memory["previous_bridge"]
            reward = self.direct.tracking_reward(x, geometry, old_geometry, goal, old_goal, command, memory["cmd"],
                                                previous_state=prior_x, phase=phase, old_phase=old_phase,
                                                target_jump=target_jump, requested_action=action, delayed=memory["delay"])
            if "elbow_hold_anchor" in memory:
                from .stability import stationary_elbow_anchor_cost
                hold_cost, _, _ = stationary_elbow_anchor_cost(
                    geometry, old_geometry, goal, old_goal, memory["elbow_hold_anchor"],
                    memory["stationary_steps"] > 0,
                )
                reward -= self.direct.elbow_hold_drift_weight * hold_cost
            if "teacher_target" in memory:
                cost = self.arm_teacher_cost(action, memory)
                if self.bank.source_weights is not None:
                    cost = cost * (~self.bank.egopro[memory["frame"]])
                reward = reward - cost
        else:
            reward = self.direct.tracking_reward(x, geometry, old, goal, old_goal, command, memory["cmd"],
                                                 phase=(memory["frame"], memory["bridge"]),
                                                 target_jump=target_jump, requested_action=action, delayed=memory["delay"])
        wrist_dist = squared(measured.wrist - goal.wrist)
        wrist_rot = chord2(measured.rotation, goal.rotation)
        lost = torch.sigmoid((wrist_dist.amax(-1) - RECOVERY_DISTANCE_M**2) / .008)
        turned = torch.sigmoid((wrist_rot.amax(-1) - RECOVERY_ROTATION_CHORD2) / .12)
        recovering = memory["switch_recovery"] > 0
        reward -= .20 * contact[:, 0] + contact[:, 1] + .75 * torch.where(recovering, 0., torch.maximum(lost, turned))
        memory["switch_eligible"] = contact[:, 1].detach() < .5
        if self.refine_posture and self.direct.posture.mixed_wrist:
            reward -= .18 * contact[:, 0] + .65 * contact[:, 1]
        return reward, prior_x, {}

    def rotation_error(self, x, memory):
        return self.env.transition["rollout_stats"]["wrist_deg"]


class ImaginationEnv:
    def __init__(self, backend, bank, model, seed, robust, *, posture_profile=None, switch=None, finger_reward=None,
                 reference_hand_fk=False, observation_version=OBSERVATION_VERSION, positive_tracking_reward=False,
                 rsi_task_space_weight=0., rsi_joint_prior_weight=0., rsi_live_fraction=0., rsi_hand_weight=1.,
                 rsi_tip_tail_weight=0., wrist_execution_weight=0., rsi_arm_motion_gate=0.,
                 hand_execution_weight=0., rsi_hand_motion_gate=0., rsi_hand_hardness=0.,
                 rsi_arm_lead=0, rsi_hand_lead=0, rsi_hold_fraction=0., rsi_arm_actuation_weight=0.,
                 rsi_hold_recovery_noise=0., stationary_reset_fraction=0., rsi_elbow_weight=0.,
                 rsi_hand_feedback_weight=0., rsi_relative_pose_weight=1.,
                 rsi_hard_live_fraction=0., rsi_recovery_frames=0,
                 rsi_stationary_live_fraction=0., rsi_independent_feedback=False, rsi_hand_delay_lead=0,
                 rsi_arm_delay_lead=0,
                 rsi_elbow_nullspace=False, wrist_rotation_tail_weight=0., fast_hand_weight=0.,
                 rsi_hand_physical_weight=0., rsi_wrist_physical_weight=0., recorded_clock_fraction=0.,
                 rsi_live_recovery_weight=0., rsi_elbow_physical_weight=0., rsi_live_contact_weight=0.,
                 rsi_recorded_hand_lead=0, rsi_execution_priority=0., rsi_wrist_recovery_priority=0.):
        self.direct = DirectEnv(backend, bank, model, seed, robust, record_transition=True,
                                posture_profile=posture_profile, switch=switch, finger_reward=finger_reward,
                                reference_hand_fk=reference_hand_fk, observation_version=observation_version,
                                positive_tracking_reward=positive_tracking_reward,
                                wrist_execution_weight=wrist_execution_weight,
                                hand_execution_weight=hand_execution_weight,
                                stationary_reset_fraction=stationary_reset_fraction,
                                wrist_rotation_tail_weight=wrist_rotation_tail_weight, fast_hand_weight=fast_hand_weight,
                                recorded_clock_fraction=recorded_clock_fraction)
        self.num_envs, self.device = backend.num_envs, backend.device
        self.task = ImaginationTask(self)
        if rsi_task_space_weight or rsi_joint_prior_weight:
            self.task.actor_reference_anchor = ReferenceAnchor(
                bank, self.direct.task.observe, rsi_task_space_weight, seed + 193,
                joint_weight=rsi_joint_prior_weight, batch=512 if rsi_joint_prior_weight else 256,
                live_env=self.direct, live_fraction=rsi_live_fraction, hand_weight=rsi_hand_weight,
                tip_tail_weight=rsi_tip_tail_weight, arm_motion_gate=rsi_arm_motion_gate,
                hand_motion_gate=rsi_hand_motion_gate, hand_hardness=rsi_hand_hardness,
                arm_lead=rsi_arm_lead, hand_lead=rsi_hand_lead, hold_fraction=rsi_hold_fraction,
                arm_actuation_weight=rsi_arm_actuation_weight, hold_recovery_noise=rsi_hold_recovery_noise,
                elbow_weight=rsi_elbow_weight, hand_feedback_weight=rsi_hand_feedback_weight,
                relative_pose_weight=rsi_relative_pose_weight, hard_live_fraction=rsi_hard_live_fraction,
                recovery_frames=rsi_recovery_frames, stationary_live_fraction=rsi_stationary_live_fraction,
                independent_feedback=rsi_independent_feedback, hand_delay_lead=rsi_hand_delay_lead,
                arm_delay_lead=rsi_arm_delay_lead,
                elbow_nullspace=rsi_elbow_nullspace, hand_physical_weight=rsi_hand_physical_weight,
                wrist_physical_weight=rsi_wrist_physical_weight, live_recovery_weight=rsi_live_recovery_weight,
                elbow_physical_weight=rsi_elbow_physical_weight, live_contact_weight=rsi_live_contact_weight,
                recorded_hand_lead=rsi_recorded_hand_lead, execution_priority=rsi_execution_priority,
                wrist_recovery_priority=rsi_wrist_recovery_priority)
        self.disturbance = SimpleNamespace(adaptive_scale=float(robust), success_scale=0.)

    @property
    def common_step_counter(self):
        return self.direct.steps

    @common_step_counter.setter
    def common_step_counter(self, steps):
        self.direct.steps = steps

    @property
    def transition(self):
        return self.direct.transition

    def state(self):
        return self.direct.state

    def reset(self, env_ids=None):
        self.direct.reset(env_ids)
        return {"policy": self.direct.observation()}, {}

    def step(self, action):
        reward, done, timeout, _ = self.direct.step(action)
        return {"policy": self.direct.observation()}, reward, done & ~timeout, timeout, {}


class Learner(SuperMimicEngine):
    def __init__(self, task, cfg, log_dir):
        super().__init__(task, cfg, log_dir)
        visits = self.env.direct.goal_visits
        self.actor_starts = torch.zeros_like(visits) if visits is not None else None
        self.task.actor_goal_counts = torch.zeros_like(visits, dtype=torch.int64) if visits is not None else None
        self.fast_start_fraction = 0.
        self.fast_start_metrics = {}
        self.fast_start_generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 421)
        anchor = getattr(self.task, "actor_reference_anchor", None)
        if anchor is not None and (anchor.hand_physical_weight or anchor.wrist_physical_weight or anchor.live_recovery_weight):
            anchor.world_model = self.step_wm
            anchor.world_features = self.wm_features

    def collect(self, steps):
        self.env.direct.set_sampling_iteration(1 + self.iteration)
        return super().collect(steps)

    @torch.no_grad()
    def sample_fast_starts(self, batch):
        task = self.task
        memory = task.decode_memory(batch["m"])
        frame, previous = memory["frame"], memory["previous_frame"]
        count = round(len(frame) * self.fast_start_fraction)
        if not count:
            return batch
        bank = task.bank
        goal = bank.pose.hand[frame]
        old_goal = bank.pose.hand[previous]
        actual = task.direct.observe.fk(batch["x"][:, :DOF]).pose.hand
        velocity = (goal[..., -1, :] - old_goal[..., -1, :]).norm(dim=-1) / DT
        error = (actual[..., -1, :] - goal[..., -1, :]).norm(dim=-1)
        same_clip = bank.clip_id[frame] == bank.clip_id[previous]
        exact_clock = (memory["bridge"] == 0) & same_clip & (memory["switch_recovery"] == 0)
        moving = velocity / .08
        per_digit = moving.clamp(max=3.) * (error / .008).clamp(max=4.)
        difficulty = per_digit.flatten(1).topk(2, dim=-1).values.mean(-1)
        weight = (difficulty + .05) * (1. + .75 * memory["delay"])
        active = exact_clock & (velocity.amax((1, 2)) >= .04)
        if not active.any():
            return batch
        weight = weight.masked_fill(~active, 0.)
        source = bank.egopro[frame].long()
        mass = weight.new_zeros(2).scatter_add_(0, source, weight)
        weight = weight / mass[source].clamp_min(1e-9)
        indices = torch.randperm(len(frame), generator=self.fast_start_generator, device=frame.device)
        selected = torch.multinomial(weight, count, replacement=True, generator=self.fast_start_generator)
        indices[:count] = selected
        self.fast_start_metrics["imagination_fast_sample_fraction"] = difficulty.new_tensor(count / len(frame))
        self.fast_start_metrics["imagination_fast_sample_tip_speed_m_s"] = velocity[selected].amax((1, 2)).mean()
        self.fast_start_metrics["imagination_fast_sample_delayed_fraction"] = memory["delay"][selected].float().mean()
        return {key: value[indices] for key, value in batch.items()}

    def sample_starts(self):
        history_frames = self.task.direct.observe.history_frames
        if history_frames == 1:
            batch = super().sample_starts()
        else:
            from .temporal import replay_history
            b = self.buffer
            rows = (b.ptr - b.size + torch.arange(b.size, device=self.device)) % b.capacity
            mask = b.data.get("imagination_valid", b.data["valid"])
            eligible = mask[rows, :b.train_end].nonzero()
            if not len(eligible):
                raise RuntimeError("Replay contains no finite imagination starts")
            chosen = eligible[torch.randint(len(eligible), (self.cfg.pi_batch,), device=self.device)]
            chronological, env = chosen.unbind(-1)
            row = rows[chronological]
            batch = {key: value[row, env] for key, value in b.data.items()}
            batch.update(env=env, sim_context=self.task.simulator_context[env])
            batch["obs"] = replay_history(b.data, rows, chronological, env, history_frames).flatten(1)
        if self.fast_start_fraction:
            batch = self.sample_fast_starts(batch)
        if self.actor_starts is not None:
            frame, bridge = batch["m"][:, :2].long().unbind(-1)
            hit = torch.ones_like(frame, dtype=self.actor_starts.dtype)
            self.actor_starts.scatter_add_(0, frame, hit)
        return batch

    def policy_update(self):
        self.fast_start_metrics.clear()
        metrics = super().policy_update()
        metrics.update(self.fast_start_metrics)
        anchor = getattr(self.task, "actor_reference_anchor", None)
        if anchor is not None:
            metrics.update(anchor.metrics)
        return metrics

    def save(self, path, next_iteration=None):
        super().save(path, next_iteration)
        self.save_actor(Path(path).with_name(Path(path).stem + "_actor.pt"), next_iteration)
        self.save_coverage(next_iteration)

    def save_actor(self, path, next_iteration=None):
        metadata = self.task.bank.metadata
        atomic_checkpoint({
            "task_id": TASK_ID, "method": "supermimic",
            "iteration": self.iteration if next_iteration is None else next_iteration,
            "features": self.task.policy_dim, "actor_architecture": ACTOR_ARCHITECTURE,
            "observation_version": self.task.direct.observe.version,
            "reward_version": self.reward_version,
            "finger_reward": self.task.direct.finger_reward.to_dict(),
            "hand_target_mode": self.task.bank.hand_target_mode,
            "bank_sha256": self.bank_sha256, "model_sha256": metadata["teleop_model_sha256"],
            "sim_steps": self.sim_steps, "actor": self.actor.net.state_dict(),
        }, path)

    def source_coverage(self, counts, count_name):
        summary = {}
        for source in ("teleop", "egopro"):
            rows = [row for row in self.task.bank.clips if row.get("source", "teleop") == source]
            clips = [(row, counts[row["start"]:row["stop"]]) for row in rows]
            seen = [row for row, frames in clips if frames.any()]
            summary[source] = {
                "frames": sum(len(frames) for _, frames in clips),
                "visited_frames": sum(int(np.count_nonzero(frames)) for _, frames in clips),
                count_name: sum(int(frames.sum()) for _, frames in clips),
                "clips": len(rows), "visited_clips": len(seen),
                "episodes": len({row["episode"] for row in rows}),
                "visited_episodes": len({row["episode"] for row in seen}),
            }
        return summary

    def save_coverage(self, next_iteration):
        visits = self.env.direct.goal_visits
        if visits is None:
            return
        counts = visits.cpu().numpy()
        starts = self.actor_starts.cpu().numpy()
        goals = self.task.actor_goal_counts.cpu().numpy()
        write_npz(self.log_dir / "goal_visits.npz", {"counts": counts, "actor_starts": starts, "actor_goals": goals})
        write_json(self.log_dir / "goal_coverage.json", {
            "iteration": next_iteration, "bank_sha256": self.bank_sha256,
            "sources": self.source_coverage(counts, "valid_reference_steps"),
            "actor_imagination": {"starts": self.source_coverage(starts, "sampled_starts"),
                                  "unblended_goals": self.source_coverage(goals, "imagined_goal_steps")},
        })


def verify_bank_controls(previous, training):
    contract = ("schema", "rate_hz", "teleop_model_sha256", "home", "joint_names", "frame",
                "position_lower_rad", "position_upper_rad", "position_rate_rad_s", "wrist_calibration",
                "train_source_weights")
    for field in contract:
        if previous.get(field) != training.get(field):
            raise ValueError(f"Reference bank differs in {field}")


def verify_superset(path, training):
    previous = json.loads(path.read_text(encoding="utf-8"))
    verify_bank_controls(previous, training)
    assignments = {}
    for split in ("train", "validation", "test"):
        current = {row["id"]: row for row in training["splits"][split]["clips"]}
        if len(current) != len(training["splits"][split]["clips"]):
            raise ValueError(f"Duplicate clip IDs in expanded {split}")
        for row in current.values():
            key = (row["source"], row["episode"])
            if assignments.setdefault(key, split) != split:
                raise ValueError(f"Expanded bank leaks an episode across splits: {key}")
        for entry in previous["splits"][split]["clips"]:
            match = current.get(entry["id"])
            if match is None or any(match[key] != entry[key] for key in
                    ("episode", "source", "reference_sha256")) or (
                    entry["stop"] - entry["start"] != match["stop"] - match["start"]):
                raise ValueError(f"Expanded bank changed original {split} clip {entry['id']}")
    return digest(path)


def verify_aligned_bank(path, training):
    previous = json.loads(path.read_text(encoding="utf-8"))
    verify_bank_controls(previous, training)
    prior_hash = digest(path)
    if (training["previous_bank_sha256"] != prior_hash
            or training["teleop_start_alignment"] != "per_side_physical_wrist_anchor"
            or training["source_manifest_sha256"] != previous["source_manifest_sha256"]
            or training["position_rate_contract"] != previous["position_rate_contract"]):
        raise ValueError("Expected a per-side START rebuild of the preceding reference bank")
    for split in ("train", "validation", "test"):
        old, new = (bank["splits"][split]["clips"] for bank in (previous, training))
        raw_teleop = lambda rows: Counter((row["episode"], row["id"].split("_wrist_")[0])
                                          for row in rows if row["source"] == "teleop")
        egopro = lambda rows: {row["id"]: (row["episode"], row["reference_sha256"]) for row in rows
                               if row["source"] == "egopro"}
        if raw_teleop(old) != raw_teleop(new) or egopro(old) != egopro(new):
            raise ValueError(f"The START alignment rebuild changed source coverage in {split}")
    return prior_hash


def verify_refit_bank(path, bank_path, training):
    """Bind initialization to the old weights only after verifying this bank's unchanged goals."""
    previous = json.loads(path.read_text(encoding="utf-8"))
    verify_bank_controls(previous, training)
    prior_hash = digest(path)
    refit = training.get("wrist_refit", {})
    if (refit.get("input_bank_sha256") != prior_hash
            or previous["source_manifest_sha256"] != training["source_manifest_sha256"]
            or previous["position_rate_contract"] != training["position_rate_contract"]):
        raise ValueError("Expected a wrist refit of the initializing checkpoint's archived bank")

    fields = ("wrist", "rotation", "hand", "time_ns")
    clip_identity = ("id", "episode", "source", "task", "start", "stop", "timing")
    for split in ("train", "validation", "test"):
        source, updated = (bank["splits"][split] for bank in (previous, training))
        pack = bank_path.parent / updated["file"]
        if source["frames"] != updated["frames"] or len(source["clips"]) != len(updated["clips"]):
            raise ValueError(f"Wrist refit changed coverage in {split}")
        if digest(pack) != updated["sha256"]:
            raise ValueError(f"Refitted split checksum differs: {pack}")
        with np.load(pack, allow_pickle=False) as archive:
            goals = {key: archive[key] for key in fields}
            left_inverse = goals["rotation"][:, 0].swapaxes(-1, -2)
            relative_position = (left_inverse @ (goals["wrist"][:, 1] - goals["wrist"][:, 0])[..., None])[..., 0]
            if (not np.allclose(archive["relative_position"], relative_position, rtol=0., atol=2e-7)
                    or not np.allclose(archive["relative_rotation"], left_inverse @ goals["rotation"][:, 1],
                                       rtol=0., atol=2e-7)):
                raise ValueError(f"Refitted relative wrist goals disagree with the wrist pair in {split}")

        # The original per-clip goal archives retain their training manifest checksums.
        # This verifies the new packed goals without keeping the old duplicate training pack.
        for old, new in zip(source["clips"], updated["clips"], strict=True):
            if any(old[key] != new[key] for key in clip_identity):
                raise ValueError(f"Wrist refit changed a clip identity in {split}: {old['id']}")
            reference = Path(old["reference_file"])
            if digest(reference) != old["reference_sha256"]:
                raise ValueError(f"Archived source reference checksum differs: {reference}")
            frames = slice(old["start"], old["stop"])
            with np.load(reference, allow_pickle=False) as record:
                for key in fields:
                    expected, actual = record[key], goals[key][frames]
                    if expected.dtype != actual.dtype or not np.array_equal(expected, actual):
                        raise ValueError(f"Wrist refit changed {key} in {split}: {old['id']}")
        print(f"Verified checkpoint goal alignment: {split} {len(updated['clips'])} clips", flush=True)
    return prior_hash


def verify_thumb_bank(previous_path, bank_path, training):
    """Allow a warm start after checking the projected bank against its source."""
    prior = json.loads(previous_path.read_text(encoding="utf-8"))
    verify_bank_controls(prior, training)
    prior_hash = digest(previous_path)
    projection = training.get("thumb_projection", {})
    if (projection.get("version") != THUMB_PROJECTION_VERSION
            or projection.get("input_bank_sha256") != prior_hash
            or prior["source_manifest_sha256"] != training["source_manifest_sha256"]
            or prior["splits"].keys() != training["splits"].keys()):
        raise ValueError("Expected a Wuji thumb projection of the initializing checkpoint's bank")
    for split, original in prior["splits"].items():
        projected = training["splits"][split]
        if original["clips"] != projected["clips"] or original["frames"] != projected["frames"]:
            raise ValueError(f"Thumb projection changed clip coverage in {split}")
        paths = (previous_path.parent / original["file"], bank_path.parent / projected["file"])
        for path, info in zip(paths, (original, projected), strict=True):
            if digest(path) != info["sha256"]:
                raise ValueError(f"Bank checksum differs: {path}")
        with np.load(paths[0], allow_pickle=False) as before, np.load(paths[1], allow_pickle=False) as after:
            for key in PRESERVED_FIELDS:
                if not np.array_equal(before[key], after[key]):
                    raise ValueError(f"Thumb projection changed {key} in {split}")
            if not np.array_equal(before["hand"][:, :, 1:], after["hand"][:, :, 1:]):
                raise ValueError(f"Thumb projection changed non-thumb goals in {split}")
            if not (after["reset_depth_m"] <= CONTACT_TOLERANCE_M).all():
                raise ValueError(f"Thumb projection contains a penetrating pose in {split}")
        print(f"Verified thumb projection source: {split} {original['frames']} frames", flush=True)
    wrist_refit = prior.get("wrist_refit")
    if wrist_refit:
        # This bank kept all original goals but was rebuilt with cleaner RSI states.
        return verify_refit_bank(Path(wrist_refit["input_bank"]), previous_path, prior)
    return prior_hash


def restrict_thumb_distal_updates(actor, model):
    """Adapt MCP/IP predictions while keeping the shared encoder and feedback gains."""
    net = actor.net
    net.encoder.requires_grad_(False)
    rows = [int(model.joint(f"{side}_thumb_{joint}").qposadr[0])
            for side in "lr" for joint in ("mcp", "ip")]
    weight = torch.zeros_like(net.head.weight, dtype=torch.bool)
    bias = torch.zeros_like(net.head.bias, dtype=torch.bool)
    weight[rows, :net.head.in_features - 2 * DOF] = True
    bias[rows] = True
    net.head.weight.register_hook(lambda grad: grad * weight)
    net.head.bias.register_hook(lambda grad: grad * bias)
    return {"net.head.weight": weight, "net.head.bias": bias}


def optimizer_steps(engine):
    return {name: max((int(state["step"]) for state in opt.state.values() if "step" in state), default=0)
            for name, opt in engine.optimizers.items()}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    bank = TaskBank(args.bank, "train", "cuda:0",
                    mapped_hand_clip_fraction=args.mapped_hand_clip_fraction,
                    hand_style_seed=args.seed)
    bank.stability = args.stability_v1
    bank.stability_v2 = args.stability_v2
    metadata = bank.metadata
    refit_manifest = getattr(args, "init_refit_bank", None)
    refit_source_hash = verify_refit_bank(refit_manifest, args.bank, metadata) if refit_manifest else None
    thumb_manifest = getattr(args, "init_thumb_bank", None)
    thumb_source_hash = verify_thumb_bank(thumb_manifest, args.bank, metadata) if thumb_manifest else None
    adapt_thumb_distal = getattr(args, "adapt_thumb_distal", False)
    if adapt_thumb_distal and not metadata.get("thumb_projection"):
        raise ValueError("Distal thumb adaptation requires projected thumb targets")
    if digest(args.model) != metadata["teleop_model_sha256"]:
        raise ValueError("Train with the teleoperation MJCF used to prepare targets")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if joint_names(model) != metadata["joint_names"]:
        raise ValueError("MJCF joint order differs from the task-space bank")
    args.output.mkdir(parents=True, exist_ok=False)
    with open_physics("newton", model, metadata["joint_names"], args.num_envs, robot_usd=args.usd) as backend:
        finger_reward = finger_reward_config(args)
        env = ImaginationEnv(backend, bank, model, args.seed, args.robust, posture_profile=args.refine_posture,
                             switch=switch_config(args), finger_reward=finger_reward,
                             reference_hand_fk=getattr(args, "reference_hand_fk", False),
                             observation_version=args.observation_version,
                             positive_tracking_reward=args.positive_tracking_reward,
                             rsi_task_space_weight=args.rsi_task_space_weight,
                             rsi_joint_prior_weight=args.rsi_joint_prior_weight,
                             rsi_live_fraction=args.rsi_live_fraction, rsi_hand_weight=args.rsi_hand_weight,
                             rsi_tip_tail_weight=args.rsi_tip_tail_weight,
                             wrist_execution_weight=args.wrist_execution_weight,
                             rsi_arm_motion_gate=args.rsi_arm_motion_gate,
                             hand_execution_weight=args.hand_execution_weight,
                             rsi_hand_motion_gate=args.rsi_hand_motion_gate,
                             rsi_hand_hardness=args.rsi_hand_hardness,
                             rsi_arm_lead=args.rsi_arm_lead,
                             rsi_hand_lead=args.rsi_hand_lead, rsi_hold_fraction=args.rsi_hold_fraction,
                             rsi_arm_actuation_weight=args.rsi_arm_actuation_weight,
                             rsi_hold_recovery_noise=args.rsi_hold_recovery_noise,
                             stationary_reset_fraction=args.stationary_reset_fraction,
                             rsi_elbow_weight=args.rsi_elbow_weight,
                             rsi_hand_feedback_weight=args.rsi_hand_feedback_weight,
                             rsi_relative_pose_weight=args.rsi_relative_pose_weight,
                             rsi_hard_live_fraction=args.rsi_hard_live_fraction,
                             rsi_recovery_frames=args.rsi_recovery_frames,
                             rsi_stationary_live_fraction=args.rsi_stationary_live_fraction,
                             rsi_independent_feedback=args.rsi_independent_feedback,
                             rsi_hand_delay_lead=args.rsi_hand_delay_lead,
                             rsi_arm_delay_lead=args.rsi_arm_delay_lead,
                             rsi_elbow_nullspace=args.rsi_elbow_nullspace,
                             wrist_rotation_tail_weight=args.wrist_rotation_tail_weight,
                             fast_hand_weight=args.fast_hand_weight,
                             rsi_hand_physical_weight=args.rsi_hand_physical_weight,
                             rsi_wrist_physical_weight=args.rsi_wrist_physical_weight,
                             recorded_clock_fraction=args.recorded_clock_fraction,
                             rsi_live_recovery_weight=args.rsi_live_recovery_weight,
                             rsi_elbow_physical_weight=args.rsi_elbow_physical_weight,
                             rsi_live_contact_weight=args.rsi_live_contact_weight,
                             rsi_recorded_hand_lead=args.rsi_recorded_hand_lead,
                             rsi_execution_priority=args.rsi_execution_priority,
                             rsi_wrist_recovery_priority=args.rsi_wrist_recovery_priority)
        if args.opposition_sample_fraction:
            bank.prepare_opposition_sampling(args.opposition_sample_fraction)
        if args.elbow_hold_drift_weight:
            env.direct.task.elbow_hold_drift_weight = args.elbow_hold_drift_weight
            env.direct.elbow_hold_anchor = torch.zeros((args.num_envs, 2, 7), device=env.device)
        if args.track_coverage:
            env.direct.goal_visits = torch.zeros(len(bank.end), dtype=torch.int32, device=env.device)
        witness = check_reset(env.direct)
        cfg = SuperMimicConfig(
            seed=args.seed, max_iterations=args.iterations, warmup_steps=64, buffer_steps=args.buffer_steps,
            utd=.4, wm_only_iters=args.wm_only_iters, wm_hidden=(2048,) * 3, wm_window=8,
            wm_batches=10, wm_batch=2048, pi_batch=1024, horizon=args.horizon,
            max_step_reward=env.task.max_step_reward, actor_bootstrap_weight=1.,
            actor_lr=1.5e-5 if adapt_thumb_distal else 8e-5, action_noise=0., contact_weight=.20,
            contact_partial_intervals=True, contact_depth_weight=.03,
            failure_warmup_iters=max(4, args.wm_only_iters // 2), failure_clip=.35,
            critic_epochs=5, optimizer="fused-adam", control_error_features=True, isolate_wm_init=True,
            amp=False, compile=False, compile_policy=False, profile_training=True,
            save_interval=args.checkpoint_every, log_interval=max(1, min(20, args.iterations // 8)),
        )
        engine = Learner(env.task, cfg, args.output)
        engine.fast_start_fraction = args.fast_start_fraction
        engine.bank_sha256 = digest(args.bank)
        engine.reward_version = PRIOR_VERSIONS.get(args.refine_posture, "task_space_v2")
        if args.stability_v1:
            from .stability import VERSION, VERSION_V2
            engine.reward_version += "+" + (VERSION_V2 if args.stability_v2 else VERSION)
        elbow_settings = None
        if args.elbow_swivel_weight or args.elbow_envelope_weight:
            from .stability import ELBOW_TRANSPORT_VERSION, fit_elbow_envelope
            direct_task = env.direct.task
            direct_task.elbow_swivel_weight = args.elbow_swivel_weight
            direct_task.elbow_envelope_weight = args.elbow_envelope_weight
            elbow_settings = {"swivel_weight": args.elbow_swivel_weight,
                              "envelope_weight": args.elbow_envelope_weight}
            if args.elbow_envelope_weight:
                direct_task.elbow_bounds, counts = fit_elbow_envelope(
                    bank, direct_task.observe.fk, direct_task.posture.outward)
                elbow_settings.update(bounds_m=direct_task.elbow_bounds.cpu().tolist(),
                                      pose_counts=counts)
            engine.reward_version += "+" + ELBOW_TRANSPORT_VERSION
        if args.elbow_hold_drift_weight:
            from .stability import ELBOW_HOLD_ANCHOR_VERSION
            elbow_settings = {**(elbow_settings or {}), "hold_drift_weight": args.elbow_hold_drift_weight}
            engine.reward_version += "+" + ELBOW_HOLD_ANCHOR_VERSION
        if finger_reward.tracking_boost:
            engine.reward_version += f"+{REWARD_VARIANT}"
        if finger_reward.still_command_weight:
            still_version = STILL_HUBER_VERSION if finger_reward.still_command_loss == "pseudo_huber" else STILL_COMMAND_VERSION
            engine.reward_version += f"+{still_version}"
        if finger_reward.still_landmark_weight:
            engine.reward_version += f"+{STILL_LANDMARK_VERSION}"
        if finger_reward.opposition_weight:
            engine.reward_version += f"+{OPPOSITION_REWARD_VERSION}"
        if args.reference_hand_fk or bank.hand_target_mode == REFERENCE_HAND_FK:
            engine.reward_version += f"+{bank.hand_target_mode}"
        if args.positive_tracking_reward:
            engine.reward_version += f"+{POSITIVE_TRACKING_REWARD_VERSION}"
        if args.rsi_task_space_weight:
            engine.reward_version += f"+{REFERENCE_ANCHOR_VERSION}"
        if args.rsi_joint_prior_weight:
            engine.reward_version += f"+{REFERENCE_JOINT_PRIOR_VERSION}"
        if args.rsi_live_fraction:
            engine.reward_version += f"+{ON_POLICY_ANCHOR_VERSION}"
        if args.rsi_hand_weight != 1.:
            engine.reward_version += f"+{HAND_PRIORITY_ANCHOR_VERSION}"
        if args.rsi_tip_tail_weight:
            engine.reward_version += f"+{TIP_TAIL_ANCHOR_VERSION}"
        if args.wrist_execution_weight:
            engine.reward_version += f"+{WRIST_EXECUTION_REWARD_VERSION}"
        if args.rsi_arm_motion_gate:
            engine.reward_version += f"+{ARM_MOTION_GATE_VERSION}"
        if args.hand_execution_weight:
            engine.reward_version += f"+{HAND_EXECUTION_REWARD_VERSION}"
        if args.rsi_hand_motion_gate:
            engine.reward_version += f"+{HAND_MOTION_GATE_VERSION}"
        if args.rsi_hand_hardness:
            engine.reward_version += f"+{HAND_HARDNESS_VERSION}"
        if args.rsi_arm_lead or args.rsi_hand_lead:
            engine.reward_version += f"+{TEMPORAL_ANCHOR_VERSION}"
        if args.rsi_hold_fraction:
            engine.reward_version += f"+{HOLD_ANCHOR_VERSION}"
        if args.rsi_arm_actuation_weight:
            engine.reward_version += f"+{ACTUATION_ANCHOR_VERSION}"
        if args.rsi_hold_recovery_noise:
            engine.reward_version += f"+{RECOVERY_ANCHOR_VERSION}"
        if args.stationary_reset_fraction:
            engine.reward_version += "+stationary_rsi_rollouts_v1"
        if args.recorded_clock_fraction:
            engine.reward_version += "+recorded_goal_tempo_v1"
        if args.rsi_elbow_weight:
            engine.reward_version += f"+{ELBOW_ANCHOR_VERSION}"
        if args.rsi_hand_feedback_weight:
            engine.reward_version += f"+{FEEDBACK_ANCHOR_VERSION}"
        if args.rsi_relative_pose_weight != 1.:
            engine.reward_version += f"+{RELATIVE_ANCHOR_VERSION}"
        if args.rsi_hard_live_fraction or args.rsi_recovery_frames:
            engine.reward_version += f"+{LIVE_RECOVERY_ANCHOR_VERSION}"
        if args.rsi_stationary_live_fraction or args.rsi_independent_feedback:
            engine.reward_version += f"+{STILL_LIVE_ANCHOR_VERSION}"
        if args.rsi_arm_delay_lead:
            engine.reward_version += "+delay_conditioned_arm_pose_anchor_v1"
        if args.rsi_recorded_hand_lead:
            engine.reward_version += "+recorded_hand_effect_interval_v1"
        if args.rsi_hand_delay_lead:
            engine.reward_version += f"+{DELAYED_HAND_ANCHOR_VERSION}"
        if args.rsi_elbow_nullspace:
            engine.reward_version += f"+{ELBOW_NULLSPACE_VERSION}"
        if args.wrist_rotation_tail_weight:
            engine.reward_version += f"+{WRIST_ROTATION_REWARD_VERSION}"
        if args.fast_hand_weight:
            engine.reward_version += f"+{FAST_HAND_REWARD_VERSION}"
        if args.fast_start_fraction:
            engine.reward_version += "+imagination_fast_reference_sampling_v1"
        if args.rsi_hand_physical_weight:
            engine.reward_version += f"+{LIVE_EXECUTION_ANCHOR_VERSION}"
        if args.rsi_wrist_physical_weight:
            engine.reward_version += f"+{LIVE_WRIST_EXECUTION_ANCHOR_VERSION}"
        if args.rsi_live_recovery_weight:
            engine.reward_version += "+causal_live_recovery_v1"
        if args.rsi_execution_priority:
            engine.reward_version += "+elbow_and_stationary_execution_sampling_v1"
        if args.rsi_elbow_physical_weight:
            engine.reward_version += "+wrist_preserving_elbow_execution_v1"
        if args.rsi_wrist_recovery_priority:
            engine.reward_version += "+measured_wrist_recovery_v1"
        if args.rsi_arm_rowspace_weight:
            env.task.actor_reference_anchor.arm_rowspace_weight = args.rsi_arm_rowspace_weight
            engine.reward_version += "+wrist_observable_joint_prior_v1"
        if args.rsi_live_contact_weight:
            engine.reward_version += "+cross_hand_landmark_clearance_v2"
        if args.rsi_same_hand_contact_focus:
            env.task.actor_reference_anchor.same_hand_contact_focus = args.rsi_same_hand_contact_focus
            engine.reward_version += "+same_hand_contact_sampling_v1"
        if env.direct.task.thumb_ip_reference is not None:
            engine.reward_version += f"+{THUMB_IP_REWARD_VERSION}"
        with torch.no_grad():
            engine.fail_head.net[-1].bias.fill_(-4.)
        initialization = "random_no_checkpoint"
        source = args.resume or args.init
        if source:
            initial = json.loads((source.parent / "provenance.json").read_text(encoding="utf-8"))
            expected_bank = metadata.get("previous_bank_sha256") if args.init_precalibration else engine.bank_sha256
            if args.init_bank:
                expected_bank = verify_superset(args.init_bank, metadata)
            if args.init_aligned_bank:
                expected_bank = verify_aligned_bank(args.init_aligned_bank, metadata)
            if refit_source_hash:
                expected_bank = refit_source_hash
            if thumb_source_hash:
                expected_bank = thumb_source_hash
            expected_model = metadata.get("previous_model_sha256") if args.init_precalibration else digest(args.model)
            expected_usd = digest(args.init_precalibration) if args.init_precalibration else digest(args.usd)
            if args.init_precalibration and not (env.direct.task.observe.physical_wrist and expected_bank and expected_model):
                raise ValueError("Expected a calibrated bank bound to the previous native assets")
            history_frames = env.direct.task.observe.history_frames
            expand_single_frame = (
                args.init is not None and history_frames > 1
                and env.direct.task.observe.use_command_error
                and initial.get("observation_version") == FOUR_POINT_OBSERVATION
            )
            for key, expected in {
                "task_id": TASK_ID, "bank_sha256": expected_bank, "actor_architecture": ACTOR_ARCHITECTURE,
                "observation_version": FOUR_POINT_OBSERVATION if expand_single_frame else args.observation_version,
                "model_sha256": expected_model,
                "usd_sha256": expected_usd, "robust": args.robust,
            }.items():
                if initial.get(key) != expected:
                    raise ValueError(f"Initializer differs in {key}")
            keep_critic = (args.refine_posture in ARM_TEACHER_PROFILES
                           and (not args.resume or initial.get("train_only_arm_teacher", False) or args.arm_teacher))
            if args.stability_v1:
                keep_critic = False  # Shared-policy ablation adds no frozen arm teacher.
            teacher_state = None
            if args.resume:
                if initial.get("rsi_same_hand_contact_focus", 0.) != args.rsi_same_hand_contact_focus:
                    raise ValueError("Resume requires the same closing-gesture contact objective")
                if initial.get("rsi_arm_rowspace_weight", 0.) != args.rsi_arm_rowspace_weight:
                    raise ValueError("Resume requires the same wrist observable prior")
                if initial.get("elbow_continuity") != elbow_settings:
                    raise ValueError("Resume requires the same elbow geometry objective")
                if (initial.get("hand_target_mode", MAPPED_HAND) != bank.hand_target_mode
                        or initial.get("mapped_hand_clip_fraction", 0.) != args.mapped_hand_clip_fraction):
                    raise ValueError("Resume requires the same hand target mode; use --init when changing targets")
                prior_finger_reward = initial.get("finger_reward", {})
                opposition_options = ("opposition_weight", "opposition_wrist_position_relaxation",
                                      "opposition_wrist_rotation_relaxation")
                if (prior_finger_reward.get("opposition_wrist_relaxation", 0.)
                        or any(prior_finger_reward.get(k, 0.) != getattr(finger_reward, k) for k in opposition_options)
                        or initial.get("opposition_sampling") != bank.opposition_sampling):
                    raise ValueError("Resume requires the same opposition task; use --init to change it")
                if bool(initial.get("adapt_thumb_distal", False)) != adapt_thumb_distal:
                    raise ValueError("Resume requires the same actor update scope")
                if initial.get("cross_trajectory_switch") != env.task.reference_switch.config.to_dict():
                    raise ValueError("Resume requires the same switch contract; use --init to change task sampling")
                allowed_changes = {"max_iterations", "save_interval", "log_interval"}
                settings = json.loads(json.dumps(cfg.to_dict()))
                if initial["num_envs"] != args.num_envs or any(
                        settings[k] != initial["config"][k] for k in settings.keys() - allowed_changes):
                    raise ValueError("Resume requires the preceding SuperMimic settings")
                engine.load(args.resume)
                if engine.iteration >= cfg.max_iterations or engine.iteration % 2:
                    raise ValueError("Resume needs an earlier checkpoint at the collection cadence boundary")
                initialization = "optimizer_resume_replay_refilled"
            else:
                payload = torch.load(source, map_location="cpu", weights_only=False)
                modules = ("actor", "wm", "fail_head", "critic") if keep_critic or args.stability_v1 else ("actor", "wm", "fail_head")
                if args.init_iteration and payload.get("iteration") != args.init_iteration:
                    raise ValueError("Warm-start iteration must match the checkpoint")
                for key in modules:
                    weights = payload["modules"][key]
                    if key == "actor" and expand_single_frame:
                        from .temporal import expand_actor_weights
                        weights = expand_actor_weights(weights, engine.actor.state_dict(), history_frames)
                    engine.modules()[key].load_state_dict(weights)
                teacher_state = payload["modules"]["actor"] if keep_critic else None
                del payload
                initialization = ("shared_actor_critic_physics_weights_optimizer_reset" if args.stability_v1 else
                                  "actor_critic_and_physics_from_checkpoint" if keep_critic
                                  else "actor_and_physical_models_from_checkpoint")
            if keep_critic:
                teacher = env.task.make_actor().to(env.device)
                teacher_file = args.arm_teacher or source
                if args.arm_teacher:
                    teacher_info = torch.load(teacher_file, map_location="cpu", weights_only=True)
                    if (teacher_info["task_id"] != TASK_ID or teacher_info["actor_architecture"] != ACTOR_ARCHITECTURE
                            or teacher_info["observation_version"] != args.observation_version
                            or teacher_info["model_sha256"] != digest(args.model)):
                        raise ValueError("Arm teacher geometry and actor must match the current platform")
                    teacher.net.load_state_dict(teacher_info["actor"])
                else:
                    teacher.load_state_dict(engine.actor.state_dict() if teacher_state is None else teacher_state)
                env.task.use_arm_teacher(teacher.eval().requires_grad_(False), model, metadata["joint_names"])
        if args.init_iteration:
            if not args.init or args.resume or args.init_iteration % 2:
                raise ValueError("init-iteration requires --init and an even collection boundary")
            engine.iteration = args.init_iteration
        adaptation_masks = restrict_thumb_distal_updates(engine.actor, model) if adapt_thumb_distal else None
        starting = {name: p.detach().clone() for name, p in engine.actor.named_parameters()}
        start_iteration, start_steps = engine.iteration, engine.sim_steps
        optimizer_start = optimizer_steps(engine)
        provenance = {
            "task_id": TASK_ID, "method": "supermimic", "initialization": initialization,
            "init_checkpoint": str(source.resolve()) if source else None,
            "init_sha256": digest(source) if source else None,
            "init_precalibration": bool(args.init_precalibration),
            "init_start_realignment": bool(args.init_aligned_bank),
            "init_refit_bank": str(refit_manifest.resolve()) if refit_manifest else None,
            "thumb_projection": metadata.get("thumb_projection"),
            "adapt_thumb_distal": adapt_thumb_distal,
            "hand_target_mode": bank.hand_target_mode,
            "mapped_hand_clip_fraction": args.mapped_hand_clip_fraction,
            "opposition_sampling": bank.opposition_sampling,
            "training_hand_mix": bank.training_hand_mix,
            "stability_v1": args.stability_v1,
            "elbow_continuity": elbow_settings,
            "stability_revision": 2 if args.stability_v2 else 1 if args.stability_v1 else 0,
            "init_bank_sha256": expected_bank if source else None,
            "start_iteration": start_iteration,
            "optimizer_steps_at_start": optimizer_start,
            "arm_teacher_sha256": digest(teacher_file) if env.task.arm_teacher else None,
            "reward_version": engine.reward_version, "posture_profile": args.refine_posture,
            "positive_tracking_reward": args.positive_tracking_reward,
            "rsi_task_space_weight": args.rsi_task_space_weight,
            "rsi_joint_prior_weight": args.rsi_joint_prior_weight,
            "rsi_live_fraction": args.rsi_live_fraction, "rsi_hand_weight": args.rsi_hand_weight,
            "rsi_tip_tail_weight": args.rsi_tip_tail_weight,
            "wrist_execution_weight": args.wrist_execution_weight, "rsi_arm_motion_gate": args.rsi_arm_motion_gate,
            "hand_execution_weight": args.hand_execution_weight, "rsi_hand_motion_gate": args.rsi_hand_motion_gate,
            "rsi_hand_hardness": args.rsi_hand_hardness,
            "rsi_arm_lead": args.rsi_arm_lead, "rsi_hand_lead": args.rsi_hand_lead,
            "rsi_hold_fraction": args.rsi_hold_fraction,
            "rsi_arm_actuation_weight": args.rsi_arm_actuation_weight,
            "rsi_hold_recovery_noise": args.rsi_hold_recovery_noise,
            "stationary_reset_fraction": args.stationary_reset_fraction,
            "recorded_clock_fraction": args.recorded_clock_fraction,
            "rsi_elbow_weight": args.rsi_elbow_weight,
            "rsi_hand_feedback_weight": args.rsi_hand_feedback_weight,
            "rsi_relative_pose_weight": args.rsi_relative_pose_weight,
            "rsi_arm_rowspace_weight": args.rsi_arm_rowspace_weight,
            "rsi_hard_live_fraction": args.rsi_hard_live_fraction,
            "rsi_recovery_frames": args.rsi_recovery_frames,
            "rsi_stationary_live_fraction": args.rsi_stationary_live_fraction,
            "rsi_independent_feedback": args.rsi_independent_feedback,
            "rsi_hand_delay_lead": args.rsi_hand_delay_lead,
            "rsi_arm_delay_lead": args.rsi_arm_delay_lead,
            "rsi_recorded_hand_lead": args.rsi_recorded_hand_lead,
            "rsi_elbow_nullspace": args.rsi_elbow_nullspace,
            "wrist_rotation_tail_weight": args.wrist_rotation_tail_weight,
            "fast_hand_weight": args.fast_hand_weight,
            "fast_start_fraction": args.fast_start_fraction,
            "rsi_hand_physical_weight": args.rsi_hand_physical_weight,
            "rsi_wrist_physical_weight": args.rsi_wrist_physical_weight,
            "rsi_live_recovery_weight": args.rsi_live_recovery_weight,
            "rsi_execution_priority": args.rsi_execution_priority,
            "rsi_wrist_recovery_priority": args.rsi_wrist_recovery_priority,
            "rsi_elbow_physical_weight": args.rsi_elbow_physical_weight,
            "rsi_live_contact_weight": args.rsi_live_contact_weight,
            "rsi_same_hand_contact_focus": args.rsi_same_hand_contact_focus,
            "stationary_steps": ([250, 500] if args.stability_v1 else STATIONARY_STEPS) if args.stationary_reset_fraction else 0,
            "causal_goal": args.refine_posture is not None,
            "train_only_arm_teacher": env.task.arm_teacher is not None,
            "cross_trajectory_switch": env.task.reference_switch.config.to_dict(),
            "finger_reward": finger_reward.to_dict(),
            "actor_architecture": ACTOR_ARCHITECTURE, "observation_version": args.observation_version,
            "observation_features": env.task.policy_dim,
            "bank": str(args.bank.resolve()), "bank_sha256": engine.bank_sha256,
            "model": str(args.model.resolve()), "model_sha256": digest(args.model),
            "usd": str(args.usd.resolve()), "usd_sha256": digest(args.usd),
            "num_envs": args.num_envs, "robust": args.robust, "seed": args.seed,
            "control_dt": DT, "q_default": metadata["home"], "joint_names": metadata["joint_names"],
            "action_scale": env.task.direct.command.scale.tolist(), "sampling_ramp_updates": SAMPLING_RAMP_UPDATES,
            "wrist_calibration": metadata.get("wrist_calibration"), "train_source_weights": bank.source_weights,
            "position_rate_rad_s": metadata.get("position_rate_rad_s"),
            "newton": witness, "config": cfg.to_dict(),
        }
        write_json(args.output / "provenance.json", provenance)
        if source:
            engine.save_actor(args.output / "model_start_actor.pt")
        print("S0_1_SUPERMIMIC_NEWTON", json.dumps(witness), flush=True)
        start = time.perf_counter()
        engine.learn()
        steps = optimizer_steps(engine)
        updates = {name: value - optimizer_start[name] for name, value in steps.items()}
        change = max(float((p.detach() - starting[name]).abs().max())
                     for name, p in engine.actor.named_parameters())
        if adaptation_masks is not None:
            for name, p in engine.actor.named_parameters():
                mask = adaptation_masks.get(name)
                unchanged, before = (p, starting[name]) if mask is None else (p[~mask], starting[name][~mask])
                if not torch.equal(unchanged, before):
                    raise RuntimeError(f"Thumb adaptation changed frozen actor parameters: {name}")
        expected = start_steps + args.num_envs * (cfg.warmup_steps + collection_steps(args.iterations, cfg.utd)
                                                  - collection_steps(start_iteration, cfg.utd))
        if not all(updates.values()) or not math.isfinite(change) or change <= 0 or engine.sim_steps != expected:
            raise RuntimeError(f"SuperMimic training proof failed: {updates}, {change}, {engine.sim_steps}/{expected}")
        proof = dict(iterations=args.iterations, start_iteration=start_iteration,
                     additional_iterations=args.iterations - start_iteration, sim_transitions=engine.sim_steps,
                     additional_sim_transitions=engine.sim_steps - start_steps, num_envs=args.num_envs,
                     wall_seconds=time.perf_counter() - start, actor_parameter_max_change=change,
                     optimizer_steps=steps, additional_optimizer_steps=updates,
                     peak_torch_bytes=torch.cuda.max_memory_allocated(),
                     replay_bytes=sum(v.numel() * v.element_size() for v in engine.buffer.data.values()),
                     initialization=provenance["initialization"], adapt_thumb_distal=adapt_thumb_distal)
        write_json(args.output / "training_proof.json", proof)
        print("S0_1_SUPERMIMIC_PROOF", json.dumps(proof), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("bank", "model", "usd", "output"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--wm-only-iters", type=int, default=STANDARD.wm_only_iters)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--buffer-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=49)
    p.add_argument("--robust", action="store_true")
    p.add_argument("--refine-posture", nargs="?", const="balanced", choices=tuple(PRIOR_VERSIONS))
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--init", type=Path, help="Initialize the actor and physics predictors from a full checkpoint")
    initialization.add_argument("--resume", type=Path, help="Resume a full checkpoint and refill the replay buffer")
    bank_change = p.add_mutually_exclusive_group()
    bank_change.add_argument("--init-bank", type=Path, help="Verify the old library as a subset of the current one")
    bank_change.add_argument("--init-aligned-bank", type=Path,
                             help="Verify the previous library when switching to per-side START targets")
    bank_change.add_argument("--init-refit-bank", type=Path, metavar="ARCHIVED_MANIFEST",
                             help="Verify unchanged goals before initializing with refitted RSI states")
    bank_change.add_argument("--init-thumb-bank", type=Path, metavar="ARCHIVED_MANIFEST",
                             help="Verify a thumb-only goal projection before initializing from that bank")
    p.add_argument("--arm-teacher", type=Path, help="An explicit frozen actor-only teacher checkpoint")
    p.add_argument("--adapt-thumb-distal", action="store_true",
                   help="Update only the MCP and IP task-space output weights of an existing actor")
    p.add_argument("--track-coverage", action="store_true",
                   help="Count physical reference visits and actor imagination samples per frame")
    p.add_argument("--init-precalibration", type=Path, metavar="OLD_USD",
                   help="Validate a warm start against the previous bank and Newton asset")
    add_switch_arguments(p)
    add_finger_reward_arguments(p)
    p.add_argument("--observation-version", choices=(OBSERVATION_VERSION, *FOUR_POINT_HISTORY_FRAMES),
                   help="Defaults to four-point current-frame observations for fresh training")
    p.add_argument("--rsi-task-space-weight", type=float, default=0.,
                   help="Task-space loss at perturbed reference-initialized states")
    p.add_argument("--rsi-joint-prior-weight", type=float, default=0.,
                   help="Use geometrically valid reference commands as an offline action prior")
    p.add_argument("--rsi-live-fraction", type=float, default=0.,
                   help="Fraction of auxiliary samples drawn from the live simulation states")
    p.add_argument("--rsi-hand-weight", type=float, default=1.,
                   help="Hand position and feasible hand-command weight within the auxiliary loss")
    p.add_argument("--rsi-tip-tail-weight", type=float, default=0.,
                   help="Extra auxiliary weight on larger fingertip placement errors across all fingers")
    p.add_argument("--wrist-execution-weight", type=float, default=0.,
                   help="Share of task reward for actual bilateral wrist and relative pose tracking")
    p.add_argument("--rsi-arm-motion-gate", type=float, default=0.,
                   help="Wrist speed scale in m/s for downweighting dynamic arm reference priors")
    p.add_argument("--hand-execution-weight", type=float, default=0.,
                   help="Share of task reward for the measured finger landmark and fingertip tracking")
    p.add_argument("--rsi-hand-motion-gate", type=float, default=0.,
                   help="Finger target speed scale in m/s for downweighting dynamic command priors")
    p.add_argument("--rsi-hand-hardness", type=float, default=0.,
                   help="Within-source weight for samples with larger requested fingertip errors")
    p.add_argument("--rsi-arm-lead", type=int, default=0,
                   help="Training-only arm command supervision lead in source frames")
    p.add_argument("--rsi-hand-lead", type=int, default=0,
                   help="Training-only hand command supervision lead in source frames")
    p.add_argument("--rsi-hold-fraction", type=float, default=0.,
                   help="Fraction of synthetic RSI supervision using a stationary recorded goal")
    p.add_argument("--rsi-arm-actuation-weight", type=float, default=0.,
                   help="Use the measured steady arm actuator offset in the live auxiliary loss")
    p.add_argument("--rsi-hold-recovery-noise", type=float, default=0.,
                   help="Hand position noise in radians for half the stationary RSI examples")
    p.add_argument("--stability-v2", action="store_true", help="Observable nullspace continuity; no archived arm-pose attraction")
    p.add_argument("--stability-v1", action="store_true",
                   help="Reference FK hand goals, causal static elbow supervision and 5-10 s holds")
    p.add_argument("--init-iteration", type=int, default=0,
                   help="Keep the source iteration/LR schedule for an explicit weight warm start")
    p.add_argument("--stationary-reset-fraction", type=float, default=0.,
                   help="Probability of an initial stationary goal at an RSI reset during training")
    p.add_argument("--recorded-clock-fraction", type=float, default=0.,
                   help="Fraction of eligible moving RSI resets following the recording timestamp map")
    p.add_argument("--rsi-elbow-weight", type=float, default=0.,
                   help="Auxiliary symmetric elbow tracking weight in wrist-compatible poses")
    p.add_argument("--rsi-hand-feedback-weight", type=float, default=0.,
                   help="Penalize amplification of small hand feedback changes in stationary RSI examples")
    p.add_argument("--rsi-relative-pose-weight", type=float, default=1.,
                   help="Relative wrist position and rotation multiplier for the task-space auxiliary loss")
    p.add_argument("--rsi-hard-live-fraction", type=float, default=0.,
                   help="Fraction of live auxiliary examples selected by measured hand tracking difficulty")
    p.add_argument("--rsi-recovery-frames", type=int, default=0,
                   help="Sample stationary recovery states from nearby verified frames within the same clip")
    p.add_argument("--rsi-stationary-live-fraction", type=float, default=0.,
                   help="Fraction of hard live examples from stationary reference intervals")
    p.add_argument("--rsi-independent-feedback", action="store_true",
                   help="Include independent changes to measured position and previous command in stationary supervision")
    p.add_argument("--rsi-arm-delay-lead", type=int, default=0,
                   help="Additional training-only arm pose lead in samples with actuator delay")
    p.add_argument("--rsi-hand-delay-lead", type=int, default=0,
                   help="Additional hand supervision lead for live samples with actuator delay")
    p.add_argument("--rsi-elbow-nullspace", action="store_true",
                   help="Use the wrist-preserving joint direction for auxiliary elbow supervision")
    p.add_argument("--wrist-rotation-tail-weight", type=float, default=0.,
                   help="Extra actual bilateral and relative wrist orientation penalty")
    p.add_argument("--fast-hand-weight", type=float, default=0.,
                   help="Extra actual four-point and fingertip tracking weight during fast motion")
    p.add_argument("--fast-start-fraction", type=float, default=0.,
                   help="Fraction of imagination starts emphasizing fast hand motion and actual tracking error")
    p.add_argument("--rsi-hand-physical-weight", type=float, default=0.,
                   help="Fit predicted finger tracking at the interval where a live action takes effect")
    p.add_argument("--rsi-wrist-physical-weight", type=float, default=0.,
                   help="Fit predicted bilateral and relative wrist poses at the effective command interval")
    p.add_argument("--rsi-live-recovery-weight", type=float, default=0.,
                   help="Task-space execution loss from three closed-loop intervals starting at live RSI states")
    p.add_argument("--rsi-elbow-physical-weight", type=float, default=0.,
                   help="Elbow execution loss after local wrist-preserving joint projection")
    p.add_argument("--rsi-live-contact-weight", type=float, default=0.,
                   help="Learn clearance for nearby finger goals and avoid predicted hard contact in recovery")
    p.add_argument("--rsi-recorded-hand-lead", type=int, default=0,
                   help="Additional hand supervision lead on recording-clock rollout samples")
    p.add_argument("--rsi-execution-priority", type=float, default=0.,
                   help="Weight elbow drift and stationary fingertip errors in physically matched live samples")
    p.add_argument("--rsi-wrist-recovery-priority", type=float, default=0.,
                   help="Prioritize measured bilateral wrist displacement and separation in live recovery")
    p.add_argument("--positive-tracking-reward", action="store_true",
                   help="Use a positive tracking score before applying collision and terminal penalties")
    p.add_argument("--reference-hand-fk", action="store_true",
                   help="Build four wrist-local landmarks per finger from the active library's reference pose")
    p.add_argument("--mapped-hand-clip-fraction", type=float, default=0.,
                   help="Fraction of each training source's clips using online-style mapped hand landmarks")
    p.add_argument("--opposition-sample-fraction", type=float, default=0.,
                   help="Within-source RSI draws reserved for near-opposition FK references")
    p.add_argument("--elbow-swivel-weight", type=float, default=0.,
                   help="Penalize tangential elbow swivel in a wrist-ray frame")
    p.add_argument("--elbow-envelope-weight", type=float, default=0.,
                   help="Apply a weak lateral envelope fitted from training poses")
    p.add_argument("--elbow-hold-drift-weight", type=float, default=0.,
                   help="Penalize accumulated elbow swivel drift during long aligned wrist holds")
    p.add_argument("--rsi-arm-rowspace-weight", type=float, default=0.,
                   help="Use a wrist-only joint prior near a matching RSI pose")
    p.add_argument("--rsi-same-hand-contact-focus", type=float, default=0.,
                   help="Increase physical recovery coverage of closing gestures")
    args = p.parse_args()
    if not 0. <= args.rsi_same_hand_contact_focus <= 3.:
        p.error("Same-hand contact focus must be in [0,3]")
    if args.rsi_same_hand_contact_focus and not args.rsi_live_contact_weight:
        p.error("Closing-gesture focus requires the physical contact loss")
    if not 0. <= args.rsi_arm_rowspace_weight <= 4.:
        p.error("RSI arm rowspace weight must be in [0,4]")
    if args.rsi_arm_rowspace_weight and not (args.stability_v2 and args.rsi_joint_prior_weight):
        p.error("The wrist rowspace prior uses the v2 redundancy-aware RSI joint loss")
    elbow_weights = (args.elbow_swivel_weight, args.elbow_envelope_weight, args.elbow_hold_drift_weight)
    if not all(0. <= w <= 1. for w in elbow_weights):
        p.error("Elbow objective weights must be in [0,1]")
    if any(elbow_weights) and (
            not args.stability_v2 or args.refine_posture != "natural_point_tracking"):
        p.error("Elbow continuity uses the v9 natural four-point stability task")
    if args.elbow_hold_drift_weight and not args.stationary_reset_fraction:
        p.error("Elbow hold drift requires stationary reset episodes")
    if not 0. <= args.mapped_hand_clip_fraction <= 1. or (args.mapped_hand_clip_fraction and not args.reference_hand_fk):
        p.error("Mapped hand mixtures require --reference-hand-fk and a fraction in [0,1]")
    if not 0. <= args.opposition_sample_fraction <= 1. or (args.opposition_sample_fraction and (
            not args.reference_hand_fk or args.mapped_hand_clip_fraction)):
        p.error("Opposition sampling requires FK hand targets and a fraction in [0,1]")
    args.stability_v1 = args.stability_v1 or args.stability_v2
    checkpoint = args.resume or args.init
    if args.observation_version is None:
        args.observation_version = FOUR_POINT_OBSERVATION
        if checkpoint is not None:
            prior = json.loads((checkpoint.parent / "provenance.json").read_text(encoding="utf-8"))
            args.observation_version = prior.get("observation_version", OBSERVATION_VERSION)
    if (checkpoint is None and args.observation_version in (
            FOUR_POINT_OBSERVATION, HISTORY5_OBSERVATION, HISTORY5_NO_COMMAND_ERROR_OBSERVATION,
            HISTORY5_ABSOLUTE_COMMAND_OBSERVATION, HISTORY5_ABSOLUTE_OBSERVATION)
            and args.refine_posture is None):
        args.refine_posture = "natural_point_tracking"
    try:
        switch_config(args)
        finger_reward_config(args)
    except ValueError as error:
        p.error(str(error))
    if not math.isfinite(args.rsi_task_space_weight) or args.rsi_task_space_weight < 0.:
        p.error("RSI task-space weight must be finite and nonnegative")
    if not math.isfinite(args.rsi_joint_prior_weight) or args.rsi_joint_prior_weight < 0.:
        p.error("RSI joint prior weight must be finite and nonnegative")
    if (args.rsi_task_space_weight or args.rsi_joint_prior_weight) and args.observation_version not in FOUR_POINT_HISTORY_FRAMES:
        p.error("RSI task-space supervision requires four-point current-frame observations")
    if not 0 <= args.rsi_live_fraction < 1:
        p.error("The live auxiliary sample fraction must be in [0,1)")
    if args.rsi_live_fraction and not (args.rsi_task_space_weight or args.rsi_joint_prior_weight):
        p.error("Live auxiliary samples require a task-space or joint-prior weight")
    if not math.isfinite(args.rsi_hand_weight) or args.rsi_hand_weight <= 0.:
        p.error("RSI hand weight must be finite and positive")
    if not math.isfinite(args.rsi_tip_tail_weight) or args.rsi_tip_tail_weight < 0.:
        p.error("RSI fingertip tail weight must be finite and nonnegative")
    if args.rsi_tip_tail_weight and not args.rsi_task_space_weight:
        p.error("The auxiliary fingertip tail requires a task-space anchor weight")
    if not 0. <= args.wrist_execution_weight <= 1.:
        p.error("Wrist execution reward weight must be in [0,1]")
    if not math.isfinite(args.rsi_arm_motion_gate) or args.rsi_arm_motion_gate < 0.:
        p.error("Arm reference speed scale must be finite and nonnegative")
    if not 0. <= args.hand_execution_weight <= 1. - args.wrist_execution_weight:
        p.error("Hand and wrist execution weights must be nonnegative and sum to at most one")
    if not math.isfinite(args.rsi_hand_motion_gate) or args.rsi_hand_motion_gate < 0.:
        p.error("Finger reference speed scale must be finite and nonnegative")
    if not math.isfinite(args.rsi_hand_hardness) or args.rsi_hand_hardness < 0.:
        p.error("RSI hand hardness weight must be finite and nonnegative")
    if min(args.rsi_arm_lead, args.rsi_hand_lead) < 0:
        p.error("Reference command supervision leads must be nonnegative")
    if (args.rsi_arm_lead or args.rsi_hand_lead) and not (args.rsi_task_space_weight or args.rsi_joint_prior_weight):
        p.error("Command supervision leads require an auxiliary reference loss")
    if not 0 <= args.rsi_hold_fraction <= 1.:
        p.error("Stationary RSI sample fraction must be in [0,1]")
    if args.rsi_hold_fraction and not (args.rsi_task_space_weight or args.rsi_joint_prior_weight):
        p.error("Stationary RSI samples require an auxiliary reference loss")
    if not 0 <= args.rsi_arm_actuation_weight <= 1.:
        p.error("The arm actuation weight must be in [0,1]")
    if args.rsi_arm_actuation_weight and not (args.rsi_live_fraction and args.rsi_task_space_weight):
        p.error("The arm actuation weight requires live task-space auxiliary samples")
    if not math.isfinite(args.rsi_hold_recovery_noise) or args.rsi_hold_recovery_noise < 0.:
        p.error("The stationary hand recovery noise must be finite and nonnegative")
    if args.rsi_hold_recovery_noise and not args.rsi_hold_fraction:
        p.error("Stationary hand recovery noise requires stationary RSI samples")
    if not 0 <= args.stationary_reset_fraction <= 1.:
        p.error("The stationary reset fraction must be in [0,1]")
    if not math.isfinite(args.rsi_elbow_weight) or args.rsi_elbow_weight < 0.:
        p.error("The elbow supervision weight must be finite and nonnegative")
    if args.rsi_elbow_weight and not (args.rsi_task_space_weight and args.refine_posture == "natural_point_tracking"):
        p.error("Elbow supervision requires the four-point posture and task-space auxiliary loss")
    if not math.isfinite(args.rsi_hand_feedback_weight) or args.rsi_hand_feedback_weight < 0.:
        p.error("The stationary hand feedback weight must be finite and nonnegative")
    if args.rsi_hand_feedback_weight and not ((args.rsi_hold_recovery_noise or args.rsi_recovery_frames)
                                               and args.rsi_task_space_weight):
        p.error("Hand feedback supervision requires stationary recovery examples and a task-space auxiliary loss")
    if not math.isfinite(args.rsi_relative_pose_weight) or args.rsi_relative_pose_weight <= 0.:
        p.error("The relative-pose supervision weight must be finite and positive")
    if args.rsi_relative_pose_weight != 1. and not args.rsi_task_space_weight:
        p.error("The relative-pose multiplier requires a task-space auxiliary loss")
    if not 0. <= args.rsi_hard_live_fraction <= 1.:
        p.error("The hard live sample fraction must be in [0,1]")
    if args.rsi_hard_live_fraction and not args.rsi_live_fraction:
        p.error("Hard live samples require live auxiliary examples")
    if args.rsi_recovery_frames < 0 or (args.rsi_recovery_frames and not args.rsi_hold_fraction):
        p.error("Nearby recovery frames require a nonnegative offset and stationary RSI examples")
    if args.rsi_recovery_frames and args.rsi_hold_recovery_noise:
        p.error("Choose recorded recovery frames or synthetic recovery noise")
    if not 0. <= args.rsi_stationary_live_fraction <= 1.:
        p.error("The stationary fraction of difficult live examples must be in [0,1]")
    if args.rsi_stationary_live_fraction and not (args.rsi_hard_live_fraction and args.stationary_reset_fraction):
        p.error("Hard stationary examples require stationary rollouts and hard live sampling")
    if args.rsi_independent_feedback and not args.rsi_hand_feedback_weight:
        p.error("Independent feedback changes require feedback supervision")
    if args.rsi_recorded_hand_lead < 0 or (args.rsi_recorded_hand_lead and not (
            args.recorded_clock_fraction and args.rsi_live_fraction and args.rsi_hand_lead)):
        p.error("Extra hand lead on the recording clock requires live and recorded-clock examples")
    if args.rsi_arm_delay_lead < 0 or (args.rsi_arm_delay_lead and not (args.rsi_live_fraction and args.rsi_arm_lead)):
        p.error("Extra arm lead for actuator delay requires live temporal supervision")
    if args.rsi_hand_delay_lead < 0 or (args.rsi_hand_delay_lead and not (args.rsi_live_fraction and args.rsi_hand_lead)):
        p.error("Extra hand lead for actuator delay requires live temporal supervision and a nonnegative frame count")
    if args.rsi_elbow_nullspace and not args.rsi_elbow_weight:
        p.error("Elbow null-space supervision requires an elbow tracking weight")
    if not math.isfinite(args.wrist_rotation_tail_weight) or args.wrist_rotation_tail_weight < 0.:
        p.error("The wrist rotation tail weight must be finite and nonnegative")
    if not math.isfinite(args.fast_hand_weight) or args.fast_hand_weight < 0.:
        p.error("The fast hand tracking weight must be finite and nonnegative")
    if not 0. <= args.recorded_clock_fraction <= 1.:
        p.error("The recorded clock fraction must be in [0,1]")
    if not 0. <= args.fast_start_fraction <= 1.:
        p.error("The fast imagination start fraction must be in [0,1]")
    if not math.isfinite(args.rsi_hand_physical_weight) or args.rsi_hand_physical_weight < 0.:
        p.error("The live execution weight must be finite and nonnegative")
    if not math.isfinite(args.rsi_wrist_physical_weight) or args.rsi_wrist_physical_weight < 0.:
        p.error("The live wrist execution weight must be finite and nonnegative")
    if not math.isfinite(args.rsi_live_recovery_weight) or args.rsi_live_recovery_weight < 0.:
        p.error("The live recovery weight must be finite and nonnegative")
    if not math.isfinite(args.rsi_elbow_physical_weight) or args.rsi_elbow_physical_weight < 0.:
        p.error("The elbow execution weight must be finite and nonnegative")
    if args.rsi_elbow_physical_weight and not (args.rsi_live_recovery_weight and args.rsi_elbow_nullspace):
        p.error("Elbow execution supervision requires live recovery and the wrist-preserving elbow projection")
    if not math.isfinite(args.rsi_live_contact_weight) or args.rsi_live_contact_weight < 0.:
        p.error("The live contact weight must be finite and nonnegative")
    if not math.isfinite(args.rsi_execution_priority) or args.rsi_execution_priority < 0.:
        p.error("Execution priority must be finite and nonnegative")
    if args.rsi_execution_priority and not (args.rsi_live_recovery_weight and args.rsi_elbow_physical_weight):
        p.error("Execution priority requires live recovery and elbow execution supervision")
    if not math.isfinite(args.rsi_wrist_recovery_priority) or args.rsi_wrist_recovery_priority < 0.:
        p.error("Wrist recovery priority must be finite and nonnegative")
    if args.rsi_wrist_recovery_priority and not args.rsi_execution_priority:
        p.error("Wrist recovery priority requires execution priority")
    if args.rsi_live_contact_weight and not args.rsi_live_recovery_weight:
        p.error("The live contact cost requires the closed-loop recovery objective")
    if (args.rsi_hand_physical_weight or args.rsi_wrist_physical_weight or args.rsi_live_recovery_weight) and not (
            args.rsi_live_fraction and args.rsi_task_space_weight and args.refine_posture == "natural_point_tracking"):
        p.error("Live execution supervision requires live four-point task-space samples")
    if not 0 <= args.wm_only_iters < args.iterations:
        p.error("Include actor updates after the world-model-only stage")
    if args.arm_teacher and (args.refine_posture not in ARM_TEACHER_PROFILES or not (args.init or args.resume)):
        p.error("An explicit arm teacher requires a checkpoint and a teacher-enabled posture profile")
    if args.init_precalibration and args.init is None:
        p.error("--init-precalibration requires --init")
    if args.init_refit_bank and (args.init is None or args.init_precalibration):
        p.error("--init-refit-bank requires --init and the same calibration")
    if args.adapt_thumb_distal and not (args.init or args.resume):
        p.error("--adapt-thumb-distal requires an existing actor checkpoint")
    if args.init_thumb_bank and (args.init is None or args.init_precalibration):
        p.error("--init-thumb-bank requires --init and the same calibration")
    if (args.init_bank or args.init_aligned_bank) and (args.init_precalibration or not (args.init or args.resume)):
        p.error("A changed reference bank needs an initializer and a matching control contract")
    if min(args.num_envs, args.checkpoint_every) < 1 or not 1 <= args.horizon <= 32 or args.buffer_steps < 64:
        p.error("Expected positive environment and save counts, horizon 1–32, and at least 64 buffer steps")
    train(args)


if __name__ == "__main__":
    main()
