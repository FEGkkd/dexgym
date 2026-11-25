"""Shared observation, command and reward for Newton and SuperMimic."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from ..wholebody_s0_1 import DT
from ..wholebody_s0_1.cross_trajectory import SwitchConfig
from ..wholebody_s0_1.supermimic import ImaginationEnv, ImaginationTask
from ..wholebody_s0_1.task import DirectEnv, RECOVERY_DISTANCE_M
from ..wholebody_s0_2.interface import world
from . import ACTION_DIM, ARM_RESIDUAL_DEG, CONTRACT, HISTORY, TARGET_MM, TASK_ID


class ArmCommand(nn.Module):
    """Project only the 14 arm requests. Copy all other columns from ref."""

    def __init__(self, model, metadata, command):
        super().__init__()
        names = metadata["joint_names"]
        arms = [names.index(f"Joint{i}_{side}") for side in "LR" for i in range(1, 8)]
        if sorted(arms) != command.arm.nonzero().flatten().tolist():
            raise ValueError("Expected seven named joints on each arm")
        self.register_buffer("arm", torch.tensor(arms, dtype=torch.long, device=command.home.device))
        self.register_buffer("hand", (~command.arm).nonzero().flatten())
        joints = [model.joint(names[i]) for i in arms]
        model_bounds = command.home.new_tensor([joint.range.tolist() for joint in joints])
        self.register_buffer("lower", torch.maximum(command.lower[self.arm], model_bounds[:, 0]))
        self.register_buffer("upper", torch.minimum(command.upper[self.arm], model_bounds[:, 1]))
        self.radius = math.radians(ARM_RESIDUAL_DEG)
        self.coupling_limit = math.radians(110.5)
        self.register_buffer("coupling_weights", command.home.new_tensor([1.025, 1.0]))

    def bounds(self, arm_ref):
        return (torch.maximum(self.lower, arm_ref - self.radius),
                torch.minimum(self.upper, arm_ref + self.radius))

    def coupled_cost(self, paired):
        return (paired.abs() * self.coupling_weights).sum(-1, keepdim=True)

    @torch.no_grad()
    def check_bank(self, references):
        for ref in references.split(65536):
            lower, upper = self.bounds(ref[:, self.arm])
            safe = torch.zeros_like(lower).clamp(lower, upper).reshape(-1, 2, 7)
            if (not torch.isfinite(ref).all() or (lower > upper).any()
                    or (self.coupled_cost(safe[..., 5:]) > self.coupling_limit + 1e-6).any()):
                raise ValueError("Bank contains a frame with no legal arm command inside its ±8° bounds")

    def forward(self, action, ref):
        if action.shape != (len(ref), ACTION_DIM):
            raise ValueError("Expected a batch of 14 arm residuals")
        arm_ref = ref[:, self.arm]
        low, high = self.bounds(arm_ref)
        arm = (arm_ref + self.radius * action.clamp(-1, 1)).clamp(low, high).reshape(-1, 2, 7)
        low_pair, high_pair = (limit.reshape(-1, 2, 7)[..., 5:] for limit in (low, high))
        closest = torch.zeros_like(low_pair).clamp(low_pair, high_pair)
        proposed = arm[..., 5:]
        floor = self.coupled_cost(closest)
        excess = self.coupled_cost(proposed) - floor
        alpha = ((self.coupling_limit - floor) / excess.clamp_min(1e-8)).clamp(0, 1)
        pair = torch.lerp(closest, proposed, alpha)
        command = torch.cat((arm[..., :5], pair), -1).flatten(1)
        return ref.index_copy(1, self.arm, command)


from ..wholebody_s0_1.kinematics import RobotKinematics
from .hand_points import (
    BASE_FRAME_FEATURES, EXTRA_FRAME_FEATURES, GOAL_POINTS, articulated_points,
    arm_landmark_cost, worst_rotation_deg,
)


class Observation(nn.Module):
    current_only = True
    history_frames = HISTORY
    frame_features = BASE_FRAME_FEATURES + (EXTRA_FRAME_FEATURES if GOAL_POINTS==20 else 0)
    features = history_frames * frame_features
    version = CONTRACT["observation_version"]

    def __init__(self, fk, proximal):
        super().__init__()
        self.fk = fk
        self.register_buffer("proximal", proximal)

    def frame(self, state, previous_command, goal, ref):
        q, qd = state.split(54, -1)
        target = world(goal).flatten(1)
        poses = self.fk(torch.cat((q, ref))).pose
        measured, nominal = world(poses).flatten(1).chunk(2)
        base = (q, .1 * qd, previous_command, ref,
                target, 10. * (target - measured), 10. * (target - nominal))
        if GOAL_POINTS==5:
            return torch.cat(base, -1)
        extra_goal=articulated_points(goal,self.proximal).flatten(1)
        sensed, nominal=articulated_points(poses,self.proximal).flatten(1).chunk(2)
        return torch.cat((*base,extra_goal,10. * (extra_goal-sensed),10. * (extra_goal-nominal)),-1)


def world_errors(measured, goal):
    return (world(measured) - world(goal)).square().sum(-1).flatten(1)


def tracking_value(error2):
    ratio = error2 / (TARGET_MM / 1000.) ** 2
    score = 1. / (1. + ratio)
    return .65 * score.mean(-1) + .35 * score.amin(-1)


def arm_recovery_cost(actual, goal, interaction_gate=False):
    """Differentiate wrist-induced fingertip error on arm-dominant states."""
    desired_local = goal.hand[..., -1, :]
    measured_local = actual.hand[..., -1, :]
    desired_tips = goal.wrist[:, :, None] + desired_local @ goal.rotation.transpose(-1, -2)
    same_hand_tips = actual.wrist[:, :, None] + desired_local @ actual.rotation.transpose(-1, -2)
    arm_sq = (same_hand_tips - desired_tips).square().sum(-1)
    shape_peak = (measured_local - desired_local).norm(dim=-1).amax(-1)
    arm_peak = arm_sq.amax(-1).clamp_min(1e-12).sqrt()
    onset = ((arm_peak - .009) / .021).clamp(0., 1.)
    ratio = ((.63 * arm_peak - shape_peak + .004) / .013).clamp(0., 1.)
    gate = (onset * ratio).detach()
    robust = torch.sqrt(1. + arm_sq / .025**2) - 1.
    per_hand = .65 * robust.mean(-1) + .35 * robust.amax(-1)
    result = (gate * per_hand).mean(-1)
    if interaction_gate:
        with torch.no_grad():
            local = goal.hand.flatten(2, 3)
            points = goal.wrist[:, :, None] + local @ goal.rotation.transpose(-1, -2)
            interhand = (points[:, 0, :, None] - points[:, 1, None]).norm(dim=-1).amin((1, 2))
            clearance_gate = .25 + .75 * ((interhand - .014) / .026).clamp(0., 1.)
        result = result * clearance_gate
    return result


def interhand_clearance_cost(actual, goal):
    """Penalize bringing the hands closer than the target geometry calls for."""
    def gap(pose):
        local = pose.hand.flatten(2, 3)
        points = pose.wrist[:, :, None] + local @ pose.rotation.transpose(-1, -2)
        distances_sq = (points[:, 0, :, None] - points[:, 1, None, :]).square().sum(-1)
        return distances_sq.amin((1, 2)).clamp_min(1e-12).sqrt()

    target_clearance = (gap(goal).detach() - .002).clamp(min=.006, max=.018)
    deficit = (target_clearance - gap(actual)).clamp_min(0.)
    return (deficit / .008).square().clamp(max=4.)


def residual_rate(command, previous, reference, previous_reference, arm):
    correction = command[:, arm] - reference[:, arm]
    old_correction = previous[:, arm] - previous_reference[:, arm]
    return (correction - old_correction) / DT


def smoothness_cost(rate, error2, goal, previous_goal, quiet_scale=1.):
    ordinary = (1. - torch.exp(-rate.square() / math.radians(25.) ** 2)).mean(-1)
    # Quiet, already-aligned fingertips allow stronger damping. A moving goal
    # or a large tracking error still has the full per-frame residual budget.
    target_motion = (world(goal) - world(previous_goal)).norm(dim=-1).flatten(1).amax(-1)
    quiet = (1. - target_motion / .002).clamp(0., 1.)
    aligned = ((.025 - error2.amax(-1).sqrt()) / .015).clamp(0., 1.)
    relative = (rate.abs() / math.radians(10.)).clamp(max=3.)
    huber = torch.where(relative < 1., .5 * relative.square(), relative - .5).mean(-1)
    return ordinary + quiet_scale * (quiet * aligned).detach() * huber


def joint_limit_risk(state, command):
    q, qd = state.split(54, -1)
    to_low, to_high = q - command.lower, command.upper - q
    clearance = torch.minimum(to_low, to_high)
    outward = torch.where(to_high < to_low, qd, -qd).clamp_min(0.)
    return ((.035 - clearance + .012 * outward) / .05).clamp(0., 2.).square().amax(-1)


def fingertip_tail_cost(error_sq):
    """Maintain a useful gradient just below and above the acceptance threshold."""
    error = (error_sq + 1e-12).sqrt()
    excess = nn.functional.softplus((error - TARGET_MM / 1000.) / .006)
    return .7 * excess.amax(-1) + .3 * excess.mean(-1)


def wrist_pose_cost(actual, goal):
    """Scale translation and orientation by the wrist precision needed for tip tracking."""
    position_sq = (actual.wrist - goal.wrist).square().sum(-1) / .020**2
    rotation_sq = (actual.rotation - goal.rotation).square().sum((-1, -2))
    rotation_sq = rotation_sq / (8. * math.sin(math.radians(8.) / 2.)**2)
    robust = (1. + .5 * (position_sq + rotation_sq)).sqrt() - 1.
    return .7 * robust.mean(-1) + .3 * robust.amax(-1)


class TrackingEnv(DirectEnv):
    def __init__(self, backend, bank, model, seed, *, evaluation=False,
                 smoothness_weight=.04, limit_weight=.045, stationary_reset_fraction=0.):
        original_goals = bank.pose
        super().__init__(backend, bank, model, seed, not evaluation,
                         record_transition=True, switch=SwitchConfig(probability=0.),
                         stationary_reset_fraction=stationary_reset_fraction)
        if stationary_reset_fraction:
            self.hold_min_steps, self.hold_max_steps = 45, 100
        if bank.pose is not original_goals:
            raise ValueError("S04 requires the bank's original world fingertip targets")
        self.decoder = ArmCommand(model, bank.metadata, self.task.command)
        four = RobotKinematics(model,bank.metadata["joint_names"],physical_wrist=True,proximal=True)
        with torch.no_grad():
            origin=torch.zeros(1,54,dtype=torch.float32)
            proximal=four(origin).pose.hand[0,:,:,0,:].to(self.device)
        self.task.observe = Observation(self.task.observe.fk,proximal)
        self.full_hand_weight=.20 if GOAL_POINTS==20 else 0.
        self.wrist_pose_weight = 0.
        self.tip_tail_weight = 0.
        self.arm_recovery_weight = 0.
        self.interhand_clearance_weight = 0.
        self.arm_recovery_interaction_gate = False
        self.quiet_smoothness_scale = 1.
        self.smoothness_weight = smoothness_weight
        self.limit_weight = limit_weight
        self.task.tracking_reward = self.tracking_reward
        self.policy_history = self.state.new_zeros(self.num_envs, HISTORY, Observation.frame_features)

    def set_sampling_iteration(self, iteration):
        super().set_sampling_iteration(iteration)
        self.home_fraction = 0.

    def observation(self, ids=None, *, current_only=False):
        rows = slice(None) if ids is None else ids
        if not current_only:
            return self.policy_history[rows].flatten(1)
        frame = self.frame[rows]
        return self.task.observe.frame(
            self.state[rows] + self.domain.observation_bias[rows],
            self.history.previous_command[rows], self.bank.at(frame), self.bank.rsi_q[frame],
        )

    def command_targets(self, action):
        command = self.decoder(action, self.bank.rsi_q[self.frame])
        return command, command

    def tracking_failure(self, measured, goal, wrist_dist, wrist_rot):
        return world_errors(measured, goal).amax(-1) > RECOVERY_DISTANCE_M ** 2

    def tracking_objective(self, error2, actual, goal, stationary=None):
        value = tracking_value(error2)
        if self.tip_tail_weight:
            value = value - self.tip_tail_weight * fingertip_tail_cost(error2)
        if self.full_hand_weight:
            value = value - self.full_hand_weight * arm_landmark_cost(
                actual,goal,self.task.observe.proximal)
        if self.arm_recovery_weight:
            value = value - self.arm_recovery_weight * arm_recovery_cost(
                actual, goal, self.arm_recovery_interaction_gate)
        if self.wrist_pose_weight:
            hold_gain = 1. if stationary is None else 1. + .6 * (stationary > 0).float()
            value = value - self.wrist_pose_weight * hold_gain * wrist_pose_cost(actual, goal)
        return value

    def correction_rate(self, command, previous, frame, previous_frame):
        return residual_rate(command, previous, self.bank.rsi_q[frame],
                             self.bank.rsi_q[previous_frame], self.decoder.arm)

    def tracking_reward(self, state, geometry, previous, goal, previous_goal, command, previous_command,
                        *, phase, **kwargs):
        rate = self.correction_rate(command, previous_command, phase[0], self.previous_frame)
        error2 = world_errors(geometry.pose, goal)
        tracking = self.tracking_objective(error2, geometry.pose, goal,
                                           getattr(self, "stationary_steps", None))
        if self.interhand_clearance_weight:
            tracking = tracking - self.interhand_clearance_weight * interhand_clearance_cost(geometry.pose, goal)
        return (tracking
                - self.smoothness_weight * smoothness_cost(rate, error2, goal, previous_goal, self.quiet_smoothness_scale)
                - self.limit_weight * joint_limit_risk(state, self.task.command))

    @torch.no_grad()
    def step(self, action):
        stationary = getattr(self, "stationary_steps", None)
        goal = self.goal
        frame, previous_frame = self.frame, self.previous_frame
        previous_tip = world(self.actual)
        previous_command, previous_state = self.history.previous_command, self.state
        transition = super().step(action)
        error2 = world_errors(self.actual, goal)
        rate = self.correction_rate(self.transition["c"], previous_command, frame, previous_frame)
        arm = self.decoder.arm
        acceleration = (self.state[:, 54 + arm] - previous_state[:, 54 + arm]) / DT
        self.transition["rollout_stats"].update(
            tracking_score=tracking_value(error2),
            wrist_rotation_worst_deg=worst_rotation_deg(self.actual,goal),
            tip_speed_mm_s=(1000./DT)*(world(self.actual)-previous_tip).square().sum(-1).mean((1,2)).sqrt(),
            wrist_translation_worst_mm=1000.*(self.actual.wrist-goal.wrist).norm(dim=-1).amax(-1),
            tip_world_mean_mm=1000. * error2.sqrt().mean(-1),
            tip_world_worst_mm=1000. * error2.amax(-1).sqrt(),
            ten_tip_hit_15mm=error2.amax(-1) <= (TARGET_MM / 1000.) ** 2,
            arm_residual_rate_deg_s=rate.rad2deg().square().mean(-1).sqrt(),
            arm_acceleration_deg_s2=acceleration.rad2deg().square().mean(-1).sqrt(),
            joint_limit_risk=joint_limit_risk(self.state, self.task.command),
            joint_limit_overflow=self.task.position_violation(self.state) > .05,
        )
        if stationary is not None:
            self.transition["rollout_stats"]["gravity_hold_rollout_fraction"] = (stationary > 0).float()
        return transition


class Task(ImaginationTask):
    task_id = TASK_ID
    action_dim = ACTION_DIM
    command_dim = 54
    max_step_reward = 1.

    def make_actor(self):
        actor = nn.Sequential(nn.Linear(self.policy_dim, 512), nn.LayerNorm(512), nn.SiLU(),
                              nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
                              nn.Linear(256, ACTION_DIM), nn.Tanh())
        nn.init.normal_(actor[-2].weight, std=.0005)
        nn.init.zeros_(actor[-2].bias)
        return actor

    def policy_observation(self, x, memory, aux):
        frame = memory["frame"]
        return self.direct.observe.frame(x + aux, memory["cmd"], self.bank.at(frame),
                                         self.bank.rsi_q[frame])

    def control_target(self, action, memory, bias=None):
        command = self.env.direct.decoder(action, self.bank.rsi_q[memory["frame"]])
        return command, command

    def collection_action(self, action, memory):
        return (action + .08 * torch.randn_like(action)).clamp(-1., 1.)

    def reward(self, x, memory, action, command, contact, torque, soft=False):
        measured = self.direct.observe.fk(x[:, :54]).pose
        goal, previous_goal = self.goals(memory)
        error2 = world_errors(measured, goal)
        score = tracking_value(error2)
        rate = self.env.direct.correction_rate(command, memory["cmd"], memory["frame"],
                                               memory["previous_frame"])
        lost = torch.sigmoid((error2.amax(-1) - RECOVERY_DISTANCE_M ** 2) / .008)
        memory["switch_eligible"] = contact[:, 1].detach() < .5
        metrics = {} if soft else {"tracking_score": score.detach()}
        physical = self.env.direct
        value = physical.tracking_objective(error2, measured, goal, memory.get("stationary_steps"))
        if physical.interhand_clearance_weight:
            value = value - physical.interhand_clearance_weight * interhand_clearance_cost(measured, goal)
        value = value - self.env.direct.smoothness_weight * smoothness_cost(
            rate, error2, goal, previous_goal, self.env.direct.quiet_smoothness_scale)
        value = value - self.env.direct.limit_weight * joint_limit_risk(x, self.direct.command)
        value = value - .20 * contact[:, 0] - contact[:, 1] - .75 * lost
        return value, memory["interval_start"], metrics


class Env(ImaginationEnv):
    def __init__(self, backend, bank, model, seed, *, evaluation=False, stationary_reset_fraction=0.):
        self.direct = TrackingEnv(backend, bank, model, seed, evaluation=evaluation,
                                  stationary_reset_fraction=stationary_reset_fraction)
        self.num_envs, self.device = backend.num_envs, backend.device
        self.task = Task(self)
        self.disturbance = SimpleNamespace(adaptive_scale=float(not evaluation), success_scale=0.)
