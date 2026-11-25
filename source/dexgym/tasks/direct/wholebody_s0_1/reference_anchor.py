"""Task-space supervision at reference-initialized states for SuperMimic."""

import torch

from . import DOF, DT
from .kinematics import Pose, make_pose
from .task import chord2, fast_hand_tracking_cost, fingertip_position_tail_cost, squared

REFERENCE_ANCHOR_VERSION = "rsi_task_space_anchor_v1"
REFERENCE_JOINT_PRIOR_VERSION = "rsi_feasible_command_prior_v1"
ON_POLICY_ANCHOR_VERSION = "on_policy_task_space_anchor_v1"
HAND_PRIORITY_ANCHOR_VERSION = "hand_priority_anchor_v1"
TIP_TAIL_ANCHOR_VERSION = "fingertip_tail_anchor_v1"
ARM_MOTION_GATE_VERSION = "arm_motion_gated_anchor_v1"
HAND_MOTION_GATE_VERSION = "hand_motion_gated_anchor_v1"
HAND_HARDNESS_VERSION = "source_balanced_hand_error_weight_v1"
TEMPORAL_ANCHOR_VERSION = "causal_policy_temporal_command_anchor_v1"
HOLD_ANCHOR_VERSION = "stationary_reference_anchor_v1"
ACTUATION_ANCHOR_VERSION = "observed_arm_actuation_anchor_v1"
RECOVERY_ANCHOR_VERSION = "stationary_hand_recovery_anchor_v1"
ELBOW_ANCHOR_VERSION = "bilateral_elbow_anchor_v1"
FEEDBACK_ANCHOR_VERSION = "stationary_hand_feedback_anchor_v1"
RELATIVE_ANCHOR_VERSION = "relative_wrist_pose_anchor_v1"
LIVE_RECOVERY_ANCHOR_VERSION = "measured_hand_recovery_anchor_v1"
STILL_LIVE_ANCHOR_VERSION = "measured_stationary_hand_feedback_v1"
DELAYED_HAND_ANCHOR_VERSION = "actuation_delayed_hand_targets_v1"
ELBOW_NULLSPACE_VERSION = "wrist_preserving_elbow_anchor_v1"
LIVE_EXECUTION_ANCHOR_VERSION = "effect_time_hand_supervision_v1"
LIVE_WRIST_EXECUTION_ANCHOR_VERSION = "effect_time_wrist_supervision_v1"


def robust_distance(error2, scale):
    return torch.sqrt(1. + error2 / scale**2) - 1.


def cross_hand_landmark_distances(pose):
    local = pose.hand.flatten(2, 3)
    world = local @ pose.rotation.transpose(-1, -2) + pose.wrist[:, :, None, :]
    return torch.cdist(world[:, 0], world[:, 1]).flatten(1)


def same_hand_proximity(pose):
    """Tip-to-other-finger segment proximity without a desired separation."""
    tips = pose.hand[:, :, :, None, None, -1, :]
    starts = pose.hand[:, :, None, :, :-1, :]
    segments = pose.hand.diff(dim=-2)[:, :, None, :, :, :]
    offset = tips-starts
    alpha = ((offset*segments).sum(-1, keepdim=True)
             / segments.square().sum(-1, keepdim=True).clamp_min(1e-8)).clamp(0., 1.)
    separation2 = (offset-alpha*segments).square().sum(-1)
    same_digit = torch.eye(pose.hand.shape[2], device=separation2.device, dtype=torch.bool)
    separation2 = separation2.masked_fill(same_digit[None,None,:,:,None], float("inf"))
    gap = separation2.flatten(1).amin(-1).sqrt()
    return ((.022-gap)/.012).clamp(0., 1.)


class ReferenceAnchor:
    def __init__(self, bank, observation, weight, seed, *, joint_weight=0., batch=256,
                 live_env=None, live_fraction=0., hand_weight=1., tip_tail_weight=0., arm_motion_gate=0.,
                 hand_motion_gate=0., hand_hardness=0., arm_lead=0, hand_lead=0, hold_fraction=0.,
                 arm_actuation_weight=0., hold_recovery_noise=0., elbow_weight=0., hand_feedback_weight=0.,
                 relative_pose_weight=1., hard_live_fraction=0., recovery_frames=0,
                 stationary_live_fraction=0., independent_feedback=False, hand_delay_lead=0,
                 elbow_nullspace=False, hand_physical_weight=0., wrist_physical_weight=0., arm_delay_lead=0,
                 live_recovery_weight=0., elbow_physical_weight=0., live_contact_weight=0., recorded_hand_lead=0,
                 execution_priority=0., wrist_recovery_priority=0.):
        if not observation.current_only:
            raise ValueError("The reference anchor requires four-point current-frame observations")
        self.bank = bank
        self.observation = observation
        self.weight = weight
        self.joint_weight = joint_weight
        self.batch = batch
        self.live_env = live_env
        self.live_count = round(batch * live_fraction)
        self.hand_weight = hand_weight
        self.tip_tail_weight = tip_tail_weight
        self.arm_motion_gate = arm_motion_gate
        self.hand_motion_gate = hand_motion_gate
        self.hand_hardness = hand_hardness
        self.arm_lead = arm_lead
        self.hand_lead = hand_lead
        self.hold_fraction = hold_fraction
        self.arm_actuation_weight = arm_actuation_weight
        self.hold_recovery_noise = hold_recovery_noise
        self.elbow_weight = elbow_weight
        self.hand_feedback_weight = hand_feedback_weight
        self.relative_pose_weight = relative_pose_weight
        self.hard_live_fraction = hard_live_fraction
        self.recovery_frames = recovery_frames
        self.stationary_live_fraction = stationary_live_fraction
        self.independent_feedback = independent_feedback
        self.arm_delay_lead = arm_delay_lead
        self.hand_delay_lead = hand_delay_lead
        self.recorded_hand_lead = recorded_hand_lead
        self.elbow_nullspace = elbow_nullspace
        self.hand_physical_weight = hand_physical_weight
        self.wrist_physical_weight = wrist_physical_weight
        self.live_recovery_weight = live_recovery_weight
        self.elbow_physical_weight = elbow_physical_weight
        self.live_contact_weight = live_contact_weight
        self.arm_rowspace_weight = 0.
        self.same_hand_contact_focus = 0.
        self.execution_priority = execution_priority
        self.wrist_recovery_priority = wrist_recovery_priority
        if elbow_weight:
            self.elbow_reference = live_env.task.posture
        self.metrics = {}
        self.generator = torch.Generator(device=bank.rsi_q.device).manual_seed(seed)
        self.command = observation.command
        if hand_feedback_weight:
            self.hand_indices = (~self.command.arm).nonzero().flatten()
        self.noise = torch.where(self.command.arm, .025, .060)
        if joint_weight or arm_actuation_weight or elbow_nullspace:
            names = bank.metadata["joint_names"]
            arms = [[i for i, name in enumerate(names) if name.startswith("Joint") and name.endswith(f"_{side}")]
                    for side in "LR"]
            if any(len(indices) != 7 for indices in arms):
                raise ValueError("Expected seven joints per arm for the feasible command prior")
            self.arms = torch.tensor(arms, device=bank.rsi_q.device)
            self.hand = ~self.command.arm
            self.mapped_frames = getattr(bank, "mapped_hand_frames", None)
            if self.mapped_frames is not None:
                digits = ("thumb", "index", "middle", "ring", "pinky")
                joints = [[i for i, n in enumerate(names) if n.startswith(f"{side}_{digit}_")]
                          for side in "lr" for digit in digits]
                if any(len(group) != 4 for group in joints):
                    raise ValueError("Mapped hand prior requires four joints per finger")
                self.digit_joints = torch.tensor(joints, device=bank.rsi_q.device).reshape(2, 5, 4)

    def arm_position_offset(self, env, indices, state):
        q, qd = state.split(DOF, dim=-1)
        offset = torch.zeros_like(q)
        transition = getattr(env, "transition", None)
        if not self.arm_actuation_weight or not transition:
            return offset
        # Use the setpoint applied in the completed Newton interval.
        # A reset or incomplete interval has no matched actuator measurement.
        same_state = (transition["x1"][indices, :DOF] - q).abs().amax(-1) < 1e-6
        valid = same_state & transition["full_step"][indices] & ~transition["drop"][indices]
        observed_offset = transition["u"][indices] - q
        span = 2. * self.command.slew[self.arms]
        observed_offset = observed_offset[:, self.arms].clamp(-span, span)
        arm_speed2 = qd[:, self.arms].square().mean(-1)
        weight = self.arm_actuation_weight / (1. + (arm_speed2 / .35**2).square())
        offset[:, self.arms] = (observed_offset * weight[..., None]).masked_fill(~valid[:, None, None], 0.)
        self.metrics["anchor_arm_position_offset_rad"] = offset[:, self.arms].abs().mean()
        return offset

    def sample_live_indices(self, env, available, frames, goal):
        indices = available[torch.randint(len(available), (self.live_count,), device=frames.device,
                                         generator=self.generator)]
        hard_count = round(self.live_count * self.hard_live_fraction)
        if not hard_count:
            return indices
        eligible = available[env.switch_recovery[available] == 0]
        if not eligible.numel():
            return indices
        errors = squared(env.actual.hand[eligible] - goal.hand[eligible])
        digits = (.6 * errors[..., -1] + .4 * errors.mean(-1)).sqrt()
        difficulty = digits.flatten(1).topk(2, dim=-1).values.mean(-1)
        weights = (.5 + difficulty / .008).clamp(max=3.5).square()
        source = self.bank.egopro[frames[eligible]].long()
        normalizer = weights.new_zeros(2).scatter_add_(0, source, weights)
        weights = weights / normalizer[source].clamp_min(1e-9)
        chosen = torch.multinomial(weights, hard_count, replacement=True, generator=self.generator)
        stationary_count = round(hard_count * self.stationary_live_fraction)
        stationary = getattr(env, "stationary_steps", None)
        if stationary_count and stationary is not None:
            held = stationary[eligible] > 0
            if held.any():
                held_weights = weights * held
                held_mass = weights.new_zeros(2).scatter_add_(0, source, held_weights)
                probabilities = held_weights / held_mass[source].clamp_min(1e-9)
                chosen[:stationary_count] = torch.multinomial(
                    probabilities, stationary_count, replacement=True, generator=self.generator)
                self.metrics["anchor_hard_stationary_fraction"] = held[chosen].float().mean()
        indices[:hard_count] = eligible[chosen]
        self.metrics["anchor_hard_live_fraction"] = difficulty.new_tensor(hard_count / self.live_count)
        self.metrics["anchor_live_hard_error_mm"] = difficulty[chosen].mean() * 1000.
        return indices

    def sample_live(self):
        if not self.live_count:
            return None
        env = self.live_env
        if env.refine_posture:
            frames, bridge, goal = env.previous_frame, env.previous_bridge, env.history.previous_goal
        else:
            frames, bridge, goal = env.frame, env.bridge, env.goal
        available = (bridge == 0).nonzero().flatten()
        if not available.numel():
            return None
        indices = self.sample_live_indices(env, available, frames, goal)
        actual = env.state[indices]
        state = actual + env.domain.observation_bias[indices]
        target = Pose(*(value[indices] for value in goal))
        offset = self.arm_position_offset(env, indices, actual)
        stationary = getattr(env, "stationary_steps", None)
        hold = torch.zeros_like(indices, dtype=torch.bool) if stationary is None else stationary[indices] > 0
        delayed = env.domain.delay[indices]
        return actual[:, :DOF], state, env.history.previous_command[indices], target, frames[indices], offset, hold, delayed, indices

    def recovery_neighbor(self, frames):
        size = torch.randint(max(1, self.recovery_frames // 2), self.recovery_frames + 1,
                             frames.shape, device=frames.device, generator=self.generator)
        later = torch.rand(frames.shape, device=frames.device, generator=self.generator) < .5
        indices = frames + torch.where(later, size, -size)
        start = self.bank.clip_starts[self.bank.clip_id[frames]]
        indices = torch.maximum(start, torch.minimum(indices, self.bank.end[frames]))
        return torch.where(self.bank.safe[indices], indices, frames)

    def sample(self):
        live = self.sample_live()
        count = self.batch - (0 if live is None else len(live[0]))
        frames = self.bank.sample(count, self.generator)
        offsets = torch.randint(5, (count,), device=frames.device, generator=self.generator)
        # Reuse recorded poses as stationary goals so an accurately reached
        # pose is stable even if its recorded trajectory was moving.
        ordinal = torch.arange(count, device=frames.device)
        hold = ordinal < round(count * self.hold_fraction)
        recovery = torch.zeros_like(hold)
        if self.hold_recovery_noise or self.recovery_frames:
            recovery = hold & ordinal.remainder(2).bool()
        target_frames = frames + offsets.masked_fill(hold, 0)
        target = self.bank.at(target_frames)
        reference_q = self.bank.rsi_q[frames]
        noise = torch.randn(reference_q.shape, device=reference_q.device, generator=self.generator)
        recovery_joint = recovery[:, None] & ~self.command.arm[None]
        sample_noise = torch.where(recovery_joint, self.hold_recovery_noise, self.noise)
        q = (reference_q + sample_noise * noise).clamp(self.command.lower, self.command.upper)
        if self.recovery_frames:
            # Recover toward the held goal from a nearby pose verified in the same clip.
            nearby = self.bank.rsi_q[self.recovery_neighbor(frames)]
            q = torch.where(recovery[:, None], nearby, q)
        qd = self.bank.rsi_qd[frames].masked_fill(hold[:, None], 0.)
        state = torch.cat((q, qd), dim=-1)
        command = q
        if self.hold_recovery_noise:
            # A recorded stationary goal can arrive while the robot is off
            # pose and its prior command still differs from its measurement.
            sign = 2. * torch.rand((count, 1), device=q.device, generator=self.generator) - 1.
            history_offset = (.5 * sign * (q - reference_q)).clamp(-self.command.slew, self.command.slew)
            command = (q + history_offset.masked_fill(~recovery_joint, 0.)).clamp(
                self.command.lower, self.command.upper)
        offset = torch.zeros_like(q)
        delayed = torch.zeros_like(hold)
        live_indices = None
        if live is not None:
            live_q, live_state, live_command, live_target, live_frames, live_offset, live_hold, live_delay, live_indices = live
            delayed = torch.cat((delayed, live_delay))
            offset = torch.cat((offset, live_offset))
            q = torch.cat((q, live_q))
            state = torch.cat((state, live_state))
            command = torch.cat((command, live_command))
            target = Pose(*(torch.cat((a, b)) for a, b in zip(target, live_target, strict=True)))
            target_frames = torch.cat((target_frames, live_frames))
            hold = torch.cat((hold, live_hold))
            recovery = torch.nn.functional.pad(recovery, (0, len(live_frames)))
        recorded = getattr(self.live_env, "recorded_clock", None)
        if recorded is not None:
            recorded = (torch.zeros_like(hold) if live_indices is None else
                        torch.nn.functional.pad(recorded[live_indices], (count, 0)))
        self.metrics["anchor_live_fraction"] = state.new_tensor(1. - count / self.batch)
        observation = self.observation.forward_current(state, target, command)
        if self.observation.history_frames > 1:
            # Synthetic RSI is an explicit cold start. Real samples carry the real
            # measured/command/target history, not a reconstructed expert future.
            history = observation[:, None].expand(-1, self.observation.history_frames, -1).clone()
            if live_indices is not None:
                history[count:] = self.live_env.policy_history[live_indices]
                history[count:, -1] = observation[count:]
            observation = history.flatten(1)
        self.sample_observation = observation
        return q, target, observation, target_frames, hold, offset, recovery, command, state, delayed, live_indices, recorded

    def arm_anchor_gate(self, target, frames, recorded):
        if not self.arm_motion_gate:
            return target.wrist.new_ones(target.wrist.shape[:2])
        following = self.bank.advance_frame(frames, 1, recorded)
        translation = squared(self.bank.pose.wrist[following] - target.wrist) / (DT * self.arm_motion_gate)**2
        rotation = chord2(self.bank.pose.rotation[following], target.rotation) / (DT * 2.5)**2
        gate = .20 + .80 / (1. + translation + rotation)
        self.metrics["anchor_arm_motion_gate"] = gate.mean()
        return gate

    def hand_anchor_gate(self, target, frames, recorded):
        if not self.hand_motion_gate:
            return target.hand.new_ones(target.hand.shape[:3])
        following = self.bank.advance_frame(frames, 1, recorded)
        speed = squared(self.bank.pose.hand[following] - target.hand).amax(-1) / (DT * self.hand_motion_gate)**2
        gate = .35 + .65 / (1. + speed)
        self.metrics["anchor_hand_motion_gate"] = gate.mean()
        return gate

    def hand_sample_weight(self, tip_error2, frames):
        if not self.hand_hardness:
            return tip_error2.new_ones(len(frames))
        with torch.no_grad():
            largest = tip_error2.flatten(1).topk(2, dim=-1).values.sqrt().mean(-1)
            weight = 1. + self.hand_hardness * ((largest - .0065) / .008).clamp(0., 2.)
            source = self.bank.egopro[frames].long()
            sum_by_source = weight.new_zeros(2).scatter_add_(0, source, weight)
            count = weight.new_zeros(2).scatter_add_(0, source, torch.ones_like(weight))
            scale = (sum_by_source / count.clamp_min(1.)).clamp_min(1.)
            result = weight / scale[source]
            self.metrics["anchor_max_hand_sample_weight"] = result.max()
            return result

    def command_targets(self, observed, frames, hold, delayed, recorded=None):
        arm_lead = self.arm_lead + self.arm_delay_lead * delayed
        arm_frames = self.bank.advance_frame(frames, arm_lead * (~hold), recorded,
                                            max_steps=self.arm_lead + self.arm_delay_lead)
        hand_lead = self.hand_lead + self.hand_delay_lead * delayed
        if self.recorded_hand_lead and recorded is not None:
            hand_lead = hand_lead + self.recorded_hand_lead * recorded
            self.metrics["anchor_recorded_hand_lead_fraction"] = (recorded & ~hold).float().mean()
        hand_frames = self.bank.advance_frame(frames, hand_lead * (~hold), recorded,
                                             max_steps=self.hand_lead + self.hand_delay_lead + self.recorded_hand_lead)
        if self.arm_delay_lead:
            self.metrics["anchor_delayed_arm_target_fraction"] = (delayed & ~hold).float().mean()
        if self.hand_delay_lead:
            self.metrics["anchor_delayed_hand_target_fraction"] = (delayed & ~hold).float().mean()
        arm_target = self.arm_lead or self.arm_delay_lead
        hand_target = self.hand_lead or self.hand_delay_lead or self.recorded_hand_lead
        if not (arm_target or hand_target):
            return observed, arm_frames, hand_frames
        pose = self.bank.pose
        wrist = pose.wrist[arm_frames] if arm_target else observed.wrist
        rotation = pose.rotation[arm_frames] if arm_target else observed.rotation
        hand = pose.hand[hand_frames] if hand_target else observed.hand
        return make_pose(wrist, rotation, hand), arm_frames, hand_frames

    def joint_prior(self, request, target, arm_frames, hand_frames, arm_gate, hand_gate, hand_sample_weight, q=None, hold=None):
        reference = torch.where(self.command.arm, self.bank.rsi_q[arm_frames], self.bank.rsi_q[hand_frames])
        rowspace_weight = getattr(self, "arm_rowspace_weight", 0.)
        if getattr(self.bank, "stability", False) and not rowspace_weight:
            reference = torch.where(hold[:, None] & self.command.arm, q.detach(), reference)
        reference = reference.clamp(self.command.lower, self.command.upper)
        with torch.no_grad():
            target_pose = self.observation.fk(reference).pose
            position = squared(target_pose.wrist - target.wrist) / .008**2
            orientation = chord2(target_pose.rotation, target.rotation) / .10**2
            confidence = 1. / (1. + position.square() + orientation.square())
        residual2 = ((request - reference) / self.command.rate_scale).square()
        cost = robust_distance(residual2, 1.)
        arms = (cost[:, self.arms].mean(-1) * confidence * arm_gate).mean(-1)
        if getattr(self.bank, "stability_v2", False):
            from .stability import nullspace_displacement_cost
            # Only redundant movement is regularized; task-space corrections are
            # free to move away from q. No unobserved archived arm label remains.
            if rowspace_weight:
                redundant_cost, jacobian = nullspace_displacement_cost(
                    self.observation.fk, request, q, self.arms, self.command.rate_scale,
                    return_jacobian=True)
                delta = (request - reference)[:, self.arms]
                observable = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
                branch_gap = (reference-q.detach())[:, self.arms].abs().amax(-1)
                prior_gate = confidence / (1. + (branch_gap/.25).pow(4))
                tracking_cost = (robust_distance(squared(observable[..., :3]), .012)
                                 + .55 * robust_distance(squared(observable[..., 3:]), .095))
                arms = (redundant_cost + rowspace_weight * prior_gate * tracking_cost).mean(-1)
                self.metrics["anchor_wrist_rowspace_gate"] = prior_gate.mean()
                self.metrics["anchor_wrist_rowspace_mm"] = 1000. * observable[..., :3].detach().norm(dim=-1).mean()
            else:
                arms = nullspace_displacement_cost(self.observation.fk, request, q, self.arms,
                                                   self.command.rate_scale).mean(-1)
        if getattr(self, "mapped_frames", None) is None:
            hand_prior = cost[:, self.hand].mean(-1)
        else:
            # The target's four points are authoritative if an archived joint
            # label was retargeted to a different hand shape.
            geometry_error2 = squared(target_pose.hand - target.hand).amax(-1)
            confidence = 1. / (1. + (geometry_error2 / .008**2).square())
            confidence = torch.where(self.mapped_frames[hand_frames, None, None], confidence, 1.)
            hand_prior = (confidence * cost[:, self.digit_joints].mean(-1)).mean((1, 2))
            self.metrics["anchor_hand_prior_confidence"] = confidence.mean()
        hands = hand_prior * hand_gate.mean((1, 2)) * hand_sample_weight
        return .5 * arms.mean() + .5 * self.hand_weight * hands.mean()

    def elbow_geometry(self, estimated):
        # Project the elbow loss gradient onto the arm direction that
        # preserves the wrist pose to first order at this configuration.
        with torch.no_grad():
            jacobian = self.observation.fk.wrist_jacobian(estimated.detach())
            _, _, basis = torch.linalg.svd(jacobian, full_matrices=True)
            null = basis[..., -1, :]
            delta = (jacobian @ null[..., None]).abs().max()
            self.metrics["anchor_elbow_wrist_jacobian_error"] = delta
        q_arm = estimated[:, self.arms]
        change = q_arm - q_arm.detach()
        shift = (change * null).sum(-1, keepdim=True) * null
        indices = self.arms.flatten().expand(len(estimated), -1)
        q = estimated.detach().scatter(1, indices, (q_arm.detach() + shift).flatten(1))
        return self.observation.fk(q)

    def elbow_tracking_cost(self, geometry, goal, frames, q=None, hold=None):
        reference = self.elbow_reference
        target_upper = reference.reference.upper_arm[frames]
        if getattr(self.bank, "stability", False):
            current_upper = self.observation.fk(q.detach()).upper_arm
            target_upper = (current_upper if getattr(self.bank, "stability_v2", False) else
                            torch.where(hold[:, None, None], current_upper, target_upper))
        delta = geometry.upper_arm - target_upper
        ray = geometry.ray.detach()
        swivel_error = delta - (delta * ray).sum(-1, keepdim=True) * ray
        lateral_error = (delta * reference.outward).sum(-1)
        with torch.no_grad():
            position = squared(geometry.pose.wrist - goal.wrist) / .018**2
            orientation = chord2(geometry.pose.rotation, goal.rotation) / .15**2
            priority = .20 + .80 / (1. + position + orientation)
            if self.relative_pose_weight != 1.:
                relative_position = squared(geometry.pose.relative_position - goal.relative_position) / .021**2
                relative_orientation = chord2(geometry.pose.relative_rotation, goal.relative_rotation) / .14**2
                priority = priority / (1. + .5 * (relative_position + relative_orientation))[:, None]
            priority = priority * reference.elbow_confidence[frames]
            self.metrics["anchor_elbow_mm"] = delta.norm(dim=-1).mean() * 1000.
        swivel = robust_distance(squared(swivel_error), .035)
        lateral = robust_distance(lateral_error.square(), .030)
        return (priority * (.60 * swivel + .40 * lateral)).mean(-1)

    def hand_feedback_cost(self, actor, state, goal, previous, request, stationary):
        ids = stationary.nonzero().flatten()
        if not ids.numel():
            return request.new_zeros(())
        q, qd = state[ids].split(DOF, dim=-1)
        previous, request = previous[ids], request[ids]
        goal = Pose(*(part[ids] for part in goal))
        joint = self.hand_indices[torch.randint(len(self.hand_indices), (len(ids),),
                                                generator=self.generator, device=q.device)]
        lo, hi = self.command.lower[joint], self.command.upper[joint]
        angle = q.gather(1, joint[:, None]).squeeze(1)
        step = .035
        perturbation = torch.where(angle < .5 * (lo + hi), step, -step)
        delta = torch.zeros_like(q).scatter(1, joint[:, None], perturbation[:, None])
        q_delta, previous_delta, threshold = delta, delta, .75
        if self.independent_feedback:
            mode = torch.randint(3, (len(ids),), generator=self.generator, device=q.device)
            q_delta = delta.masked_fill((mode == 1)[:, None], 0.)
            previous_delta = delta.masked_fill((mode == 0)[:, None], 0.)
            threshold = torch.where(mode == 2, .75, .40)
        moved = (q + q_delta).clamp(self.command.lower, self.command.upper)
        q2 = torch.where(q_delta != 0., moved, q)
        previous2 = (previous + previous_delta).clamp(self.command.lower, self.command.upper)
        # The static goal and the velocity match across each observation pair.
        obs2 = self.observation.forward_current(torch.cat((q2, qd), -1), goal, previous2)
        if self.observation.history_frames > 1:
            past = self.sample_observation[ids].reshape(-1, self.observation.history_frames, self.observation.frame_features)
            obs2 = torch.cat((past[:, :-1], obs2[:, None]), dim=1).flatten(1)
        request2 = self.command.desired(actor(obs2)).clamp(self.command.lower, self.command.upper)
        response = (request2 - request)[:, self.hand_indices]
        gain = response.norm(dim=-1) / step
        self.metrics["anchor_hand_feedback_gain_p95"] = torch.quantile(gain.detach(), .95)
        return robust_distance((gain - threshold).clamp_min(0.).square(), .40).mean()

    def live_execution_cost(self, action, indices):
        env = self.live_env
        frame, previous = env.frame[indices], env.previous_frame[indices]
        delay = env.domain.delay[indices]
        clock = getattr(env, "recorded_clock", None)
        recorded = None if clock is None else clock[indices]
        effect_frame = self.bank.advance_frame(frame, delay.long(), recorded, max_steps=1)
        goal = self.bank.at(effect_frame)
        last_goal = self.bank.pose.hand[previous]
        rate = (goal.hand - last_goal) / (DT * (1. + delay[:, None, None, None, None]))
        continuous = ((frame > previous) & (self.bank.clip_id[frame] == self.bank.clip_id[previous])
                      & (frame == self.bank.advance_frame(previous, 1, recorded)))
        moving = rate.norm(dim=-1).amax((1, 2, 3)) >= .04
        valid = continuous & (env.bridge[indices] == 0) & (env.switch_recovery[indices] == 0)
        if not self.wrist_physical_weight:
            valid = valid & moving
        if not valid.any():
            return action.new_zeros(())
        selected = indices[valid]
        q_state = env.state[selected]
        previous_command = env.history.previous_command[selected]
        sim = env.domain.physical[selected]
        bias = env.domain.observation_bias[selected]
        candidate = self.command(action[-len(indices):][valid], previous_command)
        delay = delay[valid]
        goal = Pose(*(part[valid] for part in goal))
        if delay.any():
            with torch.no_grad():
                before, _ = self.world_model(q_state, previous_command, self.world_features(q_state, sim, bias))
                q_state = torch.where(delay[:, None], before, q_state)
        prediction, _ = self.world_model(q_state, candidate, self.world_features(q_state, sim, bias))
        measured = self.observation.fk(prediction[:, :DOF]).pose
        tip_mm = (measured.hand[..., -1, :] - goal.hand[..., -1, :]).norm(dim=-1).detach() * 1000.
        self.metrics["anchor_predicted_live_tip_p95_mm"] = torch.quantile(tip_mm.flatten(), .95)
        self.metrics["anchor_live_execution_fraction"] = valid.float().mean()
        hand_cost = fast_hand_tracking_cost(measured, goal, rate[valid], delay)
        hand_loss = (hand_cost * moving[valid]).sum() / moving[valid].sum().clamp_min(1)
        loss = self.hand_physical_weight * hand_loss
        if self.wrist_physical_weight:
            wrist_error = squared(measured.wrist - goal.wrist)
            rotation_error = chord2(measured.rotation, goal.rotation)
            relative_position = squared(measured.relative_position - goal.relative_position)
            relative_rotation = chord2(measured.relative_rotation, goal.relative_rotation)
            wrist_loss = (.35 * robust_distance(wrist_error, .013).mean()
                          + .25 * robust_distance(rotation_error, .09).mean()
                          + .22 * robust_distance(relative_position, .019).mean()
                          + .18 * robust_distance(relative_rotation, .10).mean())
            loss = loss + self.wrist_physical_weight * wrist_loss
            self.metrics["anchor_predicted_live_wrist_p95_mm"] = torch.quantile(
                wrist_error.detach().sqrt().flatten() * 1000., .95)
            self.metrics["anchor_live_wrist_execution_loss"] = wrist_loss.detach()
        return loss

    def live_recovery_cost(self, actor, first_action, indices):
        """Optimize the measured response over three causal control intervals."""
        env = self.live_env
        frame, previous = env.frame[indices], env.previous_frame[indices]
        same_clip = self.bank.clip_id[frame] == self.bank.clip_id[previous]
        valid = same_clip & (env.bridge[indices] == 0) & (env.switch_recovery[indices] == 0)
        valid &= (frame >= previous) & (env.task.position_violation(env.state[indices]) <= .05)
        if not valid.any():
            return first_action.new_zeros(())
        rows = valid.nonzero().flatten()
        goal, received = self.bank.at(frame[rows]), self.bank.at(previous[rows])
        speed = ((goal.hand[..., -1, :] - received.hand[..., -1, :]).norm(dim=-1) / DT).amax((1, 2))
        delay = env.domain.delay[indices[rows]]
        # Sample across both sources and give moving, delayed goals additional coverage.
        weight = (.6 + (speed / .12).clamp(max=2.5)) * (1. + .8 * delay)
        if self.live_contact_weight:
            nearest = cross_hand_landmark_distances(goal).amin(-1)
            weight = weight * (1. + .50 * ((.035 - nearest) / .025).clamp(0., 1.))
        stationary = getattr(env, "stationary_steps", None)
        if self.execution_priority:
            live_ids = indices[rows]
            actual, upper_arm = env.actual, env.geometry.upper_arm[live_ids]
            reference = self.elbow_reference
            lateral = ((upper_arm - reference.reference.upper_arm[frame[rows]]) * reference.outward).sum(-1).abs()
            if getattr(self.bank, "stability_v2", False):
                old_q = (env.history.previous_state - env.domain.observation_bias)[live_ids, :DOF]
                old_upper = self.observation.fk(old_q).upper_arm
                # Prioritize observed motion, not distance to a hidden IK branch.
                lateral = 2. * ((upper_arm - old_upper) * reference.outward).sum(-1).abs() / DT
            wrist_error = (actual.wrist[live_ids] - goal.wrist).norm(dim=-1)
            wrist_close = 1. / (1. + (wrist_error / .018).square())
            lateral_urgency = ((lateral - .028).clamp_min(0.) / .030).clamp(max=3.)
            confidence = reference.elbow_confidence[frame[rows]]
            elbow_priority = (wrist_close * confidence * lateral_urgency).amax(-1)
            hold_priority = torch.zeros_like(elbow_priority)
            if stationary is not None:
                point_delta = actual.hand[live_ids, ..., -1, :] - received.hand[..., -1, :]
                tip_error = point_delta.norm(dim=-1).flatten(1).topk(2, dim=-1).values.mean(-1)
                held = stationary[live_ids] > 0
                hold_priority = held * ((tip_error - .0065).clamp_min(0.) / .007).clamp(max=2.5)
            weight = weight + self.execution_priority * (2.0 * elbow_priority + 1.5 * hold_priority)
            self.metrics["anchor_recovery_elbow_priority"] = elbow_priority.mean()
            self.metrics["anchor_recovery_stationary_priority"] = hold_priority.mean()
            if self.wrist_recovery_priority:
                relative_error = (actual.relative_position[live_ids] - goal.relative_position).norm(dim=-1)
                lag = ((wrist_error.amax(-1) - .008).clamp_min(0.) / .018).clamp(max=2.)
                separation = ((relative_error - .012).clamp_min(0.) / .026).clamp(max=2.)
                wrist_speed = (goal.wrist - received.wrist).norm(dim=-1).amax(-1) / DT
                need = (.60 * lag + .40 * separation) * (1. + .5 * (wrist_speed / .25).clamp(max=1.))
                weight = weight + self.wrist_recovery_priority * need / (1. + .40 * elbow_priority)
                self.metrics["anchor_recovery_wrist_priority"] = need.mean()
        if self.same_hand_contact_focus:
            with torch.no_grad():
                actual_pose = Pose(*(value[indices[rows]] for value in env.actual))
                proximity = torch.maximum(same_hand_proximity(goal), same_hand_proximity(actual_pose))
                weight = weight * (1. + .75*self.same_hand_contact_focus*proximity)
                self.metrics["anchor_recovery_same_hand_focus"] = proximity.mean()
        source = self.bank.egopro[frame[rows]].long()
        mass = weight.new_zeros(2).scatter_add_(0, source, weight)
        count = min(128, len(rows))
        chosen = torch.multinomial(weight / mass[source].clamp_min(1e-8), count,
                                   generator=self.generator, replacement=False)
        rows = rows[chosen]
        indices = indices[rows]
        clock = getattr(env, "recorded_clock", None)
        recorded = None if clock is None else clock[indices]
        frame, previous = frame[rows], previous[rows]
        delay = env.domain.delay[indices]
        state = env.state[indices]
        physical, bias = env.domain.physical[indices], env.domain.observation_bias[indices]
        command = env.history.previous_command[indices]
        action = first_action[rows]
        history = env.policy_history[indices]
        previous_goal = self.bank.at(previous)
        current_geometry = self.observation.fk(state[:, :DOF])
        current_pose = current_geometry.pose
        terms, tips = [], []
        elbow_cost = first_action.new_zeros(())
        for step in range(3):
            goal = self.bank.at(frame)
            next_command = self.command(action, command)
            applied = torch.where(delay[:, None], command, next_command)
            following, contacts = self.world_model(state, applied, self.world_features(state, physical, bias))
            measured_geometry = self.observation.fk(following[:, :DOF])
            measured = measured_geometry.pose
            hand_error = measured.hand - goal.hand
            tip_error = hand_error[..., -1, :]
            shape = robust_distance(squared(hand_error), .010).mean((1, 2, 3))
            point_speed = (goal.hand - previous_goal.hand) / DT
            moving = (point_speed.norm(dim=-1).amax(-1) / .15).clamp(max=1.)
            by_digit = robust_distance(squared(tip_error), .008) * (.75 + .50 * moving)
            fingertip = .68 * by_digit.mean((1, 2)) + .32 * by_digit.topk(2, dim=-1).values.mean((1, 2))
            rate_error = (measured.hand - current_pose.hand) / DT - point_speed
            smooth = robust_distance(squared(rate_error), .13).mean((1, 2, 3))
            wrists = robust_distance(squared(measured.wrist - goal.wrist), .012).mean(-1)
            orientation = robust_distance(chord2(measured.rotation, goal.rotation), .085).mean(-1)
            relative = robust_distance(squared(measured.relative_position - goal.relative_position), .018)
            relative_rotation = robust_distance(chord2(measured.relative_rotation, goal.relative_rotation), .095)
            finger_cost = .30 * shape + .55 * fingertip + .45 * fingertip_position_tail_cost(tip_error)
            if self.execution_priority and stationary is not None:
                held = stationary[indices] > step
                finger_cost = finger_cost * (1. + .4 * self.execution_priority * held)
            cost = (finger_cost + .075 * smooth + .16 * wrists + .10 * orientation
                    + .12 * relative + .08 * relative_rotation)
            cost = cost + self.wrist_recovery_priority * (.16 * wrists + .10 * relative)
            if self.live_contact_weight:
                goal_gap = cross_hand_landmark_distances(goal)
                actual_gap = cross_hand_landmark_distances(measured)
                nearby = ((.045 - goal_gap) / .025).clamp(0., 1.)
                gap_error = ((goal_gap - actual_gap).clamp_min(0.) / .007).square()
                gap_cost = (nearby * (torch.sqrt(1. + gap_error) - 1.)).sum(-1) / (nearby.sum(-1) + 1.)
                estimated_depth = torch.nn.functional.softplus(contacts[:, 2])
                penetration = (estimated_depth - 1.15).clamp_min(0.)
                hard_contact = torch.nn.functional.softplus(contacts[:, 1])
                hazard = .65 * hard_contact + .35 * robust_distance(penetration.square(), .85)
                if self.same_hand_contact_focus:
                    with torch.no_grad():
                        proximity = torch.maximum(same_hand_proximity(goal), same_hand_proximity(measured))
                    hazard = hazard * (1. + self.same_hand_contact_focus*proximity)
                cost = cost + self.live_contact_weight * (.65 * gap_cost + .35 * hazard)
                if step == 2:
                    self.metrics["anchor_recovery_contact_depth_mm"] = estimated_depth.detach().mean()
                    self.metrics["anchor_recovery_hard_contact_probability"] = contacts[:, 1].sigmoid().detach().mean()
            # In the first delayed interval the new action has not yet been applied.
            if step == 0:
                cost = cost.masked_fill(delay, 0.)
            terms.append(cost)
            tips.append(tip_error.norm(dim=-1).detach())
            if step == 2 and self.elbow_physical_weight:
                geometry = self.elbow_geometry(following[:, :DOF])
                reference = self.elbow_reference
                target_upper = reference.reference.upper_arm[frame]
                target_speed = (target_upper - reference.reference.upper_arm[previous]) / DT
                if getattr(self.bank, "stability", False):
                    from .stability import stationary_wrist_gate
                    gate = stationary_wrist_gate(goal, previous_goal, current_pose).detach()[..., None]
                    target_upper = torch.lerp(target_upper, current_geometry.upper_arm.detach(), gate)
                    target_speed = target_speed * (1. - gate)
                if getattr(self.bank, "stability_v2", False):
                    target_upper = current_geometry.upper_arm.detach()
                    target_speed = torch.zeros_like(target_upper)
                lateral = ((geometry.upper_arm - target_upper) * reference.outward).sum(-1)
                speed = (geometry.upper_arm - current_geometry.upper_arm.detach()) / DT
                lateral_speed_error = ((speed - target_speed) * reference.outward).sum(-1)
                with torch.no_grad():
                    position_error = squared(measured.wrist - goal.wrist) / .019**2
                    rotation_error = chord2(measured.rotation, goal.rotation) / .115**2
                    priority = (.15 + .85 / (1. + position_error + rotation_error))
                    priority = priority * reference.elbow_confidence[frame]
                slack = .001 if getattr(self.bank, "stability_v2", False) else .018
                displacement = (lateral.abs() - slack).clamp_min(0.)
                position_cost = priority * robust_distance(displacement.square(), .028)
                speed_cost = priority * robust_distance(lateral_speed_error.square(), .12)
                elbow_cost = (.61 * position_cost.mean(-1) + .22 * position_cost.amax(-1)
                              + .17 * speed_cost.mean(-1)).mean()
                self.metrics["anchor_actual_elbow_lateral_p95_mm"] = torch.quantile(lateral.detach().abs(), .95) * 1000.
                self.metrics["anchor_actual_elbow_lateral_speed_error_m_s"] = lateral_speed_error.detach().abs().mean()
            if step < 2:
                observed = self.observation.forward_current(following + bias, goal, next_command)
                if self.observation.history_frames > 1:
                    history = torch.cat((history[:, 1:], observed[:, None]), dim=1)
                    observed = history.flatten(1)
                action = actor(observed)
                advance = 1 if stationary is None else (stationary[indices] <= step + 1).long()
                previous, frame = frame, self.bank.advance_frame(frame, advance, recorded, max_steps=1)
                state, command, previous_goal, current_pose = following, next_command, goal, measured
                current_geometry = measured_geometry
        self.metrics["anchor_recovery_final_tip_p95_mm"] = torch.quantile(tips[-1].flatten(), .95) * 1000.
        self.metrics["anchor_recovery_delayed_fraction"] = delay.float().mean()
        return (.20 * terms[0] + .35 * terms[1] + .45 * terms[2]).mean() + self.elbow_physical_weight * elbow_cost

    def __call__(self, actor):
        sample = self.sample()
        q, observed, observation, frames, hold, offset, recovery, previous, state, delayed, live_indices, recorded = sample
        target, arm_frames, hand_frames = self.command_targets(observed, frames, hold, delayed, recorded)
        action = actor(observation)
        request = self.command.desired(action)
        bounded = request.clamp(self.command.lower, self.command.upper)
        # Estimate the steady measured posture for training supervision.
        estimated = (bounded - offset).clamp(self.command.lower, self.command.upper)
        geometry = self.observation.fk(estimated)
        pose = geometry.pose
        arm_gate = self.arm_anchor_gate(observed, frames, recorded).masked_fill(hold[:, None], 1.)
        wrist = (arm_gate * robust_distance(squared(pose.wrist - target.wrist), .010)).mean(-1)
        wrist_rotation = (arm_gate * robust_distance(chord2(pose.rotation, target.rotation), .085)).mean(-1)
        hand_error = pose.hand - target.hand
        hand_error2 = squared(hand_error)
        hand = robust_distance(hand_error2, .010)
        hand_gate = self.hand_anchor_gate(observed, frames, recorded).masked_fill(hold[:, None, None], 1.)
        weighted_hand = hand_gate[..., None] * hand
        hand_score = .25 * weighted_hand.mean((1, 2, 3)) + .15 * weighted_hand[..., -1].mean((1, 2))
        if self.tip_tail_weight:
            tail = fingertip_position_tail_cost(hand_error[..., -1, :]) * hand_gate.mean((1, 2))
            hand_score = hand_score + self.tip_tail_weight * tail
        tip_error2 = hand_error2[..., -1]
        hand_sample_weight = self.hand_sample_weight(tip_error2, frames)
        tip_mm = tip_error2.detach().sqrt().flatten() * 1000.
        self.metrics["anchor_request_tip_mm"] = tip_mm.mean()
        self.metrics["anchor_request_tip_p95_mm"] = torch.quantile(tip_mm, .95)
        if self.hold_fraction:
            held_tip = tip_error2[hold].detach().sqrt().flatten() * 1000.
            self.metrics["anchor_hold_fraction"] = hold.float().mean()
            if held_tip.numel():
                self.metrics["anchor_hold_tip_p95_mm"] = torch.quantile(held_tip, .95)
        if self.hold_recovery_noise or self.recovery_frames:
            recovery_tip = tip_error2[recovery].detach().sqrt().flatten() * 1000.
            self.metrics["anchor_hold_recovery_fraction"] = recovery.float().mean()
            if recovery_tip.numel():
                self.metrics["anchor_hold_recovery_tip_p95_mm"] = torch.quantile(recovery_tip, .95)
        relative_gate = arm_gate.amin(-1)
        relative = relative_gate * robust_distance(squared(pose.relative_position - target.relative_position), .015)
        relative_rotation = relative_gate * robust_distance(chord2(pose.relative_rotation, target.relative_rotation), .10)

        # A box penalty supplies gradients on requests clipped for the FK.
        box_error = (request - bounded) / self.command.rate_scale
        large_step = ((request - q).abs() / self.command.rate_scale - 2.).clamp_min(0.)
        feasible = robust_distance(box_error.square(), 1.).mean(-1)
        smooth = robust_distance(large_step.square(), 1.).mean(-1)
        score = (.28 * wrist + .14 * wrist_rotation + self.hand_weight * hand_sample_weight * hand_score
                 + self.relative_pose_weight * (.12 * relative + .06 * relative_rotation)
                 + .035 * feasible + .035 * smooth)
        if self.elbow_weight:
            elbow_geometry = self.elbow_geometry(estimated) if self.elbow_nullspace else geometry
            score = score + self.elbow_weight * self.elbow_tracking_cost(elbow_geometry, target, arm_frames, q, hold)
        pose_loss = score.mean()
        self.metrics["anchor_pose_loss"] = pose_loss.detach()
        loss = self.weight * pose_loss
        if (self.hand_physical_weight or self.wrist_physical_weight) and live_indices is not None:
            physical_loss = self.live_execution_cost(action, live_indices)
            self.metrics["anchor_live_execution_loss"] = physical_loss.detach()
            loss = loss + physical_loss
        if self.live_recovery_weight and live_indices is not None:
            recovery_loss = self.live_recovery_cost(actor, action[-len(live_indices):], live_indices)
            self.metrics["anchor_live_recovery_loss"] = recovery_loss.detach()
            loss = loss + self.live_recovery_weight * recovery_loss
        if self.hand_feedback_weight:
            stationary = hold if self.independent_feedback else recovery
            feedback_cost = self.hand_feedback_cost(actor, state, observed, previous, bounded, stationary)
            self.metrics["anchor_hand_feedback_loss"] = feedback_cost.detach()
            loss = loss + self.hand_feedback_weight * feedback_cost
        if self.joint_weight:
            joint_loss = self.joint_prior(request - offset, target, arm_frames, hand_frames,
                                          arm_gate, hand_gate, hand_sample_weight, q, hold)
            self.metrics["anchor_joint_loss"] = joint_loss.detach()
            loss = loss + self.joint_weight * joint_loss
        return loss
