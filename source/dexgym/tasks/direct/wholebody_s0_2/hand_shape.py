"""Wrist-local hand supervision on a common skeleton."""

import torch
from torch.nn.functional import normalize


def adjacent_finger_closure(actual, goal):
    """Excess closure along the target separation between neighboring finger links."""
    def vectors(p):
        return p[:, :, 2:, None, 1:, :] - p[:, :, 1:-1, 1:, None, :]
    ref = vectors(goal)
    dist = ref.norm(dim=-1)
    axis = ref / dist.clamp_min(1.e-6).unsqueeze(-1)
    actual_projection = (vectors(actual) * axis).sum(-1)
    excess = (dist - actual_projection - .00075).clamp_min(0.)
    proximity = torch.exp(-(dist / .035).square()) * (dist > .002)
    cost = (torch.sqrt(1. + (excess/.004).square()) - 1.) * proximity
    return .55*cost.flatten(2).mean(-1).mean(-1) + .45*cost.flatten(2).amax(-1).mean(-1)


def shape_components(actual, goal):
    actual_links = actual.diff(dim=-2)
    target_links = goal.diff(dim=-2)
    lengths = actual_links.norm(dim=-1, keepdim=True).detach()
    actual_direction = normalize(actual_links, dim=-1, eps=1e-6)
    target_direction = normalize(target_links, dim=-1, eps=1e-6)
    # Keep the target joint directions and express them on the physical robot skeleton.
    target_tip = actual[..., 0, :].detach() + (target_direction * lengths).sum(dim=-2)
    actual_tip = actual[..., -1, :]
    tip_error = (actual_tip - target_tip).norm(dim=-1)
    direction_error = (actual_direction - target_direction).norm(dim=-1)
    actual_bend = (actual_direction[..., 1:, :] * actual_direction[..., :-1, :]).sum(-1)
    target_bend = (target_direction[..., 1:, :] * target_direction[..., :-1, :]).sum(-1)
    sine_error = (torch.linalg.cross(actual_direction[..., :-1, :], actual_direction[..., 1:, :], dim=-1)
                  - torch.linalg.cross(target_direction[..., :-1, :], target_direction[..., 1:, :], dim=-1))
    bend_error = torch.cat((sine_error, (actual_bend - target_bend).unsqueeze(-1)), dim=-1).norm(dim=-1)
    thumb_to_finger = actual_tip[:, :, 1:] - actual_tip[:, :, :1]
    target_pair = target_tip[:, :, 1:] - target_tip[:, :, :1]
    pair_error = (thumb_to_finger - target_pair).norm(dim=-1)
    pair_weight = .2 + .8 * torch.exp(-target_pair.square().sum(-1) / .035**2)
    return tip_error, direction_error, bend_error, pair_error, pair_weight


def shape_cost(actual, goal):
    tip, direction, bend, pair, proximity = shape_components(actual, goal)

    def robust(value):
        return torch.sqrt(1. + value.square()) - 1.

    # The cross product complements the cosine near extension and retains the
    # direction of flexion in wrist coordinates. No inverse cosine at dot=1.
    points = (actual - goal).norm(dim=-1)
    finger = (.28 * robust(points / .010).mean(-1)
              + .16 * robust(tip / .010)
              + .28 * robust(direction / .25).mean(-1)
              + .28 * robust(bend / .25).mean(-1))
    cost = (.55 * finger.mean((1, 2)) + .35 * finger.amax(dim=2).mean(dim=1)
            + .10 * (proximity * robust(pair / .014)).sum((1, 2)) / proximity.sum((1, 2)))
    return cost, tip, direction


@torch.no_grad()
def vary_hand_observation(history, width, generator):
    frames = history.reshape(len(history), -1, width).clone()
    shape = frames[:, :, 180:300].reshape(len(frames), -1, 2, 5, 4, 3)
    overall = shape.new_empty(len(shape), 1, 2, 1, 1, 1).uniform_(.78, 1.24, generator=generator)
    proportions = shape.new_empty(len(shape), 1, 2, 5, 3, 1).uniform_(.72, 1.30, generator=generator)
    # Landmarks in the observation are scaled by ten. Perturb roots consistently across history.
    root_offset = shape.new_empty(len(shape), 1, 2, 1, 1, 3).uniform_(-.14, .14, generator=generator)
    finger_offset = shape.new_empty(len(shape), 1, 2, 5, 1, 3).uniform_(-.05, .05, generator=generator)
    roots = shape[..., :1, :] * overall + root_offset + finger_offset
    links = (shape.diff(dim=-2) * overall * proportions).cumsum(dim=-2)
    frames[:, :, 180:300] = torch.cat((roots, roots + links), dim=-2).flatten(2)
    return frames.flatten(1)


class HandShapeAnchor:
    hand_physical_weight = wrist_physical_weight = live_recovery_weight = 0.

    def __init__(self, env, teacher, augment):
        self.env, self.teacher, self.augment = env, teacher, augment
        self.generator = torch.Generator(device=env.device).manual_seed(701)
        self.metrics = {}

    def __call__(self, actor):
        from .interface import HAND_DEADZONE, HAND_STEP_RAD, hand_increment, robust

        env = self.env
        rows = torch.randint(env.num_envs, (min(512, env.num_envs),), device=env.device, generator=self.generator)
        observe, actuator = env.task.observe, env.task.command
        history = env.observation(rows).detach()
        previous = env.history.previous_command[rows]
        reference = env.reference(env.frame[rows], env.bridge[rows], env.lag[rows], env.ref_bias[rows])
        joints = env.state[rows, :54].detach()
        target = env.goal.hand[rows]
        if self.augment:
            count = len(rows) // 2
            augmented = vary_hand_observation(history[:count], observe.frame_features, self.generator)
            history = torch.cat((augmented, history[count:]))
        action = actor(history)
        command = actuator(observe.normalized(action, reference, previous), previous)
        increment = hand_increment(action)
        # Keep the deployed command in the forward pass and retain a corrective
        # derivative through the hard deadzone and bounds in this auxiliary only.
        dead = (action.detach().abs() <= HAND_DEADZONE).to(action.dtype)
        request = previous + increment + dead * HAND_STEP_RAD * (action - action.detach())
        hand = ~actuator.arm
        applied = command.detach() + (request - request.detach())
        q = torch.where(actuator.arm, joints, applied)
        shape, tips, direction = shape_cost(observe.fk(q).pose.hand, target)
        outside = (request - request.clamp(actuator.lower, actuator.upper))[:, hand]
        with torch.no_grad():
            arm_reference = self.teacher(env.observation(rows))[:, actuator.arm]
        arm_anchor = (action[:, actuator.arm] - arm_reference).square().mean()
        loss = .70 * shape.mean() + .18 * robust(outside / HAND_STEP_RAD).mean() + .35 * arm_anchor
        self.metrics = dict(hand_shape_anchor=float(shape.detach().mean()),
                            hand_shape_normalized_tip_mm=float(tips.detach().mean() * 1000),
                            hand_shape_direction_chord=float(direction.detach().mean()),
                            hand_shape_arm_anchor=float(arm_anchor.detach()))
        return loss
