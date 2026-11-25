"""Shared causal stability conventions, independent of archived joint labels."""
import math

import torch

from . import DT

VERSION = "static_elbow_long_hold_v1"
VERSION_V2 = "observable_nullspace_hold_v2"
ELBOW_TRANSPORT_VERSION = "transported_swivel_and_lateral_envelope_v1"
ELBOW_HOLD_ANCHOR_VERSION = "stationary_elbow_swivel_anchor_v1"


def stationary_wrist_gate(goal, previous_goal, measured):
    """Per-arm gate: fixed external target AND already-near measured wrist."""
    speed2 = (goal.wrist - previous_goal.wrist).square().sum(-1) / .0006**2
    rotation2 = (goal.rotation - previous_goal.rotation).square().sum((-1, -2)) / (2. * .006**2)
    position_error = (goal.wrist - measured.wrist).square().sum(-1) / .015**2
    rotation_error = (goal.rotation - measured.rotation).square().sum((-1, -2)) / (2. * .12**2)
    return torch.exp(-speed2 - rotation2) / (1. + position_error + rotation_error)


def capture_stationary_elbow_anchor(geometry, goal, hold):
    """Capture the reached configuration of each aligned wrist at hold reset."""
    valid = hold[:, None] & (stationary_wrist_gate(goal, goal, geometry.pose) >= .5)
    valid &= geometry.swivel.norm(dim=-1) >= .02
    return torch.cat((geometry.ray, geometry.swivel, valid[..., None].to(geometry.ray.dtype)), dim=-1).detach()


def stationary_elbow_anchor_cost(geometry, previous, goal, old_goal, anchor, hold):
    """Measure swivel drift since reset after transporting it to the current wrist ray."""
    anchor_ray, anchor_swivel, valid = anchor.split((3, 3, 1), dim=-1)
    ray = geometry.ray.detach()
    ray_alignment = (anchor_ray * ray).sum(dim=-1, keepdim=True)
    projection = (anchor_swivel * ray).sum(dim=-1, keepdim=True) / (1. + ray_alignment).clamp_min(1e-4)
    transported = anchor_swivel - projection * (anchor_ray + ray)
    old_radius = transported.norm(dim=-1)
    radius = geometry.swivel.norm(dim=-1)
    direction = geometry.swivel / radius.clamp_min(.02)[..., None]
    reference = transported / old_radius.clamp_min(.02)[..., None]
    displacement = (direction - reference).norm(dim=-1) * torch.minimum(radius, old_radius)
    usable = hold[:, None] & valid.squeeze(-1).bool() & (ray_alignment.squeeze(-1) > .5)
    usable &= radius.detach() >= .02
    gate = (stationary_wrist_gate(goal, old_goal, previous.pose) * usable).detach()
    penalty = torch.log1p(((displacement - .003).clamp_min(0.) / .018).square())
    return (gate * penalty).mean(dim=-1), displacement, gate


def static_elbow_motion_cost(current, previous, ray, goal, previous_goal, measured):
    """Quiet wrists prefer the current elbow, with no attraction to archived IK."""
    gate = stationary_wrist_gate(goal, previous_goal, measured).detach()
    tangent = torch.nn.functional.normalize(torch.linalg.cross(ray.detach(), previous.detach()), dim=-1, eps=1e-6)
    velocity = ((current - previous.detach()) * tangent).sum(-1) / .02
    return .10 * (gate * torch.log1p((velocity / .025).square())).mean(-1)


def transported_swivel_velocity(geometry, previous):
    """Elbow motion tangent to the wrist ray after transporting the old swivel."""
    old_ray, ray = previous.ray.detach(), geometry.ray.detach()
    old_swivel = previous.swivel.detach()
    denominator = (1. + (old_ray*ray).sum(-1, keepdim=True)).clamp_min(1e-4)
    transported = old_swivel - ((old_swivel*ray).sum(-1, keepdim=True)/denominator)*(old_ray+ray)
    tangent = torch.nn.functional.normalize(torch.linalg.cross(ray, transported, dim=-1), dim=-1, eps=.008)
    return ((geometry.upper_arm - transported)*tangent).sum(-1)/DT


def swivel_continuity_cost(geometry, previous, goal, old_goal, target_jump):
    # Some redundancy motion remains costly during motion, while a stationary
    # target increases the effect independently of a small wrist residual.
    speed2 = (goal.wrist-old_goal.wrist).square().sum(-1)/.0006**2
    rotation2 = (goal.rotation-old_goal.rotation).square().sum((-1, -2))/(2.*.006**2)
    quiet = torch.exp(-speed2-rotation2)
    wrist_confidence = stationary_wrist_gate(goal, goal, previous.pose)
    gate = (.14 + .86*quiet) * (.36 + .64*wrist_confidence)
    if target_jump is not None:
        gate = gate.masked_fill(target_jump[:, None], 0.)
    velocity = transported_swivel_velocity(geometry, previous)
    return (gate.detach() * torch.log1p((velocity/.025).square())).mean(-1)


@torch.no_grad()
def fit_elbow_envelope(bank, fk, outward):
    """Fit broad per-arm lateral bounds from safe, wrist-aligned training poses."""
    if bank.split != "train":
        raise ValueError("Elbow geometry bounds must use the training split")
    bands = []
    counts = {}
    for name, mask in (("teleop", ~bank.egopro), ("egopro", bank.egopro)):
        ids = bank.eligible[mask[bank.eligible]]
        if len(ids) == 0:
            continue
        ids = ids[torch.linspace(0, len(ids)-1, min(16384, len(ids)), device=ids.device).long()]
        samples = [[], []]
        for idx in ids.split(2048):
            g = fk(bank.rsi_q[idx])
            p_error = (g.pose.wrist-bank.pose.wrist[idx]).norm(dim=-1)
            r_error = (g.pose.rotation-bank.pose.rotation[idx]).square().sum((-1, -2))
            good = (p_error <= .015) & (r_error <= 8.*math.sin(math.radians(12.)/2)**2)
            lateral = (g.upper_arm*outward).sum(-1)
            for side in range(2):
                samples[side].append(lateral[good[:, side], side])
        values = [torch.cat(parts) for parts in samples]
        counts[name] = [int(len(v)) for v in values]
        if min(len(v) for v in values) < 128:
            raise ValueError(f"Too few wrist-aligned {name} poses to fit elbow bounds")
        quantile = outward.new_tensor([.01, .99])
        bands.append(torch.stack([torch.quantile(v, quantile) for v in values], dim=1))
    bands = torch.stack(bands)
    bounds = torch.stack((bands[:, 0].amin(0)-.012, bands[:, 1].amax(0)+.012))
    return bounds, counts


def elbow_envelope_cost(geometry, prior, goal, outward, bounds):
    lateral = (geometry.upper_arm*outward).sum(-1)
    excess = torch.maximum(bounds[0]-lateral, lateral-bounds[1]).clamp_min(0.)
    wrist_confidence = stationary_wrist_gate(goal, goal, prior.pose).detach()
    return ((.20+.80*wrist_confidence) * torch.log1p((excess/.03).square())).mean(-1)


def nullspace_displacement_cost(fk, request, q, arms, rate_scale, *, return_jacobian=False):
    """Minimum-motion arm redundancy, without opposing 6D wrist corrections."""
    with torch.no_grad():
        jacobian = fk.wrist_jacobian(q.detach())
        _, _, vh = torch.linalg.svd(jacobian, full_matrices=True)
        direction = vh[..., -1, :]
        scale = (direction.square() * rate_scale[arms].square()).sum(-1).sqrt().clamp_min(1e-6)
    displacement = ((request - q.detach())[:, arms] * direction).sum(-1) / scale
    cost = torch.sqrt(1. + displacement.square()) - 1.
    return (cost, jacobian) if return_jacobian else cost
