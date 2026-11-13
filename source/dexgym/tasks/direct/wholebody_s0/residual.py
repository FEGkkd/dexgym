"""Shared reference, command projection and measured-state reward for wholebody_s0."""

from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softplus

from dexgym.algorithms.supermimic.dex_reference.networks import Actor
from ..hand2_cube_supermimic.kinematics import HandKinematics
from ..hand2_cube_supermimic.rotations import axis_angle_to_mat, mm, mv
from ..marvin54_tracking.task import JointCodec
from .bag import digest
from .fingertip import FingertipActor, HAND_SCALE, PalmOrientation, TIP_ERROR_SCALE_M
from .intent import INTENT_DIM, HumanIntentGeometry, IntentTargets, source_orders
from .library import MotionLibrary
from .robot import FINGER_SITES, FIRMWARE_FINGERS

N = 54
DT = 0.02
AUDIT_DEPTH = 0.0005
HARD_DEPTH = 0.003
DR_DIM = 6
OBS_DIM = 6 * N
REWARD_WEIGHTS = {
    "wrist_position": 1.35, "wrist_rotation": .65, "left_hand": 1.0, "right_hand": 1.0,
    "relative_position": .50, "relative_rotation": .25, "joint": .06,
    "action_rate": .012, "action_magnitude": .002, "projection": .018,
    "contact": .25, "hard_contact": .85, "joint_violation": .30,
}
FINGERTIP_REWARD_WEIGHTS = {
    **REWARD_WEIGHTS, "left_hand": 2.5, "right_hand": 2.5, "action_rate": .03,
}
HUMAN_INTENT_WEIGHTS = {
    "intent_wrist_position": 1.2, "intent_wrist_rotation": .8,
    "intent_left_hand": 2.4, "intent_right_hand": 2.4,
    "intent_relative_position": .55, "intent_relative_rotation": .25,
}
INTENT_EXECUTION_WEIGHTS = {
    "wrist_position": 1.15, "wrist_rotation": .60, "left_hand": 1., "right_hand": 1.,
    "relative_position": .40, "relative_rotation": .25, "joint": .25,
}
INTENT_COST_WEIGHTS = {
    "action_rate": .8, "action_magnitude": .4, "projection": .02,
    "contact": .25, "hard_contact": 1.0, "joint_violation": .30,
}
INTENT_EXECUTION_COEF, INTENT_SMOOTHNESS_COEF, INTENT_RESIDUAL_COEF = .35, .04, .012
INTENT_COMMAND_RATE_WEIGHT = .20
POSE_GOAL_VERSION = "mjcf_tcp_wrist_local_fingertip_se3_v1"
POSE_TRACKING_WEIGHTS = {
    **REWARD_WEIGHTS, "wrist_position": 1.35, "wrist_rotation": 1.,
    "left_hand": 2.25, "right_hand": 2.25,
    "relative_position": .95, "relative_rotation": .65, "joint": .04,
    "action_rate": .035, "action_magnitude": .004, "command_rate": .008,
    "projection": .018, "hard_contact": 1.,
}
# Rotation scales are chord lengths 2*sin(theta/2). Position scales are meters.
POSE_TRACKING_SCALES = {
    "wrist_position": .016, "wrist_rotation": .095,
    "tip_position": .004, "tip_rotation": .10,
    "relative_position": .025, "relative_rotation": .125, "joint": .12,
}
POSE_HAND_WEIGHTS = {"position": .50, "rotation": .30, "worst_position": .12, "worst_rotation": .08}
REWARD_PROFILES = {"v1": REWARD_WEIGHTS, "fingertip": FINGERTIP_REWARD_WEIGHTS, "retarget_pose": POSE_TRACKING_WEIGHTS,
                   "human_intent": {
    **HUMAN_INTENT_WEIGHTS, **INTENT_EXECUTION_WEIGHTS, **INTENT_COST_WEIGHTS,
}}


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def encode_policy_observation(q, v, reference, previous_reference, command, previous_action, home):
    causal_vel = (reference - previous_reference) / DT
    return torch.cat((q - home, v, reference - q, causal_vel, command - q, previous_action), -1)


def encode_intent_offsets(wrist, rotation, local_tip, ref_wrist, ref_rotation, ref_local_tip):
    pos = (wrist - ref_wrist).flatten(1) / .030
    orientation = (ref_rotation.transpose(-1, -2) @ (rotation - ref_rotation))[..., :2].flatten(1) / .25
    tips = (local_tip - ref_local_tip).flatten(1) / .040
    return torch.cat((pos, orientation, tips), dim=-1).clamp(-5., 5.)


def project_command(reference, action, command, lo, hi, scale, slew):
    proposed = reference + scale * action.clamp(-1., 1.)
    lower = torch.maximum(lo, command - slew)
    upper = torch.minimum(hi, command + slew)
    return torch.minimum(torch.maximum(proposed, lower), upper)


def scaled_score(error2, scale):
    # Pseudo-Huber error keeps a gradient when the retargeting gap is large.
    return (2. - (1. + error2 / scale**2).sqrt()).flatten(1).mean(-1)


def human_intent_score(pose, goals):
    wrist, rotation, local, relative, relative_r, _ = pose
    hw, hr, hl, hp, ht = goals
    local_tip, human_tip = local[..., -1, :], hl[..., -1, :]
    palm_error = (local_tip - human_tip).square().sum(-1)
    links_error = (local[..., 1:3, :] - hl[..., :2, :]).square().sum(-1)
    pinch = (local_tip[:, :, 1:] - local_tip[:, :, :1]).norm(dim=-1)
    human_pinch = (human_tip[:, :, 1:] - human_tip[:, :, :1]).norm(dim=-1)
    errors = {
        "intent_wrist": (wrist - hw).square().sum(-1),
        "intent_rotation": .5 * (rotation - hr).square().sum((-1, -2)),
        "intent_local_tip": palm_error,
        "intent_relative": (relative - hp).square().sum(-1),
        "intent_relative_rotation": .5 * (relative_r - ht).square().sum((-1, -2)),
    }
    scores = {
        "intent_wrist_position": scaled_score(errors["intent_wrist"], .025),
        "intent_wrist_rotation": scaled_score(errors["intent_rotation"], .25),
        "intent_relative_position": scaled_score(errors["intent_relative"][:, None], .045),
        "intent_relative_rotation": scaled_score(errors["intent_relative_rotation"][:, None], .32),
    }
    for i, side in enumerate(("left", "right")):
        finger = palm_error[:, i]
        scores[f"intent_{side}_hand"] = (.65 * scaled_score(finger, .030)
            + .15 * scaled_score(finger.topk(2, dim=-1).values, .036)
            + .12 * scaled_score(links_error[:, i], .030)
            + .08 * scaled_score((pinch[:, i] - human_pinch[:, i]).square(), .025))
    return sum(HUMAN_INTENT_WEIGHTS[key] * value for key, value in scores.items()), errors


class ReferenceBank:
    def __init__(self, manifest, split, model, device, *, intent_model=None):
        self.library = MotionLibrary.open(manifest)
        self.names = [model.joint(int(i)).name for i in np.argsort(model.jnt_qposadr)]
        contract_names = self.library.manifest["joint_names"]
        if set(self.names) != set(contract_names) or len(self.names) != N or model.nq != N or model.nv != N:
            raise ValueError("Motion library and simulator joint names must match")
        if self.library.manifest["rate_hz"] != 50:
            raise ValueError("wholebody_s0 requires a 50 Hz motion library")
        self.entries = self.library.entries(split)
        if not self.entries:
            raise ValueError(f"No clips in {split}")
        source_groups = {}
        for entry in self.library.manifest["clips"]:
            if source_groups.setdefault(entry["source_episode"], entry["split"]) != entry["split"]:
                raise ValueError("A recording appears in multiple splits")
        self.order = np.array([contract_names.index(name) for name in self.names])
        self.manifest_sha256 = digest(manifest)
        positions, velocities, stored_wrists, stored_tips = [], [], [], []
        intent_geometry = HumanIntentGeometry(intent_model) if intent_model is not None else None
        orders, intent_goals = {}, []
        start, end, eligible, offset = [], [], [], 0
        for entry in self.entries:
            source = (self.library.root / entry["path"]).resolve()
            if not source.is_relative_to(self.library.root) or digest(source) != entry["sha256"]:
                raise ValueError(f"Invalid reference checksum: {source}")
            clip = self.library.load_clip(entry)
            n = entry["frames"]
            positions.append(clip["q_ref"][:, self.order])
            velocities.append(clip["qd_ref"][:, self.order])
            stored_wrists.append(clip["robot_wrist_pose"])
            stored_tips.append(clip["robot_fingertip_pose"])
            if intent_geometry is not None:
                if entry["source_episode"] not in orders:
                    orders[entry["source_episode"]] = source_orders(self.library, entry)
                intent_goals.append(intent_geometry.targets(clip, orders[entry["source_episode"]]))
            start.append(offset)
            end.extend([offset + n - 1] * n)
            eligible.extend(range(offset, offset + n - min(24, n - 1)))
            offset += n
        self.frames = offset
        self.q = tensor(np.concatenate(positions), device)
        self.qd = tensor(np.concatenate(velocities), device)
        self.stored_wrists = tensor(np.concatenate(stored_wrists), device)
        self.stored_tips = tensor(np.concatenate(stored_tips), device)
        self.intent_targets = (IntentTargets(*(tensor(np.concatenate(group), device)
                                               for group in zip(*intent_goals, strict=True)))
                               if intent_goals else None)
        self.start = torch.as_tensor(start, device=device, dtype=torch.long)
        self.end = torch.as_tensor(end, device=device, dtype=torch.long)
        self.reset_frames = torch.as_tensor(eligible, device=device, dtype=torch.long)

    def next(self, frame):
        return torch.minimum(frame + 1, self.end[frame])

    def sample(self, size, generator):
        return self.reset_frames[torch.randint(len(self.reset_frames), (size,), device=self.q.device, generator=generator)]


class Geometry:
    """Differentiable positions and frames from the actual MJCF hinge tree."""

    def __init__(self, model, names, device):
        if np.any(model.qpos0):
            raise ValueError("Geometry expects zero model hinge references")
        tcp = [model.site(f"tcp_{side}").id for side in "LR"]
        tips = [model.site(f"{side}_{finger}").id for side in "lr" for finger in FINGER_SITES]
        palms = [model.body(f"{side}_wrist").id for side in "lr"]
        self.tree = HandKinematics(model, range(model.nbody), [model.joint(name).id for name in names], tcp + tips, device)
        self.palms = palms
        joint_ids = [
            int(model.actuator(f"{side}_{finger}{j}").trnid[0])
            for side in "lr" for finger in FIRMWARE_FINGERS for j in (1, 2, 3)
        ]
        self.joint_bodies = model.jnt_bodyid[joint_ids].tolist()
        self.joint_offsets = tensor(model.jnt_pos[joint_ids], device)
        from ..hand2_cube_supermimic.rotations import quat_to_mat

        self.site_rot = quat_to_mat(tensor(model.site_quat[tcp + tips], device))

    def pose(self, q, *, wrist_local=False):
        # Reuse the model tree in depth order. Only positions and rotations
        # enter the objective, so the FK does not propagate spatial velocities.
        p = q.new_zeros(q.shape[0], 1, 3)
        r = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], 1, 3, 3)
        for level in self.tree.levels:
            parents, joints = level["parent"], level["joint"]
            pp, rp = p[:, parents], r[:, parents]
            rot, anchor = level["rot"], level["anchor"]
            hinge = axis_angle_to_mat(level["axis"], q[:, joints] * level["moving"])
            pos = pp + mv(rp, level["off"] + mv(rot, anchor - mv(hinge, anchor)))
            p = torch.cat((p, pos), 1)
            r = torch.cat((r, mm(mm(rp, rot), hinge)), 1)
        restore = self.tree.restore
        p, r = p[:, restore], r[:, restore]
        sb = self.tree.tip_bodies
        site = p[:, sb] + mv(r[:, sb], self.tree.tip_offset)
        site_r = mm(r[:, sb], self.site_rot)
        tip = site[:, 2:].reshape(-1, 2, 5, 3)
        finger = (p[:, self.joint_bodies] + mv(r[:, self.joint_bodies], self.joint_offsets)).reshape(-1, 2, 5, 3, 3)
        hand = torch.cat((finger, tip.unsqueeze(-2)), -2)
        palm_p, palm_r = p[:, self.palms], r[:, self.palms]
        local = mv(palm_r.transpose(-2, -1)[:, :, None, None], hand - palm_p[:, :, None, None])
        wr_p, wr_r = site[:, :2], site_r[:, :2]
        rel_p = mv(wr_r[:, 0].transpose(-2, -1), wr_p[:, 1] - wr_p[:, 0])
        rel_r = mm(wr_r[:, 0].transpose(-2, -1), wr_r[:, 1])
        pose = wr_p, wr_r, local, rel_p, rel_r, tip
        if wrist_local:
            inverse_wrist = wr_r.transpose(-1, -2)[:, :, None]
            tip_p = mv(inverse_wrist, tip - wr_p[:, :, None])
            tip_r = mm(inverse_wrist, site_r[:, 2:].reshape(-1, 2, 5, 3, 3))
            return (*pose, tip_p, tip_r)
        return pose


class ResidualTask:
    state_dim = 2 * N
    action_dim = N
    dynamics_dim = 2 * N + DR_DIM
    policy_dim = OBS_DIM
    context_dim = OBS_DIM + DR_DIM + 1
    contact_dim = 2
    termination_context_dim = N
    transition_termination = True
    error_floor = [.0008] * N + [.025] * N
    codec = JointCodec
    def __init__(self, env, model, bank, *, domain, transfer_model=None, reward_profile="v1", hand_model=None):
        self.env = self.reader = env
        self.device = env.device
        self.bank = bank
        self.domain = domain
        self.reward_profile = reward_profile
        self.reward_weights = REWARD_PROFILES[reward_profile]
        self.human_intent = reward_profile == "human_intent"
        self.retarget_pose = reward_profile == "retarget_pose"
        self.max_step_reward = sum(self.reward_weights[key] for key in (
            "wrist_position", "wrist_rotation", "left_hand", "right_hand",
            "relative_position", "relative_rotation", "joint",
        ))
        if self.human_intent:
            if bank.intent_targets is None or hand_model is None:
                raise ValueError("Human intent needs recorded goals and the target robot geometry")
            self.max_step_reward = sum(HUMAN_INTENT_WEIGHTS.values()) + INTENT_EXECUTION_COEF * sum(INTENT_EXECUTION_WEIGHTS.values())
        self.fingertip_priority = reward_profile == "fingertip"
        self.uses_hand_actor = self.fingertip_priority or self.human_intent or self.retarget_pose
        self.freeze_actor_norm = self.uses_hand_actor
        if self.uses_hand_actor and hand_model is None:
            raise ValueError("This reward needs the teleoperation model for task-space geometry")
        self.dynamics_dim = 2 * N + domain.physical.shape[-1]
        self.policy_dim = OBS_DIM + INTENT_DIM if self.human_intent else OBS_DIM
        self.context_dim = self.policy_dim + domain.physical.shape[-1] + 1
        self.geometry = Geometry(model, bank.names, self.device)
        ids = [model.joint(name).id for name in bank.names]
        controls = [int(np.flatnonzero(model.actuator_trnid[:, 0] == jid).item()) for jid in ids]
        self.joint_lo, self.joint_hi = tensor(model.jnt_range[ids], self.device).T
        control_lo, control_hi = tensor(model.actuator_ctrlrange[controls], self.device).T
        self.lo = torch.maximum(self.joint_lo, control_lo)
        self.hi = torch.minimum(self.joint_hi, control_hi)
        if ((bank.q < self.lo - 1e-6) | (bank.q > self.hi + 1e-6)).any():
            raise ValueError("The native reference exceeds joint/actuator position limits")
        self.home = tensor(model.key("home").qpos.copy(), self.device)
        self.palm_orientation = PalmOrientation(model, bank.names, self.home.cpu(), DT) if self.uses_hand_actor else None
        self.arm = torch.as_tensor([name.startswith("Joint") for name in bank.names], device=self.device)
        self.groups = [
            torch.as_tensor([i for i, name in enumerate(bank.names) if name.endswith("_" + side)], device=self.device)
            for side in "LR"
        ] + [
            torch.as_tensor([i for i, name in enumerate(bank.names) if name.startswith(side + "_")], device=self.device)
            for side in "lr"
        ]
        self.hand_groups = torch.stack(self.groups[2:])
        self.scale = torch.where(self.arm, .035, HAND_SCALE if self.uses_hand_actor else .060)
        self.slew = DT * torch.where(self.arm, 1.35, 5.0)
        self.reward_geometry = Geometry(hand_model, bank.names, self.device) if self.uses_hand_actor else self.geometry
        self.kp = tensor(model.actuator_gainprm[controls, 0], self.device)
        self.kd = tensor(-model.actuator_biasprm[controls, 2], self.device)
        self.effort_lo, self.effort_hi = tensor(model.actuator_forcerange[controls], self.device).T
        self.transfer_delta = None
        if transfer_model is not None:
            from ..marvin54_tracking.model import drive_parameters

            if transfer_model.body_gravcomp.any():
                raise ValueError("The transfer profile expects a target with gravity compensation disabled")
            target = drive_parameters(transfer_model, bank.names)
            self.transfer_delta = tuple(
                tensor([target[name][key] for name in bank.names], self.device) / original - 1.
                for key, original in (("kp", self.kp), ("kd", self.kd), ("effort", self.effort_hi))
            )
            if not all(torch.isfinite(delta).all() for delta in self.transfer_delta):
                raise ValueError("Transfer drive ratios must be finite")
        self.simulator_context = domain.physical
        self.encoder_bias = domain.observation_bias
        with torch.no_grad():
            parts = [self.geometry.pose(q) for q in self.bank.q.split(1024)]
            self.goal_geometry = tuple(torch.cat(group) for group in zip(*parts, strict=True))
            self.reward_goals = self.goal_geometry
            if self.uses_hand_actor:
                parts = [self.measured_pose(q) for q in self.bank.q.split(1024)]
                self.reward_goals = tuple(torch.cat(group) for group in zip(*parts, strict=True))
            if self.human_intent:
                goals = self.bank.intent_targets
                wp, wr, local, *_ = self.reward_goals
                self.intent_reference = wp, wr, local[..., -1, :]
                self.intent_offsets = encode_intent_offsets(goals.wrist, goals.rotation, goals.local[..., -1, :],
                                                            *self.intent_reference)
                left = goals.rotation[:, 0].transpose(-1, -2)
                relative = mv(left, goals.wrist[:, 1] - goals.wrist[:, 0])
                self.intent_geometry = (goals.wrist, goals.rotation, goals.local, relative, mm(left, goals.rotation[:, 1]))

    def make_actor(self):
        if self.uses_hand_actor:
            return FingertipActor(OBS_DIM, N, self.hand_groups, self.palm_orientation, intent=self.human_intent,
                                  train_arm=self.retarget_pose)
        return Actor(self.policy_dim, N)

    def measured_pose(self, q):
        return self.reward_geometry.pose(q, wrist_local=self.retarget_pose)

    def read(self):
        return self.env.state(), self.env.memory, self.encoder_bias

    def current_goal(self, m):
        return self.bank.next(m["frame"])

    def sample_goal(self, n):
        # The reference phase is advanced by one within each clip.
        return self.bank.reset_frames[:1].expand(n)

    def policy_observation(self, x, m, aux):
        q, v = (x + aux).split(N, -1)
        frame = m["frame"]
        goal = self.current_goal(m)
        obs = encode_policy_observation(q, v, self.bank.q[goal], self.bank.q[frame], m["cmd"], m["action"], self.home)
        return torch.cat((obs, self.intent_offsets[goal]), dim=-1) if self.human_intent else obs

    def context(self, x, m, aux, sim):
        obs = self.policy_observation(x, m, torch.zeros_like(x))
        phase = (self.bank.end[m["frame"]] - m["frame"]).unsqueeze(-1) * DT
        return torch.cat((obs, sim, phase), -1)

    @staticmethod
    def critic(context):
        return context

    @staticmethod
    def history_from_obs(obs):
        return obs.unsqueeze(1)

    @staticmethod
    def observation_from_history(history, memory):
        return history[:, -1]

    @staticmethod
    def dynamics(x, sim, aux):
        return torch.cat((x, sim), -1)

    @staticmethod
    def encode_memory(m):
        return torch.cat((m["frame"][:, None].float(), m["cmd"], m["action"], m["delay"][:, None].float()), -1)

    @staticmethod
    def decode_memory(encoded):
        frame, cmd, action, delay = encoded.split((1, N, N, 1), -1)
        return {"frame": frame[:, 0].long(), "cmd": cmd, "action": action, "delay": delay[:, 0].bool()}

    def advance_memory(self, x, m, action, target, geometry, next_goal, soft=True):
        return {"frame": self.current_goal(m), "cmd": target, "action": action, "delay": m["delay"]}

    def imagination_start_mask(self, m, horizon):
        return self.bank.end[m["frame"]] - m["frame"] >= horizon

    def control_target(self, action, m, bias=None):
        reference = self.bank.q[self.current_goal(m)]
        target = project_command(reference, action, m["cmd"], self.lo, self.hi, self.scale, self.slew)
        applied = torch.where(m["delay"][:, None], m["cmd"], target)
        return target, applied

    def torque(self, x, u, sim):
        q, v = x.split(N, -1)
        kp_scale, kd_scale, effort = self.servo_scales(sim)
        force = self.kp * kp_scale * (u - q) - self.kd * kd_scale * v
        return force.clamp(self.effort_lo * effort, self.effort_hi * effort)

    def servo_scales(self, sim):
        kp = torch.where(self.arm, sim[:, :1], sim[:, 1:2])
        kd, effort = sim[:, 2:3], sim[:, 3:4]
        if self.transfer_delta is not None:
            mix = sim[:, DR_DIM:DR_DIM + 1]
            kp = kp * (1. + mix * self.transfer_delta[0])
            kd = kd * (1. + mix * self.transfer_delta[1])
            effort = effort * (1. + mix * self.transfer_delta[2])
        return kp, kd, effort

    def position_violation(self, x):
        q = x[:, :N]
        return torch.maximum(self.joint_lo - q, q - self.joint_hi).clamp_min(0.).amax(-1)

    @staticmethod
    def replay_push(x, push, aux):
        return x + push, aux

    def imagine_push(self, x, m, aux):
        return x + self.domain.imagine_push(x), aux

    @staticmethod
    def contact_labels(depth):
        return torch.stack((depth > AUDIT_DEPTH, depth > HARD_DEPTH), -1).float()

    def termination(self, x, m):
        end = self.current_goal(m) == self.bank.end[m["frame"]]
        boundary = torch.sigmoid((self.position_violation(x) - .05) / .004)
        return torch.where(end, 1.0, boundary)

    @staticmethod
    def group_score(error2, scale):
        # Error is squared distance per feature. Average features within a group.
        return (1. / (1. + error2 / scale**2)).reshape(error2.shape[0], -1).mean(-1)

    def error_terms(self, pose, frame):
        wrist, rotation, local, relative, relative_r, tip = pose[:6]
        wr, rr, lr, pr, tr, tip_ref = (v[frame] for v in self.reward_goals[:6])
        pinch = (local[:, :, 1:, -1] - local[:, :, :1, -1]).norm(dim=-1)
        pinch_ref = (lr[:, :, 1:, -1] - lr[:, :, :1, -1]).norm(dim=-1)
        errors = {
            "wrist": (wrist - wr).square().sum(-1),
            "rotation": .5 * (rotation - rr).square().sum((-1, -2)),
            "local": (local - lr).square().sum(-1),
            "pinch": (pinch - pinch_ref).square(),
            "relative": (relative - pr).square().sum(-1),
            "relative_rotation": .5 * (relative_r - tr).square().sum((-1, -2)),
            "tip": (tip - tip_ref).square().sum(-1),
        }
        if self.retarget_pose:
            local_tip, local_rotation = pose[6:]
            ref_tip, ref_rotation = (v[frame] for v in self.reward_goals[6:])
            errors["wrist_local_tip"] = (local_tip - ref_tip).square().sum(-1)
            errors["wrist_local_rotation"] = .5 * (local_rotation - ref_rotation).square().sum((-1, -2))
        return errors

    def grouped(self, value):
        return torch.stack([value[:, group].mean(-1) for group in self.groups], -1).mean(-1)

    def pose_reward(self, x, m, action, target, contact, pose, frame, errors):
        scale = POSE_TRACKING_SCALES
        score = self.group_score
        q_error = (x[:, :N] - self.bank.q[frame]) / scale["joint"]
        scores = {
            "wrist_position": score(errors["wrist"], scale["wrist_position"]),
            "wrist_rotation": score(errors["rotation"], scale["wrist_rotation"]),
            "relative_position": score(errors["relative"], scale["relative_position"]),
            "relative_rotation": score(errors["relative_rotation"], scale["relative_rotation"]),
            "joint": self.grouped(1. / (1. + q_error.square())),
        }
        for index, side in enumerate(("left_hand", "right_hand")):
            pos = errors["wrist_local_tip"][:, index]
            rot = errors["wrist_local_rotation"][:, index]
            scores[side] = (
                POSE_HAND_WEIGHTS["position"] * score(pos, scale["tip_position"])
                + POSE_HAND_WEIGHTS["rotation"] * score(rot, scale["tip_rotation"])
                + POSE_HAND_WEIGHTS["worst_position"] * score(pos.topk(2, dim=-1).values, scale["tip_position"])
                + POSE_HAND_WEIGHTS["worst_rotation"] * score(rot.topk(2, dim=-1).values, scale["tip_rotation"])
            )
        costs = {
            "action_rate": self.grouped((action - m["action"]).square()),
            "command_rate": self.grouped(((target - m["cmd"]) / self.slew).square()),
            "action_magnitude": self.grouped(action.square()),
            "projection": self.grouped(((target - self.bank.q[frame]) / self.scale - action).square()),
            "contact": contact[:, 0],
            "hard_contact": contact[:, 1],
            "joint_violation": (self.position_violation(x) / .05).clamp(max=1.),
        }
        tracking = sum(self.reward_weights[key] * value for key, value in scores.items())
        reward = tracking - sum(self.reward_weights[key] * value for key, value in costs.items())
        return reward, pose[5], {**errors, "pose_score": tracking, "action_rate": costs["action_rate"]}

    def intent_reward(self, x, m, action, target, contact, pose, frame, execution):
        q = x[:, :N]
        human, intent_errors = human_intent_score(pose, (value[frame] for value in self.intent_geometry))
        score = scaled_score
        reference_scores = {
            "wrist_position": score(execution["wrist"], .022),
            "wrist_rotation": score(execution["rotation"], .14),
            "relative_position": score(execution["relative"][:, None], .033),
            "relative_rotation": score(execution["relative_rotation"][:, None], .20),
            "joint": self.grouped(2. - (1. + ((q - self.bank.q[frame]) / .10).square()).sqrt()),
        }
        for i, side in enumerate(("left_hand", "right_hand")):
            reference_scores[side] = (.72 * score(execution["local"][:, i, :, -1], .012)
                + .18 * score(execution["local"][:, i, :, :-1], .010)
                + .10 * score(execution["pinch"][:, i], .010))
        action_rate = self.grouped((action - m["action"]).square())
        command_change = self.grouped(((target - m["cmd"]) / self.slew).square())
        action_magnitude = self.grouped(action.square())
        projection = self.grouped(((target - self.bank.q[frame]) / self.scale - action).square())
        reference = sum(INTENT_EXECUTION_WEIGHTS[key] * value for key, value in reference_scores.items())
        reward = human + INTENT_EXECUTION_COEF * reference
        reward -= INTENT_SMOOTHNESS_COEF * (INTENT_COST_WEIGHTS["action_rate"] * action_rate
                                          + INTENT_COMMAND_RATE_WEIGHT * command_change)
        reward -= INTENT_RESIDUAL_COEF * INTENT_COST_WEIGHTS["action_magnitude"] * action_magnitude
        reward -= (INTENT_COST_WEIGHTS["projection"] * projection + INTENT_COST_WEIGHTS["contact"] * contact[:, 0]
                   + INTENT_COST_WEIGHTS["hard_contact"] * contact[:, 1]
                   + INTENT_COST_WEIGHTS["joint_violation"] * (self.position_violation(x) / .05).clamp(max=1.))
        return reward, pose[-1], {**execution, **intent_errors, "intent_score": human, "execution_score": reference,
                             "action_rate": action_rate, "action_magnitude": action_magnitude}

    def reward(self, x, m, action, target, contact, torque, soft=False, *, pose=None):
        frame = self.current_goal(m)
        q = x[:, :N]
        pose = self.measured_pose(q) if pose is None else pose
        e = self.error_terms(pose, frame)
        if self.retarget_pose:
            return self.pose_reward(x, m, action, target, contact, pose, frame, e)
        if self.human_intent:
            return self.intent_reward(x, m, action, target, contact, pose, frame, e)
        scores = {
            "wrist_position": self.group_score(e["wrist"], .022),
            "wrist_rotation": self.group_score(e["rotation"], .14),
            "relative_position": self.group_score(e["relative"], .033),
            "relative_rotation": self.group_score(e["relative_rotation"], .20),
            "joint": self.grouped(1. / (1. + ((q - self.bank.q[frame]) / .10).square())),
        }
        for i, side in enumerate(("left_hand", "right_hand")):
            tip_error2 = e["local"][:, i, :, -1]
            if self.fingertip_priority:
                error_mm = (tip_error2 + 1e-12).sqrt() * 1000.
                mean = self.group_score(tip_error2, TIP_ERROR_SCALE_M)
                tail = torch.topk(softplus((error_mm - 4.) / 2.), k=2, dim=-1).values.mean(-1)
                tail = 1. / (1. + tail - softplus(error_mm.new_tensor(-2.)))
                link = self.group_score(e["local"][:, i, :, :-1], .006)
                pinch = self.group_score(e["pinch"][:, i], .006)
                scores[side] = .48 * mean + .37 * tail + .10 * link + .05 * pinch
            else:
                tip = self.group_score(tip_error2, .012)
                link = self.group_score(e["local"][:, i, :, :-1], .010)
                pinch = self.group_score(e["pinch"][:, i], .010)
                scores[side] = .65 * tip + .25 * link + .10 * pinch
        costs = {
            "action_rate": self.grouped((action - m["action"]).square()),
            "action_magnitude": self.grouped(action.square()),
            "projection": self.grouped(((target - self.bank.q[frame]) / self.scale - action).square()),
            "contact": contact[:, 0],
            "hard_contact": contact[:, 1],
            "joint_violation": (self.position_violation(x) / .05).clamp(max=1.),
        }
        reward = sum(self.reward_weights[k] * v for k, v in scores.items())
        reward -= sum(self.reward_weights[k] * v for k, v in costs.items())
        return reward, pose[-1], e

    def rotation_error(self, x, m):
        # The shared learner's legacy aggregate column expects a scalar error.
        # Exact wrist SO(3) angles are computed separately in evaluation.
        pose = self.geometry.pose(x[:, :N])
        ref = self.goal_geometry[1][self.current_goal(m)]
        return (.5 * (pose[1] - ref).square().sum((-1, -2))).sqrt().mean(-1)
