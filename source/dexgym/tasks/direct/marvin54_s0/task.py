"""Hand-centric goals, state-centered position control and differentiable rewards."""

import torch

from ..hand2_cube_supermimic.rotations import mm
from ..marvin54_tracking.task import (
    FINGERS, JOINTS, MotionLibrary, TrackingGeometry, TrackingTask,
)


TASK_ID = "DexGym-Marvin54-HandCentric-Bimanual-S0-v1"
FUTURE_OFFSETS = (1, 2, 3, 5, 8, 12)
HISTORY_STEPS = 4
GOAL_FRAME_DIM = 30 + 60 + 90 + 75 + 6
TRACKING_LIMIT_M = 0.20
TRACKING_WIDTH_M = 0.025


class HandGoalLibrary(MotionLibrary):
    extra_fields = {"ref_shape": (2, 5, 4, 3)}


def cross_tip_vectors(tips):
    return tips[..., 0, :, None, :] - tips[..., 1, None, :, :]


class HandGeometry(TrackingGeometry):
    def __init__(self, model, library, device):
        super().__init__(model, library, device)
        self.fingers = torch.tensor([[
            [model.body(f"{side}_{finger}_{link}").id for link in ("proximal_abd", "middle", "distal")]
            for finger in FINGERS] for side in ("l", "r")], device=device)

    def hand_pose(self, q):
        kin = self.tree.forward(q, torch.zeros_like(q))
        p, r, _, _ = kin
        wrist, wrist_r = p[:, self.wrists], r[:, self.wrists]
        tips, _ = self.tree.tips(kin)
        tips = tips.reshape(-1, 2, 5, 3)
        tip_r = mm(r[:, self.tree.tip_bodies], self.tip_rotation).reshape(-1, 2, 5, 3, 3)
        chain = torch.cat((p[:, self.fingers], tips.unsqueeze(-2)), dim=-2)
        local = torch.einsum("bsji,bsfkj->bsfki", wrist_r, chain - wrist[:, :, None, None])
        return {
            "tip": tips, "tip_rot": tip_r, "wrist": wrist, "wrist_rot": wrist_r,
            "elbow": p[:, self.elbows], "shape": local,
        }


class HandCentricTask(TrackingTask):
    task_id = TASK_ID
    geometry_type = HandGeometry
    context_dim = 3 * JOINTS + GOAL_FRAME_DIM

    def __init__(self, env, model, library, target_memory=0.0):
        super().__init__(env, model, library, "home", "motion-encoder", "transition",
                         first_event=True, tracking_failure_m=TRACKING_LIMIT_M)
        # All training transitions come from exploration around the current Actor.
        self.collection_roles = None
        self.history_steps = HISTORY_STEPS
        self.history_frame_dim = 3 * JOINTS + 30
        self.history_dim = self.history_steps * self.history_frame_dim
        self.future_offsets = torch.tensor(FUTURE_OFFSETS, device=self.device)
        self.goal_dim = len(FUTURE_OFFSETS) * GOAL_FRAME_DIM
        self.policy_dim = self.history_dim + self.goal_dim
        if not 0 <= target_memory <= 1:
            raise ValueError("target_memory must be in [0,1]")
        self.target_memory = target_memory
        arm_step, hand_step = (0.05, 0.075) if target_memory else (0.095, 0.14)
        self.action_step = self.tensor([arm_step if name.startswith("Joint") else hand_step for name in library.names])

    def goal_features(self, frame):
        lib = self.library
        tips = lib.ref_tip_pos[frame]
        tip_r = lib.ref_tip_rot[frame]
        wrist_r = lib.ref_wrist_rot[frame]
        relative_r = wrist_r[..., 0, :, :].transpose(-1, -2) @ wrist_r[..., 1, :, :]
        shape = frame.shape
        return torch.cat((
            tips.reshape(*shape, 30),
            tip_r[..., :, :2].reshape(*shape, 60),
            lib.ref_shape[frame][..., 1:, :].reshape(*shape, 90),
            cross_tip_vectors(tips).reshape(*shape, 75),
            relative_r[..., :, :2].reshape(*shape, 6),
        ), dim=-1)

    def observation_from_history(self, history, memory):
        frames = self.library.future(memory["frame"], self.future_offsets)
        goal = self.goal_features(frames)
        # Express tip targets as displacements from the current measured/predicted tips.
        current_tip = history[:, -1:, -30:]
        goal = torch.cat((goal[..., :30] - current_tip, goal[..., 30:]), dim=-1)
        return torch.cat((history.flatten(1), goal.flatten(1)), dim=-1)

    def policy_observation(self, x, m, aux):
        q, qd = x.split(JOINTS, dim=-1)
        control = (m["control_center"] - q) / self.action_step if self.target_memory else m["action"]
        return torch.cat((q - self.home, qd, control, m["tip"]), dim=-1)

    def context(self, x, m, aux, sim):
        control = (m["control_center"] - x[:, :JOINTS]) / self.action_step if self.target_memory else m["action"]
        return torch.cat((x, control, self.goal_features(self.current_goal(m))), dim=-1)

    @staticmethod
    def encode_memory(m):
        # Integer phase is exactly representable at the size of this motion library.
        return torch.cat((m["frame"][:, None].float(), m["action"], m["control_center"]), dim=-1)

    @staticmethod
    def decode_memory(encoded):
        return {"frame": encoded[:, 0].long(), "action": encoded[:, 1:1 + JOINTS],
                "control_center": encoded[:, 1 + JOINTS:]}

    def advance_memory(self, x1, m, a, target, tip, next_goal, soft=True):
        q = x1[:, :JOINTS]
        center = torch.lerp(q, target, self.target_memory)
        return {"frame": self.current_goal(m), "action": a, "control_center": center, "tip": tip}

    def control_target(self, action, m, bias):
        u = (m["control_center"] + self.action_step * action).clamp(self.lo, self.hi)
        return u, u

    def tracking_error(self, pose, memory):
        target = self.library.ref_tip_pos[self.current_goal(memory)]
        return (pose["tip"].mean(-2) - target.mean(-2)).norm(dim=-1).amax(-1)

    def termination(self, x1, m):
        pose = self.geometry.hand_pose(x1[:, :JOINTS])
        lost = torch.sigmoid((self.tracking_error(pose, m) - TRACKING_LIMIT_M) / TRACKING_WIDTH_M)
        end = self.current_goal(m) == self.library.end[m["frame"]]
        return torch.where(end, 1.0, lost)

    def errors(self, x, m, pose):
        lib = self.library
        g = self.current_goal(m)
        desired_tip = lib.ref_tip_pos[g]
        palm_r = pose["wrist_rot"]
        target_palm_r = lib.ref_wrist_rot[g]
        relative = cross_tip_vectors(desired_tip)
        return {
            "tip_pos": pose["tip"] - desired_tip,
            "tip_rot": pose["tip_rot"] - lib.ref_tip_rot[g],
            "shape": pose["shape"][..., 1:, :] - lib.ref_shape[g][..., 1:, :],
            "cross_tip": cross_tip_vectors(pose["tip"]) - relative,
            "cross_weights": 0.25 + torch.exp(-relative.square().sum(-1) / 0.12**2),
            "palm_relative": palm_r[:, 0].transpose(-1, -2) @ palm_r[:, 1]
                             - target_palm_r[:, 0].transpose(-1, -2) @ target_palm_r[:, 1],
            "wrist": pose["wrist"] - lib.ref_wrist_pos[g],
            "elbow": pose["elbow"] - lib.ref_elbow_pos[g],
            "joint": x[:, :JOINTS] - lib.qpos[g],
        }

    def reward(self, x1, m, a, target, contact, torque, soft=False, pose=None):
        q = x1[:, :JOINTS]
        pose = self.geometry.hand_pose(q) if pose is None else pose
        e = self.errors(x1, m, pose)

        def similarity(squared_error):
            return 0.75 / (1.0 + squared_error) + 0.25 / (1.0 + squared_error / 12.0)

        def term(name, scale):
            return similarity((e[name] / scale).square().flatten(1).mean(-1))

        near_pairs = e["cross_weights"]
        relative = (e["cross_tip"].square().sum(-1) * near_pairs).sum((1, 2)) / near_pairs.sum((1, 2))
        goal_gap = cross_tip_vectors(self.library.ref_tip_pos[self.current_goal(m)]).norm(dim=-1)
        measured_gap = cross_tip_vectors(pose["tip"]).norm(dim=-1)
        clearance = ((0.85 * goal_gap).clamp(max=0.014) - measured_gap).clamp_min(0.) / 0.014
        margin = (0.035 - torch.minimum(q - self.joint_lo, self.joint_hi - q)).clamp_min(0.0) / 0.035
        terms = {
            "tip_pos": 3.6 * term("tip_pos", 0.025), "tip_rot": 1.05 * term("tip_rot", 0.30),
            "shape": 1.75 * term("shape", 0.016), "cross_tip": 2.35 * similarity(relative / 0.046**2),
            "palm_relative": 0.55 * term("palm_relative", 0.33),
            "wrist": 0.14 * term("wrist", 0.11), "elbow": 0.10 * term("elbow", 0.12),
            "smoothness": -0.16 * (a - m["action"]).square().mean(-1),
            "action": -0.04 * a.square().mean(-1),
            "limit": -0.07 * margin.square().mean(-1),
            "clearance": -0.35 * clearance.square().amax(dim=(1, 2)),
            "contact": -0.60 * contact[:, 1],
        }
        return sum(terms.values()), pose["tip"].flatten(1), terms

    def rotation_error(self, x, m):
        pose = self.geometry.hand_pose(x[:, :JOINTS])
        reference = self.library.ref_tip_rot[self.current_goal(m)]
        return self.angle(pose["tip_rot"] - reference).mean((1, 2))
