"""Independent learned hand residuals and a trained whole-body arm policy."""

import numpy as np
import torch
from torch import nn

from dexgym.algorithms.supermimic.dex_reference.common import mlp
from dexgym.algorithms.supermimic.dex_reference.networks import Actor
from ..hand2_cube_supermimic.rotations import axis_angle_to_mat, mm, quat_to_mat

HAND_SCALE = .10
TIP_ERROR_SCALE_M = .004


class PalmOrientation(nn.Module):
    """Express the world vertical axis in each palm for the learned hand policy."""

    def __init__(self, model, names, home, dt):
        super().__init__()
        axes, rotations, indices = [], [], []
        for side in "lr":
            body = model.body(f"{side}_wrist").id
            chain = []
            while body:
                chain.append(body)
                body = int(model.body_parentid[body])
            chain = list(reversed(chain))
            joint_ids = [int(model.body_jntadr[i]) if model.body_jntnum[i] else -1 for i in chain]
            axes.append([model.jnt_axis[j] if j >= 0 else [0., 0., 0.] for j in joint_ids])
            rotations.append(model.body_quat[chain])
            indices.append([names.index(model.joint(j).name) if j >= 0 else 0 for j in joint_ids])
        self.register_buffer("joint_indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("axes", torch.tensor(np.asarray(axes), dtype=torch.float32))
        self.register_buffer("body_rotations", quat_to_mat(torch.tensor(np.asarray(rotations), dtype=torch.float32)))
        self.register_buffer("home", home.clone())
        self.dt = dt

    def forward(self, relative_q):
        q = (relative_q + self.home)[:, self.joint_indices]
        angle = q * self.axes.square().sum(-1)
        local = mm(self.body_rotations, axis_angle_to_mat(self.axes, angle))
        rotation = local[:, :, 0]
        for i in range(1, self.axes.shape[1]):
            rotation = mm(rotation, local[:, :, i])
        return rotation[..., 2, :]


class HandResidualLogits(nn.Module):
    def __init__(self, observation_dim, dofs, hand_groups, palm_sites, *, intent=False, train_arm=False,
                 fingertip_feedback=None):
        super().__init__()
        self.dofs = dofs
        self.observation_dim = observation_dim
        self.intent = intent
        hand_count, per_hand = hand_groups.shape
        self.arm_policy = Actor(observation_dim, dofs).requires_grad_(intent or train_arm)
        self.hand_nets = nn.ModuleList(mlp(7 * per_hand + 3, [192, 96], per_hand) for _ in range(hand_count))
        for net in self.hand_nets:
            nn.init.orthogonal_(net[-1].weight, gain=.01)
            nn.init.zeros_(net[-1].bias)
        self.register_buffer("hand_groups", hand_groups)
        self.palm_sites = palm_sites
        self.register_buffer("feature_scale", torch.tensor([.5, 3., .08, 3., .08, 1.])[None, None, :, None])
        self.fingertip_feedback = fingertip_feedback
        if fingertip_feedback is not None:
            if not torch.equal(fingertip_feedback.indices.flatten(1), hand_groups.cpu()):
                raise ValueError("Fingertip feedback and hand action order must match")
            self.register_buffer("finger_identity", torch.eye(5))
            self.tip_feedback_nets = nn.ModuleList(mlp(15, [64, 48], 4) for _ in range(hand_count))
            for net in self.tip_feedback_nets:
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
        if intent:
            # Inject the task-space goals into the first hidden layers. Zero
            # adapters reproduce the starting controller on the joint inputs.
            self.arm_goal = nn.Linear(18, self.arm_policy.net[0].out_features, bias=False)
            self.hand_goals = nn.ModuleList(nn.Linear(15, net[0].out_features, bias=False) for net in self.hand_nets)
            for adapter in (self.arm_goal, *self.hand_goals):
                nn.init.zeros_(adapter.weight)

    def forward(self, observation):
        base = observation[:, :self.observation_dim]
        state = base.reshape(-1, 6, self.dofs)
        hands = state[:, :, self.hand_groups].transpose(1, 2) / self.feature_scale
        vertical = self.palm_sites(state[:, 0])
        error, reference_rate, target_offset, action = (state[:, index, self.hand_groups] for index in (2, 3, 4, 5))
        # Compare the previous projected command to its pre-projection proposal.
        projected_command_offset = target_offset - error + self.palm_sites.dt * reference_rate - HAND_SCALE * action
        hands = torch.cat((hands, projected_command_offset.unsqueeze(-2) / .08), dim=2)
        features = torch.cat((hands.flatten(2), vertical), dim=-1)
        if self.intent:
            wrist_goal = observation[:, self.observation_dim:self.observation_dim + 18]
            hand_goal = observation[:, self.observation_dim + 18:].reshape(-1, 2, 15)
            arm = self.arm_policy.net
            arm_features = arm[0](self.arm_policy.norm(base)) + self.arm_goal(wrist_goal)
            for layer in list(arm)[1:]:
                arm_features = layer(arm_features)
            arm_logits = arm_features
            hand_logits = torch.cat([self.hand_logits(net, self.hand_goals[i], features[:, i], hand_goal[:, i])
                                     for i, net in enumerate(self.hand_nets)], dim=-1)
        else:
            hand_logits = torch.cat([net(features[:, i]) for i, net in enumerate(self.hand_nets)], dim=-1)
            if self.fingertip_feedback is not None:
                measured_q = state[:, 0] + self.palm_sites.home
                reference_q = measured_q + state[:, 2]
                previous_ref = reference_q - self.palm_sites.dt * state[:, 3]
                measured, reference, previous = self.fingertip_feedback(
                    torch.stack((measured_q, reference_q, previous_ref))).unbind(0)
                finger_q = measured_q[:, self.fingertip_feedback.indices]
                identity = self.finger_identity.expand(state.shape[0], len(self.tip_feedback_nets), -1, -1)
                feedback = torch.cat(((reference - measured) / .004, (reference - previous) / .004,
                                      finger_q, identity), dim=-1)
                hand_update = torch.stack([net(feedback[:, i]) for i, net in enumerate(self.tip_feedback_nets)], dim=1)
                hand_logits = hand_logits + hand_update.flatten(1)
            arm_logits = self.arm_policy.net(self.arm_policy.norm(base))
        return arm_logits.index_copy(1, self.hand_groups.flatten(), hand_logits)

    @staticmethod
    def hand_logits(net, adapter, features, goal):
        features = net[0](features) + adapter(goal)
        for layer in list(net)[1:]:
            features = layer(features)
        return features


class FingertipActor(nn.Module):
    def __init__(self, observation_dim, dofs, hand_groups, palm_sites, *, intent=False, train_arm=False,
                 fingertip_feedback=None):
        super().__init__()
        self.norm = nn.Identity()
        self.net = HandResidualLogits(observation_dim, dofs, hand_groups, palm_sites, intent=intent, train_arm=train_arm,
                                      fingertip_feedback=fingertip_feedback)

    def forward(self, observation):
        return self.net(self.norm(observation)).tanh()

    def load_fingertip_weights(self, state):
        missing = {"net.arm_goal.weight", "net.hand_goals.0.weight", "net.hand_goals.1.weight"}
        expected = set(self.state_dict()) - missing
        if set(state) != expected:
            raise ValueError("The source actor does not match the pre-intent fingertip architecture")
        self.load_state_dict(state, strict=False)

    def load_precision_weights(self, hand_state, arm_state):
        expected = {key for key in self.state_dict() if key != "net.finger_identity"
                    and not key.startswith(("net.tip_feedback_nets.", "net.fingertip_feedback."))}
        if set(hand_state) != expected or set(arm_state) != expected:
            raise ValueError("Precision training requires the previous wrist and fingertip actors")
        merged = {key: arm_state[key] if key.startswith("net.arm_policy.") else value
                  for key, value in hand_state.items()}
        self.load_state_dict(merged, strict=False)
