"""Causal observations and absolute command decoding used in training and deployment."""

from typing import NamedTuple

import torch
from torch import nn

from ..hand2_cube_supermimic.rotations import mm, mv
from . import DOF, DT
from .kinematics import Pose, RobotKinematics, make_pose
from .reference_timing import position_rates
from .wrist_calibration import WristCalibration

ACTOR_ARCHITECTURE = "shared_mlp_54_v1"
GOAL_RATE_OBSERVATION = "task_space_goal_rate_v1"
OBSERVATION_VERSION = "task_space_error_rate_v2"
OBS_FEATURES = 5 * DOF + 54 + 270 + 27 + 6
FOUR_POINT_OBSERVATION = "task_space_four_point_sincos_v3"
FOUR_POINT_OBS_FEATURES = 5 * DOF + 2 * (18 + 2 * 5 * 4 * 3 + 9) + 6
HISTORY5_OBSERVATION = "task_space_four_point_sincos_history5_v6"
HISTORY5_NO_COMMAND_ERROR_OBSERVATION = "task_space_four_point_sincos_history5_no_command_error_v7"
HISTORY5_ABSOLUTE_COMMAND_OBSERVATION = "task_space_four_point_sincos_history5_absolute_command_v8"
HISTORY5_ABSOLUTE_OBSERVATION = "task_space_four_point_sincos_history5_absolute_observation_v9"
HISTORY_OBSERVATION = "task_space_four_point_sincos_history8_v4"
HISTORY_DELTA_OBSERVATION = "task_space_four_point_sincos_history8_delta_v5"
FOUR_POINT_HISTORY_FRAMES = {
    FOUR_POINT_OBSERVATION: 1,
    HISTORY5_OBSERVATION: 5,
    HISTORY5_NO_COMMAND_ERROR_OBSERVATION: 5,
    HISTORY5_ABSOLUTE_COMMAND_OBSERVATION: 5,
    HISTORY5_ABSOLUTE_OBSERVATION: 5,
    HISTORY_OBSERVATION: 8,
    HISTORY_DELTA_OBSERVATION: 8,
}
COMMAND_OBSERVATION_BY_VERSION = {
    HISTORY5_NO_COMMAND_ERROR_OBSERVATION: "none",
    HISTORY5_ABSOLUTE_COMMAND_OBSERVATION: "previous_command_rad",
    HISTORY5_ABSOLUTE_OBSERVATION: "previous_command_rad",
}
OBSERVATION_FEATURES = {
    GOAL_RATE_OBSERVATION: OBS_FEATURES,
    OBSERVATION_VERSION: OBS_FEATURES,
    **{
        version: frames * (FOUR_POINT_OBS_FEATURES - DOF
                           if version == HISTORY5_NO_COMMAND_ERROR_OBSERVATION else FOUR_POINT_OBS_FEATURES)
        for version, frames in FOUR_POINT_HISTORY_FRAMES.items()
    },
}


def rotation6(rotation):
    return rotation[..., :, :2].flatten(-2)


class History(NamedTuple):
    previous_state: torch.Tensor
    previous_command: torch.Tensor
    previous_goal: Pose


class Command(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        for key, field in (("home", "home"), ("lower", "position_lower_rad"), ("upper", "position_upper_rad")):
            self.register_buffer(key, torch.tensor(metadata[field], dtype=torch.float32))
        arm = torch.tensor([name.startswith("Joint") for name in metadata["joint_names"]])
        self.register_buffer("arm", arm)
        self.register_buffer("scale", torch.maximum(self.home - self.lower, self.upper - self.home))
        names = metadata["joint_names"]
        rate_step = torch.as_tensor(position_rates(names, metadata) * DT, dtype=torch.float32)
        self.register_buffer("slew", torch.minimum(rate_step, self.upper - self.lower))
        # The real command limits change independently of the existing reward normalization.
        smoothness = torch.as_tensor(position_rates(names) * DT, dtype=torch.float32)
        self.register_buffer("rate_scale", smoothness)
        if torch.any(self.scale <= 0.) or torch.any(self.lower > self.home) or torch.any(self.home > self.upper):
            raise ValueError("Recorded Home and command ranges do not agree")

    def desired(self, action):
        return self.home + self.scale * action.clamp(-1., 1.)

    def forward(self, action, previous_command):
        previous_command = torch.minimum(torch.maximum(previous_command, self.lower), self.upper)
        desired = self.desired(action)
        lo = torch.maximum(self.lower, previous_command - self.slew)
        hi = torch.minimum(self.upper, previous_command + self.slew)
        return torch.minimum(torch.maximum(desired, lo), hi)


class Observation(nn.Module):
    def __init__(self, model, metadata, version=OBSERVATION_VERSION, *, proximal=False):
        super().__init__()
        if version not in OBSERVATION_FEATURES:
            raise ValueError(f"Unknown task-space observation version: {version}")
        self.version = version
        self.features = OBSERVATION_FEATURES[version]
        self.history_frames = FOUR_POINT_HISTORY_FRAMES.get(version, 1)
        self.frame_features = self.features // self.history_frames
        self.current_only = version in FOUR_POINT_HISTORY_FRAMES
        self.absolute_coordinates = version == HISTORY5_ABSOLUTE_OBSERVATION
        self.command_observation = COMMAND_OBSERVATION_BY_VERSION.get(version, "command_error")
        self.use_command_error = self.command_observation == "command_error"
        self.feedback_rates = version == OBSERVATION_VERSION
        self.physical_wrist = "wrist_calibration" in metadata
        self.command = Command(metadata)
        self.fk = RobotKinematics(model, metadata["joint_names"], self.physical_wrist,
                                  proximal=proximal or self.current_only)
        if self.physical_wrist:
            cal = WristCalibration.from_metadata(metadata)
            self.register_buffer("tcp_to_wrist", torch.tensor(cal.translation, dtype=torch.float32))
            self.register_buffer("tcp_from_wrist_rotation", torch.tensor(cal.rotation, dtype=torch.float32))
        if not self.absolute_coordinates:
            home = self.actor_frame(self.fk(self.command.home[None]).pose)
            self.register_buffer("home_wrist", home.wrist)
            self.register_buffer("home_hand", home.hand)
            self.register_buffer("home_relative", home.relative_position)
        self.register_buffer("identity6", rotation6(torch.eye(3)))

    def actor_frame(self, pose):
        if not self.current_only:
            pose = pose._replace(hand=pose.hand[..., -3:, :])
        if not self.physical_wrist:
            return pose
        rotation = mm(pose.rotation, self.tcp_from_wrist_rotation.transpose(-1, -2))
        wrist = pose.wrist - mv(rotation, self.tcp_to_wrist)
        # Keep the actor's hand-mount coordinates consistent with the goal transform.
        hand = (mv(self.tcp_from_wrist_rotation[None, :, None, None], pose.hand)
                + self.tcp_to_wrist[None, :, None, None])
        return make_pose(wrist, rotation, hand)

    def current_values(self, q, qd, goal, measured):
        wrist_error = mm(measured.rotation.transpose(-1, -2), goal.rotation)
        relative_error = mm(measured.relative_rotation.transpose(-1, -2), goal.relative_rotation)
        q_position, wrist_goal, hand_goal, relative_goal = q, goal.wrist, goal.hand, goal.relative_position
        if not self.absolute_coordinates:
            q_position = (q - self.command.home) / self.command.scale
            wrist_goal = (wrist_goal - self.home_wrist) / .25
            hand_goal = (hand_goal - self.home_hand) / .06
            relative_goal = (relative_goal - self.home_relative) / .25
        joints = (q_position, qd / 4.)
        wrists = (wrist_goal, rotation6(goal.rotation),
                  (goal.wrist - measured.wrist) / .06, (rotation6(wrist_error) - self.identity6) / .35)
        hands = (hand_goal, (goal.hand - measured.hand) / .025)
        relative = (relative_goal, rotation6(goal.relative_rotation),
                    (goal.relative_position - measured.relative_position) / .08,
                    (rotation6(relative_error) - self.identity6) / .35)
        return joints, wrists, hands, relative, measured.rotation[..., 2, :]

    @staticmethod
    def flatten(values):
        return torch.cat(tuple(value.flatten(1) for value in values), dim=-1).clamp(-12., 12.)

    def forward_current(self, state, goal, previous_command):
        q, qd = state.split(DOF, dim=-1)
        measured = self.actor_frame(self.fk(q).pose)
        joints, wrists, hands, relative, up = self.current_values(q, qd, self.actor_frame(goal), measured)
        joint_features = (*joints, q.sin(), q.cos())
        if self.use_command_error:
            joint_features += ((previous_command - q) / .15,)
        frame = self.flatten((*joint_features, *wrists, *hands, *relative, up))
        if self.command_observation == "previous_command_rad":
            return torch.cat((frame, previous_command), dim=-1)
        return frame

    def forward(self, state, goal, history):
        if self.current_only:
            return self.forward_current(state, goal, history.previous_command)
        q, qd = state.split(DOF, dim=-1)
        q0, qd0 = history.previous_state.split(DOF, dim=-1)
        goal = self.actor_frame(goal)
        old = self.actor_frame(history.previous_goal)
        wrist_delta = goal.wrist - old.wrist
        wrist_rot_delta = rotation6(goal.rotation) - rotation6(old.rotation)
        hand_delta = goal.hand - old.hand
        relative_delta = goal.relative_position - old.relative_position
        relative_rot_delta = rotation6(goal.relative_rotation) - rotation6(old.relative_rotation)
        if self.feedback_rates:
            pose = self.fk(torch.cat((q, q0), dim=0)).pose
            pose = self.actor_frame(pose)
            measured, previous = (Pose(*parts) for parts in zip(*(value.chunk(2, dim=0) for value in pose), strict=True))
            wrist_delta = wrist_delta - (measured.wrist - previous.wrist)
            wrist_rot_delta = wrist_rot_delta - (rotation6(measured.rotation) - rotation6(previous.rotation))
            hand_delta = hand_delta - (measured.hand - previous.hand)
            relative_delta = relative_delta - (measured.relative_position - previous.relative_position)
            relative_rot_delta = relative_rot_delta - (rotation6(measured.relative_rotation) - rotation6(previous.relative_rotation))
        else:
            measured = self.actor_frame(self.fk(q).pose)
        joints, wrists, hands, relative, up = self.current_values(q, qd, goal, measured)
        command_error = (history.previous_command - q) / .15
        return self.flatten((*joints, (q - q0) / .08, (qd - qd0) / 3., command_error,
                             *wrists, wrist_delta / .02, wrist_rot_delta / .12,
                             *hands, hand_delta / .012,
                             *relative, relative_delta / .02, relative_rot_delta / .12, up))


class AbsoluteAction(nn.Module):
    """One joint actor for both arms and both hands."""

    def __init__(self, features, scale, *, history_delta=False, observation_version=None, action_home=None):
        super().__init__()
        if features not in OBSERVATION_FEATURES.values():
            raise ValueError(f"Unsupported direct-control observation size: {features}")
        if observation_version is not None and OBSERVATION_FEATURES.get(observation_version) != features:
            raise ValueError("Actor observation version and feature count differ")
        self.history_delta = history_delta
        if history_delta and features != OBSERVATION_FEATURES[HISTORY_DELTA_OBSERVATION]:
            raise ValueError("Delta history requires eight complete observation frames")
        frame_sizes = {
            OBSERVATION_FEATURES[version]: OBSERVATION_FEATURES[version] // frames
            for version, frames in FOUR_POINT_HISTORY_FRAMES.items()
        }
        self.frame_features = frame_sizes.get(features, features)
        # Older constructors used the feature count. Explicit versions distinguish
        # the equally sized five-frame layouts when building or loading a policy.
        legacy_mode = "none" if features == OBSERVATION_FEATURES[HISTORY5_NO_COMMAND_ERROR_OBSERVATION] else "command_error"
        self.use_command_error = COMMAND_OBSERVATION_BY_VERSION.get(observation_version, legacy_mode) == "command_error"
        self.absolute_coordinates = observation_version == HISTORY5_ABSOLUTE_OBSERVATION
        if self.absolute_coordinates and action_home is None:
            raise ValueError("Absolute observations require the joint command origin for action decoding")
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("action_home", action_home.detach().clone() if self.absolute_coordinates else None)
        self.encoder = nn.Sequential(
            nn.Linear(features, 512), nn.ELU(),
            nn.Linear(512, 384), nn.ELU(),
            nn.Linear(384, 256), nn.ELU(),
        )
        head_features = 256
        if not self.absolute_coordinates:
            head_features += (2 if self.use_command_error else 1) * DOF
        self.head = nn.Linear(head_features, DOF)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.4)
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            nn.init.orthogonal_(self.head.weight[:, :256], gain=.012)
            identity = torch.eye(DOF)
            if self.use_command_error:
                self.head.weight[:, 256:256 + DOF].copy_(.35 * identity)
                self.head.weight[:, 256 + DOF:].copy_(.65 * identity)
            elif not self.absolute_coordinates:
                # Initialize near the measured joint positions.
                self.head.weight[:, 256:].copy_(identity)
            self.head.bias.zero_()

    def forward(self, observation):
        current = observation[:, -self.frame_features:]
        q = current[:, :DOF]
        encoded = observation
        if self.history_delta:
            frames = observation.reshape(-1, 8, self.frame_features)
            # An invertible change of coordinates: seven past-minus-current
            # frames and one current frame. It does not discard past state.
            encoded = torch.cat((frames[:, :-1] - frames[:, -1:], frames[:, -1:]), dim=1).flatten(1)
        features = self.encoder(encoded)
        if self.absolute_coordinates:
            # Keep the measured starting command in the existing action coordinates.
            measured_action = ((q - self.action_home) / self.scale).clamp(-.98, .98)
            return self.head(features) + measured_action.atanh()
        inputs = (features, q.clamp(-.98, .98).atanh())
        if self.use_command_error:
            offset = current[:, 4 * DOF:5 * DOF] * .15 / self.scale
            previous = (q + offset).clamp(-.98, .98).atanh()
            inputs += (previous,)
        return self.head(torch.cat(inputs, dim=-1))


class OnlineController(nn.Module):
    """Stateless exported step with explicit causal history carried by the caller."""

    def __init__(self, observation, actor):
        super().__init__()
        self.observation = observation
        self.actor = actor

    def forward(self, wrist, rotation, hand, q, qd, previous_wrist, previous_rotation, previous_hand, previous_q,
                previous_qd, previous_command):
        state = torch.cat((q, qd), dim=-1)
        previous_state = torch.cat((previous_q, previous_qd), dim=-1)
        goal = make_pose(wrist, rotation, hand)
        old_goal = make_pose(previous_wrist, previous_rotation, previous_hand)
        obs = self.observation(state, goal, History(previous_state, previous_command, old_goal))
        action = self.actor(obs).tanh()
        return self.observation.command(action, previous_command), action


class CurrentOnlineController(OnlineController):
    """Four-point controller with just the last constrained command as memory."""

    def forward(self, wrist, rotation, hand, q, qd, previous_command):
        state = torch.cat((q, qd), dim=-1)
        goal = make_pose(wrist, rotation, hand)
        obs = self.observation.forward_current(state, goal, previous_command)
        action = self.actor(obs).tanh()
        return self.observation.command(action, previous_command), action
