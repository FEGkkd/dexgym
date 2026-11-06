"""Immutable dimensions and schema identities at the robot/task boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    state_dim: int
    action_dim: int
    delta_dim: int
    context_dim: int
    action_schema: str
    state_schema: str
    control_dt: float = 0.02
    physics_dt: float = 0.002
    asset_sha256: str = ""
    command_dim: int | None = None
    joint_order: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.robot_id or not self.action_schema or not self.state_schema:
            raise ValueError("Robot ID and state/action schemas are required")
        if min(self.state_dim, self.action_dim) < 1 or self.context_dim < 0:
            raise ValueError("Robot dimension must be positive except optional context")
        if not 1 <= self.delta_dim <= self.state_dim:
            raise ValueError("delta_dim must be within the physical state")
        if self.command_dim is not None and self.command_dim < 1:
            raise ValueError("Command dimension must be positive")
        if self.joint_order and len(self.joint_order) != self.action_dim:
            raise ValueError("Joint order must match action dimension when specified")
        if self.control_dt <= 0 or self.physics_dt <= 0 or self.control_dt < self.physics_dt:
            raise ValueError("Robot control and physics dt must be positive and ordered")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    observation_dim: int
    history_length: int
    goal_dim: int
    goal_horizon: int
    task_data_dim: int
    critic_dim: int
    goal_schema: str
    observation_schema: str = ""
    critic_schema: str = ""
    termination_feature_dim: int | None = None

    def __post_init__(self):
        if not self.task_id or not self.goal_schema:
            raise ValueError("Task ID and goal schema are required")
        if min(self.observation_dim, self.history_length, self.goal_dim,
               self.goal_horizon, self.critic_dim) < 1 or self.task_data_dim < 0:
            raise ValueError("Task dimension must be positive except optional data")
        if self.termination_feature_dim is not None and self.termination_feature_dim < 1:
            raise ValueError("Termination feature dimension must be positive")


@dataclass(frozen=True)
class NetworkSpec:
    world_hidden: tuple[int, ...] = (1024, 1024, 1024)
    goal_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    actor_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    termination_hidden: tuple[int, ...] = (128, 128)
    latent_dim: int = 32
    goal_latent_tanh: bool = True
    normalize: bool = True

    def __post_init__(self):
        if self.latent_dim < 1 or any(width < 1 for widths in (
            self.world_hidden, self.goal_hidden, self.actor_hidden,
            self.critic_hidden, self.termination_hidden) for width in widths):
            raise ValueError("Network hidden dimensions and latent width must be positive")

