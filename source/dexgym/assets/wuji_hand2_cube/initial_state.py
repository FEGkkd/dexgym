"""Asset initialization metadata; independent of simulation backend."""

from dataclasses import dataclass, field


@dataclass
class InitialState:
    pos: tuple = (0.0, 0.0, 0.0)
    rot: tuple = (1.0, 0.0, 0.0, 0.0)
    joint_pos: dict = field(default_factory=dict)
    joint_vel: dict = field(default_factory=dict)
    lin_vel: tuple = (0.0, 0.0, 0.0)
    ang_vel: tuple = (0.0, 0.0, 0.0)
