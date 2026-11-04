"""External Marvin M6 + WujiHand2 Beta 2 whole-body asset contract."""

from .joint_mapping import JointContract, inspect_joint_contract
from .robot_cfg import make_robot_cfg, resolve_source_mjcf

__all__ = ["JointContract", "inspect_joint_contract", "make_robot_cfg", "resolve_source_mjcf"]
