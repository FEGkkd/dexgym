"""Marvin M6 + Cube-aligned Wuji Hand2.2 whole-body asset contract."""

from .joint_mapping import JointContract, inspect_joint_contract

__all__ = ["JointContract", "inspect_joint_contract", "make_robot_cfg", "resolve_source_mjcf"]


def __getattr__(name):
    if name in ("make_robot_cfg", "resolve_source_mjcf"):
        from . import robot_cfg

        return getattr(robot_cfg, name)
    raise AttributeError(name)
