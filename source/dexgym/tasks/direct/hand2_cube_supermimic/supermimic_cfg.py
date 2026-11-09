"""Cube task's agent entry point; optimization defaults live in the reusable core."""

from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig


def default_experiment():
    return SuperMimicConfig()
