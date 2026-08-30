"""Local MDP namespace for the Manager-Based migration.

The baseline imports upstream in-hand terms. New Wuji-specific terms will be
implemented here and can then be selected from the task configuration without
changing the training scripts.
"""

from isaaclab_tasks.manager_based.manipulation.inhand import mdp as upstream_mdp

__all__ = ["upstream_mdp"]
