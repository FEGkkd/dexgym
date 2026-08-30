"""ManagerBasedRLEnv task definitions.

Manager-Based tasks are the componentized path for the final arm-hand tasks.
The first baseline reuses Isaac Lab's tested in-hand MDP terms while keeping
the task registration and configuration in this extension.
"""

def register_tasks():
    """Register all Manager-Based environments."""

    from . import hand_repose

    hand_repose.register_task()
    return hand_repose


__all__ = ["register_tasks"]
