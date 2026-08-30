"""DirectRLEnv task definitions.

Direct tasks are the primary migration path for the original DexterousHands
environments. Each task owns its scene, action processing, observations,
rewards, termination logic, and reset logic explicitly.
"""

def register_tasks():
    """Register all DirectTask environments."""

    from . import shadow_hand_pose, shadow_hand_reorientation, wuji_hand_reorientation

    shadow_hand_pose.register_task()
    shadow_hand_reorientation.register_task()
    wuji_hand_reorientation.register_task()
    return shadow_hand_pose, shadow_hand_reorientation, wuji_hand_reorientation


__all__ = ["register_tasks"]
