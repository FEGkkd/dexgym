"""DirectRLEnv task definitions.

Direct tasks are the primary migration path for the original DexterousHands
environments. Each task owns its scene, action processing, observations,
rewards, termination logic, and reset logic explicitly.
"""

def register_tasks():
    """Register all DirectTask environments."""

    from . import allegro_hand_track, shadow_hand_pose, shadow_hand_reorientation, wuji_hand_reorientation, bimanual_bottle, bimanual_pregrasp_capturn

    allegro_hand_track.register_task()
    shadow_hand_pose.register_task()
    shadow_hand_reorientation.register_task()
    wuji_hand_reorientation.register_task()
    bimanual_bottle.register_tasks()
    bimanual_pregrasp_capturn.register_tasks()
    return allegro_hand_track, shadow_hand_pose, shadow_hand_reorientation, wuji_hand_reorientation


__all__ = ["register_tasks"]
