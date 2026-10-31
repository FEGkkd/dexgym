"""Phase 1 observation schema; actor object poses are oracle state inputs."""

ACTION_DIM = 52
ACTION_FIELDS = {"left_wrist_translation_world": 3, "left_wrist_rotation_local": 3,
                 "left_finger_target_increment": 20,
                 "right_wrist_translation_world": 3, "right_wrist_rotation_local": 3,
                 "right_finger_target_increment": 20}

ACTOR_FIELDS = {
    "left_joint_position": 20,
    "right_joint_position": 20,
    "left_joint_velocity": 20,
    "right_joint_velocity": 20,
    "left_joint_target_error": 20,
    "right_joint_target_error": 20,
    "left_wrist_pose": 7,
    "right_wrist_pose": 7,
    "left_wrist_target_error": 6,
    "right_wrist_target_error": 6,
    "left_wrist_velocity": 6,
    "right_wrist_velocity": 6,
    "left_fingertips_relative_to_bottle": 15,
    "right_fingertips_relative_to_cap": 15,
    "bottle_relative_to_left_wrist": 7,
    "cap_relative_to_right_wrist": 7,
    "cap_relative_to_bottle": 7,
    "bottle_and_cap_velocity": 12,
    "thread_progress_and_goal": 4,
    "previous_action": ACTION_DIM,
}

CRITIC_FIELDS = {
    "left_bottle_forces": 21,
    "right_cap_forces": 21,
    "left_cap_forces": 21,
    "right_bottle_forces": 21,
    "finger_applied_torque": 40,
    "thread_internal_state": 4,
    "object_mass": 2,
    "relative_slip": 2,
    "full_body_velocity": 12,
}


def actor_dim() -> int:
    return sum(ACTOR_FIELDS.values())


def critic_dim() -> int:
    return actor_dim() + sum(CRITIC_FIELDS.values())


def schema_dict() -> dict:
    """Serializable task-owned schema; no legacy 207/413 constants."""
    return {
        "version": "phase1-bimanual-bottle-v0",
        "tasks": ["DexterousHands-WujiHand2-Bimanual-GraspHold-Direct-v0",
                  "DexterousHands-WujiHand2-Bimanual-CapTurn30-Direct-v0"],
        "units": {"distance": "m", "angle": "rad", "force": "N", "torque": "N m"},
        "quaternion_convention": "xyzw",
        "action": {"dimension": ACTION_DIM, "normalized_range": [-1, 1],
                   "fields": ACTION_FIELDS,
                   "wrist_translation_scale_m_per_action": 0.002,
                   "wrist_rotation_scale_rad_per_action": 0.02,
                   "finger_target_increment_rad_per_action": 0.05},
        "actor": {"dimension": actor_dim(), "fields": ACTOR_FIELDS,
                  "oracle_object_state": True,
                  "deployment_note": "Bottle and cap true poses/velocities are simulator oracle state, not a robot sensor."},
        "critic": {"dimension": critic_dim(), "privileged_fields": CRITIC_FIELDS,
                   "inherits_actor": True,
                   "contact_semantics": "Exact PhysX object/body normal-force matrix; no tactile hardware."},
    }
