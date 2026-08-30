from ....utils.data_contract import ObservationField, TaskDataContract


ACTUATED_JOINT_ORDER = (
    "robot0_WRJ1", "robot0_WRJ0", "robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1",
    "robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1", "robot0_RFJ3", "robot0_RFJ2",
    "robot0_RFJ1", "robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1",
    "robot0_THJ4", "robot0_THJ3", "robot0_THJ2", "robot0_THJ1", "robot0_THJ0",
)


REORIENTATION_CONTRACT = TaskDataContract(
    task_id="DexterousHands-Shadow-Hand-Reorientation-Direct-v0",
    asset_variant="isaaclab.shadow_hand",
    joint_order=ACTUATED_JOINT_ORDER,
    action_mode="joint_position_target",
    physics_dt=1.0 / 120.0,
    control_dt=2.0 / 120.0,
    observation_fields=(
        ObservationField("joint_pos_normalized", 20, description="Actuated joint positions in [-1, 1]."),
        ObservationField("joint_vel_scaled", 20, description="Actuated joint velocities."),
        ObservationField("object_pos_env", 3, frame="env", description="Object position in environment frame."),
        ObservationField("object_quat_wxyz", 4, frame="world", description="Object orientation."),
        ObservationField("object_lin_vel", 3, frame="world"),
        ObservationField("object_ang_vel_scaled", 3, frame="world"),
        ObservationField("goal_pos_env", 3, frame="env"),
        ObservationField("goal_quat_wxyz", 4, frame="world"),
        ObservationField("object_goal_quat_error_wxyz", 4, frame="world"),
        ObservationField("previous_action", 20),
    ),
)
