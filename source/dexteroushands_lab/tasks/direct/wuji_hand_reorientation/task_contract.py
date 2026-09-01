"""Canonical Wuji Hand action and observation contract."""

# The policy acts in the native official USD articulation order.
WUJI_POLICY_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)

# The task uses an asymmetric actor-critic. The actor consumes
# the 3 x 69 policy history; the critic consumes the shared clean history plus
# privileged state terms defined by the Wuji reorientation task.
WUJI_CRITIC_OBSERVATION_DIM = 413
from ....utils.data_contract import ObservationField, TaskDataContract


WUJI_REORIENTATION_CONTRACT = TaskDataContract(
    task_id="DexterousHands-Wuji-Hand-Reorientation-Direct-v0",
    asset_variant="wuji-official-right-native-usd-isaac-physx",
    joint_order=WUJI_POLICY_JOINT_NAMES,
    action_mode="joint_position_offset_ema",
    physics_dt=1.0 / 120.0,
    control_dt=6.0 / 120.0,
    observation_history_length=3,
    critic_observation_dim=WUJI_CRITIC_OBSERVATION_DIM,
    observation_fields=(
        ObservationField(
            "joint_angles",
            20,
            description="Soft-limit normalized joint positions with +/-0.06 noise.",
        ),
        ObservationField(
            "qpos_error",
            20,
            description="Normalized current position minus processed EMA target.",
        ),
        ObservationField("cube_pos_in_tag", 3, frame="tag"),
        ObservationField(
            "cube_ori_error_6d",
            6,
            frame="tag",
            description="Last two rows of cube-to-goal residual rotation matrix.",
        ),
        ObservationField("previous_raw_action", 20),
    ),
)
