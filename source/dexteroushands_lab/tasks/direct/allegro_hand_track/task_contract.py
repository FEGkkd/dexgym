"""Backend-independent Allegro tracking action and observation contract."""

from __future__ import annotations

from ....utils.data_contract import ObservationField, TaskDataContract


ALLEGRO_ROOT_CHANNEL_NAMES = (
    "root_x",
    "root_y",
    "root_z",
    "root_rx",
    "root_ry",
    "root_rz",
)
ALLEGRO_FINGER_JOINT_NAMES = tuple(f"joint_{index}" for index in range(16))
ALLEGRO_ACTION_NAMES = ALLEGRO_ROOT_CHANNEL_NAMES + ALLEGRO_FINGER_JOINT_NAMES


ALLEGRO_TRACK_CONTRACT = TaskDataContract(
    task_id="DexterousHands-Allegro-Hand-Track-Direct-v0",
    asset_variant="allegro-fly-right-urdf-converted-usd",
    joint_order=ALLEGRO_ACTION_NAMES,
    action_mode="cumulative_reference_residual",
    physics_dt=1.0 / 120.0,
    control_dt=6.0 / 120.0,
    observation_fields=(
        ObservationField("joint_position", 16),
        ObservationField("joint_velocity", 16),
        ObservationField("hand_pose_velocity", 13),
        ObservationField("object_pose_velocity", 13),
        ObservationField("reference_hand_target", 22),
        ObservationField("object_goal_delta", 3),
        ObservationField("previous_action", 22),
    ),
)


ALLEGRO_ACTION_DIM = len(ALLEGRO_ACTION_NAMES)
ALLEGRO_OBSERVATION_DIM = ALLEGRO_TRACK_CONTRACT.observation_dim
