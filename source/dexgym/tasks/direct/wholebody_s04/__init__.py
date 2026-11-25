"""S0-4 world fingertip tracking with a fixed hand reference."""

from .hand_points import GOAL_POINTS

TASK_ID = "wholebody-s04"
ARM_RESIDUAL_DEG = 8.0
ACTION_DIM = 14
HISTORY = 5
TARGET_MM = 15.0
CONTRACT = {
    "task_id": TASK_ID,
    "action_dim": ACTION_DIM,
    "command_dim": 54,
    "residual_bound_deg": ARM_RESIDUAL_DEG,
    "action_mode": "arm_ref_residual8_fixed_hand_ref_v1",
    "observation_version": ("s04_world_hand20_history5_v1" if GOAL_POINTS==20
                            else "s04_world_tip_history5_v1"),
    "goal_points_per_hand": GOAL_POINTS,
    "history_frames": HISTORY,
}
