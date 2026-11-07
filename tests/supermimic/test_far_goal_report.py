"""Independent reach probe counts actual held Newton arrivals, not tracking frames."""

import numpy as np


def test_reach_probe_requires_consecutive_dual_se3_and_net_speed():
    from scripts.supermimic.analyze_far_goal_trace import summarize_reach_errors

    position = np.full((40, 2), 0.02)
    rotation = np.full((40, 2), 0.1)
    speed = np.full(40, 0.02)
    position[10:36] = 0.003
    rotation[10:36] = 0.02
    speed[24] = 0.2  # Break the first hold streak.
    result = summarize_reach_errors(position, rotation, speed, hold_steps=25,
                                    position_limit_m=0.005,
                                    rotation_limit_rad=0.035,
                                    speed_limit_rad_s=0.15)
    assert result["first_complete_arrival_step"] == 11
    assert result["longest_hold_steps"] == 14
    assert result["success"] is False
    speed[24] = 0.02
    result = summarize_reach_errors(position, rotation, speed, hold_steps=25,
                                    position_limit_m=0.005,
                                    rotation_limit_rad=0.035,
                                    speed_limit_rad_s=0.15)
    assert result["first_hold_step"] == 35
    assert result["longest_hold_steps"] == 26
    assert result["success"] is True
