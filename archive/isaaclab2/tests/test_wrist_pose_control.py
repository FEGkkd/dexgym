import math

import pytest

from dexteroushands_lab.assets.wrist_pose_control import WristPoseController, WristPoseLimits


def test_pose_trajectory_has_world_linear_and_angular_speed_bounds():
    ctrl = WristPoseController((0, 0, 0.5), (0, 0, 0, 1),
                              WristPoseLimits(max_linear_speed=0.1,
                                              max_angular_speed=0.5))
    ctrl.set_target((0.1, 0, 0.5), (0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
    pose, velocity = ctrl.step(0.02)
    assert pose[0] == pytest.approx(0.002)
    assert velocity[:3] == pytest.approx((0.1, 0, 0))
    assert velocity[3:] == pytest.approx((0, 0, 0.5))
    assert pose[5] == pytest.approx(math.sin(0.005))
    for _ in range(160):
        ctrl.step(0.02)
    assert ctrl.at_target()
    assert ctrl.current_pose[:3] == pytest.approx((0.1, 0, 0.5))


def test_relative_local_rotation_and_episode_reset_clear_trajectory_velocity():
    a = WristPoseController((0, 0, 0.5), (0, 0, 0, 1))
    b = WristPoseController((1, 0, 0.5), (0, 0, 0, 1))
    a.command_relative((0, 0.1, 0), (0, 0, 0.2))
    a.step(0.01)
    assert not a.at_target()
    assert b.at_target()
    a.reset_to((0.2, 0.3, 0.6), (0, 0, 0, 1))
    assert a.at_target()
    assert a.current_velocity == (0, 0, 0, 0, 0, 0)
    assert a.step(0.01)[1] == (0, 0, 0, 0, 0, 0)


def test_target_validation_and_short_quaternion_path():
    ctrl = WristPoseController((0, 0, 0.5), (0, 0, 0, 1))
    with pytest.raises(ValueError):
        ctrl.set_target((0, 0, 0.01), (0, 0, 0, 1))
    with pytest.raises(ValueError):
        ctrl.set_target((0, 0, 0.5), (0, 0, 0, 0))
    ctrl.set_target((0, 0, 0.5), (0, 0, 0, -1))
    assert ctrl.at_target()
    assert ctrl.step(0.002)[1] == (0, 0, 0, 0, 0, 0)
def test_split_bimanual_action_52_preserves_side_order_and_rejects_invalid():
    import pytest
    from dexteroushands_lab.assets.wrist_pose_control import split_bimanual_action_52

    left, right = split_bimanual_action_52(range(52))
    assert left == ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0), tuple(float(i) for i in range(6, 26)))
    assert right == ((26.0, 27.0, 28.0), (29.0, 30.0, 31.0),
                     tuple(float(i) for i in range(32, 52)))
    with pytest.raises(ValueError):
        split_bimanual_action_52([0.0]*51)
    with pytest.raises(ValueError):
        split_bimanual_action_52([float("nan")]+[0.0]*51)
