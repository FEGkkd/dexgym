import torch
from types import SimpleNamespace

from dexgym.motion.motion_manager import MotionManager
from dexgym.tasks.marvin_supermimic.robot_adapter import MarvinArmDynamicsAdapter


def test_legacy_action_contract_matches_m41_formula_and_ten_substeps():
    motion = MotionManager("/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/"
                           "cmu_141_16_wave_bimanual.npz", 1, "cpu")
    robot = MarvinArmDynamicsAdapter(motion, "cpu", action_schema="legacy_wave_v1")
    old_origin = motion.data["q_arm"][0]
    old_scale = (motion.data["q_arm"] - old_origin).abs().amax(0) + 0.15
    torch.testing.assert_close(robot.action_map.origin, old_origin)
    torch.testing.assert_close(robot.action_map.scale, old_scale)
    action = torch.randn(2, 14)
    previous = old_origin[None].repeat(2, 1)
    command = robot.action_map(action, previous)
    expected = (old_origin + old_scale * action.tanh()).clamp(robot.action_map.lower, robot.action_map.upper)
    expected = expected.clamp(previous - 0.04, previous + 0.04)
    torch.testing.assert_close(command, expected)
    torch.testing.assert_close(robot.interpolate_command(previous, command, 9), command)
    torch.testing.assert_close(robot.interpolate_command(previous, command, 0), previous + 0.1 * (command - previous))


def test_robot_fixed_contract_has_motion_independent_origin_and_explicit_version():
    motion = MotionManager("/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/"
                           "cmu_141_16_wave_bimanual.npz", 1, "cpu")
    robot = MarvinArmDynamicsAdapter(motion, "cpu", action_schema="robot_fixed_v2")
    assert robot.action_schema == "robot_fixed_v2"
    assert robot.state_dim == 42 and robot.action_dim == 14
    shifted = SimpleNamespace(joint_names_arm=motion.joint_names_arm,
                              data={"q_arm": motion.data["q_arm"].clone()})
    shifted.data["q_arm"][0] += 0.1
    shifted_robot = MarvinArmDynamicsAdapter(shifted, "cpu", action_schema="robot_fixed_v2")
    torch.testing.assert_close(robot.action_map.origin, shifted_robot.action_map.origin)
    assert not torch.equal(shifted_robot.action_map.origin, shifted.data["q_arm"][0])
    assert robot.physics_substeps == 10 and robot.physics_dt == 0.002


def test_fixed_v3_caps_small_raw_action_sensitivity_without_changing_v2():
    motion = MotionManager("/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/"
                           "cmu_141_16_wave_bimanual.npz", 1, "cpu")
    old = MarvinArmDynamicsAdapter(motion, "cpu", action_schema="robot_fixed_v2")
    new = MarvinArmDynamicsAdapter(motion, "cpu", action_schema="robot_fixed_v3")
    raw = torch.full((1, 14), 0.01)
    old_delta = (old.action_map(raw, old.action_map.origin[None]) - old.action_map.origin).abs().max()
    new_delta = (new.action_map(raw, new.action_map.origin[None]) - new.action_map.origin).abs().max()
    assert float(old_delta) > 0.02
    assert float(new_delta) < 0.003
    assert new.action_map.max_delta == 0.01 and old.action_map.max_delta == 0.04
