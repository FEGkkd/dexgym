#!/usr/bin/env python3
"""Bind both native arm and Wuji hand publishers to the single diagnostic 54-axis guard."""

import argparse
from pathlib import Path

from prepare_shared_rigid_overlay import replace_once


CONTROLLER_DECL = "bool commit_hand_command(const std::function<void()> & publish);"
CONTROLLER_SIG = "bool TeleopControllerNode::commit_hand_command(const std::function<void()> & publish)"
NEW_ARGS = ("const manus_wuji_interfaces::msg::HandCommand & command, int64_t source_until_ns,\n"
            "    const std::function<void()> & publish")
RETARGET_GATE = "using CommandGate=std::function<bool(const std::function<void()> &)>;"

CONTROLLER_FIELDS = '''  dcr009_diagnostic::NominalGuard diagnostic_nominal_guard_;
  std::array<rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr, 2> diagnostic_guard_arm_sub_;
  std::array<rclcpp::Subscription<manus_wuji_interfaces::msg::HandState>::SharedPtr, 2> diagnostic_guard_hand_sub_;
  dcr009_diagnostic::NominalGuard::Proof diagnostic_guard_last_arm_;
'''

GUARD_SUBSCRIBERS = '''
  if (head_enabled_ || left_joint_state_topic_ != "/left_arm/joint_states" ||
      right_joint_state_topic_ != "/right_arm/joint_states" || command_topic_ != "/tianji/joint_command") {
    throw std::invalid_argument("DCR-009 three-group guard requires the reviewed 54-axis MuJoCo graph");
  }
  const auto dcr009_stamp = [](const builtin_interfaces::msg::Time & stamp) -> int64_t {
    if (stamp.sec < 0 || stamp.nanosec >= 1000000000U) return 0;
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
  };
  for (std::size_t side = 0; side < 2; ++side) {
    const std::string label = side == 0 ? "left" : "right";
    diagnostic_guard_arm_sub_[side] = node_.create_subscription<sensor_msgs::msg::JointState>(
        "/" + label + "_arm/joint_states", reliable,
        [this, side, dcr009_stamp](sensor_msgs::msg::JointState::ConstSharedPtr feedback) {
          std::lock_guard<std::mutex> lock(mutex_);
          diagnostic_nominal_guard_.observe(side, dcr009_stamp(feedback->header.stamp), node_.now().nanoseconds(),
              feedback->name, feedback->position, true);
        }, input_options);
    diagnostic_guard_hand_sub_[side] = node_.create_subscription<manus_wuji_interfaces::msg::HandState>(
        "/" + label + "_hand/state", sensor,
        [this, side, label, dcr009_stamp](manus_wuji_interfaces::msg::HandState::ConstSharedPtr feedback) {
          const auto all = [](const auto & flags, bool expected) {
            return flags.size() == 20u && std::all_of(flags.begin(), flags.end(),
                [expected](bool value) { return value == expected; });
          };
          std::lock_guard<std::mutex> lock(mutex_);
          diagnostic_nominal_guard_.observe(side + 2u, dcr009_stamp(feedback->header.stamp), node_.now().nanoseconds(),
              feedback->joint_names, feedback->position, feedback->side == label && feedback->connected &&
              feedback->feedback_source == "mujoco" && feedback->source_clock == "simulation" &&
              feedback->source_timestamp_valid &&
              all(feedback->position_valid, true) && all(feedback->online, true) && all(feedback->stale, false));
        }, input_options);
  }
'''

ARM_PREPUBLISH = '''  diagnostic_guard_last_arm_ = diagnostic_nominal_guard_.propose(generation_, session_id_, 0u,
      node_.now().nanoseconds(), message.joint_names, message.positions);
  const auto & dcr009_proof = diagnostic_guard_last_arm_;
  const int64_t dcr009_deadline = static_cast<int64_t>(message.valid_until.sec) * 1000000000LL +
      message.valid_until.nanosec;
  const double dcr009_steady = monotonic_now();
  const bool trackers_fresh = retain_previous ?
      std::max(tracker_age(left_tracker_, dcr009_steady), tracker_age(right_tracker_, dcr009_steady)) < tracker_hold_timeout_sec_ :
      (!result.left.accepted || tracker_status(left_tracker_, dcr009_steady, tracker_timeout_sec_).safe) &&
      (!result.right.accepted || tracker_status(right_tracker_, dcr009_steady, tracker_timeout_sec_).safe);
  if (!trackers_fresh || !diagnostic_nominal_guard_.still_valid(dcr009_proof, node_.now().nanoseconds(), dcr009_deadline)) {
    RCLCPP_INFO(node_.get_logger(), "S05_DCR009_GUARD group=arm session=%s generation=%llu command=%llu proof=%llu published=0 reason=%u feedback_ns=%lld",
        session_id_.c_str(), static_cast<unsigned long long>(generation_), static_cast<unsigned long long>(message.sequence),
        static_cast<unsigned long long>(dcr009_proof.seq), dcr009_proof.permitted ?
        dcr009_diagnostic::NominalGuard::kTimeoutOrExpired : dcr009_proof.reason,
        static_cast<long long>(dcr009_proof.feedback_header));
    return reject(SafetyCode::COMMAND_STEP, "DCR-009 54-axis arm publication did not receive a fresh swept proof");
  }
'''

ARM_COMMIT = '''  diagnostic_nominal_guard_.commit(diagnostic_guard_last_arm_);
  RCLCPP_INFO(node_.get_logger(), "S05_DCR009_GUARD group=arm session=%s generation=%llu command=%llu proof=%llu published=1 reason=0 feedback_ns=%lld",
      session_id_.c_str(), static_cast<unsigned long long>(generation_), static_cast<unsigned long long>(message.sequence),
      static_cast<unsigned long long>(diagnostic_guard_last_arm_.seq),
      static_cast<long long>(diagnostic_guard_last_arm_.feedback_header));
'''

HAND_COMMIT = '''  const uint32_t group = command.side == "left" ? 1u : command.side == "right" ? 2u : 3u;
  const int64_t issued_ns = static_cast<int64_t>(command.header.stamp.sec) * 1000000000LL + command.header.stamp.nanosec;
  const int64_t deadline_ns = std::min(source_until_ns, issued_ns + static_cast<int64_t>(100000000));
  auto proof = diagnostic_nominal_guard_.propose(generation_, session_id_, group,
      node_.now().nanoseconds(), command.joint_names, command.position);
  const auto evidence = [&](bool published, uint32_t reason) {
    RCLCPP_INFO(node_.get_logger(), "S05_DCR009_GUARD group=%s session=%s generation=%llu command=%llu proof=%llu published=%d reason=%u feedback_ns=%lld",
        command.side.c_str(), session_id_.c_str(), static_cast<unsigned long long>(generation_),
        static_cast<unsigned long long>(command.sequence), static_cast<unsigned long long>(proof.seq),
        published ? 1 : 0, reason, static_cast<long long>(proof.feedback_header));
  };
  if (!diagnostic_nominal_guard_.still_valid(proof, node_.now().nanoseconds(), deadline_ns)) {
    evidence(false, proof.permitted ? dcr009_diagnostic::NominalGuard::kTimeoutOrExpired : proof.reason);
    return false;
  }
  try {
    publish();
  } catch (...) {
    evidence(false, dcr009_diagnostic::NominalGuard::kPublishException);
    throw;
  }
  diagnostic_nominal_guard_.commit(proof);
  evidence(true, 0);
  return true;
'''


def apply_guard(directory):
    directory = directory.resolve()
    if directory.name != "whole_body_teleop" or directory.is_relative_to(Path("/home/ubuntu/teleop_ws")):
        raise ValueError("Require an isolated native Teleop source copy, outside the user worktree")
    paths = {
        "controller": directory / "src/teleop_controller_node.cpp",
        "controller_header": directory / "include/tianji_teleop_controller/teleop_controller_node.hpp",
        "coordinator": directory / "src/whole_body_teleop_node.cpp",
        "retarget": directory / "src/retarget_node.cpp",
        "retarget_header": directory / "include/manus_wuji_teleop/retarget_node.hpp",
        "core": directory / "src/teleop_core.cpp",
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    if "S05_DCR009_START_MAP_V1" not in text["core"] or "home_anchored_pilot_v1" not in text["controller"]:
        raise ValueError("Apply the reviewed isolated single-session Home anchor overlay before the diagnostic command guard")

    def replace(name, before, after):
        text[name] = replace_once(text[name], before, after, paths[name])

    replace("controller_header", '#include "manus_wuji_interfaces/srv/enable_hand.hpp"',
            '#include "manus_wuji_interfaces/srv/enable_hand.hpp"\n'
            '#include "manus_wuji_interfaces/msg/hand_command.hpp"\n'
            '#include "manus_wuji_interfaces/msg/hand_state.hpp"\n'
            '#include "tianji_teleop_controller/dcr009_teleop_guard.hpp"')
    replace("controller_header", CONTROLLER_DECL, f"bool commit_hand_command({NEW_ARGS});")
    anchor = "  friend struct TeleopControllerNodeTestPeer;\n"
    replace("controller_header", anchor, anchor + CONTROLLER_FIELDS)
    replace("controller", CONTROLLER_SIG, f"bool TeleopControllerNode::commit_hand_command({NEW_ARGS})")
    replace("controller", "  publish();return true;\n", HAND_COMMIT)
    anchor = "  input_options.callback_group = input_callback_group_;\n"
    replace("controller", anchor, anchor + GUARD_SUBSCRIBERS)
    anchor = "  message.sequence = command_sequence_ + 1;\n"
    replace("controller", anchor, anchor + ARM_PREPUBLISH)
    anchor = "  command_sequence_ = message.sequence;\n"
    replace("controller", anchor, ARM_COMMIT + anchor)
    anchor = '    add("pilot_publish_advanced",pilot_published ? "true" : "false");\n'
    replace("controller", anchor, anchor +
            '    add("pilot_guard_sequence",pilot_published ? std::to_string(diagnostic_guard_last_arm_.seq) : "0");\n'
            '    add("pilot_guarded_publish",pilot_published && diagnostic_guard_last_arm_.permitted ? "true" : "false");\n')
    old = "[this](const auto &publish){return arm_->commit_hand_command(publish);}"
    new = ("[this](const auto &command, auto expiry, const auto &publish)"
           "{return arm_->commit_hand_command(command, expiry, publish);}")
    for side in ("left", "right"):
        replace("coordinator", f'"{side}",{old}', f'"{side}",{new}')
    replace("retarget_header", RETARGET_GATE,
            "using CommandGate=std::function<bool(const manus_wuji_interfaces::msg::HandCommand &,\n"
            "      int64_t source_until_ns, const std::function<void()> &)>;")
    replace("retarget", "    if(command_gate_) command_gate_(publish);else publish();",
            "    if(command_gate_) command_gate_(command, stamp_ns +\n"
            "        static_cast<int64_t>(max_input_age_ms_) * 1000000LL, publish);")
    header = directory / "include/tianji_teleop_controller/dcr009_teleop_guard.hpp"
    if header.exists() or header.is_symlink():
        raise ValueError("DCR-009 diagnostic prepublication overlay already exists")
    header.write_bytes(Path(__file__).with_name("dcr009_teleop_guard.hpp").read_bytes())
    for name, path in paths.items():
        if name != "core":
            path.write_text(text[name], encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_copy", type=Path)
    args = parser.parse_args()
    apply_guard(args.package_copy)


if __name__ == "__main__":
    main()
