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
SEGMENTED_ARGS = NEW_ARGS.replace("std::function<void()>",
                                  "std::function<void(const manus_wuji_interfaces::msg::HandCommand &)>")

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


SEGMENTED_LIMIT_LOG = '''void log_segmented_margin(const ArmSide side, const char * seam, const Transform & goal,
    const Transform & attempt, const JointVector & q, const JointVector & previous, const ArmLimits & limits,
    double margin)
{
  const auto phase = dcr009_diagnostic::segmented_phase();
  if (phase == dcr009_diagnostic::SegmentPhase::Invalid || phase == dcr009_diagnostic::SegmentPhase::History) return;
  static std::array<std::atomic_bool, 2> logged{false, false};
  const bool left = side == ArmSide::LEFT;
  for (Eigen::Index axis = 0; axis < q.size(); ++axis) {
    const double lo = limits.lower_rad[axis] + margin;
    const double hi = limits.upper_rad[axis] - margin;
    if (q[axis] >= lo && q[axis] <= hi) continue;
    if (logged[left ? 0 : 1].exchange(true)) return;
    const Eigen::Quaterniond goal_orientation(goal.linear());
    const Eigen::Quaterniond attempt_orientation(attempt.linear());
    const auto csv = [](const auto & values) {
      std::ostringstream out;
      out.precision(17);
      for (Eigen::Index j = 0; j < values.size(); ++j) { if (j) out << ','; out << values[j]; }
      return out.str();
    };
    std::ostringstream row;
    row.precision(17);
    row << "S05_DCR009_SEGMENT_MARGIN side=" << (left ? "left" : "right") << " seam=" << seam
        << " segmented=" << (phase == dcr009_diagnostic::SegmentPhase::Preposition ? 1 : 0)
        << " axis=" << axis + 1 << " joint=" << (left ? "left_j" : "right_j") << axis + 1
        << " native_joint=Joint" << axis + 1 << (left ? "_L" : "_R")
        << " request_rad=" << q[axis] << " prior_issued_rad=" << previous[axis]
        << " bound=" << (q[axis] < lo ? "lower" : "upper")
        << " margin_lower_rad=" << lo << " margin_upper_rad=" << hi << " margin_rad=" << margin
        << " hardware_lower_rad=" << limits.lower_rad[axis] << " hardware_upper_rad=" << limits.upper_rad[axis]
        << " solution_lower_rad=" << limits.solution_lower_rad[axis] << " solution_upper_rad=" << limits.solution_upper_rad[axis]
        << " raw_q_rad=" << csv(q) << " prior_q_rad=" << csv(previous)
        << " all_margin_lower_rad=" << csv((limits.lower_rad.array() + margin).matrix())
        << " all_margin_upper_rad=" << csv((limits.upper_rad.array() - margin).matrix())
        << " all_hardware_lower_rad=" << csv(limits.lower_rad) << " all_hardware_upper_rad=" << csv(limits.upper_rad)
        << " all_solution_lower_rad=" << csv(limits.solution_lower_rad) << " all_solution_upper_rad=" << csv(limits.solution_upper_rad)
        << " goal_xyz_m=" << csv(goal.translation()) << " goal_xyzw=" << csv(goal_orientation.coeffs())
        << " attempt_xyz_m=" << csv(attempt.translation()) << " attempt_xyzw=" << csv(attempt_orientation.coeffs()) << '\\n';
    std::clog << row.str();
    return;
  }
}

'''

SEGMENTED_PHASE = '''  const auto dcr009_phase = dcr009_diagnostic::segmented_phase();
  const bool dcr009_preposition = dcr009_phase == dcr009_diagnostic::SegmentPhase::Preposition;
  if (dcr009_phase == dcr009_diagnostic::SegmentPhase::Invalid ||
      (dcr009_preposition && (!config_.manimux_analytic_ik_enabled || !config_.analytic_manimux_position_output))) {
    result.safety = failure(SafetyCode::INVALID_INPUT, "segmented DCR-009 phase or ManiMux profile is unbound");
    return result;
  }
  const double dcr009_dt = dcr009_preposition ? std::min(bounded_dt, .004) : bounded_dt;
'''

SEGMENTED_HAND = '''  auto admitted_command = command;
  const auto segment_alpha = diagnostic_nominal_guard_.segment_hand(group, node_.now().nanoseconds(), admitted_command.position);
  if (!segment_alpha) {
    RCLCPP_INFO(node_.get_logger(), "S05_DCR009_GUARD group=%s session=%s generation=%llu command=%llu proof=0 published=0 reason=%u feedback_ns=0",
        command.side.c_str(), session_id_.c_str(), static_cast<unsigned long long>(generation_),
        static_cast<unsigned long long>(command.sequence), dcr009_diagnostic::NominalGuard::kSegmentUnavailable);
    return false;
  }
  if (*segment_alpha < 1.) {
    RCLCPP_INFO(node_.get_logger(), "S05_DCR009_SEGMENT_HAND side=%s generation=%llu command=%llu alpha=%.17g",
        command.side.c_str(), static_cast<unsigned long long>(generation_),
        static_cast<unsigned long long>(command.sequence), *segment_alpha);
  }
'''

SEGMENTED_HAND_RECEIPT = '''  if (dcr009_diagnostic::segmented_phase() == dcr009_diagnostic::SegmentPhase::Preposition) {
    try {
      diagnostic_msgs::msg::DiagnosticArray trace;
      trace.header.stamp = node_.now();
      diagnostic_msgs::msg::DiagnosticStatus report;
      report.name = "tianji/" + command.side + "_hand/dcr009_segmented_target";
      report.hardware_id = "dcr009_isolated_mujoco";
      const auto field = [&](const std::string & key, const std::string & value) {
        diagnostic_msgs::msg::KeyValue pair;
        pair.key = key;
        pair.value = value;
        report.values.push_back(pair);
      };
      const auto vector = [](const auto & values) {
        std::ostringstream out;
        out.precision(17);
        for (std::size_t j = 0; j < values.size(); ++j) { if (j) out << ','; out << values[j]; }
        return out.str();
      };
      field("session", session_id_);
      field("generation", std::to_string(generation_));
      field("command_sequence", std::to_string(command.sequence));
      field("command_header_ns", std::to_string(issued_ns));
      field("guard_sequence", std::to_string(proof.seq));
      field("raw_wuji_target_rad", vector(command.position));
      field("issued_target_rad", vector(admitted_command.position));
      trace.status.push_back(report);
      diagnostic_pub_->publish(trace);
    } catch (const std::exception & error) {
      RCLCPP_WARN(node_.get_logger(), "DCR-009 segmented hand target diagnostic failed: %s", error.what());
    }
  }
'''


def apply_guard(directory, *, segmented_root_five=False):
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
    args = SEGMENTED_ARGS if segmented_root_five else NEW_ARGS
    replace("controller_header", CONTROLLER_DECL, f"bool commit_hand_command({args});")
    anchor = "  friend struct TeleopControllerNodeTestPeer;\n"
    replace("controller_header", anchor, anchor + CONTROLLER_FIELDS)
    replace("controller", CONTROLLER_SIG, f"bool TeleopControllerNode::commit_hand_command({args})")
    hand = HAND_COMMIT
    if segmented_root_five:
        hand = replace_once(hand, "  auto proof =", SEGMENTED_HAND + "  auto proof =", paths["controller"])
        hand = replace_once(hand, "command.joint_names, command.position);",
                            "admitted_command.joint_names, admitted_command.position);", paths["controller"])
        hand = replace_once(hand, "    publish();", "    publish(admitted_command);", paths["controller"])
        hand = replace_once(hand, "  evidence(true, 0);\n", "  evidence(true, 0);\n" + SEGMENTED_HAND_RECEIPT,
                            paths["controller"])
    replace("controller", "  publish();return true;\n", hand)
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
    hand_publisher = ("std::function<void(const manus_wuji_interfaces::msg::HandCommand &)>"
                      if segmented_root_five else "std::function<void()>")
    replace("retarget_header", RETARGET_GATE,
            "using CommandGate=std::function<bool(const manus_wuji_interfaces::msg::HandCommand &,\n"
            f"      int64_t source_until_ns, const {hand_publisher} &)>;")
    replace("retarget", "    if(command_gate_) command_gate_(publish);else publish();",
            "    if(command_gate_) command_gate_(command, stamp_ns +\n"
            "        static_cast<int64_t>(max_input_age_ms_) * 1000000LL, publish);")
    if segmented_root_five:
        # Only this isolated build can see the opt-in segmentation and altered hand callback.
        replace("retarget", "    const auto publish=[&] {publisher_->publish(command);};",
                "    const auto publish=[&](const auto & admitted) {publisher_->publish(admitted);};")
        anchor = "  input_options.callback_group = input_callback_group_;\n"
        replace("controller", anchor, anchor +
                "  const auto diagnostic_phase = dcr009_diagnostic::segmented_phase();\n"
                "  if (diagnostic_phase != dcr009_diagnostic::SegmentPhase::Disabled &&\n"
                "      diagnostic_phase != dcr009_diagnostic::SegmentPhase::Preposition)\n"
                "    throw std::invalid_argument(\"DCR-009 segmented mode requires the owned PREPOSITION state\");\n")
        text["core"] = ('#include <iostream>\n#include "tianji_teleop_controller/dcr009_teleop_guard.hpp"\n' + text["core"])
        anchor = "Transform scaled_delta_target("
        replace("core", anchor, SEGMENTED_LIMIT_LOG + anchor)
        anchor = "  Transform step = desired;\n  const double translation_budget = config_.max_translation_rate_m_sec * bounded_dt;"
        replace("core", anchor, SEGMENTED_PHASE + "  Transform step = desired;\n"
                "  const double translation_budget = (dcr009_preposition ? std::min(config_.max_translation_rate_m_sec, .060) :\n"
                "      config_.max_translation_rate_m_sec) * dcr009_dt;")
        anchor = "  const double rotation_budget = config_.max_rotation_rate_rad_sec * bounded_dt;"
        replace("core", anchor, "  const double rotation_budget = (dcr009_preposition ?\n"
                "      std::min(config_.max_rotation_rate_rad_sec, .30) : config_.max_rotation_rate_rad_sec) * dcr009_dt;")
        anchor = """        const auto status = validate_command_joints(*kinematics_, side, joints, limits,
                                                    config_.joint_limit_margin_rad);
        if (!status.safe) return status;"""
        replace("core", anchor, anchor.replace("        if (!status.safe) return status;", """        if (!status.safe && status.code == SafetyCode::JOINT_LIMIT &&
            status.reason == "joint command violates uniform joint limit margin") {
          log_segmented_margin(side, "raw_ik", requested, result.attempt_target, joints, q_cmd_previous,
              limits, config_.joint_limit_margin_rad);
        }
        if (!status.safe) return status;"""))
        anchor = """        arm.safety=validate_command_joints(*kinematics_,side,requested,
            limits,config_.joint_limit_margin_rad);
        if(!arm.safety.safe)return;"""
        replace("core", anchor, anchor.replace("        if(!arm.safety.safe)return;", """        if (!arm.safety.safe && arm.safety.code == SafetyCode::JOINT_LIMIT &&
            arm.safety.reason == "joint command violates uniform joint limit margin") {
          log_segmented_margin(side, "finalize", arm.ee_target, arm.attempt_target, requested, previous,
              limits, config_.joint_limit_margin_rad);
        }
        if(!arm.safety.safe)return;"""))
    header = directory / "include/tianji_teleop_controller/dcr009_teleop_guard.hpp"
    if header.exists() or header.is_symlink():
        raise ValueError("DCR-009 diagnostic prepublication overlay already exists")
    header.write_bytes(Path(__file__).with_name("dcr009_teleop_guard.hpp").read_bytes())
    for name, path in paths.items():
        if name != "core" or segmented_root_five:
            path.write_text(text[name], encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_copy", type=Path)
    parser.add_argument("--segmented-root-five", action="store_true")
    args = parser.parse_args()
    apply_guard(args.package_copy, segmented_root_five=args.segmented_root_five)


if __name__ == "__main__":
    main()
