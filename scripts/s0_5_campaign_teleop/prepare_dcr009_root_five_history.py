#!/usr/bin/env python3
"""Bind the isolated root-five replay to a native STOP publication barrier."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from prepare_shared_rigid_overlay import replace_once


SOURCES = {
    "src/teleop_controller_node.cpp": "ba43d3413c60b16e8b46cf6bb29f702e1035a221569d640b600a00349277f268",
    "include/tianji_teleop_controller/teleop_controller_node.hpp":
        "3cb33fc6509f3d19386e6a594b55d089626792e3b1232d92424418a57a73aa9d",
    "include/tianji_teleop_controller/dcr009_teleop_guard.hpp":
        "5995ef2830a216d3787bd86b3cc55b25726621e9da61ca5dd1b87c984654ce82",
}


STOP_BARRIER = """    if (report_diagnostic_stop) {
      const auto barrier_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      RCLCPP_INFO(node_.get_logger(),
          "S05_DCR009_STOP_BARRIER session=%s generation=%llu last_arm=%llu last_left=%llu last_right=%llu mono_ns=%lld",
          stopped_session.c_str(), static_cast<unsigned long long>(stopped_generation),
          static_cast<unsigned long long>(diagnostic_last_arm_sequence_),
          static_cast<unsigned long long>(diagnostic_last_hand_sequence_[0]),
          static_cast<unsigned long long>(diagnostic_last_hand_sequence_[1]), static_cast<long long>(barrier_ns));
    }
    diagnostic_last_hand_sequence_.fill(0);
"""


def patch(package, receipt):
    for name, expected in SOURCES.items():
        if sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"The paired controller source differs: {name}")
    controller_name, header_name, _ = SOURCES
    controller, header = package / controller_name, package / header_name
    cpp = controller.read_text(encoding="utf-8")
    declaration = "  dcr009_diagnostic::NominalGuard::Proof diagnostic_guard_last_arm_;\n"
    hpp = replace_once(header.read_text(encoding="utf-8"), declaration, declaration +
                       "  std::array<uint64_t,2> diagnostic_last_hand_sequence_{};\n"
                       "  uint64_t diagnostic_last_arm_sequence_{0}, diagnostic_publish_generation_{0};\n"
                       "  std::string diagnostic_publish_session_;\n", header)
    anchor = """void TeleopControllerNode::invalidate_workflow_session(const std::string & reason,bool stop_hands)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
"""
    cpp = replace_once(cpp, anchor, anchor + """    const std::string stopped_session =
        diagnostic_publish_session_.empty() ? session_id_ : diagnostic_publish_session_;
    const auto stopped_generation =
        diagnostic_publish_session_.empty() ? generation_ : diagnostic_publish_generation_;
    const bool report_diagnostic_stop = reason == "teleoperation stopped" &&
        std::getenv("S05_DCR009_HISTORY_AUDIT") != nullptr && !stopped_session.empty();
""", controller)
    anchor = "    last_head_cycle_steady_sec_ = 0.0;\n  }\n  publish_state(reason);"
    cpp = replace_once(cpp, anchor, "    last_head_cycle_steady_sec_ = 0.0;\n" + STOP_BARRIER +
                       "  }\n  publish_state(reason);", controller)
    cpp = replace_once(cpp, '  const char * finger_path = std::getenv("S05_DCR009_FINGER_STAGE");',
                       """  const bool reviewing_history = std::getenv("S05_DCR009_HISTORY_AUDIT") != nullptr &&
      dcr009_diagnostic::segmented_phase() == dcr009_diagnostic::SegmentPhase::History;
  const char * finger_path = reviewing_history ? nullptr : std::getenv("S05_DCR009_FINGER_STAGE");""",
                       controller)
    cpp = replace_once(cpp, "  int finger_stage = 0;\n", "  int finger_stage = 0;\n  int64_t finger_deadline_ns = 0;\n",
                       controller)
    cpp = replace_once(cpp, "        finger_stage = requested_stage;", """      {
        finger_stage = requested_stage;
        finger_deadline_ns = until;
      }""", controller)
    anchor = "  if (!diagnostic_nominal_guard_.still_valid(proof, node_.now().nanoseconds(), deadline_ns)) {"
    cpp = replace_once(cpp, anchor, """  // A pending nonzero digit step cannot outlive its wrist-feedback lease.
  const auto final_hand_deadline_ns = finger_delta > 0. ? std::min(deadline_ns, finger_deadline_ns) : deadline_ns;
  if (!diagnostic_nominal_guard_.still_valid(proof, node_.now().nanoseconds(), final_hand_deadline_ns)) {""", controller)
    hand_log = 'command=%llu proof=%llu published=%d reason=%u feedback_ns=%lld",\n        command.side.c_str(), session_id_.c_str(),'
    cpp = replace_once(cpp, hand_log, hand_log.replace('feedback_ns=%lld"', 'feedback_ns=%lld command_header_ns=%lld"'),
                       controller)
    cpp = replace_once(cpp, "        published ? 1 : 0, reason, static_cast<long long>(proof.feedback_header));",
                       """        published ? 1 : 0, reason, static_cast<long long>(proof.feedback_header),
        static_cast<long long>(issued_ns));""", controller)
    anchor = "  diagnostic_nominal_guard_.commit(proof);\n  evidence(true, 0);"
    cpp = replace_once(cpp, anchor, "  diagnostic_nominal_guard_.commit(proof);\n"
                       "  diagnostic_last_hand_sequence_.at(group - 1u) = command.sequence;\n"
                       "  diagnostic_publish_session_ = session_id_;\n"
                       "  diagnostic_publish_generation_ = generation_;\n"
                       "  evidence(true, 0);", controller)
    arm_log = 'command=%llu proof=%llu published=1 reason=0 feedback_ns=%lld",'
    cpp = replace_once(cpp, arm_log, arm_log.replace('feedback_ns=%lld"', 'feedback_ns=%lld command_header_ns=%lld"'),
                       controller)
    anchor = "      static_cast<long long>(diagnostic_guard_last_arm_.feedback_header));"
    cpp = replace_once(cpp, anchor, """      static_cast<long long>(diagnostic_guard_last_arm_.feedback_header),
      static_cast<long long>(message.header.stamp.sec) * 1000000000LL + message.header.stamp.nanosec);""", controller)
    anchor = "  command_sequence_ = message.sequence;\n"
    cpp = replace_once(cpp, anchor, anchor + """  diagnostic_last_arm_sequence_ = message.sequence;
  diagnostic_publish_session_ = session_id_;
  diagnostic_publish_generation_ = generation_;
""", controller)
    updated = {controller_name: cpp.encode(), header_name: hpp.encode()}
    for name, content in updated.items():
        (package / name).write_bytes(content)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"scope": "ROOT_FIVE_GUARDED_HISTORY_STOP_AUDIT", "baseline": SOURCES,
                   "updated": {name: sha256(content).hexdigest() for name, content in updated.items()}}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isolated_package_copy", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    patch(args.isolated_package_copy, args.receipt)
