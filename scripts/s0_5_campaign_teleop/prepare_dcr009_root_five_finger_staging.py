#!/usr/bin/env python3
"""Add an isolated START scheduler ahead of the unchanged native guard."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from prepare_shared_rigid_overlay import replace_once


SOURCES = {
    "src/teleop_controller_node.cpp": "25420682b2b6bbc1b0f6ec8b92cdb17fb30fac99f84b8185806f8992c2b02c5c",
    "include/tianji_teleop_controller/dcr009_teleop_guard.hpp": "5995ef2830a216d3787bd86b3cc55b25726621e9da61ca5dd1b87c984654ce82",
}
STAGE_GATE = """  // Scope this selector to the isolated wrist-first diagnostic. The existing
  // fresh-state, slope, command range and swept checks still see a full group.
  const char * finger_path = std::getenv("S05_DCR009_FINGER_STAGE");
  int finger_stage = 0;
  double finger_delta = 0.;
  std::array<double, 20> * finger_previous = nullptr;
  if (finger_path) {
    if ((group != 1u && group != 2u) ||
        dcr009_diagnostic::segmented_phase() != dcr009_diagnostic::SegmentPhase::Preposition ||
        command.position.size() != 20u ||
        !std::all_of(command.position.begin(), command.position.end(), [](double q){return std::isfinite(q);})) {
      return false;
    }
    struct stat meta{};
    int requested_stage = 0;
    int64_t until = 0;
    const auto now = node_.now().nanoseconds();
    if (finger_path[0] == '/' && ::lstat(finger_path, &meta) == 0 && S_ISREG(meta.st_mode) &&
        meta.st_uid == ::geteuid() && !(meta.st_mode & 077u) && meta.st_size <= 48) {
      std::ifstream input(finger_path);
      std::string extra;
      if (input >> requested_stage >> until && !(input >> extra) &&
          requested_stage >= 1 && requested_stage <= 11 && until > now && until - now <= 40000000LL)
        finger_stage = requested_stage;
    }
    // These priors are initialized from the guard's actual 54-axis envelope
    // and advanced only on a successful original native publish and commit.
    static std::array<std::optional<std::array<double, 20>>, 2> previous{};
    static std::string owner;
    static uint64_t owner_generation{0};
    if (owner != session_id_ || owner_generation != generation_) {
      previous = {};
      owner = session_id_;
      owner_generation = generation_;
    }
    auto & selected = previous[group - 1u];
    if (!selected) {
      if (!diagnostic_guard_last_arm_.permitted) return false;
      selected.emplace();
      const std::size_t offset = group == 1u ? 14u : 34u;
      std::copy_n(diagnostic_guard_last_arm_.candidate.begin() + offset, 20u, selected->begin());
    }
    finger_previous = &selected.value();
    static constexpr std::array<int, 12> digits{-1, 3, 0, 1, 2, 4, 0, 1, 2, 3, 4, -1};
    const auto digit = digits.at(finger_stage);
    const bool this_hand = (finger_stage >= 1 && finger_stage <= 5 && group == 2u) ||
                           (finger_stage >= 6 && finger_stage <= 10 && group == 1u);
    for (std::size_t j = 0; j < 20u; ++j) {
      if (!this_hand || static_cast<int>(j / 4) != digit)
        admitted_command.position[j] = (*finger_previous)[j];
    }
  }
"""


def patch(package, receipt):
    for name, expected in SOURCES.items():
        if sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"The previous verified source differs: {name}")
    controller = package / next(iter(SOURCES))
    source = controller.read_text(encoding="utf-8")
    source = replace_once(source, "  auto admitted_command = command;\n",
                          "  auto admitted_command = command;\n" + STAGE_GATE, controller)
    anchor = "  const auto evidence = [&](bool published, uint32_t reason) {\n"
    source = replace_once(source, anchor, """  if (finger_previous)
    for (std::size_t j = 0; j < 20u; ++j)
      finger_delta = std::max(finger_delta, std::abs(admitted_command.position[j] - (*finger_previous)[j]));
""" + anchor + """    if (finger_path)
      RCLCPP_INFO(node_.get_logger(),
          "S05_DCR009_FINGER_STEP stage=%d side=%s command=%llu proof=%llu published=%d reason=%u delta_rad=%.17g",
          finger_stage, command.side.c_str(), static_cast<unsigned long long>(command.sequence),
          static_cast<unsigned long long>(proof.seq), published ? 1 : 0, reason, finger_delta);
""", controller)
    anchor = "  diagnostic_nominal_guard_.commit(proof);\n  evidence(true, 0);\n"
    source = replace_once(source, anchor, anchor + """  if (finger_previous)
    std::copy(admitted_command.position.begin(), admitted_command.position.end(), finger_previous->begin());
""", controller)
    anchor = '      field("guard_sequence", std::to_string(proof.seq));\n'
    source = replace_once(source, anchor, anchor +
                          '      if (finger_path) field("finger_stage", std::to_string(finger_stage));\n', controller)
    body = source.encode()
    controller.write_bytes(body)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"baseline_source_sha256": SOURCES, "new_controller_source_sha256": sha256(body).hexdigest(),
                   "scope": "WRIST_FIRST_ROOT_FIVE_MUJOCO_DIAGNOSTIC"}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isolated_package_copy", type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args()
    patch(arguments.isolated_package_copy, arguments.receipt)
