#!/usr/bin/env python3
"""Install the single-session DCR-009 target hook into an isolated Teleop source copy."""

import argparse
from pathlib import Path

from prepare_shared_rigid_overlay import replace_once


PILOT_CORE = r'''struct HomeAnchoredPilot {
  std::string session;
  Transform world_from_human;
  std::array<Transform, 2> world_from_base;
  std::array<Transform, 2> tracker_from_palm;
  std::array<Transform, 2> palm_from_native_tcp;

  Transform target(const SessionReference & reference, const Transform & tracker,
                   std::size_t side) const
  {
    if (reference.session_id != session)
      throw std::runtime_error("DCR-009 runtime session differs from the sealed Home anchor");
    return world_from_base[side].inverse() * world_from_human * tracker *
        tracker_from_palm[side] * palm_from_native_tcp[side];
  }
};

const HomeAnchoredPilot & home_anchored_pilot()
{
  static const HomeAnchoredPilot pilot = []() {
    const char * path = std::getenv("S05_HOME_ANCHORED_SESSION_MAP");
    const char * domain = std::getenv("ROS_DOMAIN_ID");
    const char * local = std::getenv("ROS_LOCALHOST_ONLY");
    if (path == nullptr || domain == nullptr || local == nullptr || std::string(local) != "1" ||
        std::string(domain).size() != 3 || std::string(domain) < "206" || std::string(domain) > "213" ||
        std::getenv("S05_SHARED_RIGID_MAP") != nullptr)
      throw std::runtime_error("DCR-009 requires one sealed anchor in an isolated MuJoCo domain 206–213");
    std::ifstream input(path);
    HomeAnchoredPilot value;
    std::string version;
    if (!(input >> version >> value.session) || version != "S05_DCR009_START_MAP_V1" ||
        value.session.rfind("s05-dcr009-", 0) != 0)
      throw std::runtime_error("Invalid DCR-009 one-session map header");
    const auto read = [&]() {
      Transform matrix = Transform::Identity();
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          if (!(input >> matrix.matrix()(i, j)))
            throw std::runtime_error("Missing DCR-009 frame matrix");
        }
      }
      const auto rotation = matrix.linear();
      if (!matrix.matrix().allFinite() ||
          !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
          std::abs(rotation.determinant() - 1.) > 1e-8)
        throw std::runtime_error("DCR-009 frame matrix is not a proper rigid transform");
      return matrix;
    };
    value.world_from_human = read();
    for (auto & frame : value.world_from_base) frame = read();
    for (auto & frame : value.tracker_from_palm) frame = read();
    for (auto & frame : value.palm_from_native_tcp) frame = read();
    std::string extra;
    if (input >> extra) throw std::runtime_error("Trailing fields in DCR-009 session map");
    return value;
  }();
  return pilot;
}

'''


PILOT_DIAGNOSTICS = '''    const auto pilot_stamp = to_builtin_time(current.now_sec);
    const bool pilot_published = command_sequence_ == current.command_sequence + 1;
    add("pilot_generation",std::to_string(current.generation));
    add("pilot_previous_sequence",std::to_string(current.command_sequence));
    add("pilot_publish_advanced",pilot_published ? "true" : "false");
    add("pilot_command_header_ns",std::to_string(
        static_cast<int64_t>(pilot_stamp.sec) * 1000000000LL + pilot_stamp.nanosec));
    add("pilot_arm_status",!pilot_published ? "NO_PUBLISH" : !arm.accepted ? "HOLD" :
        (arm.trajectory_braking || arm.servo_braking || !arm.ik_target_accepted) ? "FALLBACK" :
        (arm.rate_limited || arm.workspace_projected || arm.backoff_scale < 1.0 ||
         arm.projected_axis >= 0 || arm.task_projection >= 0) ? "LIMITED" : "ACCEPTED");
    add("pilot_q_nominal_candidate_rad",vector_text(arm.joint_candidate));
    add("pilot_q_post_finalize_rad",vector_text(arm.joint_command));
    add("pilot_q_previous_issued_rad",vector_text(previous));
'''


def prepare(directory):
    directory = directory.resolve()
    if directory.name != "whole_body_teleop" or directory.is_relative_to(Path("/home/ubuntu/teleop_ws")):
        raise ValueError("Use a fresh isolated whole_body_teleop source copy")
    core_file = directory / "src" / "teleop_core.cpp"
    node_file = directory / "src" / "teleop_controller_node.cpp"
    core = core_file.read_text(encoding="utf-8")
    node = node_file.read_text(encoding="utf-8")
    if "shared_rigid_replay()" in core or "HomeAnchoredPilot" in core:
        raise ValueError("Prepare the isolated pilot from the original reviewed native source")
    core = replace_once(core, "Transform scaled_delta_target(", PILOT_CORE + "Transform scaled_delta_target(", core_file)
    seam = "      auto wrist = scaled_delta_target("
    mapped = ('      if (std::getenv("S05_HOME_ANCHORED_SESSION_MAP") != nullptr)\n'
              '        return home_anchored_pilot().target(reference, pose, side);\n')
    # The coordinator requires the pilot map at startup. Keep historical math
    # visible in the isolated diff and preserve the original mapping structure.
    core = replace_once(core, seam, mapped + seam, core_file)
    core = "#include <cstdlib>\n#include <fstream>\n#include <stdexcept>\n" + core
    mode = "  config.absolute_orientation = bimanual_.t_heading_enabled ||"
    guard = '''  if (std::getenv("S05_HOME_ANCHORED_SESSION_MAP") == nullptr || bimanual_.t_heading_enabled)
    throw std::invalid_argument("DCR-009 requires its single-session map and registered tracker payload seam");
'''
    node = replace_once(node, mode, guard + mode, node_file)
    mapping = 'add("mapping_policy",core_->config().absolute_orientation ? "world_absolute_orientation_v1" : "world_spatial_v1");'
    node = replace_once(node, mapping, 'add("mapping_policy","home_anchored_pilot_v1");', node_file)
    accepting = '    add("arm_command_accepted",arm.accepted ? "true" : "false");\n'
    node = replace_once(node, accepting, accepting + PILOT_DIAGNOSTICS, node_file)
    node = "#include <cstdlib>\n" + node
    core_file.write_text(core, encoding="utf-8")
    node_file.write_text(node, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_copy", type=Path)
    args = parser.parse_args()
    prepare(args.package_copy)


if __name__ == "__main__":
    main()
