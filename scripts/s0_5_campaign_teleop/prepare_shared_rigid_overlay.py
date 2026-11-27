#!/usr/bin/env python3
"""Patch an isolated Teleop source copy for the same-binary nominal mapping A/B."""

import argparse
from pathlib import Path

CORE_HOOK = r'''struct SharedRigidReplay {
  Transform world_from_tracker;
  std::array<Transform, 2> world_from_base;
  std::array<Transform, 2> tracker_from_wuji;
  std::array<Transform, 2> wuji_from_native_tcp;

  Transform map(const SessionReference & reference, const Transform & tracker,
                std::size_t side) const
  {
    // One world transform serves both hands. Only the pair midpoint is latched.
    const Transform left_home = world_from_base[0] * reference.left_robot_start_ee *
        wuji_from_native_tcp[0].inverse();
    const Transform right_home = world_from_base[1] * reference.right_robot_start_ee *
        wuji_from_native_tcp[1].inverse();
    const Transform left_human_start = world_from_tracker *
        reference.left_tracker_start.pose_world * tracker_from_wuji[0];
    const Transform right_human_start = world_from_tracker *
        reference.right_tracker_start.pose_world * tracker_from_wuji[1];
    Transform shared = world_from_tracker;
    shared.translation() += 0.5 * (left_home.translation() + right_home.translation() -
                                   left_human_start.translation() - right_human_start.translation());
    return world_from_base[side].inverse() * shared * tracker *
        tracker_from_wuji[side] * wuji_from_native_tcp[side];
  }
};

const std::optional<SharedRigidReplay> & shared_rigid_replay()
{
  static const auto diagnostic = []() -> std::optional<SharedRigidReplay> {
    const char * path = std::getenv("S05_SHARED_RIGID_MAP");
    if (path == nullptr) return std::nullopt;
    const char * domain = std::getenv("ROS_DOMAIN_ID");
    const char * localhost = std::getenv("ROS_LOCALHOST_ONLY");
    if (domain == nullptr || std::string(domain) != "205" ||
        localhost == nullptr || std::string(localhost) != "1")
      throw std::runtime_error("S05 shared mapper requires isolated simulation ROS domain 205");
    std::ifstream input(path);
    const auto load = [&]() {
      Transform matrix = Transform::Identity();
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          if (!(input >> matrix.matrix()(i, j)))
            throw std::runtime_error("Incomplete S05 shared rigid map");
        }
      }
      const auto rotation = matrix.linear();
      if (!matrix.matrix().allFinite() ||
          !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
          std::abs(rotation.determinant() - 1.) > 1e-8)
        throw std::runtime_error("S05 shared rigid map contains a non-rigid transform");
      return matrix;
    };
    SharedRigidReplay value;
    value.world_from_tracker = load();
    for (auto & frame : value.world_from_base) frame = load();
    for (auto & frame : value.tracker_from_wuji) frame = load();
    for (auto & frame : value.wuji_from_native_tcp) frame = load();
    std::string extra;
    if (input >> extra) throw std::runtime_error("Unexpected fields in S05 shared rigid map");
    return value;
  }();
  return diagnostic;
}

'''


def replace_once(content, original, replacement, file):
    if content.count(original) != 1:
        raise ValueError(f"Source does not match the reviewed hook: {file}")
    return content.replace(original, replacement, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_copy", type=Path)
    parser.add_argument("--cohort-domains", action="store_true",
                        help="Limit the full-cohort diagnostic to localhost domains 206–213")
    args = parser.parse_args()
    directory = args.package_copy.resolve()
    if str(directory).startswith("/home/ubuntu/teleop_ws/") or directory.name != "whole_body_teleop":
        raise ValueError("Point this script at a fresh isolated whole_body_teleop source copy")

    core_file = directory / "src" / "teleop_core.cpp"
    node_file = directory / "src" / "teleop_controller_node.cpp"
    hook = CORE_HOOK
    if args.cohort_domains:
        hook = replace_once(
            hook, 'domain == nullptr || std::string(domain) != "205" ||',
            'domain == nullptr || std::string(domain).size() != 3 ||\n'
            '        std::string(domain) < "206" || std::string(domain) > "213" ||', core_file,
        ).replace("isolated simulation ROS domain 205", "isolated simulation ROS domain 206-213")
    core = "#include <cstdlib>\n#include <fstream>\n#include <optional>\n#include <stdexcept>\n" + core_file.read_text()
    original = "Transform scaled_delta_target("
    core = replace_once(core, original, hook + original, core_file)
    original = "      auto wrist = scaled_delta_target("
    replacement = """      if (const auto & diagnostic = shared_rigid_replay())
        return diagnostic->map(reference, pose, side);
      auto wrist = scaled_delta_target("""
    core = replace_once(core, original, replacement, core_file)
    node = "#include <cstdlib>\n" + node_file.read_text()
    original = '  config.absolute_orientation = bimanual_.t_heading_enabled ||'
    replacement = '''  if (std::getenv("S05_SHARED_RIGID_MAP") != nullptr && bimanual_.t_heading_enabled)
    throw std::invalid_argument("S05 shared mapping requires the original registered tracker payload seam");
  config.absolute_orientation = bimanual_.t_heading_enabled ||'''
    node = replace_once(node, original, replacement, node_file)
    original = 'add("mapping_policy",core_->config().absolute_orientation ? "world_absolute_orientation_v1" : "world_spatial_v1");'
    replacement = '''add("mapping_policy",std::getenv("S05_SHARED_RIGID_MAP") != nullptr ?
        "shared_rigid_diagnostic_v1" :
        (core_->config().absolute_orientation ? "world_absolute_orientation_v1" : "world_spatial_v1"));'''
    node = replace_once(node, original, replacement, node_file)
    core_file.write_text(core)
    node_file.write_text(node)


if __name__ == "__main__":
    main()
