// Narrow, offline SI interface to the reviewed production Marvin adapter.
// No ROS runtime or robot driver is linked.
#include <memory>
#include <string>

#include "tianji_teleop_controller/kinematics_adapter.hpp"
#include "tianji_teleop_controller/marvin_kine_sdk.hpp"

using tianji_teleop_controller::ArmSide;
using tianji_teleop_controller::JointVector;
using tianji_teleop_controller::MarvinKinematicsAdapter;
using tianji_teleop_controller::Transform;

namespace {
using RowMajorPose = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
thread_local std::string last_error;
ArmSide arm(int side) {
  return side == 0 ? ArmSide::LEFT : ArmSide::RIGHT;
}
}  // namespace

extern "C" void *s05_marvin_open(const char *config) noexcept {
  try {
    std::string reason;
    auto sdk = tianji_teleop_controller::create_marvin_kine_backend(config, 0, 1, reason);
    if (!sdk) return nullptr;
    auto adapter = std::make_unique<MarvinKinematicsAdapter>(config, sdk, 0, 1);
    return adapter->initialized() ? adapter.release() : nullptr;
  } catch (...) {
    return nullptr;
  }
}

extern "C" void s05_marvin_close(void *handle) noexcept {
  delete static_cast<MarvinKinematicsAdapter *>(handle);
}

extern "C" int s05_marvin_fk(void *handle, int side, const double *q, double *out) noexcept {
  try {
    const auto result = static_cast<MarvinKinematicsAdapter *>(handle)->fk(
        arm(side), Eigen::Map<const JointVector>(q));
    if (!result.success) return 1;
    Eigen::Map<RowMajorPose> pose_out(out);
    pose_out = result.transform.matrix();
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" int s05_marvin_workspace_probe(void *handle, int side, const double *target,
                                          double *projection) noexcept {
  try {
    const auto *adapter = static_cast<MarvinKinematicsAdapter *>(handle);
    const Transform goal{Eigen::Map<const RowMajorPose>(target)};
    const auto envelope = adapter->workspace(arm(side));
    const auto result = tianji_teleop_controller::project_workspace_target(goal, envelope, 0.0005, 0.005);
    if (!result.valid) return 1;
    projection[0] = result.offset_m;
    projection[1] = envelope.outer_radius_m;
    projection[2] = (goal.translation() - goal.linear() * envelope.tool_offset - envelope.shoulder).norm();
    Eigen::Map<RowMajorPose> result_pose(projection + 3);
    result_pose = result.pose.matrix();
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" const char *s05_marvin_last_error() noexcept {
  return last_error.c_str();
}

extern "C" int s05_marvin_ik(void *handle, int side, const double *target, const double *seed,
                            double *out) noexcept {
  try {
    const auto *adapter = static_cast<MarvinKinematicsAdapter *>(handle);
    const auto which = arm(side);
    const JointVector start = Eigen::Map<const JointVector>(seed);
    const Transform goal{Eigen::Map<const RowMajorPose>(target)};
    // The preferred elbow branch follows the previous command's SDK NSP plane.
    const auto plane = adapter->fk_nsp(which, start);
    last_error.clear();
    auto result = plane.success ? adapter->ik_nsp(which, goal, start, plane.direction)
                                : adapter->ik(which, goal, start);
    if (!result.success && plane.success) {
      last_error = "NSP: " + result.reason + "; ";
      result = adapter->ik(which, goal, start);
    }
    if (!result.success) {
      last_error += "standard: " + result.reason;
      return 1;
    }
    const auto valid = adapter->validate_joints(which, result.joints);
    if (!valid.success) {
      last_error = "SDK limit: " + valid.reason;
      return 2;
    }
    Eigen::Map<JointVector> joints_out(out);
    joints_out = result.joints;
    return 0;
  } catch (...) {
    last_error = "Native adapter exception";
    return 3;
  }
}
