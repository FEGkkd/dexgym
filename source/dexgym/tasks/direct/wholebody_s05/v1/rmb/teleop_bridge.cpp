// A narrow offline adapter around the pinned Teleop command kernel.
// No ROS node, driver, publisher, or device session is linked.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <utility>

#include "tianji_teleop_controller/marvin_kine_sdk.hpp"
#include "tianji_teleop_controller/teleop_core.hpp"

namespace {
using namespace tianji_teleop_controller;
using Pose = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using Joints = Eigen::Map<const JointVector>;

struct Bridge {
  std::shared_ptr<MarvinKinematicsAdapter> marvin;
  TeleopCore core;
  SessionReference reference;
  std::string reasons[2];
};

// This ABI mirrors the checked sim.yaml-over-real.yaml arm configuration.
struct Profile {
  double lower[7], upper[7], solution_lower[7], solution_upper[7];
  double fk_position_tolerance_m, fk_orientation_tolerance_rad;
  double max_raw_joint_step_rad, max_command_rate_rad_sec, max_command_dt_sec;
  double max_tracking_error_rad, hard_tracking_error_rad;
  double max_translation_rate_m_sec, max_rotation_rate_rad_sec;
  double workspace_reserve_m, workspace_transition_m, joint_limit_margin_deg;
  double analytic_joint_rate_deg_sec, analytic_max_step_dt_sec;
  double nullspace_activation_deg, nullspace_rate_deg_sec, nullspace_probe_deg, arm_angle_limit_deg;
  double axis_projection_min_retained_ratio, arm_angle_centering_weight, tracker_timeout_sec;
  double zsp_angle_deg, left_zsp_para[6], right_zsp_para[6], dgr_deg[3];
  int zsp_type, held_boundary_brake;
};

struct ArmOutput {
  double joint_candidate[7], joint_command[7];
  double step_target[16], workspace_target[16], attempt_target[16];
  double arm_angle_command_rad, workspace_projection_m, position_lag_m, rotation_lag_rad, backoff_scale;
  int projected_axis, safety_code;
  std::uint32_t flags;
};

constexpr double radians(double degrees) {
  return degrees * (std::numbers::pi / 180.);
}

CoreConfig core_config(const Profile & p) {
  CoreConfig c;
  c.fk_position_tolerance_m = p.fk_position_tolerance_m;
  c.fk_orientation_tolerance_rad = p.fk_orientation_tolerance_rad;
  c.max_raw_joint_step_rad = p.max_raw_joint_step_rad;
  c.max_command_rate_rad_sec = p.max_command_rate_rad_sec;
  c.max_command_dt_sec = p.max_command_dt_sec;
  c.max_tracking_error_rad = p.max_tracking_error_rad;
  c.hard_tracking_error_rad = p.hard_tracking_error_rad;
  c.max_translation_rate_m_sec = p.max_translation_rate_m_sec;
  c.max_rotation_rate_rad_sec = p.max_rotation_rate_rad_sec;
  c.workspace_boundary_enabled = true;
  c.workspace_boundary_reserve_m = p.workspace_reserve_m;
  c.workspace_boundary_transition_m = p.workspace_transition_m;
  c.joint_limit_margin_rad = radians(p.joint_limit_margin_deg);
  c.manimux_analytic_ik_enabled = true;
  c.analytic_uniform_joint_scaling = true;
  c.analytic_manimux_position_output = true;
  c.analytic_source_nullspace_enabled = true;
  c.analytic_joint_rate_rad_sec = radians(p.analytic_joint_rate_deg_sec);
  c.analytic_max_step_dt_sec = p.analytic_max_step_dt_sec;
  c.nullspace_activation_rad = radians(p.nullspace_activation_deg);
  c.nullspace_rate_rad_sec = radians(p.nullspace_rate_deg_sec);
  c.nullspace_probe_rad = radians(p.nullspace_probe_deg);
  c.arm_angle_limit_rad = radians(p.arm_angle_limit_deg);
  c.arm_angle_centering_weight = p.arm_angle_centering_weight;
  c.axis_projection_enabled = true;
  c.axis_projection_min_retained_ratio = p.axis_projection_min_retained_ratio;
  c.independent_arm_execution = true;
  c.servo_held_boundary_brake_enabled = p.held_boundary_brake != 0;
  c.tracker_timeout_sec = p.tracker_timeout_sec;
  c.backoff_scales = {1., .5, .25, .1};
  c.left_limits = {Joints(p.lower), Joints(p.upper), Joints(p.solution_lower), Joints(p.solution_upper)};
  c.right_limits = c.left_limits;
  return c;
}

void put_pose(double * destination, const Transform & value) {
  Eigen::Map<Pose> mapped(destination);
  mapped = value.matrix();
}

// See the pinned TeleopCore local_arm_block. This only labels the final
// aggregate denial and cannot grant authority the native kernel withheld.
bool local_code(SafetyCode code) {
  switch (code) {
    case SafetyCode::OK:
    case SafetyCode::IK_FAILURE:
    case SafetyCode::FK_POSITION_RESIDUAL:
    case SafetyCode::FK_ORIENTATION_RESIDUAL:
    case SafetyCode::JOINT_LIMITING:
    case SafetyCode::JOINT_LIMIT:
    case SafetyCode::COUPLED_JOINT_LIMIT:
    case SafetyCode::JOINT_DISCONTINUITY:
    case SafetyCode::ELBOW_BRANCH:
    case SafetyCode::ELBOW_WORKSPACE:
    case SafetyCode::TRACKER_INVALID:
    case SafetyCode::STALE_TRACKER:
    case SafetyCode::TRACKING_ERROR: return true;
    default: return false;
  }
}

void export_arm(ArmOutput & out, const ArmCommandResult & arm, bool pair_accepted, bool pair_hard_fault) {
  std::copy_n(arm.joint_candidate.data(), 7, out.joint_candidate);
  std::copy_n(arm.joint_command.data(), 7, out.joint_command);
  put_pose(out.step_target, arm.step_target);
  put_pose(out.workspace_target, arm.workspace_target);
  put_pose(out.attempt_target, arm.attempt_target);
  out.arm_angle_command_rad = arm.arm_angle_command_rad;
  out.workspace_projection_m = arm.workspace_projection_m;
  out.position_lag_m = arm.position_lag_m;
  out.rotation_lag_rad = arm.rotation_lag_rad;
  out.backoff_scale = arm.backoff_scale;
  out.projected_axis = arm.projected_axis;
  out.safety_code = static_cast<int>(arm.safety.code);
  out.flags = std::uint32_t(arm.accepted) | (std::uint32_t(arm.ik_target_accepted) << 1)
      | (std::uint32_t(arm.analytic_braking) << 2) | (std::uint32_t(arm.workspace_projected) << 3)
      | (std::uint32_t(arm.rate_limited) << 4) | (std::uint32_t(arm.analytic_exact_stop_used) << 5)
      | (std::uint32_t(pair_accepted) << 6) | (std::uint32_t(pair_hard_fault) << 7);
}

ArmCommandResult target(const double * pose) {
  ArmCommandResult arm;
  arm.ee_target = Transform(Eigen::Map<const Pose>(pose));
  arm.raw_input_target = arm.ee_target;
  arm.raw_input_target_available = arm.target_available = true;
  arm.safety = {true, SafetyCode::OK, {}};
  return arm;
}
}  // namespace

extern "C" std::size_t s05_candidate_arm_output_size() noexcept { return sizeof(ArmOutput); }
extern "C" std::size_t s05_candidate_profile_size() noexcept { return sizeof(Profile); }
extern "C" int s05_candidate_abi_version() noexcept { return 2; }

extern "C" void * s05_candidate_open(const char * config, const Profile * profile) noexcept {
  try {
    std::string reason;
    auto sdk = create_marvin_kine_backend(config, 0, 1, reason);
    if (!sdk) return nullptr;
    auto marvin = std::make_shared<MarvinKinematicsAdapter>(config, sdk, 0, 1);
    if (!marvin->initialized()) return nullptr;
    MarvinIkConfig ik;
    ik.zsp_type = profile->zsp_type;
    ik.zsp_angle_deg = profile->zsp_angle_deg;
    std::copy_n(profile->dgr_deg, 3, ik.dgr_deg.begin());
    std::copy_n(profile->left_zsp_para, 6, ik.zsp_para.begin());
    marvin->set_ik_config(ArmSide::LEFT, ik);
    std::copy_n(profile->right_zsp_para, 6, ik.zsp_para.begin());
    marvin->set_ik_config(ArmSide::RIGHT, ik);
    return new Bridge{marvin, TeleopCore(core_config(*profile), marvin), {}, {}};
  } catch (...) {
    return nullptr;
  }
}

extern "C" void s05_candidate_close(void * handle) noexcept { delete static_cast<Bridge *>(handle); }

extern "C" int s05_candidate_home(void * handle, const double * joints) noexcept {
  try {
    auto & b = *static_cast<Bridge *>(handle);
    SessionReference reference;
    reference.left_q_start = Joints(joints);
    reference.right_q_start = Joints(joints + 7);
    const auto left = b.marvin->fk_nsp(ArmSide::LEFT, reference.left_q_start);
    const auto right = b.marvin->fk_nsp(ArmSide::RIGHT, reference.right_q_start);
    if (!left.success || !right.success) return 1;
    reference.left_robot_start_ee = left.transform;
    reference.right_robot_start_ee = right.transform;
    reference.left_nsp_direction = left.direction;
    reference.right_nsp_direction = right.direction;
    reference.session_id = "s05-offline-candidate";
    b.reference = std::move(reference);
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" int s05_candidate_fk(void * handle, int side, const double * joints, double * out) noexcept {
  try {
    const auto & b = *static_cast<Bridge *>(handle);
    const auto result = b.marvin->fk(side == 0 ? ArmSide::LEFT : ArmSide::RIGHT, Joints(joints));
    if (!result.success) return 1;
    put_pose(out, result.transform);
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" int s05_candidate_propose(
    void * handle, const double * goals, const double * measured, const double * prior,
    const double * angles, double dt, ArmOutput * outputs) noexcept {
  try {
    auto & b = *static_cast<Bridge *>(handle);
    Joints lp(prior), rp(prior + 7), lm(measured), rm(measured + 7);
    DualCommandResult request{false, target(goals), target(goals + 16), b.reference.session_id};
    auto solved = b.core.process_targets(b.reference, std::move(request), lm, rm, lp, rp, dt, angles[0], angles[1]);
    auto final = b.core.finalize_command(std::move(solved), lm, rm, lp, rp, dt);
    const bool hard_fault = !final.accepted
        && (!local_code(final.left.safety.code) || !local_code(final.right.safety.code));
    export_arm(outputs[0], final.left, final.accepted, hard_fault);
    export_arm(outputs[1], final.right, final.accepted, hard_fault);
    b.reasons[0] = std::move(final.left.safety.reason);
    b.reasons[1] = std::move(final.right.safety.reason);
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" const char * s05_candidate_reason(void * handle, int side) noexcept {
  return static_cast<const Bridge *>(handle)->reasons[side == 0 ? 0 : 1].c_str();
}
