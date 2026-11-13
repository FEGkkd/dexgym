// Sensor-only offline reference replay. Links the teleop math, never ROS or a driver.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "tianji_teleop_controller/bimanual_registration.hpp"
#include "tianji_teleop_controller/marvin_kine_sdk.hpp"
#include "tianji_teleop_controller/teleop_core.hpp"
#include "tianji_teleop_controller/tracker_position_filter.hpp"

using namespace tianji_teleop_controller;

namespace {
constexpr std::array sides{ArmSide::LEFT, ArmSide::RIGHT};
constexpr double rad = std::numbers::pi / 180.0;

struct Options {
  CoreConfig core;
  BimanualRegistration bimanual;
  std::array<JointVector, 2> home;
  std::array<double, 2> cutoff;
  double filter_reset_s;
  bool adaptive_filter;
  int ticks_per_frame;
};

template<typename Vector>
void read_vector(std::istream & in, Vector & vector) {
  for (double & value : vector) in >> value;
}

void read_matrix(std::istream & in, Eigen::Matrix3d & matrix) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) in >> matrix(row, col);
  }
}

Options read_options(std::istream & in) {
  Options out;
  auto & cfg = out.core;
  auto & reg = out.bimanual;
  read_vector(in, out.home[0]);
  read_vector(in, out.home[1]);
  read_matrix(in, cfg.left_world_to_base);
  read_matrix(in, cfg.right_world_to_base);
  read_matrix(in, reg.world_from_left);
  read_vector(in, reg.left_origin);
  read_matrix(in, reg.world_from_right);
  read_vector(in, reg.right_origin);
  read_vector(in, cfg.translation_scale);
  in >> cfg.rotation_scale >> cfg.absolute_orientation;
  read_matrix(in, cfg.left_hand_to_tool);
  read_matrix(in, cfg.right_hand_to_tool);
  for (auto * limit : {&cfg.left_limits, &cfg.right_limits}) {
    read_vector(in, limit->lower_rad);
    read_vector(in, limit->upper_rad);
    limit->solution_lower_rad = limit->lower_rad;
    limit->solution_upper_rad = limit->upper_rad;
  }
  in >> cfg.joint_limit_margin_rad >> cfg.fk_position_tolerance_m >> cfg.fk_orientation_tolerance_rad
     >> cfg.max_raw_joint_step_rad >> cfg.max_command_rate_rad_sec >> cfg.max_command_dt_sec
     >> cfg.max_translation_rate_m_sec >> cfg.max_rotation_rate_rad_sec
     >> cfg.max_tracking_error_rad >> cfg.hard_tracking_error_rad >> cfg.tracker_timeout_sec
     >> cfg.analytic_max_step_dt_sec >> cfg.analytic_source_nullspace_enabled
     >> cfg.nullspace_activation_rad >> cfg.nullspace_rate_rad_sec >> cfg.nullspace_probe_rad
     >> cfg.arm_angle_limit_rad >> cfg.arm_angle_centering_weight
     >> cfg.axis_projection_enabled >> cfg.axis_projection_min_retained_ratio
     >> cfg.workspace_boundary_enabled >> cfg.workspace_boundary_reserve_m >> cfg.workspace_boundary_transition_m
     >> out.cutoff[0] >> out.cutoff[1] >> out.filter_reset_s >> out.adaptive_filter
     >> out.ticks_per_frame >> reg.enabled >> reg.tolerance;
  int backoff_count = 0;
  in >> backoff_count;
  if (!in || backoff_count < 1 || backoff_count > 8 || out.ticks_per_frame < 1) {
    throw std::runtime_error("Malformed offline IK configuration");
  }
  cfg.backoff_scales.resize(backoff_count);
  read_vector(in, cfg.backoff_scales);
  if (!in) throw std::runtime_error("Malformed backoff scales");
  cfg.independent_arm_execution = true;
  cfg.manimux_analytic_ik_enabled = true;
  cfg.analytic_manimux_position_output = true;
  cfg.analytic_uniform_joint_scaling = true;
  cfg.analytic_joint_rate_rad_sec = cfg.max_command_rate_rad_sec;
  reg.left_mapping = cfg.left_world_to_base;
  reg.right_mapping = cfg.right_world_to_base;
  reg.scale = cfg.translation_scale;
  reg.validate();
  return out;
}

struct Frame {
  double time;
  std::array<TrackerSample, 2> wrist;
};

Frame read_frame(std::istream & in) {
  Frame frame;
  in >> frame.time;
  for (auto & sample : frame.wrist) {
    Eigen::Vector4d q;
    in >> sample.valid;
    auto translation = sample.pose_world.translation();
    read_vector(in, translation);
    read_vector(in, q);
    if (!in || !q.allFinite() || q.norm() < 0.5 || !sample.pose_world.translation().allFinite()) {
      throw std::runtime_error("Malformed offline wrist sample");
    }
    sample.pose_world.linear() = Eigen::Quaterniond(q).normalized().toRotationMatrix();
    sample.stamp_sec = sample.arrival_sec = frame.time;
    sample.quality = sample.valid ? 1.0 : 0.0;
  }
  return frame;
}

std::optional<JointVector> start_ik(const KinematicsBackend & kin, ArmSide side,
                                   const Transform & target, const JointVector & home,
                                   const NspResult & home_nsp, const ArmLimits & limit,
                                   const CoreConfig & cfg) {
  std::optional<JointVector> best;
  double cost = std::numeric_limits<double>::infinity();
  for (int angle : {0, -10, 10, -20, 20, -30, 30, -40, 40}) {
    const auto solved = kin.ik_nsp_angle(side, target, home, home_nsp.direction, angle * rad);
    if (!solved.success) continue;
    auto candidates = solved.candidates;
    candidates.push_back(solved.joints);
    for (const auto & q : candidates) {
      if (!q.allFinite() ||
          (q.array() < limit.lower_rad.array() + cfg.joint_limit_margin_rad).any() ||
          (q.array() > limit.upper_rad.array() - cfg.joint_limit_margin_rad).any() ||
          !kin.validate_joints(side, q).success) continue;
      const double distance = (q - home).squaredNorm();
      if (distance >= cost) continue;
      const auto fk = kin.fk(side, q);
      if (!fk.success || (fk.transform.translation() - target.translation()).norm() > cfg.fk_position_tolerance_m ||
          rotation_error_rad(fk.transform.linear(), target.linear()) > cfg.fk_orientation_tolerance_rad) continue;
      cost = distance;
      best = q;
    }
  }
  return best;
}

struct Start {
  SessionReference reference;
  bool paired;
  double relative_error_m;
};

Start make_start(const Options & opt, const KinematicsBackend & kin, const Frame & first) {
  const auto & reg = opt.bimanual;
  auto joints = opt.home;
  const auto left = kin.fk_nsp(sides[0], joints[0]);
  const auto right = kin.fk_nsp(sides[1], joints[1]);
  if (!left.success || !right.success) throw std::runtime_error("Native Home FK_NSP failed");
  bool paired = false;
  const Eigen::Vector3d human_left = first.wrist[0].pose_world.translation();
  const Eigen::Vector3d human_right = first.wrist[1].pose_world.translation();
  if (reg.enabled) {
    const Eigen::Vector3d home_mid = (reg.world_from_left * left.transform.translation() + reg.left_origin
        + reg.world_from_right * right.transform.translation() + reg.right_origin) / 2.0;
    const Eigen::Vector3d half = reg.world_from_left * reg.left_mapping *
        reg.scale.cwiseProduct(human_right - human_left) / 2.0;
    Transform left_goal = left.transform;
    Transform right_goal = right.transform;
    left_goal.translation() = reg.world_from_left.transpose() * (home_mid - half - reg.left_origin);
    right_goal.translation() = reg.world_from_right.transpose() * (home_mid + half - reg.right_origin);
    const auto lq = start_ik(kin, sides[0], left_goal, joints[0], left, opt.core.left_limits, opt.core);
    const auto rq = start_ik(kin, sides[1], right_goal, joints[1], right, opt.core.right_limits, opt.core);
    if (lq && rq) {
      joints = {*lq, *rq};
      paired = true;
    }
  }
  SessionReference reference;
  std::string reason;
  if (!capture_session_reference(first.wrist[0], first.wrist[1], joints[0], joints[1], kin,
                                 "wholebody_s0_offline", reference, &reason)) {
    throw std::runtime_error("Offline START capture failed: " + reason);
  }
  const double error = reg.error(human_left, human_right, reference.left_robot_start_ee.translation(),
                                 reference.right_robot_start_ee.translation()).norm();
  return {reference, paired, error};
}

TrackerSample subframe(const TrackerSample & previous, const TrackerSample & next, double fraction,
                       double now) {
  if (fraction == 1.0) return next;
  TrackerSample result = next;
  result.valid = previous.valid && next.valid;
  result.stamp_sec = result.arrival_sec = now;
  result.pose_world.translation() = (1 - fraction) * previous.pose_world.translation()
      + fraction * next.pose_world.translation();
  result.pose_world.linear() = Eigen::Quaterniond(previous.pose_world.linear())
      .slerp(fraction, Eigen::Quaterniond(next.pose_world.linear())).toRotationMatrix();
  return result;
}

void write_pose(std::ostream & out, const Transform & pose, bool valid) {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  const Eigen::Quaterniond q(pose.linear());
  for (double value : pose.translation()) out << ' ' << (valid ? value : nan);
  for (double value : q.coeffs()) out << ' ' << (valid ? value : nan);
}

void write_arm(std::ostream & out, const KinematicsBackend & kin, ArmSide side,
               const ArmCommandResult & raw, const ArmCommandResult & result, const JointVector & q) {
  const auto fk = kin.fk(side, q);
  const bool compare = raw.target_available && fk.success;
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  out << ' ' << result.accepted << ' ' << static_cast<int>(result.safety.code) << ' ' << result.rate_limited
      << ' ' << (compare ? (fk.transform.translation() - raw.ee_target.translation()).norm() : nan)
      << ' ' << (compare ? rotation_error_rad(fk.transform.linear(), raw.ee_target.linear()) : nan);
  for (double value : q) out << ' ' << value;
  write_pose(out, raw.ee_target, raw.target_available);
  write_pose(out, fk.transform, fk.success);
}
}  // namespace

int main(int argc, char ** argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument("usage: wholebody_s0_ik CCS_CONFIG SETTINGS WRISTS OUTPUT START_INFO");
    }
    std::ifstream options_file(argv[2]), wrists(argv[3]);
    std::ofstream output(argv[4]), start_file(argv[5]);
    if (!options_file || !wrists || !output || !start_file) throw std::runtime_error("Cannot open offline replay files");
    std::string reason;
    auto sdk = create_marvin_kine_backend(argv[1], 0, 1, reason);
    if (!sdk) throw std::runtime_error("Cannot initialize libKine: " + reason);
    auto kin = std::make_shared<MarvinKinematicsAdapter>(argv[1], sdk, 0, 1);
    const auto opt = read_options(options_file);
    const TeleopCore core(opt.core, kin);
    std::vector<Frame> frames;
    while (wrists >> std::ws && wrists.peek() != EOF) frames.push_back(read_frame(wrists));
    const auto start = std::find_if(frames.begin(), frames.end(), [](const Frame & frame) {
      return frame.wrist[0].valid && frame.wrist[1].valid;
    });
    if (start == frames.end()) throw std::runtime_error("No synchronized wrist pair for START");
    const auto anchor = make_start(opt, *kin, *start);
    start_file << std::setprecision(17) << (start - frames.begin()) << ' ' << anchor.paired
               << ' ' << anchor.relative_error_m << '\n';
    auto reference = anchor.reference;
    std::array q{reference.left_q_start, reference.right_q_start};
    std::array<double, 2> angle{0, 0};
    std::array<TrackerPositionFilter, 2> filters{
      TrackerPositionFilter(opt.cutoff[0], opt.filter_reset_s, opt.adaptive_filter),
      TrackerPositionFilter(opt.cutoff[1], opt.filter_reset_s, opt.adaptive_filter),
    };
    output << std::setprecision(17);
    for (auto it = frames.begin(); it != frames.end(); ++it) {
      DualCommandResult result;
      DualCommandResult raw;
      if (it >= start) {
        const auto & previous = it == start ? *it : *(it - 1);
        const double dt = it == start ? 0.02 / opt.ticks_per_frame : (it->time - previous.time) / opt.ticks_per_frame;
        if (dt <= 0) throw std::runtime_error("Nonpositive offline control interval");
        const int first_tick = it == start ? opt.ticks_per_frame : 1;
        for (int tick = first_tick; tick <= opt.ticks_per_frame; ++tick) {
          const double now = it->time - (opt.ticks_per_frame - tick) * dt;
          std::array<TrackerSample, 2> pose;
          for (int side = 0; side < 2; ++side) {
            pose[side] = subframe(previous.wrist[side], it->wrist[side],
                                 static_cast<double>(tick) / opt.ticks_per_frame, now);
            if (pose[side].valid) {
              pose[side].pose_world.translation() = filters[side].update(pose[side].pose_world.translation(), now);
            }
          }
          auto target = core.map_targets(reference, pose[0], pose[1], now);
          result = core.process_targets(reference, std::move(target), q[0], q[1], q[0], q[1], dt, angle[0], angle[1]);
          result = core.finalize_command(std::move(result), q[0], q[1], q[0], q[1], dt,
                                         JointVector::Zero(), JointVector::Zero(),
                                         JointVector::Zero(), JointVector::Zero(), nullptr, nullptr);
          for (const auto [side, arm] : {std::pair{0, &result.left}, std::pair{1, &result.right}}) {
            if (arm->accepted) {
              q[side] = arm->joint_command;
              angle[side] = arm->arm_angle_command_rad;
            }
          }
        }
        raw = core.map_targets(reference, it->wrist[0], it->wrist[1], it->time);
      }
      output << it->time;
      write_arm(output, *kin, sides[0], raw.left, result.left, q[0]);
      write_arm(output, *kin, sides[1], raw.right, result.right, q[1]);
      output << '\n';
    }
    if (!output || !start_file) throw std::runtime_error("Offline replay output write failed");
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
