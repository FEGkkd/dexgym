// Standalone diagnostic seam for original MuJoCo driver's hand acceptance.
#include "whole_body_mujoco_driver/model_joints.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace manus_wuji_interfaces::msg {
struct HandCommand {
  struct Stamp { int64_t ns; };
  struct { Stamp stamp; } header;
  std::string side;
  uint64_t sequence;
  std::array<std::string, 20> joint_names;
  std::array<double, 20> position;
};
}

namespace whole_body_mujoco_driver {
namespace {
constexpr std::size_t kArmJointCount = 14;
int64_t valid_until_ns(manus_wuji_interfaces::msg::HandCommand::Stamp stamp) { return stamp.ns; }
struct CommandResult { bool ok; std::string message; };
}

struct MujocoBackend {
  struct Data { std::array<double, 54> qpos{}; } data;
  struct Model { std::array<double, 108> actuator_ctrlrange{}; } model;
  struct Config {
    double hand_command_timeout_sec;
    double hand_handoff_hold_sec;
    double hand_handoff_ramp_sec;
  } config_{};
  Data *data_{&data};
  Model *model_{&model};
  std::mutex mutex_;
  bool fault_{false};
  std::array<int, 54> actuator_indices_{};
  std::array<int, 54> joint_qpos_addresses_{};
  std::array<int, 2> hand_home_mode_{};
  std::array<uint64_t, 2> hand_sequences_{};
  std::array<int64_t, 2> hand_command_ns_{};
  std::array<int64_t, 2> hand_enable_ns_{};
  std::array<bool, 2> hand_enabled_{};
  std::array<bool, 2> hand_fault_{};
  std::array<std::array<double, 20>, 2> hand_seed_{};
  std::array<double, 54> target_positions_{};
  std::array<double, 14> arm_filtered_targets_{};
  std::vector<double> arm_target_history_;

  CommandResult accept_hand(std::size_t side, const manus_wuji_interfaces::msg::HandCommand & c,
                            int64_t ros_ns, int64_t steady_ns);
  void hold_range_locked(std::size_t begin, std::size_t end);
  void watchdog(std::size_t side, int64_t steady_ns) {
    #include "native_hand_watchdog.inc"
  }
};

#include "native_hand_accept.inc"
#include "native_hand_hold.inc"
}

using whole_body_mujoco_driver::MujocoBackend;
using whole_body_mujoco_driver::kModelJointNames;
using manus_wuji_interfaces::msg::HandCommand;

extern "C" {
const char *s05_driver_alias(int axis) {
  return axis >= 0 && axis < 54 ? kModelJointNames[axis].c_str() : nullptr;
}
MujocoBackend *s05_driver_create(const double *q, const double *ctrlrange, int64_t now,
                                double timeout, double hold, double ramp) {
  auto *b = new MujocoBackend;
  b->config_ = {timeout, hold, ramp};
  std::copy_n(q, 54, b->data.qpos.begin());
  std::copy_n(q, 54, b->target_positions_.begin());
  std::copy_n(ctrlrange, 108, b->model.actuator_ctrlrange.begin());
  std::iota(b->actuator_indices_.begin(), b->actuator_indices_.end(), 0);
  std::iota(b->joint_qpos_addresses_.begin(), b->joint_qpos_addresses_.end(), 0);
  for (int side = 0; side < 2; ++side) {
    b->hand_enabled_[side] = true;
    b->hand_enable_ns_[side] = b->hand_command_ns_[side] = now;
    std::copy_n(q + 14 + 20 * side, 20, b->hand_seed_[side].begin());
  }
  return b;
}
void s05_driver_destroy(MujocoBackend *b) { delete b; }
int s05_driver_receive(MujocoBackend *b, int side, int64_t stamp, int64_t receive, uint64_t sequence,
                       const double *q) {
  if (side < 0 || side > 1) return 0;
  HandCommand msg;
  msg.header.stamp.ns = stamp;
  msg.side = side == 0 ? "left" : "right";
  msg.sequence = sequence;
  std::copy_n(kModelJointNames.begin() + 14 + 20 * side, 20, msg.joint_names.begin());
  std::copy_n(q, 20, msg.position.begin());
  return b->accept_hand(side, msg, receive, receive).ok;
}
void s05_driver_snapshot(MujocoBackend *b, int64_t now, const double *measured,
                         double *q, int *fault) {
  std::copy_n(measured, 54, b->data.qpos.begin());
  for (int side = 0; side < 2; ++side) {
    b->watchdog(side, now);
    fault[side] = b->hand_fault_[side];
  }
  std::copy_n(b->target_positions_.begin(), 54, q);
}
}
