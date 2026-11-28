// Offline seam for the pinned RetargetNode callback. The Python build step
// extracts its mapping and callback bodies and supplies the diagnostic taps.
#include "manus_wuji_teleop/command_smoother.hpp"
#include "wuji_sdk_cpp/retarget_session.hpp"
#include "wuji_sdk_cpp/sdk_manager.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Keypoints = std::array<float, 63>;
using Positions = std::array<float, 20>;

extern "C" {
struct ProbeRawNode {
  int32_t node_id;
  const char *chain_type;
  const char *joint_type;
  double xyz[3];
};

struct ProbeOutput {
  int32_t stage;
  int32_t ids[21];
  float points[63];
  float raw[20];
  float filtered[20];
  char error[512];
};
}

namespace {
enum Stage { Rejected = 0, Mapped = 1, Retargeted = 2, Filtered = 3, Published = 4 };

template <typename ...Args>
void record_error(ProbeOutput &output, const char *format, Args ...args) {
  std::snprintf(output.error, sizeof(output.error), format, args...);
}

// A deterministic adapter for the fields read or written by the callback.
struct Stamp {
  int32_t sec;
  uint32_t nanosec;
};

struct ClockReading {
  int64_t ns;
  int64_t nanoseconds() const { return ns; }
  operator Stamp() const {
    return {static_cast<int32_t>(ns / 1000000000), static_cast<uint32_t>(ns % 1000000000)};
  }
};

struct CallbackClock {
  int64_t before_ns;
  int64_t after_ns;
  int reads;
  ClockReading now() { return {reads++ == 0 ? before_ns : after_ns}; }
};

struct CallbackNode {
  CallbackClock clock{};
  CallbackClock *get_clock() { return &clock; }
};
}

namespace manus_ros2_msgs::msg {
struct ManusRawNode {
  int32_t node_id;
  std::string chain_type;
  std::string joint_type;
  struct { struct { double x, y, z; } position; } pose;
};

struct ManusGlove {
  using ConstSharedPtr = std::shared_ptr<const ManusGlove>;
  struct { Stamp stamp; } header;
  std::string side;
  int32_t glove_id;
  std::vector<ManusRawNode> raw_nodes;
};
}

namespace manus_wuji_interfaces::msg {
struct HandCommand {
  struct { Stamp stamp; } header;
  std::string side;
  std::array<std::string, 20> joint_names;
  Positions position;
  uint64_t sequence;
};
}

namespace manus_wuji_teleop {
class RetargetNode final {
public:
  RetargetNode(int side, double rotation, int max_age, double alpha, uint32_t glove_id)
      : handedness_(side == 0 ? wuji_sdk_cpp::Handedness::Left : wuji_sdk_cpp::Handedness::Right),
        retarget_(std::make_unique<wuji_sdk_cpp::RetargetSession>(sdk_manager_, handedness_)),
        side_(side == 0 ? "left" : "right"), glove_id_(glove_id), rotation_z_deg_(rotation),
        max_input_age_ms_(max_age) {
    command_smoother_.configure(alpha, max_age / 1000.0);
  }

  void reset(bool recreate_session) {
    if (recreate_session) {
      retarget_ = std::make_unique<wuji_sdk_cpp::RetargetSession>(sdk_manager_, handedness_);
    } else {
      retarget_->reset();
    }
    command_smoother_.reset();
    keypoint_map_initialized_ = false;
    cached_keypoint_map_.clear();
    sequence_ = 0;
  }

  void run(const manus_ros2_msgs::msg::ManusGlove::ConstSharedPtr &msg, double receive_time,
           int64_t before_ns, int64_t after_ns, bool gate_open, ProbeOutput &output) {
    receive_time_sec_ = receive_time;
    node_.clock = {before_ns, after_ns, 0};
    output_ = &output;
    command_gate_ = [gate_open](const std::function<void()> &publish) {
      if (gate_open) publish();
      return gate_open;
    };
    callback(msg);
  }

private:
  void callback(const manus_ros2_msgs::msg::ManusGlove::ConstSharedPtr &msg);
  Keypoints keypoints(const manus_ros2_msgs::msg::ManusGlove &msg) const;
  static std::string normalize_side(std::string value);
  static std::string normalize_token(std::string value);
  static std::vector<int32_t>
  build_keypoint_map(const std::vector<manus_ros2_msgs::msg::ManusRawNode> &nodes);

  Keypoints tap_keypoints(const manus_ros2_msgs::msg::ManusGlove &msg) {
    const auto points = keypoints(msg);
    std::copy(points.begin(), points.end(), output_->points);
    std::copy(cached_keypoint_map_.begin(), cached_keypoint_map_.end(), output_->ids);
    output_->stage = Mapped;
    return points;
  }

  void tap_raw(const Positions &positions) {
    std::copy(positions.begin(), positions.end(), output_->raw);
    output_->stage = Retargeted;
  }

  void tap_filtered(const Positions &positions) {
    std::copy(positions.begin(), positions.end(), output_->filtered);
    output_->stage = Filtered;
  }

  void tap_publish(const manus_wuji_interfaces::msg::HandCommand &) {
    output_->stage = Published;
  }

  CallbackNode node_;
  wuji_sdk_cpp::SdkManager sdk_manager_;
  wuji_sdk_cpp::Handedness handedness_;
  std::unique_ptr<wuji_sdk_cpp::RetargetSession> retarget_;
  std::string side_;
  uint32_t glove_id_;
  double rotation_z_deg_;
  int max_input_age_ms_;
  CommandSmoother command_smoother_;
  mutable std::vector<int32_t> cached_keypoint_map_;
  mutable bool keypoint_map_initialized_{false};
  uint64_t sequence_{0};
  double receive_time_sec_{};
  ProbeOutput *output_{};
  std::function<bool(const std::function<void()> &)> command_gate_;
};

#define RCLCPP_WARN_THROTTLE(logger, clock, period, ...) record_error(*output_, __VA_ARGS__)
#include "parity_wuji_callback.inc"
#undef RCLCPP_WARN_THROTTLE
}

extern "C" {
void *s05_wuji_probe_create(int side, double rotation, int max_age, double alpha, uint32_t glove_id,
                           ProbeOutput *output) {
  try {
    return new manus_wuji_teleop::RetargetNode(side, rotation, max_age, alpha, glove_id);
  } catch (const std::exception &error) {
    record_error(*output, "%s", error.what());
    return nullptr;
  }
}

void s05_wuji_probe_destroy(void *handle) {
  delete static_cast<manus_wuji_teleop::RetargetNode *>(handle);
}

int s05_wuji_probe_reset(void *handle, int recreate_session, ProbeOutput *output) {
  try {
    static_cast<manus_wuji_teleop::RetargetNode *>(handle)->reset(recreate_session != 0);
    return 0;
  } catch (const std::exception &error) {
    record_error(*output, "%s", error.what());
    return -1;
  }
}

int s05_wuji_probe_step(void *handle, const ProbeRawNode *nodes, size_t count, const char *message_side,
                       int32_t glove_id, double receive_time_sec, int64_t header_ns, int64_t before_ns,
                       int64_t after_ns, int gate_open, ProbeOutput *output) {
  try {
    auto parsed = std::make_shared<manus_ros2_msgs::msg::ManusGlove>();
    parsed->side = message_side;
    parsed->glove_id = glove_id;
    parsed->header.stamp = ClockReading{header_ns};
    parsed->raw_nodes.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const auto &node = nodes[i];
      parsed->raw_nodes.push_back({node.node_id, node.chain_type, node.joint_type,
                                   {{node.xyz[0], node.xyz[1], node.xyz[2]}}});
    }
    static_cast<manus_wuji_teleop::RetargetNode *>(handle)->run(
        parsed, receive_time_sec, before_ns, after_ns, gate_open != 0, *output);
    return 0;
  } catch (const std::exception &error) {
    record_error(*output, "%s", error.what());
    return -1;
  }
}
}
