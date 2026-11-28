#pragma once

// DCR-009 diagnostic-only publication gate. The owning controller holds its
// lifecycle mutex during propose, native publish, and commit. This class owns
// no ROS publisher and cannot synthesize, replay, or issue a robot command.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace dcr009_diagnostic {

class NominalGuard {
public:
  using Clock = std::chrono::steady_clock;
  using Vector = std::array<double, 54>;
  static constexpr int64_t kFreshNs = 40'000'000;
  static constexpr uint32_t kNoProof = 100;
  static constexpr uint32_t kInvalidCandidate = 101;
  static constexpr uint32_t kRearm = 102;
  static constexpr uint32_t kTransport = 103;
  static constexpr uint32_t kTimeoutOrExpired = 104;
  static constexpr uint32_t kPublishException = 105;

  struct Proof {
    Vector candidate{};
    uint64_t seq{0};
    int64_t feedback_header{0};
    uint32_t reason{kNoProof};
    bool permitted{false};
  };

  NominalGuard()
  {
    const char * socket = std::getenv("S05_DCR009_GUARD_SOCKET");
    const char * domain = std::getenv("ROS_DOMAIN_ID");
    const char * local = std::getenv("ROS_LOCALHOST_ONLY");
    const char * discovery = std::getenv("ROS_AUTOMATIC_DISCOVERY_RANGE");
    if (!socket || socket[0] != '/' || std::strlen(socket) >= sizeof(sockaddr_un{}.sun_path) ||
        !domain || std::strlen(domain) != 3 || std::strcmp(domain, "206") < 0 || std::strcmp(domain, "213") > 0 ||
        !local || std::strcmp(local, "1") != 0 || !discovery || std::strcmp(discovery, "LOCALHOST") != 0 ||
        has_env("ROS_STATIC_PEERS") || has_env("ROS_DISCOVERY_SERVER") || has_env("CYCLONEDDS_URI") ||
        has_env("FASTRTPS_DEFAULT_PROFILES_FILE") || has_env("FASTDDS_DEFAULT_PROFILES_FILE"))
      throw std::invalid_argument("DCR-009 guard requires an owned UNIX socket and localhost simulation domain 206-213");
    socket_path_ = socket;
    if (const char * configured = std::getenv("S05_DCR009_GUARD_TIMEOUT_MS")) {
      std::size_t parsed = 0;
      const int value = std::stoi(configured, &parsed);
      if (parsed != std::strlen(configured) || value < 1 || value > 40)
        throw std::invalid_argument("DCR-009 guard timeout must be in 1-40 ms");
      timeout_ms_ = value;
    }
  }

  template<typename Names, typename Positions>
  void observe(std::size_t group, int64_t header, int64_t observed_ros_now,
    const Names & names, const Positions & positions, bool valid)
  {
    if (group >= 4 || header <= 0 || observed_ros_now < header || observed_ros_now - header > kFreshNs) return;
    auto & entry = feedback_[header];
    static constexpr std::array<std::size_t, 4> offsets{0, 7, 14, 34};
    static constexpr std::array<std::size_t, 4> sizes{7, 7, 20, 20};
    valid = valid && write_group(entry.q, offsets[group], sizes[group], names, positions);
    const unsigned bit = 1u << group;
    if ((entry.groups & bit) != 0u || !valid) entry.valid = false;
    entry.groups |= bit;
    if (entry.groups == 15u && (!feedback_complete_ || header >= feedback_complete_->header)) {
      feedback_complete_ = Measured{header, Clock::now(), entry.q, entry.valid};
    }
    while (feedback_.size() > 16u || (!feedback_.empty() && header - feedback_.begin()->first > kFreshNs))
      feedback_.erase(feedback_.begin());
  }

  template<typename Names, typename Positions>
  Proof propose(uint64_t generation, const std::string & session, uint32_t group,
    int64_t now_ros, const Names & names, const Positions & positions)
  {
    Proof result;
    if (group > 2u || session.rfind("s05-dcr009-", 0) != 0 || generation == 0 ||
        (initialized_ && (generation != generation_ || session != session_))) {
      result.reason = kRearm;
      return result;
    }
    if (now_ros < last_proposal_ros_) clock_failed_ = true;
    last_proposal_ros_ = now_ros;
    if (clock_failed_ || !fresh(now_ros)) return result;
    const auto & measured = *feedback_complete_;
    result.feedback_header = measured.header;
    result.candidate = initialized_ ? issued_ : measured.q;
    static constexpr std::array<std::size_t, 3> offsets{0, 14, 34};
    static constexpr std::array<std::size_t, 3> sizes{14, 20, 20};
    if (!write_group(result.candidate, offsets[group], sizes[group], names, positions)) {
      result.reason = kInvalidCandidate;
      return result;
    }
    if (!initialized_) {
      initialized_ = true;
      generation_ = generation;
      session_ = session;
      issued_ = measured.q;
    }
    result.seq = ++sequence_;
    result.reason = send(group, result.seq, measured.q, result.candidate);
    result.permitted = result.reason == 0u;
    return result;
  }

  bool still_valid(const Proof & proof, int64_t now_ros, int64_t valid_until_ros) const
  {
    return proof.permitted && proof.seq == sequence_ && now_ros >= last_proposal_ros_ &&
      now_ros < valid_until_ros && fresh(now_ros) && feedback_complete_->header == proof.feedback_header;
  }

  void commit(const Proof & proof) {
    if (proof.permitted && proof.seq != 0 && proof.seq == sequence_) issued_ = proof.candidate;
  }

private:
  struct Sample {
    Vector q{};
    unsigned groups{0};
    bool valid{true};
  };

  struct Measured {
    int64_t header;
    Clock::time_point observed;
    Vector q;
    bool valid;
  };

  struct Socket {
    int fd{-1};
    ~Socket() { if (fd >= 0) ::close(fd); }
  };

  static bool has_env(const char * key) {
    const char * value = std::getenv(key);
    return value && value[0] != '\0';
  }

  static const std::array<std::string, 54> & wire_joint_names()
  {
    static const auto names = [] {
      std::array<std::string, 54> result;
      for (int j = 0; j < 7; ++j) {
        result[j] = "left_j" + std::to_string(j + 1);
        result[7 + j] = "right_j" + std::to_string(j + 1);
      }
      const std::array<const char *, 5> digits{"THJ", "FFJ", "MFJ", "RFJ", "LFJ"};
      for (int side = 0; side < 2; ++side)
        for (int finger = 0; finger < 5; ++finger)
          for (int joint = 0; joint < 4; ++joint)
            result[14 + 20 * side + 4 * finger + joint] =
              std::string(side == 0 ? "l_" : "r_") + digits[finger] + std::to_string(joint);
      return result;
    }();
    return names;
  }

  template<typename Names, typename Positions>
  static bool write_group(Vector & q, std::size_t offset, std::size_t count,
    const Names & names, const Positions & positions)
  {
    if (names.size() != count || positions.size() != count) return false;
    const auto & expected = wire_joint_names();
    // Validate explicitly named axes, even if a publisher reorders an arm group.
    // The Wuji driver requires hand payloads in manifest order, so refuse reordering.
    for (std::size_t j = 0; j < count; ++j) {
      if (names[j] != expected[offset + j] || !std::isfinite(positions[j])) return false;
      q[offset + j] = positions[j];
    }
    return true;
  }

  bool fresh(int64_t now_ros) const {
    return feedback_complete_ && feedback_complete_->valid && now_ros >= feedback_complete_->header &&
      now_ros - feedback_complete_->header <= kFreshNs &&
      Clock::now() - feedback_complete_->observed <= std::chrono::nanoseconds(kFreshNs);
  }

  template<typename Integer>
  static void encode(unsigned char * to, Integer value) {
    for (std::size_t i = 0; i < sizeof(Integer); ++i) to[i] = (value >> (8u * i)) & 0xffu;
  }

  template<typename Integer>
  static Integer decode(const unsigned char * data) {
    Integer value = 0;
    for (std::size_t i = 0; i < sizeof(Integer); ++i) value |= static_cast<Integer>(data[i]) << (8u * i);
    return value;
  }

  static bool ready(int fd, short conditions, Clock::time_point deadline) {
    for (;;) {
      const auto time = deadline - Clock::now();
      if (time <= Clock::duration::zero()) return false;
      pollfd event{fd, conditions, 0};
      const int status = ::poll(&event, 1, static_cast<int>(std::chrono::ceil<std::chrono::milliseconds>(time).count()));
      if (status < 0 && errno == EINTR) continue;
      return status > 0 && (event.revents & conditions) != 0;
    }
  }

  template<bool Transmit, std::size_t Count>
  static bool transfer(int fd, std::array<unsigned char, Count> & data, Clock::time_point deadline) {
    for (std::size_t i = 0; i < Count;) {
      if (!ready(fd, Transmit ? POLLOUT : POLLIN, deadline)) return false;
      const ssize_t n = Transmit ? ::send(fd, data.data() + i, Count - i, MSG_NOSIGNAL) :
        ::recv(fd, data.data() + i, Count - i, 0);
      if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
      if (n <= 0) return false;
      i += static_cast<std::size_t>(n);
    }
    return Clock::now() < deadline;
  }

  uint32_t send(uint32_t group, uint64_t seq, const Vector & measured, const Vector & candidate) const
  {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "Use IEEE-754 binary64");
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms_);
    struct stat path_metadata{};
    if (::lstat(socket_path_.c_str(), &path_metadata) != 0 || !S_ISSOCK(path_metadata.st_mode) ||
        path_metadata.st_uid != ::geteuid()) return kTransport;
    Socket socket{::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0)};
    if (socket.fd < 0) return kTransport;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
    if (::connect(socket.fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0) {
      if ((errno != EINPROGRESS && errno != EAGAIN) || !ready(socket.fd, POLLOUT, deadline)) return kTransport;
    }
    int error = 0;
    socklen_t error_size = sizeof(error);
    if (::getsockopt(socket.fd, SOL_SOCKET, SO_ERROR, &error, &error_size) != 0 || error != 0) return kTransport;
    struct ucred peer{};
    socklen_t peer_size = sizeof(peer);
    if (::getsockopt(socket.fd, SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) != 0 || peer.uid != ::geteuid()) return kTransport;
    constexpr char magic[] = "S05SWP01";
    std::array<unsigned char, 20 + 162 * sizeof(double)> request{};
    std::array<unsigned char, 24> reply{};
    std::memcpy(request.data(), magic, 8);
    encode(request.data() + 8, seq);
    encode(request.data() + 16, group);
    std::size_t offset = 20;
    for (const auto * values : {&issued_, &measured, &candidate}) {
      for (const auto value : *values) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        encode(request.data() + offset, bits);
        offset += sizeof(bits);
      }
    }
    if (!transfer<true>(socket.fd, request, deadline) || !transfer<false>(socket.fd, reply, deadline)) return kTimeoutOrExpired;
    if (std::memcmp(reply.data(), magic, 8) != 0 || decode<uint64_t>(reply.data() + 8) != seq) return kTransport;
    const auto allowed = decode<uint32_t>(reply.data() + 16);
    const auto reason = decode<uint32_t>(reply.data() + 20);
    if (allowed == 1u && reason == 0u) return 0;
    return allowed == 0u && reason > 0u && reason <= 10u ? reason : kTransport;
  }

  std::string socket_path_;
  std::string session_;
  int timeout_ms_{30};
  Vector issued_{};
  std::map<int64_t, Sample> feedback_;
  std::optional<Measured> feedback_complete_;
  uint64_t sequence_{0}, generation_{0};
  int64_t last_proposal_ros_{0};
  bool initialized_{false}, clock_failed_{false};
};

}  // namespace dcr009_diagnostic
