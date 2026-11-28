if(hand_enabled_[side] && hand_home_mode_[side]==0 && steady_ns-hand_command_ns_[side]>static_cast<std::int64_t>(config_.hand_command_timeout_sec*1e9)) {
      hold_range_locked(14+20*side,34+20*side);hand_enabled_[side]=false;hand_fault_[side]=true;
}
