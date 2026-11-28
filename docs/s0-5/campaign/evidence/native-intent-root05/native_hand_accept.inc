CommandResult MujocoBackend::accept_hand(std::size_t side,const manus_wuji_interfaces::msg::HandCommand & c,
    std::int64_t ros_ns,std::int64_t steady_ns) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(!data_ || fault_ || side>1)return {false,"hand unavailable"};
  if(c.side!=(side==0 ? "left" : "right"))return {false,"hand side mismatch"};
  const auto stamp=valid_until_ns(c.header.stamp);
  if(stamp<=0 || stamp>ros_ns || ros_ns-stamp>static_cast<std::int64_t>(config_.hand_command_timeout_sec*1e9))return {false,"stale hand command"};
  if(c.sequence!=0 && c.sequence<=hand_sequences_[side])return {false,"hand sequence is not newer"};
  const auto offset=14+20*side;
  std::array<double,20> target{};
  for(std::size_t i=0;i<20;++i) {
    if(c.joint_names[i]!=kModelJointNames[offset+i] || !std::isfinite(c.position[i]))return {false,"invalid hand joint"};
    const int a=actuator_indices_[offset+i];
    target[i]=std::clamp(c.position[i],model_->actuator_ctrlrange[2*a],model_->actuator_ctrlrange[2*a+1]);
  }
  if(hand_home_mode_[side]!=0)return {false,"simulation hand Home owns the command target"};
  if(hand_enabled_[side]) {
    const double elapsed=std::max(0.0,static_cast<double>(steady_ns-hand_enable_ns_[side])*1e-9);
    const double u=config_.hand_handoff_ramp_sec>0.0
      ? std::clamp((elapsed-config_.hand_handoff_hold_sec)/config_.hand_handoff_ramp_sec,0.0,1.0)
      : (elapsed<config_.hand_handoff_hold_sec ? 0.0 : 1.0);
    const double weight=u*u*(3.0-2.0*u);
    for(std::size_t i=0;i<20;++i)
      target_positions_[offset+i]=(1.0-weight)*hand_seed_[side][i]+weight*target[i];
  }
  hand_sequences_[side]=c.sequence;hand_command_ns_[side]=steady_ns;
  return {true,""};
}

