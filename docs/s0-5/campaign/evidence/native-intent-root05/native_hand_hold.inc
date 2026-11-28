void MujocoBackend::hold_range_locked(std::size_t begin, std::size_t end) {
  for (auto i=begin;i<end;++i) {
    target_positions_[i]=data_->qpos[joint_qpos_addresses_[i]];
    if(i<kArmJointCount)arm_filtered_targets_[i]=target_positions_[i];
  }
  if(begin==0)arm_target_history_.clear();
}

