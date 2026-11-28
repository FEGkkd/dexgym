#!/usr/bin/env python3
"""Seed the isolated MuJoCo driver at the certified named Home before any simulation step."""

import argparse
from pathlib import Path

from prepare_shared_rigid_overlay import replace_once


OLD = """  mj_forward(model_, data_);
  target_positions_.assign(kJointCount, 0.0);
"""
NEW = """  // Isolated DCR-009 diagnosis: cold reset before ownership or physics execution.
  const int diagnostic_home = mj_name2id(model_, mjOBJ_KEY, "home");
  if (diagnostic_home < 0) {
    error = "DCR009 diagnostic requires the certified Home keyframe";
    mj_deleteData(data_); data_ = nullptr; mj_deleteModel(model_); model_ = nullptr;
    return false;
  }
  mj_resetDataKeyframe(model_, data_, diagnostic_home);
  for (std::size_t index = 0; index < kArmJointCount; ++index) {
    const double from_keyframe = data_->qpos[joint_qpos_addresses_[index]];
    if (!std::isfinite(from_keyframe) || std::abs(from_keyframe-home_positions_[index]) > 1e-7) {
      error = "DCR009 diagnostic Home configuration differs from the model keyframe";
      mj_deleteData(data_); data_ = nullptr; mj_deleteModel(model_); model_ = nullptr;
      return false;
    }
  }
  mj_forward(model_, data_);
  target_positions_.assign(kJointCount, 0.0);
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isolated_driver_package", type=Path)
    args = parser.parse_args()
    source = args.isolated_driver_package / "src" / "mujoco_backend.cpp"
    text = source.read_text(encoding="utf-8")
    if "DCR009 diagnostic Home" in text:
        raise ValueError("Use a fresh isolated driver source copy")
    source.write_text(replace_once(text, OLD, NEW, source), encoding="utf-8")


if __name__ == "__main__":
    main()
