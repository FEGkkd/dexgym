#!/usr/bin/env python3
"""Prepare the one-root MuJoCo START diagnostic with a 5-degree left J6 margin."""

import argparse
import difflib
from hashlib import sha256
from pathlib import Path

from prepare_shared_rigid_overlay import replace_once


BASELINE_CORE_SHA256 = "5751c39e1d95717fee33afd37e175213bbec3b22e484cc8679c1bf882f8ff55f"
GUARD_HEADER_SHA256 = "5995ef2830a216d3787bd86b3cc55b25726621e9da61ca5dd1b87c984654ce82"


MARGIN_HELPER = """JointVector dcr009_command_margins(ArmSide side, double original_margin)
{
  JointVector margins = JointVector::Constant(original_margin);
  constexpr double degree = 3.14159265358979323846 / 180.;
  constexpr Eigen::Index left_j6 = 5;
  if (side == ArmSide::LEFT && std::abs(original_margin - 8. * degree) < 1e-12 &&
      dcr009_diagnostic::segmented_phase() == dcr009_diagnostic::SegmentPhase::Preposition) {
    margins[left_j6] = 5. * degree;
    static std::atomic_bool recorded{false};
    if (!recorded.exchange(true)) {
      std::clog << "S05_DCR009_MARGIN_PROFILE side=left joint=Joint6_L margin_deg=5"
                << " remaining_axes_margin_deg=8 phase=PREPOSITION" << std::endl;
    }
  }
  return margins;
}

"""


def patch(package_copy, patch_file):
    core = package_copy / "src/teleop_core.cpp"
    header = package_copy / "include/tianji_teleop_controller/dcr009_teleop_guard.hpp"
    baseline = core.read_bytes()
    if (sha256(baseline).hexdigest() != BASELINE_CORE_SHA256 or
            sha256(header.read_bytes()).hexdigest() != GUARD_HEADER_SHA256):
        raise ValueError("The prior isolated segmented controller source does not match")
    original = baseline.decode("utf-8")
    text = replace_once(original, "void log_segmented_margin(",
                        MARGIN_HELPER + "void log_segmented_margin(", core)
    text = replace_once(text, "  const bool left = side == ArmSide::LEFT;\n  for (Eigen::Index axis",
                        "  const bool left = side == ArmSide::LEFT;\n"
                        "  const auto margins = dcr009_command_margins(side, margin);\n"
                        "  for (Eigen::Index axis", core)
    for old, new in (
        ("limits.lower_rad[axis] + margin", "limits.lower_rad[axis] + margins[axis]"),
        ("limits.upper_rad[axis] - margin", "limits.upper_rad[axis] - margins[axis]"),
        ('<< " margin_upper_rad=" << hi << " margin_rad=" << margin',
         '<< " margin_upper_rad=" << hi << " margin_rad=" << margins[axis]'),
        ("csv((limits.lower_rad.array() + margin).matrix())", "csv(limits.lower_rad + margins)"),
        ("csv((limits.upper_rad.array() - margin).matrix())", "csv(limits.upper_rad - margins)"),
    ):
        text = replace_once(text, old, new, core)
    text = replace_once(text,
                        "  if (!inside_limits(joints, limits.lower_rad.array() + margin,\n"
                        "                     limits.upper_rad.array() - margin)) {",
                        "  const auto margins = dcr009_command_margins(side, margin);\n"
                        "  if (!inside_limits(joints, limits.lower_rad + margins,\n"
                        "                     limits.upper_rad - margins)) {", core)
    text = replace_once(text,
                        "      for (int j=0;j<7;++j) {\n"
                        "        const double deficit=std::max({0.,limits.lower_rad[j]+config_.joint_limit_margin_rad-q[j],\n"
                        "            q[j]-limits.upper_rad[j]+config_.joint_limit_margin_rad});",
                        "      const auto margins = dcr009_command_margins(side, config_.joint_limit_margin_rad);\n"
                        "      for (int j=0;j<7;++j) {\n"
                        "        const double deficit=std::max({0.,limits.lower_rad[j]+margins[j]-q[j],\n"
                        "            q[j]-limits.upper_rad[j]+margins[j]});", core)
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), text.splitlines(keepends=True),
                                        fromfile="prior/teleop_core.cpp", tofile="experiment/teleop_core.cpp"))
    with patch_file.open("x", encoding="utf-8") as output:
        output.write(diff)
    core.write_bytes(text.encode("utf-8"))
    print(f"baseline_core_sha256={BASELINE_CORE_SHA256}")
    print(f"experiment_core_sha256={sha256(core.read_bytes()).hexdigest()}")
    print(f"unchanged_guard_header_sha256={GUARD_HEADER_SHA256}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isolated_package_copy", type=Path)
    parser.add_argument("--record-patch", type=Path, required=True)
    args = parser.parse_args()
    patch(args.isolated_package_copy, args.record_patch)


if __name__ == "__main__":
    main()
