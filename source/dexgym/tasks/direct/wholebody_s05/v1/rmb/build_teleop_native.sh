#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  printf 'usage: bash build_teleop_native.sh PINNED_TELEOP_REPO NEW_BUILD_DIRECTORY\n' >&2
  exit 2
fi

teleop=$(realpath "$1")
build=$(realpath -m "$2")
pin=6d0b0b5b2601618f1b0052d93ef2403eea9b064a
if [[ $(git -C "$teleop" rev-parse HEAD) != "$pin" ]]; then
  printf 'Expected pinned Teleop commit %s\n' "$pin" >&2
  exit 2
fi
git -C "$teleop" diff --exit-code "$pin" -- src/whole_body_teleop
package="$teleop/src/whole_body_teleop"
vendor="$package/vendor/marvin/lib/libKine.so"
if [[ $(sha256sum "$vendor" | cut -d' ' -f1) != f2aec6ebc1f41302fe0126284076211c044d6c7e1c2276dc919767254e0c597f ]]; then
  printf 'Production Marvin library digest differs from the audited one\n' >&2
  exit 2
fi

mkdir "$build"
here=$(dirname "$(realpath "$0")")
cat >"$build/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(s05_phase1_candidate LANGUAGES CXX)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_subdirectory("${TELEOP_PACKAGE}/vendor/ruckig" ruckig)
set(core "${TELEOP_PACKAGE}/src")
add_library(s05_teleop SHARED
  "${S05_BRIDGE}"
  "${core}/cartesian_servo.cpp" "${core}/joint_trajectory.cpp"
  "${core}/direct_joint_safety.cpp" "${core}/kinematics_adapter.cpp"
  "${core}/marvin_kine_sdk.cpp" "${core}/teleop_core.cpp"
  "${core}/feasible_path.cpp" "${core}/recovery_policy.cpp")
target_include_directories(s05_teleop PRIVATE "${TELEOP_PACKAGE}/include" "${TELEOP_PACKAGE}/vendor/marvin/include")
target_compile_definitions(s05_teleop PRIVATE TIANJI_HAS_MARVIN_KINE=1)
target_compile_features(s05_teleop PRIVATE cxx_std_20)
target_link_options(s05_teleop PRIVATE -Wl,-z,defs)
target_link_libraries(s05_teleop PRIVATE Eigen3::Eigen teleop_ruckig "${MARVIN_LIB}")
CMAKE

cmake -S "$build" -B "$build/obj" -DTELEOP_PACKAGE="$package" -DS05_BRIDGE="$here/teleop_bridge.cpp" \
  -DMARVIN_LIB="$vendor" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG'
cmake --build "$build/obj" --parallel 2
sha256sum "$here/teleop_bridge.cpp" "$vendor" "$package/config/real.yaml" "$package/config/sim.yaml" \
  "$package/src/teleop_core.cpp" "$package/src/marvin_kine_sdk.cpp" "$build/obj/libs05_teleop.so" \
  | tee "$build/source_and_library.sha256"
