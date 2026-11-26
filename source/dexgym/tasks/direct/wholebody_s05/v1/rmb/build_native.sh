#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  printf 'usage: bash build_native.sh TELEOP_ROOT OUTPUT_DIRECTORY\n' >&2
  exit 2
fi
teleop="$(realpath "$1")"
build="$2"
package="$teleop/src/whole_body_teleop"
# Match whole_body_teleop's CMake dependency. The older third_party library
# does not export the official logging switch referenced by this adapter.
marvin="$package/vendor/marvin"
mkdir -p "$build"
native="$(dirname "$(realpath "$0")")/marvin_bridge.cpp"
if [[ ! -f "$marvin/lib/libKine.so" || ! -f "$package/src/marvin_kine_sdk.cpp" ]]; then
  printf 'Pinned Teleop Marvin assets are missing\n' >&2
  exit 2
fi
if [[ -e "$build/lib_s05_marvin.so" ]]; then
  printf 'Output already exists: %s\n' "$build/lib_s05_marvin.so" >&2
  exit 2
fi
c++ -shared -fPIC -O2 -std=c++20 -DTIANJI_HAS_MARVIN_KINE=1 \
  -I/usr/include/eigen3 -I"$package/include" -I"$marvin/include" \
  "$native" "$package/src/marvin_kine_sdk.cpp" "$package/src/kinematics_adapter.cpp" \
  "$marvin/lib/libKine.so" -Wl,-z,defs -Wl,-rpath,"$marvin/lib" -pthread \
  -o "$build/lib_s05_marvin.so"
sha256sum "$native" "$package/src/marvin_kine_sdk.cpp" \
  "$package/src/kinematics_adapter.cpp" "$marvin/lib/libKine.so" \
  "$build/lib_s05_marvin.so"
