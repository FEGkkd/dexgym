#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Usage: bash $0 YYYY-MM-DD [workers]" >&2
    exit 2
fi

day=$1
workers=${2:-4}
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$task_dir/../../../../.." && pwd)
teleop=${WHOLEBODY_S0_TELEOP_ROOT:-"$HOME/teleop_ws"}
platform=${WHOLEBODY_S0_PLATFORM_ROOT:-"$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection"}
episodes=${WHOLEBODY_S0_EPISODES:-"$platform/data/raw/episodes"}
python=${WHOLEBODY_S0_PYTHON:-/usr/bin/python3}
mcap_path=${WHOLEBODY_S0_MCAP_PATH:-"$repo/data/egopro/.venv/lib/python3.12/site-packages"}
output="$repo/data/processed/wholebody_s0/$day"

source /opt/ros/jazzy/setup.bash
source "$teleop/install/local_setup.bash"
source "$platform/ros2_ws/install/local_setup.bash"
export PYTHONPATH="$repo/source:$repo/source/dexgym/tasks/direct:$mcap_path${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
cd -- "$repo"

cmake -S "$task_dir/native" -B "$output/.build" -DTELEOP_ROOT="$teleop" -DCMAKE_BUILD_TYPE=Release
cmake --build "$output/.build" --target wholebody_s0_ik -j "$workers"
args=(--date "$day" --workers "$workers" --teleop-root "$teleop" --episodes-dir "$episodes" --data-root "$repo/data")
"$python" -m wholebody_s0.prepare audit "${args[@]}" --raw-index
"$python" -m wholebody_s0.prepare build "${args[@]}"
"$python" -m wholebody_s0.prepare verify "${args[@]}"
"$python" -m wholebody_s0.prepare viewer "${args[@]}"
