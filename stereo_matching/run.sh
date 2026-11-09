#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_dir}/build"
model="${STEREO_MODEL:-}"
left="${STEREO_LEFT:-}"
right="${STEREO_RIGHT:-}"
output="${STEREO_OUTPUT:-/tmp/stereo_matching_output/disparity.png}"
a100_cores="${STEREO_A100_CORES:-8}"
instances="${STEREO_INSTANCES:-4}"

if [[ ! -s "${model}" || -z "${left}" || -z "${right}" ]]; then
    echo "Set STEREO_MODEL to a non-empty tensor model file." >&2
    echo "Set STEREO_LEFT and STEREO_RIGHT to the stereo input images." >&2
    echo "Example:" >&2
    echo "  STEREO_MODEL=/path/to/tensor/model.onnx STEREO_LEFT=/path/im0.png STEREO_RIGHT=/path/im1.png ./run.sh" >&2
    exit 2
fi

cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DSTEREO_MATCHING_BUILD_DEMO=ON
cmake --build "${build_dir}" -j"$(nproc)"

exec taskset -c 0-7 "${build_dir}/stereo_matching_demo" \
    "${model}" "${left}" "${right}" "${output}" \
    "${a100_cores}" "${instances}" \
    "$@"
