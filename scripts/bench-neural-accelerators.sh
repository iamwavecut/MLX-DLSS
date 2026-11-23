#!/usr/bin/env bash
# Whole-network A/B of environment configurations, run in alternation in separate
# processes, plus the numerical difference of each configuration's output from
# the first one. Without arguments, on M5, M6 and later with macOS 26.2+, it
# compares three paths on the same machine: MLX's matrix kernels on the GPU Neural
# Accelerators (nax), MLX on the shader cores (shader), and nax plus the Metal 4
# tensor-operation feed-forward kernel (tensor-ops, MLXDLSS_TENSOR_OPS_FFN=1):
#
#   scripts/bench-neural-accelerators.sh                 # nax, shader, tensor-ops
#   scripts/bench-neural-accelerators.sh 'fused=' 'chain=MLXDLSS_FUSED_FFN=0'
#
# Each argument is LABEL=ENV[;ENV...]. The first configuration is the numerical
# reference. MLXDLSS_LOGICAL_WEIGHTS selects the network (default: the small one),
# BENCH_ROUNDS the number of alternating rounds (default 6), MLXDLSS_PERF_SHAPES
# the frame sizes (default 768x1024,1088x1920). Other GPU work on the Mac
# (displays, VMs, browsers) moves single runs by 30% or more; read the minimum and
# compare configurations from the same run only.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

default_weights="$project_root/weights/dlssnr-small-weights-logical.safetensors"
[[ -f "$default_weights" ]] || default_weights="$project_root/weights/dlssnr-weights-logical.safetensors"
export MLXDLSS_LOGICAL_WEIGHTS="${MLXDLSS_LOGICAL_WEIGHTS:-$default_weights}"
if [[ ! -f "$MLXDLSS_LOGICAL_WEIGHTS" ]]; then
  echo "logical weights not found: $MLXDLSS_LOGICAL_WEIGHTS (set MLXDLSS_LOGICAL_WEIGHTS)" >&2
  exit 66
fi
export MLXDLSS_PERF_SPIKE=1
rounds="${BENCH_ROUNDS:-6}"

swift build -c release --build-tests -Xswiftc -enable-testing >/dev/null
bin_path="$(swift build -c release --show-bin-path)"
MLXDLSS_PREPARE_SKIP_SWIFT_BUILD=1 MLXDLSS_BUILD_CONFIGURATION=release \
  scripts/prepare-mlx-metallib.sh "$bin_path" "$bin_path/DLSSMLXTests.xctest/Contents/MacOS" >/dev/null
bundle="$bin_path/DLSSMLXTests.xctest"

device_json="$("$bin_path/mlxdlss" device)"
echo "device: $device_json"

configs=("$@")
if [[ ${#configs[@]} -eq 0 ]]; then
  if [[ "$device_json" != *'"neuralAccelerators":true'* ]]; then
    echo "This GPU has no Neural Accelerators (M5, M6 and later have them); pass configurations to compare." >&2
    exit 0
  fi
  if [[ "$device_json" != *'"osSupportsNeuralAccelerators":true'* ]]; then
    echo "MLX uses the Neural Accelerators from macOS 26.2; update macOS to compare the paths." >&2
    exit 0
  fi
  architecture="$(sed -E 's/.*"architecture":"([^"]+)".*/\1/' <<<"$device_json")"
  # The same size class one generation before the Neural Accelerators: MLX keeps
  # its device tuning and takes the shader path.
  shader_architecture="applegpu_g16${architecture: -1}"
  configs=("nax=" "shader=MLX_METAL_GPU_ARCH=$shader_architecture" "tensor-ops=MLXDLSS_TENSOR_OPS_FFN=1")
fi

run_config() { # LABEL=ENV;ENV  TEST  [extra env...]
  local spec="$1" test="$2"; shift 2
  local assignments="${spec#*=}"
  local -a env_args=("$@")
  if [[ -n "$assignments" ]]; then
    IFS=';' read -r -a parts <<<"$assignments"
    env_args+=("${parts[@]}")
  fi
  env ${env_args[@]+"${env_args[@]}"} xcrun xctest -XCTest "DLSSMLXTests.$test" "$bundle" 2>&1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/mlxdlss-bench.XXXXXX")"
trap 'rm -rf "$work"' EXIT

echo "== outputs (first configuration is the reference)"
reference_dir=""
for spec in "${configs[@]}"; do
  label="${spec%%=*}"
  run_config "$spec" "WholeFrameDigestTests/testWholeFrameDigests" "MLXDLSS_DIGEST_DUMP=$work/$label" | grep -E "^digest |error" || true
  if [[ -z "$reference_dir" ]]; then
    reference_dir="$work/$label"
  else
    echo "-- $label vs ${configs[0]%%=*}"
    MLXDLSS_DIGEST_COMPARE="$reference_dir,$work/$label" xcrun xctest \
      -XCTest DLSSMLXTests.WholeFrameDigestTests/testCompareDumpedOutputs "$bundle" 2>&1 | grep -E "^digest-compare|error" || true
  fi
done

echo "== timing, $rounds alternating rounds (each sample is the best of 5 frames)"
results="$work/timing.txt"
: >"$results"
for ((round = 0; round < rounds; round++)); do
  ordered=("${configs[@]}")
  if ((round % 2 == 1)); then
    ordered=()
    for ((i = ${#configs[@]} - 1; i >= 0; i--)); do ordered+=("${configs[i]}"); done
  fi
  for spec in "${ordered[@]}"; do
    label="${spec%%=*}"
    run_config "$spec" "PerfSpikeKernelTimingTests/testWholeFrame" |
      sed -nE "s/^perf-spike whole frame ([0-9x]+): ([0-9.]+) ms.*/$label \1 \2/p" >>"$results"
  done
done
sort -k2,2 -k1,1 -k3,3n "$results" | awk '
  { key = $2 " " $1; n[key]++; v[key, n[key]] = $3; order[key] = NR }
  END {
    for (key in n) {
      split(key, parts, " ")
      median = v[key, int((n[key] + 1) / 2)]
      printf "%-10s %-10s min %7.1f ms  median %7.1f ms  (%d samples)\n", parts[1], parts[2], v[key, 1], median, n[key]
    }
  }' | sort
