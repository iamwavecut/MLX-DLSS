#!/usr/bin/env bash
# Kernel and whole-frame timing of the fused float16 path on the external weights.
#   scripts/bench.sh [--display-sleep] [TEST ...]
# Whole-frame numbers move by ±10% while WindowServer composites; --display-sleep
# puts the display to sleep first (the system stays awake, wake it afterwards)
# so the GPU watchdog and the desktop do not skew the run. TEST defaults to the
# per-kernel and A/B diagnostics of PerfSpikeKernelTimingTests.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
sleep_display=0
if [[ "${1:-}" == "--display-sleep" ]]; then sleep_display=1; shift; fi
tests=("$@")
if [[ ${#tests[@]} -eq 0 ]]; then
  tests=(testFusedWindowBlock1h testFusedWindowAttentionCore testGlobalAttentionV2IsBitExactAndTiming testWholeFrameAB)
fi
export MLXDLSS_PERF_SPIKE=1
export MLXDLSS_LOGICAL_WEIGHTS="${MLXDLSS_LOGICAL_WEIGHTS:-$project_root/weights/dlssnr-weights-logical.safetensors}"
filter="PerfSpikeKernelTimingTests/($(IFS='|'; echo "${tests[*]}"))"
if [[ $sleep_display -eq 1 ]]; then
  caffeinate -s -w $$ &
  pmset displaysleepnow
  sleep 3
fi
swift test -c release -Xswiftc -enable-testing --skip-build --filter "$filter" 2>&1 | grep -E "perf-spike|error:|failed"
