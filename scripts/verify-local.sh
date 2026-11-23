#!/usr/bin/env bash
# Local verification with the external weights CI cannot see: release build with
# testing enabled, every Swift suite (external-weight cases run when
# MLXDLSS_LOGICAL_WEIGHTS points at the logical safetensors), the public-tree
# audit and the Python package tests. Run before pushing kernel or pipeline changes.
# Without MLXDLSS_LOGICAL_WEIGHTS the suites run once per network found in
# weights/: the small network (the 310.9.16 default) first, then HT.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
weights=()
if [[ -n "${MLXDLSS_LOGICAL_WEIGHTS:-}" ]]; then
  weights=("$MLXDLSS_LOGICAL_WEIGHTS")
else
  for name in dlssnr-small-weights-logical dlssnr-weights-logical; do
    if [[ -f "$project_root/weights/$name.safetensors" ]]; then weights+=("$project_root/weights/$name.safetensors"); fi
  done
fi
swift build -c release --build-tests -Xswiftc -enable-testing
bin_path="$(swift build -c release --show-bin-path)"
# Always refresh: a colocated library from another MLX version would shadow the build's own.
MLXDLSS_PREPARE_SKIP_SWIFT_BUILD=1 MLXDLSS_BUILD_CONFIGURATION=release \
  scripts/prepare-mlx-metallib.sh "$bin_path" "$bin_path/MLXDLSSPackageTests.xctest/Contents/MacOS"
if [[ ${#weights[@]} -eq 0 ]]; then
  echo "external weights: not found, external-weight cases will skip" >&2
  swift test -c release -Xswiftc -enable-testing --skip-build
else
  for file in "${weights[@]}"; do
    echo "external weights: $file"
    MLXDLSS_LOGICAL_WEIGHTS="$file" swift test -c release -Xswiftc -enable-testing --skip-build
  done
fi
scripts/audit-public-tree.sh .
python="${MLXDLSS_PYTHON:-}"
if [[ -z "$python" && -x "$project_root/.venv/bin/python" ]]; then python="$project_root/.venv/bin/python"; fi
if [[ -n "$python" ]]; then
  (cd python && "$python" -m unittest discover -s tests -t . -p 'test_*.py')
else
  echo "python tests: skipped (set MLXDLSS_PYTHON or create .venv)" >&2
fi
echo "verify-local: passed"
