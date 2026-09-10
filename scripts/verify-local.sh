#!/usr/bin/env bash
# Local verification with the external weights CI cannot see: release build with
# testing enabled, every Swift suite (external-weight cases run when
# MLXDLSS_LOGICAL_WEIGHTS points at the logical safetensors), the public-tree
# audit and the Python package tests. Run before pushing kernel or pipeline changes.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
weights="${MLXDLSS_LOGICAL_WEIGHTS:-$project_root/weights/dlssnr-weights-logical.safetensors}"
if [[ -f "$weights" ]]; then
  export MLXDLSS_LOGICAL_WEIGHTS="$weights"
  echo "external weights: $weights"
else
  echo "external weights: not found, external-weight cases will skip" >&2
fi
swift build -c release --build-tests -Xswiftc -enable-testing
bin_path="$(swift build -c release --show-bin-path)"
if [[ ! -f "$bin_path/mlx.metallib" ]]; then
  scripts/prepare-mlx-metallib.sh "$bin_path" "$bin_path/MLXDLSSPackageTests.xctest/Contents/MacOS"
fi
swift test -c release -Xswiftc -enable-testing --skip-build
scripts/audit-public-tree.sh .
python="${MLXDLSS_PYTHON:-}"
if [[ -z "$python" && -x "$project_root/.venv/bin/python" ]]; then python="$project_root/.venv/bin/python"; fi
if [[ -n "$python" ]]; then
  (cd python && "$python" -m unittest discover -s tests -t . -p 'test_*.py')
else
  echo "python tests: skipped (set MLXDLSS_PYTHON or create .venv)" >&2
fi
echo "verify-local: passed"
