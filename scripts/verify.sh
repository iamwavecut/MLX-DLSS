#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "verification requires macOS" >&2
  exit 69
fi
architecture="$(uname -m)"
echo "architecture: $architecture"
if [[ "$architecture" != "arm64" ]]; then
  echo "verification requires Apple Silicon arm64" >&2
  exit 69
fi

for command_name in git swift cmake ninja xcrun python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required command is unavailable: $command_name" >&2
    exit 69
  fi
done
if ! xcrun -f metal >/dev/null 2>&1; then
  echo "the selected Xcode does not provide the Metal compiler" >&2
  exit 69
fi

swift --version
scripts/audit-public-tree.sh .
python3 -m py_compile Tools/*.py Tests/ToolsTests/*.py
python3 -m py_compile python/mlxdlss/*.py python/mlxdlss/tools/*.py python/tests/*.py
python3 -m unittest \
  Tests/ToolsTests/test_compare_f32.py \
  Tests/ToolsTests/test_compare_neural_rendering_golden_bundle.py
if python3 -c 'import numpy; import safetensors' >/dev/null 2>&1; then
  python3 -m unittest \
    Tests/ToolsTests/test_extract_dlssnr_weights.py \
    Tests/ToolsTests/test_package_neural_rendering_transformer.py \
    Tests/ToolsTests/test_unpack_dlssnr_weights.py
else
  echo "neural-rendering package tool tests: skipped (numpy and safetensors are optional)"
fi
if python3 -c 'import numpy; import torch; import safetensors' >/dev/null 2>&1; then
  python3 -m unittest \
    Tests/ToolsTests/test_convert_neural_rendering_coreml.py \
    Tests/ToolsTests/test_neural_rendering_reference.py
else
  echo "PyTorch converter tests: skipped (torch, numpy, and safetensors are optional)"
fi
if python3 -c 'import numpy; import torch; import safetensors; import PIL' >/dev/null 2>&1; then
  python3 -m unittest discover -s python -t python -p 'test_*.py'
else
  echo "Python package tests: skipped (torch, numpy, safetensors, and pillow are optional)"
fi

if [[ "${MLXDLSS_SKIP_BUILD_AND_INFERENCE:-0}" == "1" ]]; then
  echo "build and inference: skipped by test harness"
  echo "verification: passed"
  exit 0
fi

swift package resolve
scripts/prepare-mlx-metallib.sh

if [[ "${MLXDLSS_SKIP_NESTED_SWIFT_TEST:-0}" != "1" ]]; then
  SWIFTPM_MAXIMUM_CONCURRENT_JOBS=2 swift test
else
  echo "swift test: skipped by nested test harness"
fi

SWIFTPM_MAXIMUM_CONCURRENT_JOBS=2 swift build -c release --jobs 2
release_bin="$(swift build -c release --show-bin-path)"
MLXDLSS_PREPARE_SKIP_SWIFT_BUILD=1 MLXDLSS_BUILD_CONFIGURATION=release scripts/prepare-mlx-metallib.sh "$release_bin"

echo "verification: passed"
