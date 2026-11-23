#!/usr/bin/env bash
# Places mlx.metallib beside the binaries (and in the given destinations).
#
# mlx-swift builds MLX in JIT mode: GEMM, convolution, attention and elementwise
# kernels, including the Neural Accelerator (NAX) kernels of M5 and later, are
# compiled from source at first use. The default library only has to provide the
# remaining precompiled kernels. A SwiftPM build with Swift Build compiles exactly
# that library into mlx-swift_Cmlx.bundle (about 4 MB); the script copies it. When
# the bundle has no default.metallib (the native SwiftPM build system does not
# compile Metal sources), it builds MLX's full precompiled library with CMake and
# Ninja instead (about 150 MB, a superset).
#
#   MLXDLSS_BUILD_CONFIGURATION  debug (default) or release: the build to take the library from
#   MLXDLSS_PREPARE_SKIP_SWIFT_BUILD=1  do not run swift build first
#   MLXDLSS_METALLIB_SOURCE=cmake  always build the full library with CMake
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SOURCE="$PROJECT_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
METAL_BUILD="$PROJECT_ROOT/.build/mlxdlss-mlx-metallib"
CMAKE_METALLIB="$METAL_BUILD/mlx/backend/metal/kernels/mlx.metallib"
MIN_MACOS_VERSION="${MLXDLSS_MIN_MACOS_VERSION:-14.0}"
BUILD_JOBS="${MLXDLSS_BUILD_JOBS:-2}"
CONFIGURATION="${MLXDLSS_BUILD_CONFIGURATION:-debug}"

if ! command -v swift >/dev/null 2>&1; then
  echo "Required command is unavailable: swift" >&2
  exit 69
fi

if [[ "${MLXDLSS_PREPARE_SKIP_SWIFT_BUILD:-0}" != "1" ]]; then
  swift build --package-path "$PROJECT_ROOT" -c "$CONFIGURATION" --build-tests --jobs "$BUILD_JOBS"
fi
bin_path="$(swift build --package-path "$PROJECT_ROOT" -c "$CONFIGURATION" --show-bin-path)"
bundle_metallib="$bin_path/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"

if [[ -s "$bundle_metallib" && "${MLXDLSS_METALLIB_SOURCE:-}" != "cmake" ]]; then
  metallib="$bundle_metallib"
else
  for command_name in cmake ninja xcrun; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      echo "Required command is unavailable: $command_name" >&2
      exit 69
    fi
  done
  if [[ ! -f "$MLX_SOURCE/CMakeLists.txt" ]]; then
    swift package --package-path "$PROJECT_ROOT" resolve
  fi
  if [[ ! -f "$MLX_SOURCE/CMakeLists.txt" ]]; then
    echo "MLX sources were not resolved at $MLX_SOURCE" >&2
    exit 66
  fi
  cmake \
    -S "$MLX_SOURCE" \
    -B "$METAL_BUILD" \
    -G Ninja \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DMLX_BUILD_GGUF=OFF \
    -DMLX_BUILD_SAFETENSORS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_MACOS_VERSION"
  cmake --build "$METAL_BUILD" --target mlx-metallib --parallel "$BUILD_JOBS"
  metallib="$CMAKE_METALLIB"
fi

if [[ ! -s "$metallib" ]]; then
  echo "MLX metallib was not produced at $metallib" >&2
  exit 70
fi

if [[ "$#" -gt 0 ]]; then
  destinations=("$@")
else
  destinations=("$bin_path")
  while IFS= read -r destination; do
    destinations+=("$destination")
  done < <(find "$bin_path" -type d -path '*.xctest/Contents/MacOS' -print)
fi

# A colocated mlx.metallib wins over the bundle's library, so always replace it:
# a library left from another MLX version would not match the host code.
for destination in "${destinations[@]}"; do
  mkdir -p "$destination"
  cp "$metallib" "$destination/mlx.metallib"
done

echo "$metallib"
