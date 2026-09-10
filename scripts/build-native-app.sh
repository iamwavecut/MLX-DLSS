#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGURATION="${MLXDLSS_BUILD_CONFIGURATION:-release}"
BUILD_JOBS="${MLXDLSS_BUILD_JOBS:-2}"
APP_PATH="$PROJECT_ROOT/.build/MLX DLSS.app"

swift build --package-path "$PROJECT_ROOT" -c "$CONFIGURATION" --jobs "$BUILD_JOBS"
BIN_PATH="$(swift build --package-path "$PROJECT_ROOT" -c "$CONFIGURATION" --show-bin-path)"
mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"
cp "$BIN_PATH/MLXDLSSApp" "$APP_PATH/Contents/MacOS/MLXDLSSApp"
cp "$BIN_PATH/mlxdlss" "$APP_PATH/Contents/MacOS/mlxdlss"
MLXDLSS_PREPARE_SKIP_SWIFT_BUILD=1 "$PROJECT_ROOT/scripts/prepare-mlx-metallib.sh" \
  "$APP_PATH/Contents/MacOS" "$BIN_PATH"

cat > "$APP_PATH/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>MLXDLSSApp</string>
  <key>CFBundleIdentifier</key><string>org.mlxdlss.mac</string>
  <key>CFBundleName</key><string>MLX DLSS</string>
  <key>CFBundleDisplayName</key><string>MLX DLSS</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>26.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSEnvironment</key><dict>
    <key>AGX_RELAX_CDM_CTXSTORE_TIMEOUT</key><string>1</string>
  </dict>
  <key>CFBundleDocumentTypes</key><array><dict>
    <key>CFBundleTypeName</key><string>Images and Video</string>
    <key>CFBundleTypeRole</key><string>Viewer</string>
    <key>LSHandlerRank</key><string>Alternate</string>
    <key>LSItemContentTypes</key><array><string>public.image</string><string>public.movie</string></array>
  </dict></array>
</dict></plist>
PLIST

plutil -lint "$APP_PATH/Contents/Info.plist"
codesign --force --sign - "$APP_PATH/Contents/MacOS/mlx.metallib"
codesign --force --sign - "$APP_PATH/Contents/MacOS/mlxdlss"
codesign --force --sign - "$APP_PATH"
codesign --verify --strict "$APP_PATH"
echo "$APP_PATH"
