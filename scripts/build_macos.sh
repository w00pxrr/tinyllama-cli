#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist/macos"
BUILD_DIR="$ROOT_DIR/build/macos"
SPEC_FILE="$ROOT_DIR/packaging/pyinstaller-cli.spec"
CACHE_DIR="$ROOT_DIR/.pyinstaller-cache"

mkdir -p "$DIST_DIR" "$BUILD_DIR" "$CACHE_DIR"
rm -rf "$DIST_DIR/tinyllama-cli" "$DIST_DIR/tinyllama-cli-macos-arm64.zip" "$BUILD_DIR"

PYINSTALLER_CONFIG_DIR="$CACHE_DIR" \
  "$ROOT_DIR/.venv/bin/python" -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR" \
  "$SPEC_FILE"

cd "$DIST_DIR"
ditto -c -k --sequesterRsrc --keepParent tinyllama-cli tinyllama-cli-macos-arm64.zip
echo "macOS artifact ready:"
echo "  $DIST_DIR/tinyllama-cli-macos-arm64.zip"
