#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist/linux"
BUILD_DIR="$ROOT_DIR/build/linux"
CACHE_DIR="$ROOT_DIR/.pyinstaller-cache"
SPEC_FILE="$ROOT_DIR/packaging/pyinstaller-cli.spec"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
APPIMAGE_TOOL="${APPIMAGE_TOOL:-appimagetool}"
APPDIR="$DIST_DIR/AppDir"

mkdir -p "$DIST_DIR" "$BUILD_DIR" "$CACHE_DIR"
rm -rf "$DIST_DIR/tinyllama-cli" "$DIST_DIR/AppDir" "$DIST_DIR/tinyllama-cli-x86_64.AppImage" "$BUILD_DIR"

PYINSTALLER_CONFIG_DIR="$CACHE_DIR" \
  "$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR" \
  "$SPEC_FILE"

mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications"
cp -R "$DIST_DIR/tinyllama-cli/." "$APPDIR/usr/bin/"

cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
cd "$HERE/usr/bin"
exec ./tinyllama-cli "$@"
EOF
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/usr/share/applications/tinyllama-cli.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=TinyLlama CLI
Exec=tinyllama-cli
Terminal=true
Categories=Utility;
EOF

"$APPIMAGE_TOOL" "$APPDIR" "$DIST_DIR/tinyllama-cli-x86_64.AppImage"
echo "Linux artifacts ready:"
echo "  $DIST_DIR/tinyllama-cli"
echo "  $DIST_DIR/tinyllama-cli-x86_64.AppImage"
