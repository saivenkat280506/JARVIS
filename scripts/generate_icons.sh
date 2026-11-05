#!/usr/bin/env bash
set -euo pipefail

# Convert backend/assets/icons/JARVIS.jpg to macOS .icns and required Tauri PNG icons.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_IMAGE="$PROJECT_ROOT/backend/assets/icons/JARVIS.jpg"
OUT_DIR="$PROJECT_ROOT/src-tauri/icons"

if [ ! -f "$SRC_IMAGE" ]; then
  echo "Error: Source image not found at $SRC_IMAGE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
ICONSET_DIR="$OUT_DIR/JARVIS.iconset"
mkdir -p "$ICONSET_DIR"

echo "Converting $SRC_IMAGE to PNG..."
BASE_PNG="$OUT_DIR/icon.png"
sips -s format png "$SRC_IMAGE" --out "$BASE_PNG" > /dev/null

echo "Generating iconset sizes..."
sips -z 16 16     "$BASE_PNG" --out "$ICONSET_DIR/icon_16x16.png" > /dev/null
sips -z 32 32     "$BASE_PNG" --out "$ICONSET_DIR/icon_16x16@2x.png" > /dev/null
sips -z 32 32     "$BASE_PNG" --out "$ICONSET_DIR/icon_32x32.png" > /dev/null
sips -z 64 64     "$BASE_PNG" --out "$ICONSET_DIR/icon_32x32@2x.png" > /dev/null
sips -z 128 128   "$BASE_PNG" --out "$ICONSET_DIR/icon_128x128.png" > /dev/null
sips -z 256 256   "$BASE_PNG" --out "$ICONSET_DIR/icon_128x128@2x.png" > /dev/null
sips -z 256 256   "$BASE_PNG" --out "$ICONSET_DIR/icon_256x256.png" > /dev/null
sips -z 512 512   "$BASE_PNG" --out "$ICONSET_DIR/icon_256x256@2x.png" > /dev/null
sips -z 512 512   "$BASE_PNG" --out "$ICONSET_DIR/icon_512x512.png" > /dev/null
sips -z 1024 1024 "$BASE_PNG" --out "$ICONSET_DIR/icon_512x512@2x.png" > /dev/null

echo "Compiling macOS icon.icns via iconutil..."
iconutil -c icns "$ICONSET_DIR" -o "$OUT_DIR/icon.icns"
rm -rf "$ICONSET_DIR"

echo "Generating standard Tauri icon files..."
cp "$OUT_DIR/icon.png" "$OUT_DIR/1024x1024.png"
sips -z 32 32   "$BASE_PNG" --out "$OUT_DIR/32x32.png" > /dev/null
sips -z 128 128 "$BASE_PNG" --out "$OUT_DIR/128x128.png" > /dev/null
sips -z 256 256 "$BASE_PNG" --out "$OUT_DIR/128x128@2x.png" > /dev/null

# Also copy 32x32 as icon.ico placeholder if needed by tauri config
cp "$OUT_DIR/32x32.png" "$OUT_DIR/icon.ico"

echo "Successfully generated macOS icons in $OUT_DIR"
