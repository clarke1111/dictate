#!/bin/zsh
# Build ~/Applications/Dictate.app — a menu-bar-only launcher for dictate.py.
# The app runs the script in place, so edits to dictate.py apply on next launch.
set -euo pipefail

SRC="${0:A:h}"
UV="$(command -v uv)" || { echo "uv not found — install it with: brew install uv" >&2; exit 1; }
APP="$HOME/Applications/Dictate.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Dictate</string>
  <key>CFBundleIdentifier</key><string>local.dictate</string>
  <key>CFBundleExecutable</key><string>Dictate</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>Dictate records your voice to transcribe it locally.</string>
</dict>
</plist>
EOF

cat > "$APP/Contents/MacOS/Dictate" <<EOF
#!/bin/zsh
mkdir -p "\$HOME/Library/Logs"
exec "$UV" run -q --script "$SRC/dictate.py" >> "\$HOME/Library/Logs/Dictate.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/Dictate"

codesign --force --deep -s - "$APP"
echo "Built $APP"
