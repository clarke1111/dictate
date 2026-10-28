#!/bin/zsh
# One-shot setup: checks the Mac, installs uv if needed, builds Dictate.app and launches it.
set -euo pipefail

SRC="${0:A:h}"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Dictate needs an Apple Silicon Mac (M1 or later) — MLX doesn't run on Intel." >&2
  exit 1
fi

if ! command -v uv >/dev/null; then
  echo "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

"$SRC/build-app.sh"
open "$HOME/Applications/Dictate.app"

cat <<'EOF'

Dictate is starting — look for ⏳ in the menu bar. The first launch downloads
the speech model (~1.6 GB), so give it a few minutes before it turns to ◯.

When macOS asks, allow Dictate access to:
  • Microphone
  • Accessibility
  • Input Monitoring
(System Settings → Privacy & Security). If you missed a prompt, add Dictate
there by hand, then quit and relaunch it from Spotlight.

Then hold Left Control to talk, and let go to paste.
EOF
