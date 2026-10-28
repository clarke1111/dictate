# dictate

Local push-to-talk dictation for macOS (Wispr Flow stand-in). Whisper large-v3-turbo via MLX, fully offline after the first model download.

## Install

Needs an Apple Silicon Mac (M1 or later). In Terminal:

```sh
git clone https://github.com/clarke1111/dictate.git ~/dictate && ~/dictate/install.sh
```

That installs [uv](https://docs.astral.sh/uv/) if you don't have it, builds `~/Applications/Dictate.app` (menu bar only, no Dock icon) and launches it. The first launch downloads the model (~1.6 GB), so the `⏳` icon stays up for a few minutes; after that it starts in seconds.

When macOS asks, allow Dictate **Microphone**, **Accessibility** and **Input Monitoring** (System Settings → Privacy & Security). If you dismissed a prompt, add Dictate there by hand, then quit and relaunch it from Spotlight.

## Use

- **Hold Left ⌃** — talk, release to transcribe and paste at the cursor
- **Tap Left ⌃** — hands-free; tap again to stop
- Menu bar: `◯` ready · `●` listening · `…` transcribing
- A bubble at the bottom of the screen shows the words as you talk, then the final text with a **Copy** button. If no text field had focus it stays up until you copy or dismiss it
- Menu → **Copy last transcript** gets the most recent one back any time; **Live preview** turns off the as-you-talk text (saves GPU)
- Every transcript is also appended to `~/Documents/Dictation/YYYY-MM-DD.md`
- Filler words (um/uh) are stripped; your clipboard is restored after pasting

## Update

The app runs `dictate.py` from the cloned folder, so `git pull` picks up changes on next launch. If you move the folder, re-run `./build-app.sh`.

## Troubleshooting

- **Hotkey does nothing after a rebuild:** macOS sometimes forgets the permissions. Toggle Dictate off and on in Accessibility and Input Monitoring, then relaunch.
- **Text didn't paste:** it's still in the bubble, under **Copy last transcript**, and in the notes file. The log records why each clip was pasted or dropped (too short, silent, cancelled because ⌃ was used with another key)
- **Logs:** `~/Library/Logs/Dictate.log`
- **Run in a terminal to see errors live:** `uv run --script dictate.py` (grant the permissions to your terminal app instead)

Env overrides: `DICTATE_MODEL` (e.g. `mlx-community/whisper-small-mlx` for lighter), `DICTATE_LANG` (e.g. `en`), `DICTATE_LOG_DIR`.
