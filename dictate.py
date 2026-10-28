# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "mlx-whisper",
#   "numpy",
#   "sounddevice",
#   "rumps",
#   "pyobjc-framework-Quartz",
# ]
# ///
"""
Local push-to-talk dictation for macOS (a Wispr Flow stand-in).

  Hold Left Control  -> talk, release to transcribe + paste into the focused app
  Tap Left Control   -> hands-free mode; tap again to stop

Transcription runs locally with Whisper on Apple Silicon (MLX). Every
transcript is also appended to ~/Documents/Dictation/YYYY-MM-DD.md.
"""

import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import mlx_whisper
import numpy as np
import Quartz
import rumps
import sounddevice as sd
from AppKit import NSPasteboard, NSPasteboardTypeString, NSSound

MODEL = os.environ.get("DICTATE_MODEL", "mlx-community/whisper-large-v3-turbo")
LANGUAGE = os.environ.get("DICTATE_LANG") or None  # None = auto-detect
LOG_DIR = Path(os.environ.get("DICTATE_LOG_DIR", "~/Documents/Dictation")).expanduser()

SAMPLE_RATE = 16_000
HOTKEY_KEYCODE = 59          # Left Control
HOTKEY_DEVICE_MASK = 0x01    # NX_DEVICELCTLKEYMASK
TAP_THRESHOLD = 0.3          # seconds; shorter press = toggle hands-free
MIN_AUDIO_SECONDS = 0.4
SILENCE_RMS = 0.004          # below this the clip is treated as silence

FILLERS = re.compile(r"\b(?:um+|uh+|erm+|uhm+)\b[,.]?\s*", re.IGNORECASE)
# Whisper's classic hallucinations on near-silent audio
HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "."}

ICON_IDLE, ICON_REC, ICON_BUSY, ICON_LOADING = "◯", "●", "…", "⏳"


class Recorder:
    def __init__(self):
        self.chunks = []
        self.stream = None

    def start(self):
        self.chunks = []
        try:
            self._open()
        except sd.PortAudioError:
            # PortAudio caches the device list at import. When a mic disappears or
            # re-enumerates (USB/monitor sleep, AirPods, etc.) the cached IDs go stale
            # and every open fails until PortAudio is reinitialized.
            self._close()
            sd._terminate()
            sd._initialize()
            self._open()

    def _open(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=lambda data, *_: self.chunks.append(data.copy()),
        )
        self.stream.start()

    def _close(self):
        stream, self.stream = self.stream, None
        if stream:
            try:
                stream.stop()
                stream.close()
            except sd.PortAudioError as e:
                print(f"mic close error: {e}", flush=True)

    def stop(self) -> np.ndarray:
        self._close()
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks).flatten()


def clean(text: str) -> str:
    text = FILLERS.sub("", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m[1] + m[2].upper(), text)
    if text.lower() in HALLUCINATIONS:
        return ""
    return text[:1].upper() + text[1:] if text else text


def paste(text: str):
    pb = NSPasteboard.generalPasteboard()
    previous = pb.stringForType_(NSPasteboardTypeString)
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)

    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, 9, down)  # 9 = 'v'
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    # Give the target app time to read the clipboard, then restore it.
    time.sleep(0.4)
    if previous is not None:
        pb.clearContents()
        pb.setString_forType_(previous, NSPasteboardTypeString)


def log_transcript(text: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    with open(LOG_DIR / f"{now:%Y-%m-%d}.md", "a") as f:
        f.write(f"- **{now:%H:%M}** {text}\n")


def sound(name: str):
    s = NSSound.soundNamed_(name)
    if s:
        s.play()


class DictateApp(rumps.App):
    def __init__(self):
        super().__init__("Dictate", title=ICON_LOADING, quit_button="Quit")
        self.status = rumps.MenuItem("Loading model…")
        self.paste_toggle = rumps.MenuItem("Auto-paste", callback=self.toggle_paste)
        self.paste_toggle.state = 1
        self.menu = [self.status, self.paste_toggle, rumps.MenuItem("Open notes folder", callback=self.open_logs), None]

        self.recorder = Recorder()
        self.state = "loading"   # loading | idle | recording | transcribing
        self.handsfree = False
        self.pressed_at = 0.0
        self.key_down = False
        self.cancelled = False

        self._install_event_tap()
        threading.Thread(target=self._warm_up, daemon=True).start()
        rumps.Timer(self._refresh_ui, 0.1).start()

    # ---- UI ---------------------------------------------------------------

    def _refresh_ui(self, _):
        icon, label = {
            "loading": (ICON_LOADING, "Loading model…"),
            "idle": (ICON_IDLE, "Ready — hold Left ⌃ to talk"),
            "recording": (ICON_REC, "Listening (hands-free)" if self.handsfree else "Listening…"),
            "transcribing": (ICON_BUSY, "Transcribing…"),
        }[self.state]
        if self.title != icon:
            self.title = icon
        if self.status.title != label:
            self.status.title = label

    def toggle_paste(self, item):
        item.state = not item.state

    def open_logs(self, _):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{LOG_DIR}"')

    # ---- model ------------------------------------------------------------

    def _warm_up(self):
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL)
        self.state = "idle"

    def _transcribe(self, audio: np.ndarray):
        try:
            if len(audio) < MIN_AUDIO_SECONDS * SAMPLE_RATE:
                return
            if float(np.sqrt(np.mean(audio**2))) < SILENCE_RMS:
                return
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo=MODEL, language=LANGUAGE,
                condition_on_previous_text=False,
            )
            text = clean(result.get("text", ""))
            if not text:
                return
            log_transcript(text)
            if self.paste_toggle.state:
                paste(text)
        except Exception as e:
            print(f"transcription failed: {e}", flush=True)
        finally:
            self.state = "idle"

    # ---- hotkey -----------------------------------------------------------

    def _install_event_tap(self):
        # Trigger the system permission prompts on first launch.
        Quartz.CGRequestListenEventAccess()
        Quartz.CGRequestPostEventAccess()
        mask = (Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
                | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown))
        self.tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly, mask, self._on_event, None,
        )
        if self.tap is None:
            rumps.alert(
                "Dictate needs permissions",
                "Grant Accessibility and Input Monitoring to Dictate (or your terminal, "
                "if running from one) in System Settings → Privacy & Security, then relaunch.",
            )
            raise SystemExit(1)
        source = Quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetMain(), source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self.tap, True)

    def _on_event(self, _proxy, etype, event, _refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self.tap, True)
            return event

        if etype == Quartz.kCGEventKeyDown:
            # Left Control used as a modifier for a normal shortcut -> not dictation.
            if self.key_down and self.state == "recording" and not self.handsfree:
                self.cancelled = True
                self.recorder.stop()
                self.state = "idle"
            return event

        if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode) != HOTKEY_KEYCODE:
            return event

        down = bool(Quartz.CGEventGetFlags(event) & HOTKEY_DEVICE_MASK)
        if down == self.key_down:
            return event
        self.key_down = down
        if down:
            self._on_press()
        else:
            self._on_release()
        return event

    def _on_press(self):
        self.cancelled = False
        if self.state == "idle":
            self.pressed_at = time.monotonic()
            self.handsfree = False
            try:
                self.recorder.start()
            except Exception as e:
                print(f"mic error: {e}", flush=True)
                sound("Basso")
                return
            sound("Tink")
            self.state = "recording"
        elif self.state == "recording" and self.handsfree:
            self._finish()

    def _on_release(self):
        if self.state != "recording" or self.cancelled or self.handsfree:
            return
        if time.monotonic() - self.pressed_at < TAP_THRESHOLD:
            self.handsfree = True
        else:
            self._finish()

    def _finish(self):
        audio = self.recorder.stop()
        self.handsfree = False
        sound("Pop")
        self.state = "transcribing"
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()


if __name__ == "__main__":
    DictateApp().run()
