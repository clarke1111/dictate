# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "mlx-whisper",
#   "numpy",
#   "sounddevice",
#   "rumps",
#   "pyobjc-framework-Quartz",
#   "pyobjc-framework-ApplicationServices",
# ]
# ///
"""
Local push-to-talk dictation for macOS (a Wispr Flow stand-in).

  Hold Left Control  -> talk, release to transcribe + paste into the focused app
  Tap Left Control   -> hands-free mode; tap again to stop

Transcription runs locally with Whisper on Apple Silicon (MLX). A bubble at the
bottom of the screen shows the transcript as you talk, with a Copy button after.
Every transcript is also appended to ~/Documents/Dictation/YYYY-MM-DD.md.
"""

import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import mlx_whisper
import numpy as np
import objc
import Quartz
import rumps
import sounddevice as sd
from AppKit import (
    NSAppearance, NSAppearanceNameDarkAqua, NSBackingStoreBuffered, NSButton, NSColor,
    NSEvent, NSFont, NSFontManager, NSItalicFontMask, NSMakeRect, NSMouseInRect, NSObject, NSPanel, NSPasteboard,
    NSPasteboardTypeString, NSScreen, NSSound, NSStatusWindowLevel, NSTextField,
    NSVisualEffectBlendingModeBehindWindow, NSVisualEffectMaterialHUDWindow,
    NSVisualEffectStateActive, NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from ApplicationServices import (
    AXUIElementCopyAttributeValue, AXUIElementCreateSystemWide, AXUIElementIsAttributeSettable,
    kAXErrorNoValue, kAXErrorSuccess, kAXFocusedUIElementAttribute, kAXRoleAttribute,
    kAXValueAttribute,
)

MODEL = os.environ.get("DICTATE_MODEL", "mlx-community/whisper-large-v3-turbo")
LANGUAGE = os.environ.get("DICTATE_LANG") or None  # None = auto-detect
LOG_DIR = Path(os.environ.get("DICTATE_LOG_DIR", "~/Documents/Dictation")).expanduser()

SAMPLE_RATE = 16_000
HOTKEY_KEYCODE = 59          # Left Control
HOTKEY_DEVICE_MASK = 0x01    # NX_DEVICELCTLKEYMASK
TAP_THRESHOLD = 0.3          # seconds; shorter press = toggle hands-free
MIN_AUDIO_SECONDS = 0.4
SILENCE_RMS = 0.004          # loudest 100 ms below this -> the clip is treated as silence
PREVIEW_INTERVAL = 0.6       # seconds between live-preview passes while recording
PREVIEW_WINDOW = 25          # seconds; the live preview only re-transcribes the tail
CLIPBOARD_RESTORE_DELAY = 1.0
RESULT_SECONDS = 5           # how long the bubble lingers after a successful paste

TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox"}
# Focused roles that definitely don't take typing. Anything else (AXGroup, AXWebArea,
# custom terminal views…) is treated as "can't tell" rather than "no text field".
NON_TEXT_ROLES = {
    "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXSlider",
    "AXList", "AXOutline", "AXTable", "AXRow", "AXCell", "AXBrowser", "AXImage", "AXLink",
    "AXStaticText", "AXTabGroup", "AXToolbar", "AXMenu", "AXMenuItem", "AXWindow", "AXSheet",
}
SYNTHETIC_TAG = 0x44494354   # marks our own ⌘V so the event tap ignores it
MODIFIER_FLAGS = (Quartz.kCGEventFlagMaskCommand | Quartz.kCGEventFlagMaskShift
                  | Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskAlternate)

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

    def snapshot(self) -> np.ndarray:
        chunks = list(self.chunks)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).flatten()

    def stop(self) -> np.ndarray:
        self._close()
        return self.snapshot()


def loudness(audio: np.ndarray) -> float:
    """RMS of the loudest 100 ms, so long clips with pauses aren't mistaken for silence."""
    frame = SAMPLE_RATE // 10
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
    return float(np.sqrt((frames**2).mean(axis=1)).max()) if len(frames) else 0.0


def clean(text: str) -> str:
    text = FILLERS.sub("", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m[1] + m[2].upper(), text)
    if text.lower() in HALLUCINATIONS:
        return ""
    return text[:1].upper() + text[1:] if text else text


def has_text_focus():
    """True/False if Accessibility can tell whether a text input has focus, None if it can't."""
    err, el = AXUIElementCopyAttributeValue(AXUIElementCreateSystemWide(), kAXFocusedUIElementAttribute, None)
    if err == kAXErrorNoValue:
        return False             # nothing focused at all
    if err != kAXErrorSuccess:
        return None
    _, role = AXUIElementCopyAttributeValue(el, kAXRoleAttribute, None)
    if role in TEXT_ROLES:
        return True
    err, settable = AXUIElementIsAttributeSettable(el, kAXValueAttribute, None)
    if err == kAXErrorSuccess and settable:
        return True
    err, _ = AXUIElementCopyAttributeValue(el, "AXSelectedTextRange", None)
    if err == kAXErrorSuccess:
        return True
    return False if role in NON_TEXT_ROLES else None


def copy_to_clipboard(text: str):
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)


def paste(text: str):
    # A modifier still held (e.g. Left ⌃ after a hands-free stop) would turn ⌘V into
    # some other shortcut, so wait for the keyboard to go quiet first.
    deadline = time.monotonic() + 1.0
    while (Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState) & MODIFIER_FLAGS
           and time.monotonic() < deadline):
        time.sleep(0.02)

    pb = NSPasteboard.generalPasteboard()
    previous = pb.stringForType_(NSPasteboardTypeString)
    copy_to_clipboard(text)
    ours = pb.changeCount()

    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, 9, down)  # 9 = 'v'
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGEventSourceUserData, SYNTHETIC_TAG)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    # Give the target app time to read the clipboard (busy Electron apps can take a
    # while), then restore it — unless something else has been copied meanwhile.
    time.sleep(CLIPBOARD_RESTORE_DELAY)
    if previous is not None and pb.changeCount() == ours:
        copy_to_clipboard(previous)


def log_transcript(text: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    with open(LOG_DIR / f"{now:%Y-%m-%d}.md", "a") as f:
        f.write(f"- **{now:%H:%M}** {text}\n")


def sound(name: str):
    s = NSSound.soundNamed_(name)
    if s:
        s.play()


class _Action(NSObject):
    """Target/action trampoline so an NSButton can call a Python function."""

    def initWithHandler_(self, handler):
        self = objc.super(_Action, self).init()
        self.handler = handler
        return self

    def fire_(self, _sender):
        self.handler()


class _ClickThroughButton(NSButton):
    # The bubble never becomes key (so it can't steal focus from the text field),
    # which means buttons must accept the first click instead of eating it.
    def acceptsFirstMouse_(self, _event):
        return True


class Bubble:
    """Floating, non-activating panel at the bottom of the screen."""

    WIDTH, PAD, GAP = 460, 14, 8

    def __init__(self, on_copy, on_dismiss):
        self._actions = [_Action.alloc().initWithHandler_(on_copy), _Action.alloc().initWithHandler_(on_dismiss)]
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.WIDTH, 60),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False,
        )
        self.panel.setLevel_(NSStatusWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setHasShadow_(True)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setBecomesKeyOnlyIfNeeded_(True)
        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self.panel.setAppearance_(NSAppearance.appearanceNamed_(NSAppearanceNameDarkAqua))

        bg = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, self.WIDTH, 60))
        bg.setMaterial_(NSVisualEffectMaterialHUDWindow)
        bg.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        bg.setState_(NSVisualEffectStateActive)
        bg.setWantsLayer_(True)
        bg.layer().setCornerRadius_(14)
        bg.layer().setMasksToBounds_(True)
        self.panel.setContentView_(bg)

        self.caption = NSTextField.labelWithString_("")
        self.caption.setFont_(NSFont.systemFontOfSize_(11))
        self.caption.setTextColor_(NSColor.secondaryLabelColor())
        self.body = NSTextField.wrappingLabelWithString_("")
        self.body.setSelectable_(False)
        # Draft text (still being transcribed) is grey italic; the final transcript is plain.
        self.final_font = NSFont.systemFontOfSize_(14)
        self.draft_font = NSFontManager.sharedFontManager().convertFont_toHaveTrait_(self.final_font, NSItalicFontMask)
        self.copy_button = _ClickThroughButton.buttonWithTitle_target_action_("Copy", self._actions[0], "fire:")
        self.close_button = _ClickThroughButton.buttonWithTitle_target_action_("✕", self._actions[1], "fire:")
        self.close_button.setBordered_(False)
        for view in (self.caption, self.body, self.copy_button, self.close_button):
            bg.addSubview_(view)
        self.shown = None

    def show(self, caption: str, body: str = "", buttons: bool = False, draft: bool = False):
        if self.shown == (caption, body, buttons, draft):
            return
        self.shown = (caption, body, buttons, draft)
        self.caption.setStringValue_(caption)
        self.body.setFont_(self.draft_font if draft else self.final_font)
        self.body.setTextColor_(NSColor.secondaryLabelColor() if draft else NSColor.labelColor())
        self.body.setStringValue_(body)
        self.body.setHidden_(not body)
        self.copy_button.setHidden_(not buttons)
        self.close_button.setHidden_(not buttons)
        self._layout(body, buttons)
        self.panel.orderFrontRegardless()

    def hide(self):
        if self.shown is not None:
            self.shown = None
            self.panel.orderOut_(None)

    def _layout(self, body: str, buttons: bool):
        W, P, G = self.WIDTH, self.PAD, self.GAP
        copy_w, copy_h = self.copy_button.fittingSize()
        close_w, close_h = self.close_button.fittingSize()
        text_w = W - 2 * P - (copy_w + close_w + 2 * G if buttons else 0)
        cap_h = self.caption.fittingSize().height
        body_h = self.body.cell().cellSizeForBounds_(NSMakeRect(0, 0, text_w, 10_000)).height if body else 0
        H = P + cap_h + (4 + body_h if body else 0) + P

        self.caption.setFrame_(NSMakeRect(P, H - P - cap_h, text_w, cap_h))
        self.body.setFrame_(NSMakeRect(P, P, text_w, body_h))
        self.close_button.setFrame_(NSMakeRect(W - P - close_w, (H - close_h) / 2, close_w, close_h))
        self.copy_button.setFrame_(NSMakeRect(W - P - close_w - G - copy_w, (H - copy_h) / 2, copy_w, copy_h))

        # Bottom-centre of whichever screen the mouse is on, clear of the Dock.
        mouse = NSEvent.mouseLocation()
        screen = next((s for s in NSScreen.screens() if NSMouseInRect(mouse, s.frame(), False)), NSScreen.mainScreen())
        area = screen.visibleFrame()
        x = area.origin.x + (area.size.width - W) / 2
        self.panel.setFrame_display_(NSMakeRect(x, area.origin.y + 24, W, H), True)
        self.panel.invalidateShadow()


def tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-limit:].lstrip()


class DictateApp(rumps.App):
    def __init__(self):
        super().__init__("Dictate", title=ICON_LOADING, quit_button="Quit")
        self.status = rumps.MenuItem("Loading model…")
        self.paste_toggle = rumps.MenuItem("Auto-paste", callback=self.toggle)
        self.paste_toggle.state = 1
        self.preview_toggle = rumps.MenuItem("Live preview", callback=self.toggle)
        self.preview_toggle.state = 1
        self.menu = [
            self.status, self.paste_toggle, self.preview_toggle, None,
            rumps.MenuItem("Copy last transcript", callback=self.copy_last),
            rumps.MenuItem("Open notes folder", callback=self.open_logs), None,
        ]

        self.recorder = Recorder()
        self.state = "loading"   # loading | idle | recording
        self.handsfree = False
        self.pressed_at = 0.0
        self.key_down = False
        self.cancelled = False

        # Whisper runs on one worker so clips paste in order; recording a new clip
        # doesn't have to wait for the previous one to finish transcribing.
        self.jobs = queue.Queue()
        self.pending = 0
        self.pending_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.session = 0         # bumps per recording so stale previews are dropped
        self.preview = ""
        self.last_text = ""
        self.result = None       # (caption, text, copyable, hide_at or None for sticky)
        self.bubble = None

        self._install_event_tap()
        threading.Thread(target=self._worker, daemon=True).start()
        rumps.Timer(self._refresh_ui, 0.1).start()

    # ---- UI ---------------------------------------------------------------

    def _refresh_ui(self, _):
        busy = self.pending > 0
        if self.state == "recording":
            icon, label = ICON_REC, "Listening (hands-free)" if self.handsfree else "Listening…"
        elif self.state == "loading":
            icon, label = ICON_LOADING, "Loading model…"
        elif busy:
            icon, label = ICON_BUSY, "Transcribing…"
        else:
            icon, label = ICON_IDLE, "Ready — hold Left ⌃ to talk"
        if self.title != icon:
            self.title = icon
        if self.status.title != label:
            self.status.title = label
        self._refresh_bubble(busy)

    def _refresh_bubble(self, busy: bool):
        if self.bubble is None:
            self.bubble = Bubble(self.copy_result, self.dismiss_result)
        if self.state == "recording":
            self.bubble.show("● Listening (tap ⌃ to stop)" if self.handsfree else "● Listening", tail(self.preview, 240), draft=True)
        elif busy:
            self.bubble.show("Transcribing…", tail(self.preview, 240), draft=True)
        elif self.result and (self.result[3] is None or time.monotonic() < self.result[3]):
            caption, text, copyable, _ = self.result
            self.bubble.show(caption, tail(text, 600), copyable)
        else:
            self.result = None
            self.bubble.hide()

    def show_result(self, caption: str, text: str = "", copyable: bool = False, seconds=RESULT_SECONDS):
        self.result = (caption, text, copyable, None if seconds is None else time.monotonic() + seconds)

    def copy_result(self):
        if self.result:
            copy_to_clipboard(self.result[1])
            self.show_result("Copied to clipboard", self.result[1], seconds=1)

    def dismiss_result(self):
        self.result = None

    def copy_last(self, _):
        if self.last_text:
            copy_to_clipboard(self.last_text)

    def toggle(self, item):
        item.state = not item.state

    def open_logs(self, _):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{LOG_DIR}"')

    # ---- model ------------------------------------------------------------

    def _whisper(self, audio: np.ndarray) -> str:
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MODEL, language=LANGUAGE,
            condition_on_previous_text=False,
        )
        return clean(result.get("text", ""))

    def _worker(self):
        with self.model_lock:
            mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL)
        self.state = "idle"
        while True:
            audio = self.jobs.get()
            try:
                self._transcribe(audio)
            except Exception as e:
                print(f"transcription failed: {e}", flush=True)
                self.show_result("Transcription failed — see ~/Library/Logs/Dictate.log")
            finally:
                with self.pending_lock:
                    self.pending -= 1

    def _transcribe(self, audio: np.ndarray):
        seconds = len(audio) / SAMPLE_RATE
        if seconds < MIN_AUDIO_SECONDS:
            print(f"dropped {seconds:.1f}s clip: too short", flush=True)
            return
        if loudness(audio) < SILENCE_RMS:
            print(f"dropped {seconds:.1f}s clip: silent (peak rms {loudness(audio):.4f})", flush=True)
            self.show_result("No speech detected — check your mic", seconds=2)
            return
        with self.model_lock:
            text = self._whisper(audio)
        if not text:
            print(f"dropped {seconds:.1f}s clip: no text", flush=True)
            self.show_result("No speech detected", seconds=2)
            return
        log_transcript(text)
        self.last_text = text
        if not self.paste_toggle.state:
            self.show_result("Transcript", text, copyable=True, seconds=None)
            return
        focused = has_text_focus()
        paste(text)
        print(f"pasted {len(text)} chars ({seconds:.1f}s clip, text field: {focused})", flush=True)
        if focused is False:
            # ⌘V most likely went nowhere — keep the bubble up until it's copied or dismissed.
            self.show_result("No text field found — copy it from here", text, copyable=True, seconds=None)
        else:
            self.show_result("Pasted", text, copyable=True)

    def _preview_loop(self, session: int):
        while self.session == session and self.state == "recording":
            time.sleep(PREVIEW_INTERVAL)
            if not self.preview_toggle.state:
                continue
            audio = self.recorder.snapshot()
            window = audio[-PREVIEW_WINDOW * SAMPLE_RATE:]
            if len(window) < SAMPLE_RATE or loudness(window) < SILENCE_RMS:
                continue
            # Never hold up a real transcription for a preview.
            if not self.model_lock.acquire(blocking=False):
                continue
            try:
                if self.session != session or self.state != "recording":
                    break
                text = self._whisper(window)
            except Exception as e:
                print(f"preview failed: {e}", flush=True)
                continue
            finally:
                self.model_lock.release()
            if self.session == session and text:
                self.preview = ("… " if len(audio) > len(window) else "") + text
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
            # Our own ⌘V can land while the next clip is being recorded; it's not a shortcut.
            if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == SYNTHETIC_TAG:
                return event
            # Left Control used as a modifier for a normal shortcut -> not dictation.
            if self.key_down and self.state == "recording" and not self.handsfree:
                self.cancelled = True
                self.recorder.stop()
                self.session += 1
                self.state = "idle"
                print("recording cancelled: ⌃ used as a shortcut modifier", flush=True)
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
            self.session += 1
            self.preview = ""
            self.result = None
            self.state = "recording"
            threading.Thread(target=self._preview_loop, args=(self.session,), daemon=True).start()
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
        with self.pending_lock:
            self.pending += 1
        self.state = "idle"
        self.jobs.put(audio)


if __name__ == "__main__":
    DictateApp().run()
