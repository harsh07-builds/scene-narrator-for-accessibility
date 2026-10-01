"""
tts_engine.py — Stage 4: offline text-to-speech for AI Scene Narrator.

Wraps pyttsx3 on a dedicated worker thread (pyttsx3 engines must live on one
thread and `runAndWait()` blocks). Behaviour:

* Latest-wins: if a new sentence arrives while one is queued, the stale one is
  dropped, so narration never falls behind the live camera.
* Degrades gracefully: on a headless server (no audio device / no espeak) the
  engine simply reports `available == False` and `speak()` returns False.

Linux needs espeak:  sudo apt-get install espeak-ng
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, rate: int = 170, volume: float = 1.0, voice_id: Optional[str] = None) -> None:
        self.available: bool = False
        self._rate, self._volume, self._voice_id = rate, volume, voice_id
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def _new_engine(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        return engine

    def _worker(self) -> None:
        try:
            self._new_engine().stop()  # probe: is a speech driver available at all?
            self.available = True
        except Exception as exc:  # ImportError, OSError, RuntimeError (no driver) ...
            logger.warning("TTS unavailable (%s). Narration will be text-only.", exc)
            self._ready.set()
            return
        self._ready.set()

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                # A fresh engine per sentence: reusing one engine across
                # runAndWait() calls silently stops speaking on some Windows setups.
                engine = self._new_engine()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception:
                logger.exception("TTS playback failed")

    def speak(self, text: str) -> bool:
        """Queue `text` for speech. Returns True if it was accepted."""
        if not self.available or not text:
            return False
        try:  # drop any stale queued sentence — latest wins
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
