"""
live.py — real-time webcam narration (desktop use).

    python -m app.live                 # camera 0, spoken + on-screen
    python -m app.live --no-window     # headless: speak/print only
    python -m app.live --no-speech     # print only

Needs: ultralytics, opencv-python, and (for speech) pyttsx3.
"""

from __future__ import annotations

import argparse
import sys

from app.detection import CONFIDENCE_THRESHOLD, draw_detections
from app.pipeline import ChangeGate, NarrationPipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Scene Narrator — live webcam narration")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    ap.add_argument("--interval", type=float, default=3.0, help="min seconds between utterances")
    ap.add_argument("--repeat", type=float, default=0.0,
                    help="re-narrate an unchanged scene every N seconds (0 = only on change)")
    ap.add_argument("--stable", type=int, default=4,
                    help="frames a scene must stay the same before it is announced")
    ap.add_argument("--max-read-errors", type=int, default=5,
                    help="consecutive camera read failures before stopping")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-speech", action="store_true")
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1

    tts = None
    if not args.no_speech:
        from app.tts_engine import TTSEngine

        tts = TTSEngine()
        if not tts.available:
            print("TTS unavailable — continuing with text output only.", file=sys.stderr)

    pipeline = NarrationPipeline(conf_threshold=args.conf)
    gate = ChangeGate(min_interval=args.interval, repeat_after=args.repeat or None,
                      stable_frames=args.stable)
    print("Live narration running — press 'q' in the window (or Ctrl+C) to quit.")

    consecutive_read_errors = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_errors += 1
                print(
                    f"WARNING: camera read failed ({consecutive_read_errors}/{args.max_read_errors}).",
                    file=sys.stderr,
                )
                if consecutive_read_errors >= args.max_read_errors:
                    break
                continue
            consecutive_read_errors = 0

            result = pipeline.narrate_frame(frame)
            if gate.should_speak(result.scene):
                print(f"[narration] {result.narration}", flush=True)
                if tts:
                    tts.speak(result.narration)

            if not args.no_window:
                cv2.imshow("AI Scene Narrator (q to quit)", draw_detections(frame, result.detections))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        if tts:
            tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
