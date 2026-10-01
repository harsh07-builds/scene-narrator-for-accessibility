"""YOLO detection layer with safe image decoding and thread-safe model reuse."""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MAX_IMAGE_WIDTH = 8192
DEFAULT_MAX_IMAGE_HEIGHT = 8192
DEFAULT_MAX_IMAGE_PIXELS = 25_000_000

_model = None
_loaded_model_name: Optional[str] = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run: pip install opencv-python")
    return cv2


@dataclass
class Detection:
    """One object detected in a frame."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "x2": int(self.x2),
            "y2": int(self.y2),
        }

    def __repr__(self) -> str:
        return (
            f"Detection(label={self.label!r}, conf={self.confidence:.2f}, "
            f"box=({self.x1},{self.y1})→({self.x2},{self.y2}))"
        )


def load_model(model_name: Optional[str] = None):
    """Return a cached YOLO model, loading/reloading when the configured path changes."""
    global _model, _loaded_model_name
    requested = (model_name or os.getenv("SCENE_MODEL") or DEFAULT_MODEL_NAME).strip()
    if not requested:
        raise ValueError("YOLO model name/path must not be empty")

    with _model_lock:
        if _model is None or _loaded_model_name != requested:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is not installed. Run: pip install ultralytics"
                ) from exc
            print(f"[detection] Loading {requested} ...", flush=True)
            _model = YOLO(requested)
            _loaded_model_name = requested
            print("[detection] Model ready.", flush=True)
    return _model


def is_model_loaded() -> bool:
    return _model is not None


def loaded_model_name() -> Optional[str]:
    return _loaded_model_name


_get_model = load_model


def validate_image_bytes(
    data: bytes,
    *,
    max_width: int = DEFAULT_MAX_IMAGE_WIDTH,
    max_height: int = DEFAULT_MAX_IMAGE_HEIGHT,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> tuple[int, int, str]:
    """Validate image headers before OpenCV allocates a decoded pixel buffer."""
    if not data:
        raise ValueError("Empty image data.")
    if max_width <= 0 or max_height <= 0 or max_pixels <= 0:
        raise ValueError("Image safety limits must be positive")

    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            fmt = (image.format or "").lower()
            if width <= 0 or height <= 0:
                raise ValueError("Image has invalid dimensions")
            if width > max_width or height > max_height:
                raise ValueError(
                    f"Image dimensions {width}x{height} exceed the {max_width}x{max_height} limit"
                )
            pixels = width * height
            if pixels > max_pixels:
                raise ValueError(
                    f"Image contains {pixels:,} pixels; limit is {max_pixels:,}"
                )
            image.verify()
            return width, height, fmt
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not validate image (unsupported or corrupt file).") from exc


def decode_image(
    data: bytes,
    *,
    max_width: int = DEFAULT_MAX_IMAGE_WIDTH,
    max_height: int = DEFAULT_MAX_IMAGE_HEIGHT,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> np.ndarray:
    """Decode encoded image bytes into a BGR array after header-level safety checks."""
    width, height, _ = validate_image_bytes(
        data,
        max_width=max_width,
        max_height=max_height,
        max_pixels=max_pixels,
    )
    frame = _require_cv2().imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image (unsupported or corrupt file).")
    if frame.shape[1] != width or frame.shape[0] != height:
        raise ValueError("Decoded image dimensions do not match the image header")
    return frame


def detect(
    frame: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    model_name: Optional[str] = None,
) -> List[Detection]:
    """Run YOLO inference on one BGR image and return confidence-sorted detections."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a non-empty HxWx3 BGR image")
    if not math.isfinite(conf_threshold) or not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be in [0, 1]")

    model = load_model() if model_name is None else load_model(model_name)
    with _infer_lock:
        results = model(frame, verbose=False, conf=conf_threshold)

    detections: List[Detection] = []
    frame_h, frame_w = frame.shape[:2]
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            label = str(model.names[cls_id])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            # Clamp model coordinates to the decoded frame.
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(x1 + 1, min(frame_w, x2))
            y2 = max(y1 + 1, min(frame_h, y2))
            detections.append(Detection(label, conf, x1, y1, x2, y2))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of the frame."""
    _require_cv2()
    annotated = frame.copy()
    for det in detections:
        colour = (0, 200, 0)
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), colour, 2)
        label_text = f"{det.label} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_top = max(0, det.y1 - th - baseline - 4)
        cv2.rectangle(annotated, (det.x1, y_top), (det.x1 + tw + 4, det.y1), colour, cv2.FILLED)
        cv2.putText(
            annotated,
            label_text,
            (det.x1 + 2, max(th + 2, det.y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _main() -> int:
    ap = argparse.ArgumentParser(description="AI Scene Narrator — YOLO detector")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    args = ap.parse_args()

    _require_cv2()
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"ERROR: cannot read image {args.image}", file=sys.stderr)
            return 1
        for det in detect(frame, args.conf):
            print(det)
        return 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: camera read failed", file=sys.stderr)
                return 1
            detections = detect(frame, args.conf)
            cv2.imshow("YOLO detections (q to quit)", draw_detections(frame, detections))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
