"""detect() parsing logic against a stand-in for the Ultralytics API."""

import numpy as np

from app import detection
from app.detection import Detection, decode_image, detect, draw_detections


class _Box:
    def __init__(self, conf, cls, xyxy):
        self.conf = [conf]
        self.cls = [cls]
        self.xyxy = [np.array(xyxy, dtype=float)]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    names = {0: "person", 56: "chair"}

    def __call__(self, frame, verbose=False, conf=0.0):
        return [_Result([
            _Box(0.60, 56, [10.9, 20, 110, 220]),
            _Box(0.95, 0, [200, 50, 300, 400]),
            _Box(0.30, 0, [0, 0, 5, 5]),      # below threshold
        ])]


def test_detect_filters_sorts_and_parses(monkeypatch):
    monkeypatch.setattr(detection, "load_model", lambda: _FakeYOLO())
    dets = detect(np.zeros((480, 640, 3), dtype=np.uint8), conf_threshold=0.5)
    assert [d.label for d in dets] == ["person", "chair"]          # sorted by confidence
    assert dets[1].x1 == 10 and dets[1].y2 == 220                   # int-truncated coords


def test_detect_handles_empty_frame():
    assert detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_decode_image_roundtrip():
    import cv2
    ok, buf = cv2.imencode(".jpg", np.full((32, 48, 3), 127, dtype=np.uint8))
    frame = decode_image(buf.tobytes())
    assert frame.shape[:2] == (32, 48)


def test_draw_detections_returns_copy():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(frame, [Detection("x", 0.9, 10, 20, 60, 80)])
    assert out is not frame and out.sum() > 0 and frame.sum() == 0


def test_load_model_respects_requested_model_name(monkeypatch):
    import sys
    import types

    loaded = []

    class FakeConfiguredYOLO:
        def __init__(self, name):
            loaded.append(name)

    fake_ultralytics = types.SimpleNamespace(YOLO=FakeConfiguredYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(detection, "_model", None)
    monkeypatch.setattr(detection, "_loaded_model_name", None)

    first = detection.load_model("custom-a.pt")
    second = detection.load_model("custom-a.pt")
    third = detection.load_model("custom-b.pt")

    assert first is second
    assert third is not first
    assert loaded == ["custom-a.pt", "custom-b.pt"]
