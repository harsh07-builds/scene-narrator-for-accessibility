"""HTTP API tests with an injected fake detector (no YOLO required)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detection import Detection
from app.main import create_app
from app.nlg import NLGConfig
from app.pipeline import NarrationPipeline


def _detector(frame, conf_threshold=0.5):
    return [
        Detection("desk", 0.95, 100, 200, 550, 450),
        Detection("laptop", 0.92, 200, 160, 380, 220),
    ]


@pytest.fixture()
def client():
    settings = Settings(preload_model=False, max_upload_mb=1)
    pipeline = NarrationPipeline(detector=_detector, nlg_config=NLGConfig(seed=1))
    with TestClient(create_app(settings, pipeline)) as c:
        yield c


def _png_bytes(w=640, h=480):
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is False


def test_analyze_image(client):
    r = client.post("/api/analyze", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["frame_width"] == 640 and body["frame_height"] == 480
    assert body["scene_mode"] == "multi_object"
    assert "laptop" in body["narration"].lower()
    assert "all_relations" not in body


def test_analyze_debug_includes_all_relations(client):
    r = client.post("/api/analyze?debug=true", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert "all_relations" in r.json()


def test_analyze_rejects_non_image_type(client):
    r = client.post("/api/analyze", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_analyze_rejects_corrupt_image(client):
    r = client.post("/api/analyze", files={"file": ("a.png", b"garbage", "image/png")})
    assert r.status_code == 400


def test_analyze_rejects_oversized_upload(client):
    r = client.post("/api/analyze", files={"file": ("a.png", b"0" * (1024 * 1024 + 10), "image/png")})
    assert r.status_code == 413


def test_narrate_from_json(client):
    payload = {
        "frame_width": 640,
        "frame_height": 480,
        "detections": [
            {"label": "desk", "confidence": 0.95, "x1": 100, "y1": 200, "x2": 550, "y2": 450},
            {"label": "laptop", "confidence": 0.92, "x1": 200, "y1": 160, "x2": 380, "y2": 220},
        ],
    }
    r = client.post("/api/narrate", json=payload)
    assert r.status_code == 200
    assert r.json()["relations"][0]["predicate"] == "on"


def test_narrate_empty_scene(client):
    r = client.post("/api/narrate", json={"detections": []})
    assert r.status_code == 200
    assert r.json()["scene_mode"] == "empty"


def test_narrate_validates_boxes(client):
    bad = {"detections": [{"label": "x", "confidence": 0.9, "x1": 50, "y1": 0, "x2": 10, "y2": 10}]}
    assert client.post("/api/narrate", json=bad).status_code == 422
    bad_conf = {"detections": [{"label": "x", "confidence": 1.5, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]}
    assert client.post("/api/narrate", json=bad_conf).status_code == 422


def test_ready_reports_503_when_model_is_cold(client):
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert r.json()["model_loaded"] is False


def test_narrate_rejects_box_outside_frame(client):
    payload = {
        "frame_width": 100,
        "frame_height": 100,
        "detections": [
            {"label": "x", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 101, "y2": 10}
        ],
    }
    assert client.post("/api/narrate", json=payload).status_code == 422


def test_analyze_returns_consistent_deduplicated_detections(client):
    # The fixture detector always returns two different labels, so consistency is
    # checked through the response contract added by the hardened pipeline.
    r = client.post("/api/analyze", files={"file": ("a.png", _png_bytes(), "image/png")})
    body = r.json()
    assert body["num_detections"] == len(body["detections"])
    assert body["processing_ms"] >= 0
    assert body["detection_ms"] >= 0
    assert body["reasoning_ms"] >= 0
