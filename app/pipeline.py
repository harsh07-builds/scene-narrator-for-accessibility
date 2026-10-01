"""End-to-end narration pipeline and live scene-change gate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional

import numpy as np

from app.config import Settings
from app.detection import Detection, decode_image, detect
from app.nlg import NLGConfig, NLGEngine
from app.spatial import SpatialConfig, analyze_scene_spatial

Detector = Callable[..., List[Detection]]


@dataclass
class NarrationResult:
    narration: str
    scene: Dict[str, Any]
    detections: List[Detection]
    frame_width: int
    frame_height: int
    detection_ms: float = 0.0
    reasoning_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.detection_ms + self.reasoning_ms

    def to_dict(self, include_all_relations: bool = False) -> Dict[str, Any]:
        scene = dict(self.scene)
        if not include_all_relations:
            scene.pop("all_relations", None)
        # Use the post-deduplication detections from the scene whenever present.
        # This keeps num_detections, boxes, zones, and relations internally consistent.
        response_detections = scene["detections"] if "detections" in scene else [d.to_dict() for d in self.detections]
        return {
            "narration": self.narration,
            "num_detections": scene.get("num_detections", len(response_detections)),
            "scene_mode": scene.get("scene_mode", "empty"),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "detections": response_detections,
            "frame_zones": scene.get("frame_zones", []),
            "relations": scene.get("primary_relations", []),
            "processing_ms": round(self.total_ms, 2),
            "detection_ms": round(self.detection_ms, 2),
            "reasoning_ms": round(self.reasoning_ms, 2),
            **({"all_relations": scene.get("all_relations", [])} if include_all_relations else {}),
        }


class NarrationPipeline:
    def __init__(
        self,
        detector: Optional[Detector] = None,
        spatial_config: Optional[SpatialConfig] = None,
        nlg_config: Optional[NLGConfig] = None,
        conf_threshold: float = 0.50,
        max_relations: int = 5,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if not 0.0 <= float(conf_threshold) <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if max_relations <= 0:
            raise ValueError("max_relations must be > 0")
        self._detector: Detector = detector or detect
        self._uses_default_detector = detector is None
        self._spatial_config = spatial_config or SpatialConfig()
        self._nlg = NLGEngine(nlg_config or NLGConfig())
        self.conf_threshold = float(conf_threshold)
        self.max_relations = int(max_relations)
        self.model_name = model_name
        self._settings = settings or Settings(model_name=model_name or "yolov8n.pt")

    def narrate_detections(
        self, detections: List[Detection], frame_width: int, frame_height: int
    ) -> NarrationResult:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        started = time.perf_counter()
        scene = analyze_scene_spatial(
            detections, frame_width, frame_height, self._spatial_config, self.max_relations
        )
        narration = self._nlg.generate_narration(scene)
        reasoning_ms = (time.perf_counter() - started) * 1000.0
        return NarrationResult(
            narration=narration,
            scene=scene,
            detections=list(detections),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            reasoning_ms=reasoning_ms,
        )

    def narrate_frame(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> NarrationResult:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")
        h, w = frame.shape[:2]
        conf = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        started = time.perf_counter()
        if self._uses_default_detector:
            detections = detect(frame, conf_threshold=conf, model_name=self.model_name)
        else:
            detections = self._detector(frame, conf_threshold=conf)
        detection_ms = (time.perf_counter() - started) * 1000.0
        result = self.narrate_detections(detections, w, h)
        result.detection_ms = detection_ms
        return result

    def narrate_image_bytes(self, data: bytes, conf_threshold: Optional[float] = None) -> NarrationResult:
        """Validate and decode an image, then run detection + narration."""
        frame = decode_image(
            data,
            max_width=self._settings.max_image_width,
            max_height=self._settings.max_image_height,
            max_pixels=self._settings.max_image_pixels,
        )
        return self.narrate_frame(frame, conf_threshold)


def scene_signature(scene: Dict[str, Any]) -> FrozenSet[tuple]:
    """Order-independent fingerprint including coarse object positions and relations."""
    mode = scene.get("scene_mode")
    if mode == "empty":
        return frozenset()

    tokens: set[tuple] = set()
    for det in scene.get("detections", []):
        label = str(det.get("label", "")).strip().lower()
        if not label:
            continue
        # Quantized center keeps the live gate stable against 1–2 px detector jitter
        # while still noticing meaningful movement.
        cx = (float(det.get("x1", 0)) + float(det.get("x2", 0))) / 2.0
        cy = (float(det.get("y1", 0)) + float(det.get("y2", 0))) / 2.0
        fw = max(1.0, float(scene.get("frame_width", 640)))
        fh = max(1.0, float(scene.get("frame_height", 480)))
        qx = int(max(0, min(7, (cx / fw) * 8)))
        qy = int(max(0, min(7, (cy / fh) * 8)))
        tokens.add(("object", label, qx, qy))

    for r in scene.get("primary_relations", []):
        tokens.add(("relation", r.get("subject"), r.get("predicate"), r.get("object")))

    if not tokens and mode == "single_object":
        for z in scene.get("frame_zones", []):
            tokens.add(("zone", z.get("label"), z.get("zone")))
    return frozenset(tokens)


class ChangeGate:
    """Throttle live narration until a scene is stable and meaningfully changed."""

    def __init__(
        self,
        min_interval: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        repeat_after: Optional[float] = None,
        stable_frames: int = 1,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        if repeat_after is not None and repeat_after <= 0:
            raise ValueError("repeat_after must be > 0 when set")
        self.min_interval = float(min_interval)
        self.repeat_after = float(repeat_after) if repeat_after is not None else None
        self.stable_frames = max(1, int(stable_frames))
        self._cand_sig: Optional[FrozenSet[tuple]] = None
        self._cand_count = 0
        self._clock = clock
        self._last_sig: Optional[FrozenSet[tuple]] = None
        self._last_time: float = float("-inf")

    def should_speak(self, scene: Dict[str, Any]) -> bool:
        sig = scene_signature(scene)
        now = self._clock()
        if sig == self._cand_sig:
            self._cand_count += 1
        else:
            self._cand_sig, self._cand_count = sig, 1
        if self._cand_count < self.stable_frames:
            return False
        if now - self._last_time < self.min_interval:
            return False
        if sig == self._last_sig:
            due = self.repeat_after is not None and now - self._last_time >= self.repeat_after
            if not (due and sig):
                return False
        if not sig and self._last_sig is None:
            return False
        self._last_sig, self._last_time = sig, now
        return True
