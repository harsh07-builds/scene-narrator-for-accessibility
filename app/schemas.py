"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator


class BoxModel(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)
    x1: int = Field(ge=0, le=20_000)
    y1: int = Field(ge=0, le=20_000)
    x2: int = Field(ge=1, le=20_000)
    y2: int = Field(ge=1, le=20_000)

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("label must not be blank")
        return value

    @field_validator("confidence")
    @classmethod
    def _finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value

    @model_validator(mode="after")
    def _valid_box(self) -> "BoxModel":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must satisfy x2 > x1 and y2 > y1")
        return self


class NarrateRequest(BaseModel):
    detections: List[BoxModel] = Field(max_length=200)
    frame_width: int = Field(default=640, gt=0, le=20_000)
    frame_height: int = Field(default=480, gt=0, le=20_000)

    @model_validator(mode="after")
    def _boxes_fit_frame(self) -> "NarrateRequest":
        for box in self.detections:
            if box.x2 > self.frame_width or box.y2 > self.frame_height:
                raise ValueError("bounding boxes must lie within frame_width/frame_height")
        return self


class NarrateResponse(BaseModel):
    narration: str
    num_detections: int
    scene_mode: str
    frame_width: int
    frame_height: int
    detections: List[BoxModel]
    frame_zones: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    processing_ms: float = Field(ge=0.0)
    detection_ms: float = Field(ge=0.0)
    reasoning_ms: float = Field(ge=0.0)
    all_relations: List[Dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    version: str
