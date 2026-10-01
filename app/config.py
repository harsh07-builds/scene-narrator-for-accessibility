"""Runtime configuration for AI Scene Narrator."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off)")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.50
    max_upload_mb: int = 10
    max_image_width: int = 8192
    max_image_height: int = 8192
    max_image_pixels: int = 25_000_000
    cors_origins: Tuple[str, ...] = ("*",)
    preload_model: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not math.isfinite(self.conf_threshold) or not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be > 0")
        if self.max_image_width <= 0 or self.max_image_height <= 0:
            raise ValueError("max image dimensions must be > 0")
        if self.max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be > 0")
        if not self.cors_origins:
            raise ValueError("cors_origins must contain at least one origin")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            o.strip() for o in os.getenv("SCENE_CORS_ORIGINS", "*").split(",") if o.strip()
        ) or ("*",)
        conf = _env_float("SCENE_CONF_THRESHOLD", cls.conf_threshold)
        return cls(
            model_name=os.getenv("SCENE_MODEL", cls.model_name).strip(),
            conf_threshold=conf,
            max_upload_mb=_env_int("SCENE_MAX_UPLOAD_MB", cls.max_upload_mb),
            max_image_width=_env_int("SCENE_MAX_IMAGE_WIDTH", cls.max_image_width),
            max_image_height=_env_int("SCENE_MAX_IMAGE_HEIGHT", cls.max_image_height),
            max_image_pixels=_env_int("SCENE_MAX_IMAGE_PIXELS", cls.max_image_pixels),
            cors_origins=origins,
            preload_model=_env_bool("SCENE_PRELOAD_MODEL", cls.preload_model),
        )
