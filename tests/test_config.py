import os

import pytest

from app.config import Settings


def test_settings_reject_invalid_limits():
    with pytest.raises(ValueError):
        Settings(max_upload_mb=0)
    with pytest.raises(ValueError):
        Settings(max_image_pixels=0)


def test_settings_reads_image_safety_limits(monkeypatch):
    monkeypatch.setenv("SCENE_MAX_IMAGE_WIDTH", "4000")
    monkeypatch.setenv("SCENE_MAX_IMAGE_HEIGHT", "3000")
    monkeypatch.setenv("SCENE_MAX_IMAGE_PIXELS", "12000000")
    settings = Settings.from_env()
    assert (settings.max_image_width, settings.max_image_height, settings.max_image_pixels) == (4000, 3000, 12000000)
