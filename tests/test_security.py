import io

import cv2
import numpy as np
import pytest

from app.detection import decode_image


def test_decode_image_enforces_dimensions():
    ok, buf = cv2.imencode(".png", np.zeros((200, 300, 3), dtype=np.uint8))
    assert ok
    with pytest.raises(ValueError, match="dimensions"):
        decode_image(buf.tobytes(), max_width=100)


def test_decode_image_rejects_corrupt_payload():
    with pytest.raises(ValueError):
        decode_image(b"not actually an image")
