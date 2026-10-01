"""Pipeline + scene-change gate tests (no YOLO required)."""

import unittest

import numpy as np

from app.detection import Detection
from app.nlg import NLGConfig
from app.pipeline import ChangeGate, NarrationPipeline, scene_signature
from app.spatial import analyze_scene_spatial

DESK = Detection("desk", 0.95, 100, 200, 550, 450)
LAPTOP = Detection("laptop", 0.92, 200, 160, 380, 220)


def fake_detector(frame, conf_threshold=0.5):
    return [DESK, LAPTOP]


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = NarrationPipeline(detector=fake_detector, nlg_config=NLGConfig(seed=1))

    def test_narrate_frame_uses_frame_size_and_detector(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.pipeline.narrate_frame(frame)
        self.assertEqual((result.frame_width, result.frame_height), (640, 480))
        self.assertIn("laptop", result.narration.lower())
        self.assertIn("desk", result.narration.lower())

    def test_to_dict_hides_all_relations_by_default(self):
        result = self.pipeline.narrate_detections([DESK, LAPTOP], 640, 480)
        self.assertNotIn("all_relations", result.to_dict())
        self.assertIn("all_relations", result.to_dict(include_all_relations=True))

    def test_bad_image_bytes_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.pipeline.narrate_image_bytes(b"not an image")

    def test_empty_bytes_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.pipeline.narrate_image_bytes(b"")


class TestChangeGate(unittest.TestCase):
    def setUp(self):
        self.t = 0.0
        self.gate = ChangeGate(min_interval=3.0, clock=lambda: self.t)
        self.scene_a = analyze_scene_spatial([DESK, LAPTOP], 640, 480)
        self.scene_b = analyze_scene_spatial([Detection("traffic light", 0.9, 480, 30, 540, 150)], 640, 480)
        self.empty = analyze_scene_spatial([], 640, 480)

    def test_signature_is_order_independent(self):
        s1 = analyze_scene_spatial([DESK, LAPTOP], 640, 480)
        s2 = analyze_scene_spatial([LAPTOP, DESK], 640, 480)
        self.assertEqual(scene_signature(s1), scene_signature(s2))

    def test_speaks_first_scene_then_suppresses_repeat(self):
        self.assertTrue(self.gate.should_speak(self.scene_a))
        self.t = 10.0
        self.assertFalse(self.gate.should_speak(self.scene_a))  # unchanged

    def test_respects_min_interval_on_change(self):
        self.assertTrue(self.gate.should_speak(self.scene_a))
        self.t = 1.0
        self.assertFalse(self.gate.should_speak(self.scene_b))  # changed, too soon
        self.t = 4.0
        self.assertTrue(self.gate.should_speak(self.scene_b))   # changed, waited

    def test_repeat_after_renarrates_unchanged_scene(self):
        gate = ChangeGate(min_interval=3.0, clock=lambda: self.t, repeat_after=10.0)
        self.assertTrue(gate.should_speak(self.scene_a))
        self.t = 5.0
        self.assertFalse(gate.should_speak(self.scene_a))   # unchanged, not yet due
        self.t = 11.0
        self.assertTrue(gate.should_speak(self.scene_a))    # unchanged, repeat due

    def test_does_not_speak_empty_scene_at_startup(self):
        self.assertFalse(self.gate.should_speak(self.empty))


if __name__ == "__main__":
    unittest.main()


class TestStableGate(unittest.TestCase):
    def test_flicker_is_ignored_until_stable(self):
        t = [0.0]
        gate = ChangeGate(min_interval=0.0, clock=lambda: t[0], stable_frames=3)
        a = analyze_scene_spatial([DESK, LAPTOP], 640, 480)
        b = analyze_scene_spatial([Detection("traffic light", 0.9, 480, 30, 540, 150)], 640, 480)
        # a a b a a  -> 'b' appears for a single frame only: must not be announced
        results = [gate.should_speak(x) for x in (a, a, a, b, a, a)]
        self.assertEqual(results, [False, False, True, False, False, False])

    def test_no_repeat_by_default(self):
        t = [0.0]
        gate = ChangeGate(min_interval=1.0, clock=lambda: t[0])
        a = analyze_scene_spatial([DESK, LAPTOP], 640, 480)
        self.assertTrue(gate.should_speak(a))
        for i in range(1, 200):
            t[0] = i * 5.0
            self.assertFalse(gate.should_speak(a))


def test_scene_signature_changes_when_objects_move():
    a = analyze_scene_spatial([
        Detection("person", 0.9, 100, 100, 200, 300),
        Detection("dog", 0.9, 300, 100, 400, 300),
    ], 640, 480)
    b = analyze_scene_spatial([
        Detection("person", 0.9, 160, 100, 260, 300),
        Detection("dog", 0.9, 360, 100, 460, 300),
    ], 640, 480)
    assert scene_signature(a) != scene_signature(b)


def test_response_uses_deduplicated_detections():
    person1 = Detection("person", 0.95, 100, 50, 300, 450)
    person2 = Detection("person", 0.60, 110, 60, 290, 440)
    result = NarrationPipeline(
        detector=lambda frame, conf_threshold=0.5: [person1, person2]
    ).narrate_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    payload = result.to_dict()
    assert payload["num_detections"] == len(payload["detections"]) == 1


def test_empty_scene_does_not_reexpose_filtered_detections():
    low = Detection("cup", 0.1, 10, 10, 30, 30)
    result = NarrationPipeline(
        detector=lambda frame, conf_threshold=0.5: [low]
    ).narrate_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    payload = result.to_dict()
    assert payload["num_detections"] == 0
    assert payload["detections"] == []
