"""
Unit tests for the Spatial Reasoning Module.
Tests all predicates (on, next to, in front of, inside, frame_zone, directional)
and multi-object scene graph generation.
"""

import unittest
from app.detection import Detection
from app.spatial import (
    SpatialConfig,
    frame_zone,
    infer_relation,
    analyze_scene_spatial
)


class TestSpatialReasoning(unittest.TestCase):

    def setUp(self):
        self.config = SpatialConfig()

    def test_frame_zone(self):
        # 640 x 480 frame
        # Center object
        d_center = Detection("person", 0.9, 280, 200, 360, 280)
        self.assertEqual(frame_zone(d_center, 640, 480), "in the center")

        # Top-left object
        d_tl = Detection("clock", 0.85, 20, 20, 80, 80)
        self.assertEqual(frame_zone(d_tl, 640, 480), "in the top-left")

        # Bottom-right object
        d_br = Detection("chair", 0.8, 500, 380, 600, 460)
        self.assertEqual(frame_zone(d_br, 640, 480), "in the bottom-right")

        # Left-center object
        d_left = Detection("bottle", 0.8, 50, 200, 100, 280)
        self.assertEqual(frame_zone(d_left, 640, 480), "on the left")

        # Right-center object
        d_right = Detection("bottle", 0.8, 540, 200, 590, 280)
        self.assertEqual(frame_zone(d_right, 640, 480), "on the right")

    def test_predicate_on(self):
        # A table and a cup resting on it
        # Table: x: 100 to 400, y: 250 to 450 (width: 300, height: 200)
        table = Detection("dining table", 0.95, 100, 250, 400, 450)
        # Cup: x: 200 to 260, y: 190 to 255 (width: 60, height: 65, resting right on table surface at y=250)
        cup = Detection("cup", 0.90, 200, 190, 260, 255)

        rel = infer_relation(cup, table, self.config)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.predicate, "on")
        self.assertEqual(rel.subject.label, "cup")
        self.assertEqual(rel.object.label, "dining table")
        self.assertGreaterEqual(rel.salience, 0.9)

    def test_predicate_next_to(self):
        # Two chairs beside each other
        # Chair 1: x: 100 to 200, y: 200 to 400
        chair1 = Detection("chair", 0.88, 100, 200, 200, 400)
        # Chair 2: x: 230 to 330, y: 210 to 410 (gap: 30px, well within width of 100)
        chair2 = Detection("chair", 0.85, 230, 210, 330, 410)

        rel = infer_relation(chair1, chair2, self.config)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.predicate, "next to")

    def test_predicate_in_front_of(self):
        # Person standing in front of a sofa:
        # Sofa further back: y: 150 to 350, x: 150 to 500, area: 350 * 200 = 70,000
        sofa = Detection("couch", 0.92, 150, 150, 500, 350)
        # Person standing closer to camera: lower base (y2 = 460 > 350), overlaps horizontally
        person = Detection("person", 0.96, 250, 180, 400, 460)

        rel = infer_relation(person, sofa, self.config)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.predicate, "in front of")

    def test_predicate_inside(self):
        # An apple inside a bowl
        bowl = Detection("bowl", 0.9, 100, 100, 300, 300)
        apple = Detection("apple", 0.85, 150, 150, 220, 220)

        rel = infer_relation(apple, bowl, self.config)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.predicate, "inside")

    def test_directional_fallback(self):
        # Two distant objects not vertically aligned or close
        lamp = Detection("lamp", 0.8, 50, 50, 100, 150)
        dog = Detection("dog", 0.8, 500, 350, 600, 450)

        rel = infer_relation(lamp, dog, self.config)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.predicate, "to the left of")

    def test_scene_analysis_single_object(self):
        dets = [Detection("laptop", 0.9, 200, 150, 440, 330)]
        result = analyze_scene_spatial(dets, 640, 480, self.config)
        self.assertEqual(result["scene_mode"], "single_object")
        self.assertEqual(result["num_detections"], 1)
        self.assertIn("in the center", result["structured_narration"])

    def test_scene_analysis_multi_object(self):
        # Table with a cup on it and a chair next to it
        table = Detection("dining table", 0.95, 200, 250, 500, 450)
        cup = Detection("cup", 0.90, 250, 195, 310, 255)
        chair = Detection("chair", 0.85, 80, 220, 180, 430)

        result = analyze_scene_spatial([table, cup, chair], 640, 480, self.config)
        self.assertEqual(result["scene_mode"], "multi_object")
        self.assertEqual(result["num_detections"], 3)
        self.assertGreaterEqual(len(result["primary_relations"]), 2)

        predicates = [r["predicate"] for r in result["primary_relations"]]
        self.assertIn("on", predicates)
        self.assertTrue("next to" in predicates or "to the right of" in predicates)


if __name__ == "__main__":
    unittest.main()


def test_overlapping_different_classes_are_not_deleted_as_duplicates():
    from app.spatial import suppress_duplicates
    person = Detection("person", 0.95, 100, 50, 500, 450)
    couch = Detection("couch", 0.90, 120, 60, 480, 440)
    kept = suppress_duplicates([person, couch])
    assert {d.label for d in kept} == {"person", "couch"}


def test_near_vertical_objects_do_not_get_forced_left_right_relation():
    cup = Detection("cup", 0.9, 280, 100, 320, 140)
    plate = Detection("plate", 0.9, 281, 250, 321, 290)
    assert infer_relation(cup, plate) is None or infer_relation(cup, plate).predicate != "to the right of"
    assert infer_relation(plate, cup) is None or infer_relation(plate, cup).predicate != "to the left of"


def test_relation_confidence_never_exceeds_weak_detection():
    table = Detection("dining table", 0.95, 100, 250, 400, 450)
    weak_cup = Detection("cup", 0.31, 200, 190, 260, 255)
    rel = infer_relation(weak_cup, table)
    assert rel is not None
    assert rel.confidence <= weak_cup.confidence
