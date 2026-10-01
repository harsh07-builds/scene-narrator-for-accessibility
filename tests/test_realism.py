"""Regression tests: narration must stay physically plausible."""

import random
import re
import unittest

from app.detection import Detection as D
from app.nlg import NLGConfig, generate_narration
from app.spatial import CONTAINER_CLASSES, analyze_scene_spatial, suppress_duplicates

PERSON = D("person", 0.90, 100, 50, 300, 450)


def preds(dets):
    return [(r["subject"], r["predicate"], r["object"]) for r in analyze_scene_spatial(dets, 640, 480)["primary_relations"]]


class TestRealism(unittest.TestCase):
    def test_bottle_in_front_of_person_is_holding_not_inside(self):
        p = preds([PERSON, D("bottle", 0.7, 180, 200, 220, 290)])
        self.assertEqual(p, [("person", "holding", "bottle")])

    def test_person_is_never_inside_anything(self):
        car = D("car", 0.9, 50, 20, 600, 470)
        self.assertNotIn("inside", [x[1] for x in preds([D("person", 0.9, 200, 100, 300, 400), car])])

    def test_nothing_is_inside_a_person(self):
        self.assertNotIn("inside", [x[1] for x in preds([PERSON, D("dog", 0.8, 150, 300, 250, 420)])])

    def test_duplicate_person_boxes_merge(self):
        scene = analyze_scene_spatial([PERSON, D("person", 0.6, 110, 60, 290, 440)], 640, 480)
        self.assertEqual(scene["num_detections"], 1)
        self.assertEqual(scene["scene_mode"], "single_object")

    def test_nested_same_label_box_is_dropped(self):
        kept = suppress_duplicates([PERSON, D("person", 0.5, 150, 100, 250, 300)])
        self.assertEqual(len(kept), 1)

    def test_two_distinct_people_are_kept(self):
        kept = suppress_duplicates([PERSON, D("person", 0.9, 400, 60, 560, 440)])
        self.assertEqual(len(kept), 2)

    def test_real_container_still_works(self):
        p = preds([D("bowl", 0.9, 100, 100, 300, 300), D("apple", 0.85, 150, 150, 220, 220)])
        self.assertEqual(p, [("apple", "inside", "bowl")])

    def test_laptop_on_desk_still_works(self):
        p = preds([D("desk", 0.95, 100, 200, 550, 450), D("laptop", 0.92, 200, 160, 380, 220)])
        self.assertEqual(p, [("laptop", "on", "desk")])

    def test_big_object_is_not_on_small_object(self):
        p = preds([D("tv", 0.9, 100, 100, 400, 260), D("cell phone", 0.9, 180, 250, 260, 330)])
        self.assertNotIn(("tv", "on", "cell phone"), p)

    def test_person_can_sit_on_smaller_chair(self):
        p = preds([D("chair", 0.9, 150, 250, 350, 430), D("person", 0.9, 160, 60, 340, 300)])
        self.assertIn(("person", "on", "chair"), p)

    def test_two_people_side_by_side(self):
        p = preds([D("person", 0.9, 50, 100, 200, 440), D("person", 0.9, 260, 110, 410, 445)])
        self.assertIn("next to", p[0][1])


LABELS = ["person", "bottle", "cup", "chair", "couch", "laptop", "dining table", "bowl",
          "backpack", "car", "dog", "book", "tv", "bed", "cell phone", "potted plant"]
BAD = [
    re.compile(r"\b(?:a|an|the) (\w[\w ]*?) (?:is|are) (?:\w+ )*?(?:inside|within|on|on top of) (?:the|a|an) \1\b", re.I),
    re.compile(r"inside the person|within the person|on the person", re.I),
    re.compile(r"person (?:is|appears to be|seems to be) (?:\w+ )*?(?:inside|within) ", re.I),
]


class TestFuzz(unittest.TestCase):
    def test_no_impossible_sentences_in_random_scenes(self):
        rng = random.Random(1234)
        for _ in range(3000):
            dets = []
            for _ in range(rng.randint(2, 7)):
                x1, y1 = rng.randint(0, 500), rng.randint(0, 380)
                dets.append(D(rng.choice(LABELS), rng.uniform(0.5, 0.99),
                              x1, y1, x1 + rng.randint(20, 300), y1 + rng.randint(20, 300)))
            scene = analyze_scene_spatial(dets, 640, 480)
            for rel in scene["primary_relations"]:
                if rel["predicate"] == "inside":
                    self.assertIn(rel["object"], CONTAINER_CLASSES)
                    self.assertNotEqual(rel["subject"], "person")
                    self.assertNotEqual(rel["subject"], rel["object"])
            text = generate_narration(scene, NLGConfig(seed=1))
            for pat in BAD:
                self.assertIsNone(pat.search(text), f"{text!r} matched {pat.pattern}")


if __name__ == "__main__":
    unittest.main()


    def test_overlapping_same_label_objects_are_not_aggressively_merged(self):
        # Two partially occluding people; overlap is meaningful, not a duplicate.
        a = D("person", 0.95, 100, 50, 300, 450)
        b = D("person", 0.90, 210, 70, 410, 460)
        kept = suppress_duplicates([a, b])
        self.assertEqual(len(kept), 2)
