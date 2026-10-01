"""
Unit tests for the NLG (Natural Language Generation) module.
Mirrors the style of tests/test_spatial.py.
"""

import unittest

from app.detection import Detection
from app.spatial import analyze_scene_spatial
from app.nlg import (
    NLGConfig,
    NLGEngine,
    article_for,
    indefinite,
    definite,
    generate_narration,
    PREDICATE_TEMPLATES,
    PREDICATE_TEMPLATES_HEDGED,
)


class TestGrammarHelpers(unittest.TestCase):

    def test_article_basic_consonant(self):
        self.assertEqual(article_for("chair"), "a")

    def test_article_basic_vowel(self):
        self.assertEqual(article_for("umbrella"), "an")

    def test_article_consonant_sound_exception(self):
        # "university" starts with a vowel letter but a consonant sound.
        self.assertEqual(article_for("university"), "a")

    def test_article_vowel_sound_exception(self):
        # "hour" starts with a consonant letter but a vowel sound.
        self.assertEqual(article_for("hour"), "an")

    def test_article_multiword_label(self):
        # Only the leading word governs the article.
        self.assertEqual(article_for("traffic light"), "a")

    def test_indefinite_and_definite(self):
        self.assertEqual(indefinite("chair"), "a chair")
        self.assertEqual(indefinite("umbrella"), "an umbrella")
        self.assertEqual(definite("desk"), "the desk")


class TestRelationRendering(unittest.TestCase):

    def setUp(self):
        self.engine = NLGEngine(NLGConfig(seed=42))

    def test_all_predicates_have_templates(self):
        # Every predicate app/spatial.py can produce must be renderable.
        known_predicates = {
            "on", "inside", "next to", "in front of",
            "to the left of", "to the right of",
        }
        self.assertTrue(known_predicates.issubset(PREDICATE_TEMPLATES.keys()))
        self.assertTrue(known_predicates.issubset(PREDICATE_TEMPLATES_HEDGED.keys()))

    def test_render_relation_basic(self):
        relation = {"subject": "cup", "predicate": "on", "object": "desk", "confidence": 0.9}
        sentence = self.engine.render_relation(relation)
        self.assertIsNotNone(sentence)
        self.assertIn("cup", sentence.lower())
        self.assertIn("desk", sentence.lower())
        self.assertTrue(sentence.endswith("."))

    def test_render_relation_unknown_predicate_returns_none(self):
        relation = {"subject": "cup", "predicate": "orbiting", "object": "desk", "confidence": 0.9}
        self.assertIsNone(self.engine.render_relation(relation))

    def test_render_relation_missing_fields_returns_none(self):
        self.assertIsNone(self.engine.render_relation({"predicate": "on"}))

    def test_hedging_below_threshold(self):
        engine = NLGEngine(NLGConfig(seed=1, hedge_confidence_below=0.5))
        relation = {"subject": "cup", "predicate": "on", "object": "desk", "confidence": 0.2}
        sentence = engine.render_relation(relation)
        self.assertIn("appears to be", sentence)

    def test_no_hedging_above_threshold(self):
        engine = NLGEngine(NLGConfig(seed=1, hedge_confidence_below=0.5))
        relation = {"subject": "cup", "predicate": "on", "object": "desk", "confidence": 0.9}
        sentence = engine.render_relation(relation)
        self.assertNotIn("appears to be", sentence)

    def test_avoid_immediate_template_repeat(self):
        engine = NLGEngine(NLGConfig(seed=0, avoid_repeat_templates=True))
        relation = {"subject": "cup", "predicate": "on", "object": "desk", "confidence": 0.9}
        first = engine.render_relation(relation)
        second = engine.render_relation(relation)
        self.assertNotEqual(first, second)


class TestZoneAndEmptyRendering(unittest.TestCase):

    def setUp(self):
        self.engine = NLGEngine(NLGConfig(seed=3))

    def test_render_zone(self):
        zone_entry = {"label": "chair", "zone": "in the center", "confidence": 0.9}
        sentence = self.engine.render_zone(zone_entry)
        self.assertIsNotNone(sentence)
        self.assertIn("chair", sentence.lower())
        self.assertIn("in the center", sentence)

    def test_render_empty(self):
        sentence = self.engine.render_empty()
        self.assertIsInstance(sentence, str)
        self.assertGreater(len(sentence), 0)


class TestFullSceneIntegration(unittest.TestCase):
    """Integration tests against the real app/spatial.py + app/detection.py."""

    def test_multi_object_scene_produces_narration(self):
        desk = Detection("desk", 0.95, 100, 200, 550, 450)
        laptop = Detection("laptop", 0.92, 200, 160, 380, 220)
        scene = analyze_scene_spatial([desk, laptop], 640, 480)

        narration = generate_narration(scene, NLGConfig(seed=5))
        self.assertIsInstance(narration, str)
        self.assertIn("laptop", narration.lower())
        self.assertIn("desk", narration.lower())

    def test_single_object_scene_produces_zone_narration(self):
        traffic_light = Detection("traffic light", 0.89, 480, 30, 540, 150)
        scene = analyze_scene_spatial([traffic_light], 640, 480)

        narration = generate_narration(scene, NLGConfig(seed=5))
        self.assertIn("traffic light", narration.lower())

    def test_empty_scene_produces_empty_narration(self):
        scene = analyze_scene_spatial([], 640, 480)
        narration = generate_narration(scene, NLGConfig(seed=5))
        self.assertIsInstance(narration, str)
        self.assertGreater(len(narration), 0)

    def test_max_relations_spoken_is_respected(self):
        # Five objects on a desk -> spatial.py may surface several "on"
        # relations; NLG should cap how many are actually spoken.
        desk = Detection("desk", 0.95, 50, 200, 600, 460)
        items = [
            Detection("cup", 0.9, 80, 150, 130, 210),
            Detection("laptop", 0.9, 160, 160, 340, 220),
            Detection("mouse", 0.9, 360, 200, 400, 230),
            Detection("book", 0.9, 420, 170, 500, 220),
            Detection("phone", 0.9, 520, 190, 560, 230),
        ]
        scene = analyze_scene_spatial([desk] + items, 640, 480, max_relations=5)

        config = NLGConfig(seed=5, max_relations_spoken=2)
        narration = generate_narration(scene, config)
        sentence_count = narration.count(".")
        self.assertLessEqual(sentence_count, 2)


if __name__ == "__main__":
    unittest.main()
