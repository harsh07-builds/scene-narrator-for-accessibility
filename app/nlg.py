"""
app/nlg.py — Natural Language Generation (Tier 1) for AI Scene Narrator
========================================================================

Stage 3 of the pipeline (see Manual, Section 2):

    [Stage 2] Spatial Reasoning (app/spatial.py)
        Output: Spatial Graph + Predicates (on, in front of, next to, ...)
            |
            v
    [Stage 3] NLG Tier 1 (app/nlg.py)   <-- THIS MODULE
        Output: Composed natural sentence with synonym diversification
            |
            v
    [Stage 4] Offline TTS (app/tts_engine.py)

This module takes the dict returned by `app.spatial.analyze_scene_spatial()`
and converts it into a spoken-ready narration string. It has no dependency
on the camera, YOLO, or pyttsx3 -- it consumes plain data and returns a
plain string, which is what makes it fully testable in isolation (see
tests/test_nlg.py) and safe to import from app/pipeline.py or
app/tts_engine.py without pulling in any heavier dependencies.

Design principles (matching the rest of the pipeline):
    - Deterministic building blocks (template lookup), not a generative
      model -- no hallucination risk, zero inference cost, fully offline.
    - Every possible sentence the system can ever produce is enumerable
      from PREDICATE_TEMPLATES / ZONE_TEMPLATES / EMPTY_SCENE_MESSAGES,
      which makes this module straightforward to QA exhaustively.
    - Fail-safe on unknown input: an unrecognized predicate is skipped,
      never guessed at or fabricated (mirrors app/spatial.py's own
      "confident relation or nothing" philosophy).

Python API
----------
    from app.nlg import generate_narration, NLGEngine, NLGConfig

    narration = generate_narration(scene)                 # one-off, module-level
    engine = NLGEngine(NLGConfig(max_relations_spoken=2))  # stateful, reusable
    narration = engine.generate_narration(scene)
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NLGConfig:
    """Tunable behavior for narration generation.

    Attributes:
        max_relations_spoken: Cap on how many relations get turned into
            sentences per narration cycle, independent of how many
            `analyze_scene_spatial()` already capped itself (its own
            `max_relations` controls the *reasoning* fan-out; this caps
            what's actually *spoken*, which can be tighter for usability).
        avoid_repeat_templates: If True, the engine won't pick the exact
            same phrasing for the same predicate twice in a row (reduces
            the "robotic repetition" failure mode of pure template systems).
        hedge_confidence_below: If > 0.0, relations with confidence below
            this value are phrased more tentatively ("X appears to be ...")
            rather than stated flatly. Set to 0.0 (default) to disable --
            matches the rest of the pipeline's default of not editorializing
            about its own certainty unless explicitly asked to.
        seed: Optional RNG seed, for reproducible output in tests/demos.
    """
    max_relations_spoken: int = 3
    avoid_repeat_templates: bool = True
    hedge_confidence_below: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_relations_spoken <= 0:
            raise ValueError("max_relations_spoken must be > 0")
        if not 0.0 <= self.hedge_confidence_below <= 1.0:
            raise ValueError("hedge_confidence_below must be in [0, 1]")


# ---------------------------------------------------------------------------
# Grammar helpers
# ---------------------------------------------------------------------------

# Words that start with a vowel LETTER but a consonant SOUND (need "a").
_CONSONANT_SOUND_EXCEPTIONS_A = {
    "university", "unicorn", "european", "user", "one", "uniform",
}
# Words that start with a consonant LETTER but a vowel SOUND (need "an").
_VOWEL_SOUND_EXCEPTIONS_AN = {
    "hour", "honest", "honor", "honour", "heir", "mba",
}


def article_for(label: str) -> str:
    """Returns 'a' or 'an' for a COCO-style class label.

    Only looks at the first word of multi-word labels (e.g. "dining table"
    -> "table" governs nothing here; COCO labels are looked at whole-word
    for the leading token, e.g. "traffic light" -> first word "traffic").
    """
    if not label:
        return "a"
    first_word = label.strip().split()[0].lower()
    if first_word in _VOWEL_SOUND_EXCEPTIONS_AN:
        return "an"
    if first_word in _CONSONANT_SOUND_EXCEPTIONS_A:
        return "a"
    return "an" if first_word[0] in "aeiou" else "a"


def indefinite(label: str) -> str:
    """e.g. 'chair' -> 'a chair', 'umbrella' -> 'an umbrella'."""
    return f"{article_for(label)} {label}"


_IRREGULAR_PLURALS = {
    "person": "people",
    "child": "children",
    "mouse": "mice",
    "knife": "knives",
    "scissors": "scissors",
}


def pluralize(label: str) -> str:
    """Small, deterministic pluralizer for COCO-style labels."""
    words = label.strip().split()
    if not words:
        return label
    last = words[-1].lower()
    plural = _IRREGULAR_PLURALS.get(last)
    if plural is None:
        if last.endswith(("s", "x", "z", "ch", "sh")):
            plural = last + "es"
        elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
            plural = last[:-1] + "ies"
        else:
            plural = last + "s"
    words[-1] = plural
    return " ".join(words)


def definite(label: str) -> str:
    """e.g. 'chair' -> 'the chair'."""
    return f"the {label}"


def capitalize_first(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


# ---------------------------------------------------------------------------
# Template banks
# ---------------------------------------------------------------------------
# Keys MUST exactly match the predicate strings produced by app/spatial.py's
# infer_relation(): "on", "inside", "holding", "next to", "in front of",
# "to the left of", "to the right of". {subject}/{object} are filled with
# already-articled phrases (indefinite/definite), so templates should NOT
# add their own "a"/"the".

PREDICATE_TEMPLATES: Dict[str, List[str]] = {
    "on": [
        "{subject} is on {object}.",
        "{subject} is resting on {object}.",
        "{subject} sits on top of {object}.",
    ],
    "holding": [
        "{subject} appears to be holding {object}.",
        "{subject} seems to be holding {object}.",
    ],
    "inside": [
        "{subject} is inside {object}.",
        "{subject} is contained within {object}.",
        "{subject} is placed inside {object}.",
    ],
    "in front of": [
        "{subject} is in front of {object}.",
        "{subject} is ahead of {object}.",
        "{subject} is positioned in front of {object}.",
    ],
    "next to": [
        "{subject} is next to {object}.",
        "{subject} is beside {object}.",
        "{subject} is by {object}.",
    ],
    "to the left of": [
        "{subject} is to the left of {object}.",
        "{subject} is on the left side of {object}.",
    ],
    "to the right of": [
        "{subject} is to the right of {object}.",
        "{subject} is on the right side of {object}.",
    ],
}

# One tentative phrasing per predicate, used only when a relation's
# confidence falls below NLGConfig.hedge_confidence_below.
PREDICATE_TEMPLATES_HEDGED: Dict[str, str] = {
    "on": "{subject} appears to be on {object}.",
    "holding": "{subject} may be holding {object}.",
    "inside": "{subject} appears to be inside {object}.",
    "in front of": "{subject} appears to be in front of {object}.",
    "next to": "{subject} appears to be next to {object}.",
    "to the left of": "{subject} appears to be to the left of {object}.",
    "to the right of": "{subject} appears to be to the right of {object}.",
}

# {subject} = "a chair" (indefinite), {Subject} = "A chair" (capitalized),
# {zone} = a phrase already produced by app.spatial.frame_zone(), e.g.
# "in the center", "on the left", "in the top-right".
ZONE_TEMPLATES: List[str] = [
    "There is {subject} {zone}.",
    "{Subject} is {zone}.",
    "I can see {subject} {zone}.",
]

EMPTY_SCENE_MESSAGES: List[str] = [
    "No objects detected.",
    "I don't see anything recognizable right now.",
    "The scene appears to be empty.",
]

# Used only if the scene has confident detections but zero confident
# relations between them (e.g. two objects too far apart / too ambiguous
# for spatial.py to link) -- fail-safe fallback so the user still hears
# *something* useful rather than silence.
NO_RELATION_FALLBACK_TEMPLATES: List[str] = [
    "I can see {items}.",
    "In view: {items}.",
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class NLGEngine:
    """Stateful narration engine.

    Statefulness is limited to *phrasing diversification* (remembering the
    last template used per predicate/zone so consecutive narrations don't
    sound identical) -- it does NOT track scene history or decide whether
    to re-narrate an unchanged scene; that throttling decision belongs to
    app/pipeline.py or app/tts_engine.py, one layer up, per the pipeline's
    existing separation of concerns.
    """
    config: NLGConfig = field(default_factory=NLGConfig)
    _rng: random.Random = field(init=False, repr=False)
    _last_template_by_key: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed)

    # -- internal ------------------------------------------------------

    def _pick_template(self, key: str, templates: List[str]) -> str:
        if not templates:
            raise ValueError("Template list must not be empty.")
        if not self.config.avoid_repeat_templates or len(templates) == 1:
            return self._rng.choice(templates)

        last = self._last_template_by_key.get(key)
        choices = [t for t in templates if t != last] or templates
        chosen = self._rng.choice(choices)
        self._last_template_by_key[key] = chosen
        return chosen

    # -- per-unit rendering ---------------------------------------------

    def render_relation(self, relation: Dict[str, Any]) -> Optional[str]:
        """Renders one relation dict (as produced in
        `scene["primary_relations"]`) into a sentence.

        Returns None if the predicate is unrecognized or the relation is
        missing required fields -- callers should skip narrating it rather
        than treat None as an error.
        """
        predicate = relation.get("predicate")
        subject_label = relation.get("subject")
        object_label = relation.get("object")
        try:
            confidence = float(relation.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        if not predicate or not subject_label or not object_label:
            return None

        use_hedge = (
            self.config.hedge_confidence_below > 0.0
            and confidence < self.config.hedge_confidence_below
        )

        if use_hedge:
            template = PREDICATE_TEMPLATES_HEDGED.get(predicate)
            if template is None:
                return None
        else:
            templates = PREDICATE_TEMPLATES.get(predicate)
            if not templates:
                return None
            template = self._pick_template(f"predicate::{predicate}", templates)

        if str(subject_label).strip().lower() == str(object_label).strip().lower():
            subject_phrase = f"one {subject_label}"
            object_phrase = f"the other {object_label}"
        else:
            subject_phrase = capitalize_first(indefinite(subject_label))
            object_phrase = definite(object_label)

        return template.format(subject=subject_phrase, object=object_phrase)

    def render_zone(self, zone_entry: Dict[str, Any]) -> Optional[str]:
        """Renders a single-object `frame_zones` entry, e.g.
        {"label": "chair", "zone": "in the center", "confidence": 0.9}.
        """
        label = zone_entry.get("label")
        zone = zone_entry.get("zone")
        if not label or not zone:
            return None

        template = self._pick_template(f"zone::{zone}", ZONE_TEMPLATES)
        subj = indefinite(label)
        return template.format(subject=subj, Subject=capitalize_first(subj), zone=zone)

    def render_empty(self) -> str:
        return self._pick_template("empty", EMPTY_SCENE_MESSAGES)

    def render_no_relation_fallback(self, labels: List[str]) -> str:
        counts: Dict[str, int] = {}
        display: Dict[str, str] = {}
        for lbl in labels:
            key = str(lbl).strip().lower()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            display[key] = str(lbl).strip()

        parts: List[str] = []
        for key in sorted(counts):
            label = display[key]
            count = counts[key]
            if count == 1:
                parts.append(indefinite(label))
            elif count == 2:
                parts.append(f"two {pluralize(label)}")
            else:
                parts.append(f"{count} {pluralize(label)}")

        if len(parts) > 1:
            items = ", ".join(parts[:-1]) + " and " + parts[-1]
        else:
            items = parts[0] if parts else "some objects"
        template = self._pick_template("no_relation_fallback", NO_RELATION_FALLBACK_TEMPLATES)
        return template.format(items=items)

    # -- top-level entry point ------------------------------------------

    def generate_narration(self, scene: Dict[str, Any]) -> str:
        """Thread-safe wrapper (the API serves requests concurrently and the
        engine holds RNG / last-template state)."""
        with self._lock:
            return self._generate(scene)

    def _generate(self, scene: Dict[str, Any]) -> str:
        """Main entry point: `app.spatial.analyze_scene_spatial()` output
        in, a spoken-ready narration string out.

        Handles all three scene modes that analyze_scene_spatial() can
        produce: "empty", "single_object", "multi_object" -- plus the
        edge case of confident detections with zero confident relations
        between them.
        """
        scene_mode = scene.get("scene_mode")
        num_detections = scene.get("num_detections", 0)

        if scene_mode == "empty" or num_detections == 0:
            return self.render_empty()

        if scene_mode == "single_object":
            zones = scene.get("frame_zones") or []
            if not zones:
                return self.render_empty()
            sentence = self.render_zone(zones[0])
            return sentence if sentence else self.render_empty()

        # multi_object
        relations = scene.get("primary_relations", [])
        capped = relations[: self.config.max_relations_spoken]

        sentences: List[str] = []
        for relation in capped:
            sentence = self.render_relation(relation)
            if sentence:
                sentences.append(sentence)

        if sentences:
            return " ".join(sentences)

        # Confident detections exist, but spatial.py found no confident
        # relations to report (e.g. everything too far apart). Fall back
        # to naming what's in view rather than saying nothing at all.
        labels = [z["label"] for z in scene.get("frame_zones", []) if "label" in z]
        if labels:
            return self.render_no_relation_fallback(labels)

        return self.render_empty()


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------
# Mirrors the pattern used in app/detection.py (a lazily-created shared
# instance plus a plain function), so app/pipeline.py can do:
#
#     from app.nlg import generate_narration
#     sentence = generate_narration(scene)
#
# without having to manage an NLGEngine itself.

_default_engine: Optional[NLGEngine] = None
_default_engine_lock = threading.Lock()


def get_default_engine() -> NLGEngine:
    global _default_engine
    with _default_engine_lock:
        if _default_engine is None:
            _default_engine = NLGEngine()
        return _default_engine


def generate_narration(scene: Dict[str, Any], config: Optional[NLGConfig] = None) -> str:
    """Functional convenience wrapper.

    Pass `config` to use a fresh one-off NLGEngine with custom settings;
    omit it to reuse the shared default engine (recommended for the real
    pipeline, so template-repetition avoidance persists across frames).
    """
    engine = NLGEngine(config) if config is not None else get_default_engine()
    return engine.generate_narration(scene)


# ---------------------------------------------------------------------------
# __main__ — standalone demo (no camera / YOLO required)
# ---------------------------------------------------------------------------
# Reuses the exact synthetic scenarios from run_demo.py so the two demos
# are directly comparable: run_demo.py prints spatial.py's own bare
# `structured_narration`; this prints what app/nlg.py produces from the
# same underlying data, including synonym variation across repeats.

if __name__ == "__main__":
    from app.detection import Detection
    from app.spatial import analyze_scene_spatial

    print("=" * 60)
    print("AI SCENE NARRATOR — NLG TIER 1 DEMO")
    print("=" * 60)

    engine = NLGEngine(NLGConfig(seed=7))

    print("\n--- Scenario 1: Desk Setup ---")
    desk = Detection("desk", 0.95, 100, 200, 550, 450)
    laptop = Detection("laptop", 0.92, 200, 160, 380, 220)
    mouse = Detection("mouse", 0.88, 400, 200, 440, 230)
    cup = Detection("cup", 0.85, 130, 150, 170, 210)
    scene1 = analyze_scene_spatial([desk, laptop, mouse, cup], 640, 480)
    print("Spatial structured_narration:", scene1["structured_narration"])
    print("NLG narration:               ", engine.generate_narration(scene1))

    print("\n--- Scenario 2: Living Room (Depth) ---")
    sofa = Detection("couch", 0.91, 120, 160, 520, 360)
    person = Detection("person", 0.97, 240, 180, 400, 460)
    scene2 = analyze_scene_spatial([sofa, person], 640, 480)
    print("Spatial structured_narration:", scene2["structured_narration"])
    print("NLG narration:               ", engine.generate_narration(scene2))

    print("\n--- Scenario 3: Single Object (Frame Zone) ---")
    traffic_light = Detection("traffic light", 0.89, 480, 30, 540, 150)
    scene3 = analyze_scene_spatial([traffic_light], 640, 480)
    print("Spatial structured_narration:", scene3["structured_narration"])
    print("NLG narration:               ", engine.generate_narration(scene3))

    print("\n--- Scenario 4: Empty Scene ---")
    scene4 = analyze_scene_spatial([], 640, 480)
    print("NLG narration:               ", engine.generate_narration(scene4))

    print("\n--- Scenario 5: Synonym diversification across repeats ---")
    for _ in range(4):
        scene = analyze_scene_spatial([desk, laptop], 640, 480)
        print(" ", engine.generate_narration(scene))

    print("\n" + "=" * 60)
    print("NLG demo completed successfully.")
    print("=" * 60)
