"""
Demo: runs synthetic scenes through spatial reasoning AND the NLG layer.
No camera or YOLO needed.

    python -m app.demo
"""

from __future__ import annotations

from app.detection import Detection
from app.nlg import NLGConfig, NLGEngine
from app.spatial import analyze_scene_spatial


def _show(title, dets, engine, w=640, h=480):
    print(f"\n--- {title} ---")
    scene = analyze_scene_spatial(dets, w, h)
    print("Objects:   ", [d.label for d in dets])
    for r in scene["primary_relations"]:
        print(f"  * {r['subject']} -> [{r['predicate']}] -> {r['object']} (confidence: {r['confidence']})")
    print("Spatial:   ", scene["structured_narration"])
    print("Narration: ", engine.generate_narration(scene))


def run_demo() -> None:
    print("=" * 60)
    print("AI SCENE NARRATOR — END-TO-END DEMO (synthetic detections)")
    print("=" * 60)
    engine = NLGEngine(NLGConfig(seed=7))

    desk = Detection("desk", 0.95, 100, 200, 550, 450)
    laptop = Detection("laptop", 0.92, 200, 160, 380, 220)
    mouse = Detection("mouse", 0.88, 400, 200, 440, 230)
    cup = Detection("cup", 0.85, 130, 150, 170, 210)
    _show("Scenario 1: Desk setup (support + proximity)", [desk, laptop, mouse, cup], engine)

    sofa = Detection("couch", 0.91, 120, 160, 520, 360)
    person = Detection("person", 0.97, 240, 180, 400, 460)
    _show("Scenario 2: Living room (depth)", [sofa, person], engine)

    _show("Scenario 3: Single object (frame zone)", [Detection("traffic light", 0.89, 480, 30, 540, 150)], engine)
    _show("Scenario 4: Empty scene", [], engine)

    print("\n" + "=" * 60)
    print("Demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
