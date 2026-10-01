"""
Spatial Reasoning Module for AI Scene Narrator.

Implements pure 2D geometric heuristics to infer spatial predicates:
- "on" (support relationship)
- "in front of" / "behind" (depth proxy using box size and base-line ground position)
- "next to" / "beside" (horizontal proximity with vertical alignment)
- "to the left of" / "to the right of" (directional ordering fallback)
- "inside" (containment, only for real container classes)
- "holding" (person + hand-held object)
- frame_zone() for single-object positioning (e.g. center, left, right, top, bottom)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from app.detection import Detection


@dataclass
class SpatialConfig:
    """Configurable geometric thresholds for spatial predicate classification."""

    # "on" predicate thresholds
    on_overlap_threshold: float = 0.45       # Horizontal overlap ratio relative to subject width
    on_vertical_top_margin: float = 0.35     # Max penetration into object's top (relative to object height)
    on_vertical_gap_margin: float = 0.20     # Max gap above object's top (relative to subject height)
    on_max_area_ratio: float = 1.8           # Subject shouldn't be drastically larger than support object

    # "in front of" / depth proxy thresholds
    depth_size_ratio: float = 1.6            # Closer object appears significantly larger
    depth_bottom_offset: float = 0.06        # Base line lower in frame (relative to frame/box height)
    depth_h_overlap_min: float = 0.15        # Minimum horizontal overlap to consider line-of-sight depth

    # "next to" proximity thresholds
    next_to_gap_ratio: float = 1.25          # Max horizontal gap relative to min(subject.w, object.w)
    next_to_v_overlap_ratio: float = 0.30    # Min vertical overlap ratio relative to min(subject.h, object.h)
    next_to_cy_diff_ratio: float = 0.50      # Max centroid y difference relative to max height
    directional_min_center_delta_ratio: float = 0.35  # Ignore near-vertical/ambiguous left-right claims

    # Safety cap for raw pairwise graph generation in debug mode
    max_pairwise_relations: int = 5000

    # "inside" containment threshold
    inside_iou_sub_threshold: float = 0.82   # Portion of subject inside object

    # Minimum detection confidence
    min_confidence: float = 0.30

    def __post_init__(self) -> None:
        numeric = {
            "on_overlap_threshold": self.on_overlap_threshold,
            "on_vertical_top_margin": self.on_vertical_top_margin,
            "on_vertical_gap_margin": self.on_vertical_gap_margin,
            "on_max_area_ratio": self.on_max_area_ratio,
            "depth_size_ratio": self.depth_size_ratio,
            "depth_bottom_offset": self.depth_bottom_offset,
            "depth_h_overlap_min": self.depth_h_overlap_min,
            "next_to_gap_ratio": self.next_to_gap_ratio,
            "next_to_v_overlap_ratio": self.next_to_v_overlap_ratio,
            "next_to_cy_diff_ratio": self.next_to_cy_diff_ratio,
            "directional_min_center_delta_ratio": self.directional_min_center_delta_ratio,
            "inside_iou_sub_threshold": self.inside_iou_sub_threshold,
            "min_confidence": self.min_confidence,
        }
        if any(not math.isfinite(float(v)) for v in numeric.values()):
            raise ValueError("SpatialConfig thresholds must be finite")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.max_pairwise_relations <= 0:
            raise ValueError("max_pairwise_relations must be > 0")
        if self.directional_min_center_delta_ratio < 0.0:
            raise ValueError("directional_min_center_delta_ratio must be >= 0")


@dataclass
class SpatialRelation:
    """Represents an inferred spatial relationship between two objects."""
    subject: Detection
    predicate: str
    object: Detection
    confidence: float
    salience: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject.label,
            "predicate": self.predicate,
            "object": self.object.label,
            "confidence": round(float(self.confidence), 2),
            "salience": round(float(self.salience), 2),
            "subject_bbox": [round(float(v), 2) for v in (self.subject.x1, self.subject.y1, self.subject.x2, self.subject.y2)],
            "object_bbox": [round(float(v), 2) for v in (self.object.x1, self.object.y1, self.object.x2, self.object.y2)],
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        return f"{self.subject.label} is {self.predicate} {self.object.label}"


# ---------------------------------------------------------------------------
# Plausibility knowledge (COCO class names) — keeps narration physically sensible
# ---------------------------------------------------------------------------

# Things that can genuinely CONTAIN other objects. Anything else "inside" a box
# is really just overlapping in 2D (a bottle held in front of a person).
CONTAINER_CLASSES = {
    "bowl", "cup", "vase", "backpack", "handbag", "suitcase",
    "refrigerator", "oven", "microwave", "sink", "toilet",
}

# Small things a person can plausibly hold / carry.
HELD_CLASSES = {
    "bottle", "cup", "wine glass", "cell phone", "book", "remote", "umbrella",
    "handbag", "backpack", "suitcase", "knife", "fork", "spoon", "toothbrush",
    "scissors", "laptop", "frisbee", "sports ball", "tennis racket",
    "baseball bat", "skateboard", "kite", "banana", "apple", "sandwich",
    "donut", "hot dog", "teddy bear", "hair drier",
}


# Only these may rest ON something smaller than themselves (a person sits on a
# chair, a cat on a stool). A TV never "sits on" a phone.
SITTER_CLASSES = {"person", "cat", "dog", "bird", "teddy bear"}


def suppress_duplicates(
    detections: List[Detection],
    same_label_iou: float = 0.70,
    cross_label_iou: float = 0.85,
    same_label_containment: float = 0.90,
) -> List[Detection]:
    """Drops duplicate boxes YOLO sometimes emits for one physical object.

    Keeps the highest-confidence box and discards another when:
    - same label and IoU >= same_label_iou, or
    - same label and the smaller box lies (almost) entirely inside the larger.

    The intentionally conservative defaults avoid deleting legitimately overlapping
    people/animals while still removing near-identical duplicate model outputs.
    - different labels are preserved even at high IoU because overlapping classes
      can represent separate physical objects. `cross_label_iou` is retained only
      for backwards-compatible call signatures.
    """
    kept: List[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        duplicate = False
        for k in kept:
            iou = calculate_iou(det, k)
            if det.label == k.label:
                inter = (
                    calculate_overlap_1d(det.x1, det.x2, k.x1, k.x2)
                    * calculate_overlap_1d(det.y1, det.y2, k.y1, k.y2)
                )
                smaller = min(det.area, k.area)
                contained = smaller > 0 and (inter / smaller) >= same_label_containment
                if iou >= same_label_iou or contained:
                    duplicate = True
                    break
            # Do not deduplicate different classes solely because their boxes overlap.
            # Real scenes routinely contain valid overlaps (person + backpack, person +
            # bicycle, hand-held objects, furniture occlusion). Ultralytics already
            # performs class-aware NMS upstream.
        if not duplicate:
            kept.append(det)
    return kept


def frame_zone(detection: Detection, frame_width: float = 640.0, frame_height: float = 480.0) -> str:
    """
    Computes qualitative zone for a single object within the camera frame.
    Matches the deck's center, left, right, top, bottom terminology.
    """
    # Guard against invalid frame sizes
    fw = max(1.0, float(frame_width))
    fh = max(1.0, float(frame_height))

    # Normalized centroid coordinates (0.0 to 1.0)
    norm_cx = detection.cx / fw
    norm_cy = detection.cy / fh

    # Horizontal classification
    if norm_cx < 0.33:
        h_pos = "left"
    elif norm_cx > 0.67:
        h_pos = "right"
    else:
        h_pos = "center"

    # Vertical classification
    if norm_cy < 0.33:
        v_pos = "top"
    elif norm_cy > 0.67:
        v_pos = "bottom"
    else:
        v_pos = "center"

    # Compound mapping
    if h_pos == "center" and v_pos == "center":
        return "in the center"
    elif h_pos == "center" and v_pos == "top":
        return "at the top"
    elif h_pos == "center" and v_pos == "bottom":
        return "at the bottom"
    elif h_pos == "left" and v_pos == "center":
        return "on the left"
    elif h_pos == "right" and v_pos == "center":
        return "on the right"
    elif h_pos == "left" and v_pos == "top":
        return "in the top-left"
    elif h_pos == "left" and v_pos == "bottom":
        return "in the bottom-left"
    elif h_pos == "right" and v_pos == "top":
        return "in the top-right"
    elif h_pos == "right" and v_pos == "bottom":
        return "in the bottom-right"

    return "in the frame"


def calculate_overlap_1d(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    """Returns 1D overlap between two intervals."""
    return max(0.0, min(max_a, max_b) - max(min_a, min_b))


def calculate_gap_1d(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    """Returns distance gap between two intervals (0.0 if overlapping)."""
    if max_a < min_b:
        return min_b - max_a
    elif max_b < min_a:
        return min_a - max_b
    return 0.0


def calculate_iou(a: Detection, b: Detection) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    overlap_w = calculate_overlap_1d(a.x1, a.x2, b.x1, b.x2)
    overlap_h = calculate_overlap_1d(a.y1, a.y2, b.y1, b.y2)
    intersection = overlap_w * overlap_h

    if intersection <= 0.0:
        return 0.0

    union = a.area + b.area - intersection
    return (intersection / union) if union > 0 else 0.0


def _relation_confidence(geometric_confidence: float, *detections: Detection) -> float:
    """Never report a relation as more certain than its least-certain evidence."""
    detection_floor = min((float(d.confidence) for d in detections), default=1.0)
    return round(min(0.99, max(0.0, geometric_confidence), max(0.0, detection_floor)), 2)


def infer_relation(
    sub: Detection,
    obj: Detection,
    config: Optional[SpatialConfig] = None
) -> Optional[SpatialRelation]:
    """
    Infers the primary spatial relation where 'sub' is the subject and 'obj' is the reference object.
    Evaluates geometric rules in prioritized order:
    1. inside (containment)
    2. on (vertical support)
    3. in front of (depth proxy via base position & scale)
    4. next to (horizontal adjacency & vertical alignment)
    5. left of / right of (directional fallback)
    """
    if sub is obj:
        return None
    if config is None:
        config = SpatialConfig()

    # Calculate 1D overlaps and gaps
    overlap_x = calculate_overlap_1d(sub.x1, sub.x2, obj.x1, obj.x2)
    overlap_y = calculate_overlap_1d(sub.y1, sub.y2, obj.y1, obj.y2)
    gap_x = calculate_gap_1d(sub.x1, sub.x2, obj.x1, obj.x2)
    gap_y = calculate_gap_1d(sub.y1, sub.y2, obj.y1, obj.y2)

    min_w = min(sub.width, obj.width)
    min_h = min(sub.height, obj.height)
    max_h = max(sub.height, obj.height)

    # Ratios
    h_overlap_ratio_sub = (overlap_x / sub.width) if sub.width > 0 else 0.0
    v_overlap_ratio = (overlap_y / min_h) if min_h > 0 else 0.0
    area_ratio = (sub.area / obj.area) if obj.area > 0 else 1.0

    # Centroid distances
    dx = sub.cx - obj.cx
    dy = sub.cy - obj.cy
    centroid_dist = math.hypot(dx, dy)

    sub_lbl = sub.label.lower()
    obj_lbl = obj.label.lower()
    same_label = sub_lbl == obj_lbl
    inter_area = overlap_x * overlap_y
    sub_in_obj = (inter_area / sub.area) if sub.area > 0 else 0.0
    obj_in_sub = (inter_area / obj.area) if obj.area > 0 else 0.0
    min_area = min(sub.area, obj.area)
    heavy_overlap = min_area > 0 and (inter_area / min_area) >= 0.5

    # Person + hand-held item mostly inside the person's box -> "holding"
    # (never "inside": a person is not a container).
    if (
        sub_lbl == "person" and obj_lbl in HELD_CLASSES
        and obj_in_sub >= 0.6 and obj.area <= 0.25 * sub.area
    ):
        return SpatialRelation(
            subject=sub, predicate="holding", object=obj,
            confidence=_relation_confidence(min(0.95, 0.6 + 0.35 * obj_in_sub), sub, obj),
            salience=0.97,
            metadata={"obj_in_subject_ratio": round(obj_in_sub, 3)},
        )
    # Reverse direction of the above: the item is described from the person's side.
    if obj_lbl == "person" and sub_lbl in HELD_CLASSES and sub_in_obj >= 0.6 and sub.area <= 0.25 * obj.area:
        return None

    # -------------------------------------------------------------
    # 1. ON (Support relation: subject rests on top/surface of object)
    # -------------------------------------------------------------
    # Rules:
    # - Subject is higher up or resting on the surface (sub.cy < obj.cy)
    # - Large horizontal overlap (sub rests horizontally within the surface of obj)
    # - Subject's bottom (y2) is near the top edge of obj (obj.y1)
    # - Subject is not ridiculously larger than the object supporting it
    if (
        sub.cy < obj.cy and h_overlap_ratio_sub >= config.on_overlap_threshold
        and obj_lbl != "person" and not same_label
        and (area_ratio <= 1.0 or sub_lbl in SITTER_CLASSES)
    ):
        y_diff = sub.y2 - obj.y1
        is_vertically_on = (
            (-config.on_vertical_gap_margin * sub.height <= y_diff) and
            (y_diff <= config.on_vertical_top_margin * obj.height)
        )

        if is_vertically_on and area_ratio <= config.on_max_area_ratio:
            overlap_score = min(1.0, h_overlap_ratio_sub)
            contact_score = 1.0 - (abs(y_diff) / (max(1.0, obj.height * config.on_vertical_top_margin)))
            conf = max(0.5, min(0.99, 0.6 * overlap_score + 0.4 * max(0.0, contact_score)))
            return SpatialRelation(
                subject=sub,
                predicate="on",
                object=obj,
                confidence=_relation_confidence(conf, sub, obj),
                salience=1.0,  # Highest salience for physical support
                metadata={
                    "h_overlap_ratio": round(h_overlap_ratio_sub, 3),
                    "base_to_top_diff": round(y_diff, 2)
                }
            )

    # -------------------------------------------------------------
    # 2. INSIDE / CONTAINED IN
    # -------------------------------------------------------------
    intersection_area = overlap_x * overlap_y
    if (
        sub.area > 0 and (intersection_area / sub.area) >= config.inside_iou_sub_threshold
        and obj_lbl in CONTAINER_CLASSES and sub_lbl != "person" and not same_label
    ):
        if area_ratio < 0.6:
            # Must not be resting at the top surface (which is 'on')
            conf = min(0.99, (intersection_area / sub.area))
            return SpatialRelation(
                subject=sub,
                predicate="inside",
                object=obj,
                confidence=_relation_confidence(conf, sub, obj),
                salience=0.95,
                metadata={"intersection_ratio": round(intersection_area / sub.area, 3)}
            )

    # -------------------------------------------------------------
    # 3. NEXT TO / BESIDE (Checked before depth if horizontally separated)
    # -------------------------------------------------------------
    # If two objects are adjacent horizontally with significant vertical alignment,
    # they are side-by-side ("next to"), not in front of each other.
    is_vertically_aligned = (
        v_overlap_ratio >= config.next_to_v_overlap_ratio or
        abs(dy) <= (config.next_to_cy_diff_ratio * max_h)
    )

    max_allowed_gap = config.next_to_gap_ratio * min_w
    if is_vertically_aligned and gap_x <= max_allowed_gap and overlap_x < (0.3 * min_w):
        proximity_score = 1.0 - (gap_x / max(1.0, max_allowed_gap))
        conf = 0.55 + 0.40 * max(0.0, proximity_score)
        return SpatialRelation(
            subject=sub,
            predicate="next to",
            object=obj,
            confidence=_relation_confidence(conf, sub, obj),
            salience=0.80,
            metadata={
                "horizontal_gap": round(gap_x, 2),
                "v_overlap_ratio": round(v_overlap_ratio, 2)
            }
        )

    # -------------------------------------------------------------
    # 4. IN FRONT OF / BEHIND (Depth proxy without 3D depth sensor)
    # -------------------------------------------------------------
    # In standard perspective projections:
    # 1. Base contact: An object closer to the camera has its base lower in the image (sub.y2 > obj.y2)
    # 2. Shared visual line of sight: Must have significant horizontal overlap (overlap_x > 0)
    # 3. Closer objects appear larger or have significantly lower base
    base_diff = sub.y2 - obj.y2
    h_overlap_min_ratio = (overlap_x / min_w) if min_w > 0 else 0.0

    if base_diff > (config.depth_bottom_offset * max_h) and h_overlap_min_ratio >= config.depth_h_overlap_min:
        is_substantially_larger = area_ratio >= config.depth_size_ratio
        is_significantly_lower = base_diff >= (0.15 * max_h)

        if is_substantially_larger or is_significantly_lower:
            score = 0.55 + 0.25 * min(1.0, base_diff / max_h) + 0.20 * min(1.0, h_overlap_min_ratio)
            return SpatialRelation(
                subject=sub,
                predicate="in front of",
                object=obj,
                confidence=_relation_confidence(min(0.95, score), sub, obj),
                salience=0.85,
                metadata={
                    "base_diff": round(base_diff, 2),
                    "area_ratio": round(area_ratio, 2),
                    "h_overlap_ratio": round(h_overlap_min_ratio, 2)
                }
            )

    # -------------------------------------------------------------
    # 5. DIRECTIONAL FALLBACK: "to the left of" / "to the right of"
    # -------------------------------------------------------------
    if heavy_overlap:
        return None  # boxes overlap too much for left/right to be meaningful

    # A tiny horizontal centroid difference is not a meaningful left/right claim;
    # this prevents vertically stacked objects with nearly identical x-centres from
    # becoming a false directional relation.
    if abs(dx) < config.directional_min_center_delta_ratio * max(1.0, min_w):
        return None

    if sub.cx < obj.cx:
        return SpatialRelation(
            subject=sub,
            predicate="to the left of",
            object=obj,
            confidence=_relation_confidence(0.60, sub, obj),
            salience=0.50,
            metadata={"dx": round(dx, 2), "gap_x": round(gap_x, 2)}
        )
    else:
        return SpatialRelation(
            subject=sub,
            predicate="to the right of",
            object=obj,
            confidence=_relation_confidence(0.60, sub, obj),
            salience=0.50,
            metadata={"dx": round(dx, 2), "gap_x": round(gap_x, 2)}
        )


def analyze_scene_spatial(
    detections: List[Detection],
    frame_width: float = 640.0,
    frame_height: float = 480.0,
    config: Optional[SpatialConfig] = None,
    max_relations: int = 5
) -> Dict[str, Any]:
    """
    Main entry point for Spatial Reasoning Layer.
    Transforms raw detections into a prioritized spatial description graph.

    Handles:
    - 0 detections: empty scene
    - 1 detection: frame_zone positioning
    - >=2 detections: pairwise spatial reasoning with salience ranking and redundancy pruning
    """
    if config is None:
        config = SpatialConfig()

    # Filter detections by confidence
    valid_dets = suppress_duplicates(
        [d for d in detections if d.confidence >= config.min_confidence]
    )

    if not valid_dets:
        return {
            "num_detections": 0,
            "scene_mode": "empty",
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "frame_zones": [],
            "primary_relations": [],
            "all_relations": [],
            "detections": [],
            "structured_narration": "No objects detected."
        }

    # Single-object scene
    if len(valid_dets) == 1:
        det = valid_dets[0]
        zone = frame_zone(det, frame_width, frame_height)
        return {
            "num_detections": 1,
            "scene_mode": "single_object",
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "frame_zones": [{"label": det.label, "zone": zone, "confidence": det.confidence}],
            "primary_relations": [],
            "all_relations": [],
            "detections": [det.to_dict()],
            "structured_narration": f"There is a {det.label} {zone}."
        }

    # Multi-object scene
    raw_relations: List[SpatialRelation] = []
    zones = [
        {"label": d.label, "zone": frame_zone(d, frame_width, frame_height), "confidence": d.confidence}
        for d in valid_dets
    ]

    n = len(valid_dets)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel = infer_relation(valid_dets[i], valid_dets[j], config)
            if rel:
                raw_relations.append(rel)

    # Sort the complete candidate graph first, then apply the debug/raw safety cap.
    # This prevents pair iteration order from silently dropping a high-salience relation.
    raw_relations.sort(key=lambda r: (r.salience, r.confidence), reverse=True)
    raw_relations = raw_relations[:config.max_pairwise_relations]

    # -----------------------------------------------------------------
    # Salience filtering & Transitive suppression:
    # 1. Prune reciprocal/inverse pairs (e.g. keep highest salience between (A, rel, B) and (B, rel, A))
    # 2. If A is 'on' or 'inside' B, suppress (A, rel, C) if B already has a relation with C
    # -----------------------------------------------------------------
    # Identify support/containment parent relationships: child -> parent
    supported_by: Dict[int, int] = {}
    for rel in raw_relations:
        if rel.predicate in ("on", "inside"):
            supported_by[id(rel.subject)] = id(rel.object)

    primary_relations: List[SpatialRelation] = []
    seen_pairs = set()

    for rel in raw_relations:
        sub_id = id(rel.subject)
        obj_id = id(rel.object)
        pair_key = tuple(sorted([sub_id, obj_id]))

        if pair_key in seen_pairs:
            continue

        # Transitive suppression: if either entity is resting on or inside a parent entity
        # that already has an established relation with the other entity in seen_pairs
        is_transitive_redundant = False
        if sub_id in supported_by and rel.predicate not in ("on", "inside"):
            if tuple(sorted([supported_by[sub_id], obj_id])) in seen_pairs:
                is_transitive_redundant = True
        if obj_id in supported_by and rel.predicate not in ("on", "inside"):
            if tuple(sorted([supported_by[obj_id], sub_id])) in seen_pairs:
                is_transitive_redundant = True

        if is_transitive_redundant:
            continue

        # If it's a weak fallback ("left of" / "right of"), ensure they are reasonably close
        if rel.predicate in ("to the left of", "to the right of"):
            gap = rel.metadata.get("gap_x", 0.0)
            if gap > frame_width * 0.75:
                # Too far apart to be an informative relation
                continue

        primary_relations.append(rel)
        seen_pairs.add(pair_key)

        if len(primary_relations) >= max_relations:
            break

    # Build concise structured narration strings
    narrations: List[str] = []
    for rel in primary_relations:
        narrations.append(f"A {rel.subject.label} is {rel.predicate} the {rel.object.label}.")

    return {
        "num_detections": len(valid_dets),
        "scene_mode": "multi_object",
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "frame_zones": zones,
        "detections": [d.to_dict() for d in valid_dets],
        "primary_relations": [r.to_dict() for r in primary_relations],
        "all_relations": [r.to_dict() for r in raw_relations],
        "structured_narration": " ".join(narrations)
    }
