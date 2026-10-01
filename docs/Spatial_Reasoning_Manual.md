# AI Scene Narrator — Spatial Reasoning Engine Manual

**Technical Architecture & Mathematical Heuristics (from Scratch)**  
**Target Module:** `app/spatial.py`  
**Group:** 168 | **Course/Project:** PE1 Spatial Reasoning

---

## 1. Problem Statement & Motivation

Traditional computer vision detection models like YOLO output discrete object labels and bounding boxes:
$$\text{Output: } [(\text{"chair"}, [x_1, y_1, x_2, y_2]), (\text{"table"}, [x_1, y_1, x_2, y_2])]$$

For blind or low-vision users, discrete lists of object names provide very little actionable spatial context. A user cannot act on *"chair, table"*, but they can immediately act on:
> *"The chair is beside the table."* or *"A cup is on the desk in front of you."*

### Key Constraints:
- **Zero Cloud / Privacy First**: Runs 100% locally on device without network latency.
- **CPU Edge Feasible**: Runs in sub-millisecond time on ordinary CPUs without needing heavy 3D neural networks or depth sensors.

---

## 2. System Architecture & Pipeline Flow

The AI Scene Narrator follows a 4-tier pipeline:

```
Camera Frame (BGR array)
   │
   ▼
[Stage 1] Detection Layer (app/detection.py)
   │  Output: List[Detection(label, conf, x1, y1, x2, y2)]
   ▼
[Stage 2] Spatial Reasoning (app/spatial.py)   <--- YOU ARE HERE
   │  Output: Spatial Graph + Predicates (on, in front of, next to, etc.)
   ▼
[Stage 3] NLG Tier 1 (app/nlg.py)
   │  Output: Composed natural sentence with synonym diversification
   ▼
[Stage 4] Offline TTS (app/tts_engine.py)
   │  Output: Real-time spoken narration via pyttsx3
   ▼
Audio Narration Delivered to User
```

---

## 3. Mathematical Foundations & Geometric Predicates

All calculations use image coordinates where:
- $(0,0)$ is the top-left corner.
- $x$ increases from left to right.
- $y$ increases from top to bottom (so lower down in the physical world has larger $y$).

For any bounding box $A$ with $(x_1, y_1)$ and $(x_2, y_2)$:
- **Width:** $w = x_2 - x_1$
- **Height:** $h = y_2 - y_1$
- **Area:** $\text{Area} = w \times h$
- **Centroids:** $c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$
- **1D Overlap:** $\text{overlap}_x = \max(0, \min(A.x_2, B.x_2) - \max(A.x_1, B.x_1))$
- **1D Gap:** $\text{gap}_x = \max(0, \max(A.x_1, B.x_1) - \min(A.x_2, B.x_2))$

---

### Predicate Decision Matrix

| Predicate | Geometric Rule | Assistive Meaning |
| :--- | :--- | :--- |
| **`"on"`** | 1. $A.c_y < B.c_y$ (Subject is higher up)<br/>2. $\frac{\text{overlap}_x}{A.w} \ge 0.45$ (Significant horizontal support)<br/>3. Base contact: $-0.20 A.h \le (A.y_2 - B.y_1) \le 0.35 B.h$<br/>4. $A.\text{Area} \le 1.8 \times B.\text{Area}$ (Scale plausibility) | Object resting on a surface (e.g. *"cup on table"*). |
| **`"in front of"`** | 1. Base contact difference: $A.y_2 - B.y_2 > 0.06 \times \max(h)$<br/>2. Shared line of sight: $\frac{\text{overlap}_x}{\min(w)} \ge 0.15$<br/>3. Scale factor: $\frac{A.\text{Area}}{B.\text{Area}} \ge 1.6\times$ or base is $\ge 15\%$ lower | Monocular depth proxy without depth sensor (e.g. *"person in front of couch"*). |
| **`"next to"`** | 1. Vertical plane alignment: $\frac{\text{overlap}_y}{\min(h)} \ge 0.30$ or $|A.c_y - B.c_y| \le 0.50 \times \max(h)$<br/>2. Horizontal proximity: $\text{gap}_x \le 1.25 \times \min(A.w, B.w)$<br/>3. Not vertically stacked | Side-by-side objects on ground plane (e.g. *"chair next to table"*). |
| **`"inside"`** | $\frac{\text{Intersection Area}}{A.\text{Area}} \ge 0.82$ and $A.\text{Area} < 0.85 \times B.\text{Area}$ | Containment (e.g. *"apple inside bowl"*). |
| **`"to the left/right of"`** | $A.c_x < B.c_x$ or $A.c_x > B.c_x$ with horizontal separation | Fallback directional ordering when objects are further apart. |
| **`frame_zone()`** | Normalized centroid $(\frac{c_x}{W}, \frac{c_y}{H})$ mapped across a $3 \times 3$ grid | Single-object positioning (*"in the center"*, *"on the left"*, *"in the top-right"*, etc.). |

---

## 4. Anti-Clutter & Scene Graph Optimization

In real scenes with $N$ objects, a raw pairwise scan produces $N(N-1)$ combinations. An 8-object scene would generate 56 relations, confusing the user.

`app/spatial.py` solves this via 3 algorithmic steps:
1. **Salience Ranking**: Each relation is assigned an importance score:
   - Physical Support (`on`): $1.0$
   - Containment (`inside`): $0.95$
   - Depth (`in front of`): $0.85$
   - Proximity (`next to`): $0.80$
   - Directional (`left/right of`): $0.50$
2. **Reciprocal Deduplication**: Ensures only the canonical informative predicate is reported (e.g. keeps *"cup is on desk"*, removes redundant *"desk is below cup"*).
3. **Transitive Support Suppression**: If object $A$ (e.g. cup) is resting *on* object $B$ (e.g. desk), and $B$ is *next to* object $C$ (e.g. chair), the synthetic relation *"cup next to chair"* is automatically suppressed because $A$ is physically bound to $B$.

---

## 5. How to Use the Code (Step-by-Step)

### Running Automated Tests
To run the full test suite (spatial, NLG, pipeline, API):
```bash
python -m pytest
```

### Running the Interactive Demonstration
To run synthetic everyday scenes through spatial reasoning + NLG (no camera or YOLO needed):
```bash
python -m app.demo
```

### Testing the YOLO Object Detector
To run live webcam detection:
```bash
python -m app.detection
```
To run detection on a saved image:
```bash
python -m app.detection --image sample.jpg
```

### Python API Integration
```python
from app.detection import Detection
from app.spatial import analyze_scene_spatial, SpatialConfig

# Sample detections
desk = Detection(label="desk", confidence=0.95, x1=100, y1=200, x2=550, y2=450)
laptop = Detection(label="laptop", confidence=0.92, x1=200, y1=160, x2=380, y2=220)

# Run spatial analysis
scene = analyze_scene_spatial([desk, laptop], frame_width=640, frame_height=480)

# Natural language summary
print(scene["structured_narration"])
# -> "A laptop is on the desk."

# Structured relation data
print(scene["primary_relations"])
```

---

## 6. Project Roadmap

- [x] **Stage 1 (Detection)**: YOLOv8n wrapper (`app/detection.py`).
- [x] **Stage 2 (Spatial Reasoning)**: Geometric heuristics & anti-clutter engine (`app/spatial.py`).
- [x] **Stage 3 (NLG)**: Template engine + synonym banks (`app/nlg.py`).
- [x] **Stage 4 (TTS Engine)**: Offline speech synthesis (`app/tts_engine.py`).
- [x] **Pipeline & UI**: `app/pipeline.py`, FastAPI service (`app/main.py`), Streamlit frontend (`frontend/streamlit_app.py`).
- [x] **Deployment**: `Dockerfile`, `docker-compose.yml`.
