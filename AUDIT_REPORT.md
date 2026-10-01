# AI Scene Narrator — Engineering Audit & Hardening Report

## Baseline

The original repository passed its existing test suite, but the audit focused on failure modes that happy-path tests did not cover: semantic false positives, duplicate suppression, API/event-loop blocking, malformed image payloads, configuration drift, response inconsistencies, and live-scene change detection.

## High-impact findings fixed

| Area | Finding | Resolution |
|---|---|---|
| Spatial reasoning | Different classes with highly-overlapping boxes were treated as duplicate detections | Cross-class deduplication removed; overlapping objects are preserved |
| Spatial reasoning | Near-vertical objects could be forced into a left/right relation based on a tiny x-centroid difference | Minimum horizontal separation guard added |
| Spatial reasoning | Relation confidence could exceed the confidence of its weakest detection | Relation confidence now cannot exceed the least-confident evidence |
| Pipeline | API could report raw duplicate boxes while `num_detections` described deduplicated boxes | Response now uses the same post-deduplication detection set as reasoning |
| Live narration | Object movement with unchanged relation labels could be ignored | Scene signature includes coarse object positions + relation edges |
| Upload security | Small compressed images could expand to unexpectedly large decoded buffers | Pillow header validation + width/height/pixel limits added before OpenCV decode |
| API concurrency | `/api/narrate` performed CPU work directly on the FastAPI event loop | Work moved to the threadpool |
| Readiness | Liveness and model readiness were conflated | `/health` is a liveness probe; `/ready` returns 503 until the configured model is actually loaded |
| Model configuration | App-level `SCENE_MODEL` settings were not consistently passed through the preload/inference path | Model name is now part of the pipeline configuration |
| Operational visibility | Responses lacked timing information | Detection, reasoning, and total processing times are returned |
| Live camera | A single transient camera read failure stopped live narration | Consecutive read-error tolerance added |
| Speech lifecycle | TTS shutdown could leave the worker thread behind | Queue is drained, sentinel is sent, and worker is joined |
| Deployment | Docker relied on an implicit weight download/cache path | The supplied weight file is copied into the image explicitly |
| Frontend safety | Speech text was embedded directly inside a `<script>` block | Script-sensitive characters are escaped before JavaScript embedding |
| Reproducibility | No CI workflow | GitHub Actions added for Python 3.11–3.13 test/compile coverage |

## Remaining architectural limitation

The system performs 2D reasoning over object bounding boxes. `in front of` is therefore a monocular geometric proxy, not true metric depth. It can be useful for narration, but it should not be described as depth-sensor-equivalent perception.

## Verification performed

- Automated test suite: **77 passing tests** after hardening.
- End-to-end synthetic demo: completed successfully.
- NLG standalone demo: completed successfully.
- Python bytecode compilation: successful.
- Documentation PDF generation: successful.
- Adversarial spatial checks: overlapping-class preservation, vertical ambiguity, movement-sensitive scene signatures, duplicate-response consistency.
- Actual YOLO inference could not be executed in this environment because `ultralytics` is not installed and outbound package installation is unavailable here. The bundled `yolov8n.pt` file is present.
- Docker runtime could not be executed because the Docker CLI/daemon is not available in this environment.
