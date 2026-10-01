# AI Scene Narrator

**From raw object detections to spoken spatial understanding — fully local, deterministic, and deployable.**

AI Scene Narrator is a four-stage assistive computer-vision pipeline:

```text
Image / Camera
      │
      ▼
┌──────────────────────┐
│ 1. YOLO detection    │  objects + confidence + bounding boxes
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Spatial reasoning │  on / inside / holding / beside / depth proxy / L-R
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. NLG                │  safe templates + grammar + concise summaries
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Speech             │  browser TTS or offline pyttsx3
└──────────────────────┘
```

The important design choice is that the reasoning layer does **not** ask a generative model to invent relationships. It derives relations from explicit geometry, applies plausibility constraints, ranks them, removes redundancy, and only then turns them into language.

## Why this version is stronger

The hardened implementation adds several layers beyond the original happy path:

- **Class-aware duplicate handling:** two different object classes are never merged merely because their boxes overlap.
- **Ambiguity rejection:** weak/near-vertical left-right claims are suppressed instead of being forced into a sentence.
- **Evidence-bounded confidence:** a relation can never be more confident than its least-confident detection.
- **Consistent outputs:** the API returns the same post-deduplication detections that spatial reasoning actually used.
- **Live movement awareness:** scene signatures track coarse object movement as well as relation changes.
- **Image safety:** uploads are checked for valid image headers and bounded by dimensions/pixel count before OpenCV decoding.
- **Non-blocking API:** CPU-bound narration is dispatched to a threadpool.
- **Readiness endpoint:** `/ready` returns HTTP 503 until a model is loaded.
- **Operational timings:** every response exposes detection, reasoning, and total processing time.
- **Graceful startup:** model preload failure leaves the API alive and exposes degraded readiness instead of killing the server.
- **Deployment hardening:** the bundled YOLO weights are copied into the Docker image explicitly.

See `AUDIT_REPORT.md` for the engineering findings and fixes.

## Project layout

```text
app/
  config.py        validated environment configuration
  detection.py     safe image decode + YOLO wrapper + annotations
  spatial.py       geometric predicates + plausibility + graph pruning
  nlg.py           deterministic spoken-language generation
  tts_engine.py    optional offline desktop speech
  pipeline.py      end-to-end orchestration + live change gate
  schemas.py       validated API request/response contracts
  main.py          FastAPI service
  live.py          desktop webcam narration
  demo.py          synthetic reasoning/NLG demo
frontend/
  streamlit_app.py accessible upload/camera UI + browser speech
tests/
  test_*.py        unit, integration, fuzz/regression, and security tests
docs/
  Spatial_Reasoning_Manual.md
  Spatial_Reasoning_Manual.pdf
.github/workflows/ci.yml
Dockerfile
docker-compose.yml
```

## Local setup

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows cmd
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements-dev.txt
```

Run the verification suite:

```bash
python -m pytest
python -m compileall -q app frontend
```

Run the synthetic demonstrations (no YOLO inference required):

```bash
python -m app.demo
python -m app.nlg
```

Start the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for interactive API documentation.

Start the frontend in another terminal:

```bash
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

On Windows PowerShell:

```powershell
$env:API_URL="http://localhost:8000"
streamlit run frontend/streamlit_app.py
```

## API

### `GET /health`

Liveness information plus the currently loaded model name.

### `GET /ready`

Readiness probe. Returns **200** after a model is loaded and **503** while the service is cold/degraded.

### `POST /api/analyze`

Multipart image analysis.

```bash
curl -F "file=@photo.jpg" "http://localhost:8000/api/analyze?conf=0.5"
```

Optional `debug=true` includes the raw relation graph.

### `POST /api/narrate`

Useful for testing the reasoning/NLG stack without YOLO:

```json
{
  "frame_width": 640,
  "frame_height": 480,
  "detections": [
    {"label":"desk","confidence":0.95,"x1":100,"y1":200,"x2":550,"y2":450},
    {"label":"laptop","confidence":0.92,"x1":200,"y1":160,"x2":380,"y2":220}
  ]
}
```

Typical result:

```text
A laptop is resting on the desk.
```

The response also includes bounding boxes, frame zones, relation confidence, and processing timings.

## Configuration

Environment variables:

```text
SCENE_MODEL=yolov8n.pt
SCENE_CONF_THRESHOLD=0.5
SCENE_MAX_UPLOAD_MB=10
SCENE_MAX_IMAGE_WIDTH=8192
SCENE_MAX_IMAGE_HEIGHT=8192
SCENE_MAX_IMAGE_PIXELS=25000000
SCENE_CORS_ORIGINS=*
SCENE_PRELOAD_MODEL=1
API_URL=http://localhost:8000
```

For a public deployment, replace `SCENE_CORS_ORIGINS=*` with the exact frontend origin.

## Docker

The API image explicitly copies the bundled `yolov8n.pt` file and warms the model during the image build.

The container healthcheck uses `/ready`, so Compose only starts the frontend after the model-backed API reports ready.

```bash
docker compose up --build
```

Then:

```text
Frontend: http://localhost:8501
API:      http://localhost:8000/docs
```

The frontend and API are separate services, which means the UI can be hosted independently from the inference service.

## Desktop live narration

Install the optional desktop TTS dependency:

```bash
python -m pip install -r requirements-local.txt
```

On Linux, install an available speech backend such as `espeak-ng`.

Then:

```bash
python -m app.live
```

Useful flags:

```text
--camera 0          camera index
--conf 0.5          detection threshold
--interval 3        minimum seconds between utterances
--repeat 10         repeat unchanged scene after 10 seconds
--stable 4          require four stable frames
--max-read-errors 5 tolerate transient camera read failures
--no-window         no OpenCV preview
--no-speech         text-only mode
```

## Technical limitation

This implementation reasons from **2D bounding boxes**. The `in front of` predicate is a monocular geometric proxy based on projected base position, overlap, and relative scale; it is not true metric depth and does not replace a depth sensor.

For assistive deployment, the system should therefore be treated as a perception aid whose language remains conservative and whose geometric thresholds should be validated against the target camera and environment.
