# ---- AI Scene Narrator API ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# OpenCV runtime libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Keep the supplied model local so image builds do not need a runtime weight download.
COPY yolov8n.pt ./yolov8n.pt

# CPU-only PyTorch first (much smaller than the default CUDA wheels)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake YOLO weights into the image so containers start without a download
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY app ./app

RUN useradd --create-home app && chown -R app /srv
USER app

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request as u; u.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/ready', timeout=4)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
