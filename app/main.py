"""FastAPI service for AI Scene Narrator."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings
from app.detection import is_model_loaded, loaded_model_name, load_model, Detection
from app.pipeline import NarrationPipeline
from app.schemas import HealthResponse, NarrateRequest, NarrateResponse

logger = logging.getLogger("scene_narrator")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def create_app(
    settings: Optional[Settings] = None,
    pipeline: Optional[NarrationPipeline] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline = pipeline or NarrationPipeline(
        conf_threshold=settings.conf_threshold,
        model_name=settings.model_name,
        settings=settings,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.preload_model:
            logger.info("Preloading YOLO model %s ...", settings.model_name)
            try:
                await run_in_threadpool(load_model, settings.model_name)
            except RuntimeError:
                # Keep the API alive so /health can expose degraded readiness and
                # requests can return a useful 503 instead of crashing the process.
                logger.exception("YOLO model preload failed")
        yield

    app = FastAPI(
        title="AI Scene Narrator",
        version=__version__,
        description="Object detection + 2D spatial reasoning + natural-language narration.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": "ai-scene-narrator", "docs": "/docs", "health": "/health", "ready": "/ready"}

    def _health_payload(status: str = "ok") -> HealthResponse:
        loaded = is_model_loaded()
        return HealthResponse(
            status=status,
            model_loaded=loaded,
            model_name=loaded_model_name() or settings.model_name,
            version=__version__,
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # Liveness remains healthy even when the optional model backend is cold.
        return _health_payload("ok")

    @app.get("/ready", response_model=HealthResponse)
    def ready(response: Response) -> HealthResponse:
        payload = _health_payload("ready" if is_model_loaded() else "not_ready")
        if not payload.model_loaded:
            response.status_code = 503
        return payload

    @app.post("/api/narrate", response_model=NarrateResponse, response_model_exclude_none=True)
    async def narrate(
        req: NarrateRequest,
        debug: bool = Query(False, description="Include all raw relations"),
    ):
        dets = [Detection(b.label, b.confidence, b.x1, b.y1, b.x2, b.y2) for b in req.detections]
        result = await run_in_threadpool(
            pipeline.narrate_detections, dets, req.frame_width, req.frame_height
        )
        return result.to_dict(include_all_relations=debug)

    @app.post("/api/analyze", response_model=NarrateResponse, response_model_exclude_none=True)
    async def analyze(
        file: UploadFile = File(..., description="JPEG / PNG / WebP image"),
        conf: Optional[float] = Query(None, ge=0.0, le=1.0, description="Confidence threshold override"),
        debug: bool = Query(False, description="Include all raw relations"),
    ):
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(415, f"Unsupported content type: {file.content_type!r}")
        if file.size is not None and file.size > settings.max_upload_bytes:
            raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB limit")

        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB limit")

        try:
            result = await run_in_threadpool(pipeline.narrate_image_bytes, data, conf)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("Inference backend unavailable")
            raise HTTPException(503, str(exc)) from exc
        return result.to_dict(include_all_relations=debug)

    return app


app = create_app()
