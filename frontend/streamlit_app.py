"""Accessible Streamlit frontend for AI Scene Narrator."""

from __future__ import annotations

import io
import json
import os

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AI Scene Narrator", page_icon="🗣️", layout="wide")
st.title("🗣️ AI Scene Narrator")
st.caption("See the scene as objects + relationships, then hear the result.")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)
    speak = st.checkbox("Show speech controls", value=True)
    debug = st.checkbox("Show diagnostics", value=False)
    st.divider()
    st.caption(f"API: `{API_URL}`")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        if h.get("model_loaded"):
            st.success(f"API ready · {h.get('model_name', 'YOLO')}")
        else:
            st.warning("API online · model is cold; first analysis may load it")
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")


def call_api(image_bytes: bytes, filename: str, mime: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/api/analyze",
            params={"conf": conf},
            files={"file": (filename, image_bytes, mime)},
            timeout=90,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc
    try:
        body = r.json()
    except ValueError:
        body = {"detail": r.text}
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {body.get('detail', 'unknown error')}")
    return body


def annotate(image_bytes: bytes, detections: list[dict]) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in detections:
        x1, y1, x2, y2 = map(int, (d["x1"], d["y1"], d["x2"], d["y2"]))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=max(2, img.width // 320))
        text = f'{d["label"]} {d["confidence"]:.0%}'
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        top = max(0, y1 - th - 6)
        draw.rectangle([x1, top, x1 + tw + 8, max(top + th + 5, y1)], fill=(0, 200, 0))
        draw.text((x1 + 4, top + 2), text, fill=(0, 0, 0), font=font)
    return img


def browser_speech(text: str) -> None:
    payload = json.dumps(text).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    components.html(
        f"""
        <button onclick="say()" style="padding:8px 14px;cursor:pointer;border-radius:8px;border:1px solid #888">
          🔊 Replay narration
        </button>
        <script>
          function say() {{
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(new SpeechSynthesisUtterance({payload}));
          }}
        </script>""",
        height=48,
    )


def relation_cards(relations: list[dict]) -> None:
    for r in relations:
        st.markdown(
            f"**{r['subject']}** → *{r['predicate']}* → **{r['object']}** · {r['confidence']:.0%}"
        )


tab_up, tab_cam = st.tabs(["📁 Upload image", "📷 Camera snapshot"])
image_bytes = filename = mime = None

with tab_up:
    up = st.file_uploader("Choose a JPG / PNG / WebP image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if up:
        image_bytes, filename, mime = up.getvalue(), up.name, up.type or "image/jpeg"

with tab_cam:
    shot = st.camera_input("Take a photo")
    if shot:
        image_bytes, filename, mime = shot.getvalue(), "camera.jpg", "image/jpeg"

if image_bytes:
    with st.spinner("Analyzing scene..."):
        try:
            data = call_api(image_bytes, filename, mime)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    left, right = st.columns([3, 2])
    with left:
        st.image(annotate(image_bytes, data["detections"]), use_container_width=True)
        st.caption(
            f"{data['frame_width']}×{data['frame_height']} · "
            f"{data['processing_ms']:.1f} ms total"
        )

    with right:
        st.subheader("Narration")
        st.info(data["narration"])
        if speak:
            browser_speech(data["narration"])

        st.subheader(f"Detections ({data['num_detections']})")
        if data["detections"]:
            st.dataframe(
                [
                    {"object": d["label"], "confidence": round(d["confidence"], 2)}
                    for d in data["detections"]
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Nothing recognizable above the selected threshold.")

        if data["relations"]:
            st.subheader("What is where")
            relation_cards(data["relations"])
        else:
            st.subheader("Spatial reasoning")
            st.caption("No high-confidence spatial relationship was established.")

    if debug:
        st.divider()
        st.subheader("Diagnostics")
        st.json(
            {
                "processing_ms": data["processing_ms"],
                "detection_ms": data["detection_ms"],
                "reasoning_ms": data["reasoning_ms"],
                "scene_mode": data["scene_mode"],
                "primary_relation_count": len(data["relations"]),
                "all_relation_count": len(data.get("all_relations", [])),
                "all_relations": data.get("all_relations"),
            }
        )
else:
    st.info("Upload an image or take a snapshot to get a spoken-style scene description.")
