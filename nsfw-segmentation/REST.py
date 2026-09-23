#!/usr/bin/env python3
"""NSFW anatomical segmentation service for image-content moderation."""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFile, UnidentifiedImageError
from ultralytics import YOLO

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("nsfw-segmentation")

SERVICE_NAME = "nsfw-segmentation"
MODEL_ID = "NSFW-API/NSFW_Segmentation"
TASKS = ("breast", "vagina", "penis")
SERVICE_DIR = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", "7806"))
PRIVATE = os.getenv("PRIVATE", "false").lower() in {"1", "true", "yes"}
MODEL_SIZE = os.getenv("MODEL_SIZE", "s").lower()
_model_dir = Path(os.getenv("MODEL_DIR", "models")).expanduser()
MODEL_DIR = (_model_dir if _model_dir.is_absolute() else SERVICE_DIR / _model_dir).resolve()
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.1"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "832"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "100"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(32 * 1024 * 1024)))
ALLOW_PRIVATE_URLS = os.getenv("ALLOW_PRIVATE_URLS", "false").lower() in {"1", "true", "yes"}
URL_TIMEOUT = float(os.getenv("URL_TIMEOUT", "15"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

if MODEL_SIZE not in {"s", "x"}:
    raise ValueError("MODEL_SIZE must be 's' or 'x'")
if not 0 <= CONFIDENCE_THRESHOLD <= 1:
    raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")

Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))
ImageFile.LOAD_TRUNCATED_IMAGES = False

RAW_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
models: dict[str, YOLO] = {}
model_error: str | None = None
inference_lock = threading.Lock()


def model_path(task: str) -> Path:
    override = os.getenv(f"{task.upper()}_MODEL_PATH")
    return Path(override) if override else MODEL_DIR / f"nsfw-seg-{task}-{MODEL_SIZE}.pt"


def load_models() -> None:
    global model_error
    missing = [str(model_path(task)) for task in TASKS if not model_path(task).is_file()]
    if missing:
        model_error = "Missing model checkpoints: " + ", ".join(missing)
        logger.error(model_error)
        return

    try:
        for task in TASKS:
            path = model_path(task)
            logger.info("Loading %s model from %s", task, path)
            models[task] = YOLO(str(path))
        model_error = None
        logger.info("Loaded %d segmentation heads on %s", len(models), DEVICE)
    except Exception as exc:
        models.clear()
        model_error = str(exc)
        logger.exception("Unable to load segmentation models")


def is_raw_image_request() -> bool:
    content_type = (request.content_type or "").split(";", 1)[0].strip().lower()
    return content_type in RAW_IMAGE_CONTENT_TYPES


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    if ALLOW_PRIVATE_URLS:
        return
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError("URL hostname could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private, loopback, and link-local URLs are not allowed")


def _read_remote_image(url: str) -> bytes:
    _validate_remote_url(url)
    with requests.get(url, stream=True, timeout=URL_TIMEOUT, allow_redirects=False) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if not content_type.startswith("image/"):
            raise ValueError("URL response is not an image")
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise ValueError("Downloaded image is too large")
        chunks, total = [], 0
        for chunk in response.iter_content(64 * 1024):
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise ValueError("Downloaded image is too large")
            chunks.append(chunk)
        return b"".join(chunks)


def _open_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("No image data provided")
    if len(data) > MAX_FILE_SIZE:
        raise ValueError("Image is too large")
    try:
        image = Image.open(BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or unsupported image") from exc


def image_from_request() -> Image.Image:
    if request.method == "POST" and is_raw_image_request():
        return _open_image(request.get_data(cache=False))
    if request.method == "POST" and "file" in request.files:
        return _open_image(request.files["file"].read(MAX_FILE_SIZE + 1))

    url, file_name = request.args.get("url"), request.args.get("file")
    if bool(url) == bool(file_name):
        raise ValueError("Provide exactly one image source: 'url', 'file', or a POST body")
    if url:
        return _open_image(_read_remote_image(url))

    path = Path(file_name).expanduser()
    if not path.is_file():
        raise ValueError("Image file does not exist")
    if path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("Image is too large")
    return _open_image(path.read_bytes())


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    values = np.asarray(points, dtype=np.float64)
    return float(abs(np.dot(values[:, 0], np.roll(values[:, 1], 1)) -
                     np.dot(values[:, 1], np.roll(values[:, 0], 1))) / 2)


def process_results(task: str, result: Any, include_masks: bool) -> list[dict[str, Any]]:
    if result.boxes is None or result.masks is None:
        return []
    detections = []
    normalized_polygons = result.masks.xyn
    pixel_polygons = result.masks.xy
    for index, box in enumerate(result.boxes):
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        normalized = normalized_polygons[index].tolist() if index < len(normalized_polygons) else []
        pixels = pixel_polygons[index].tolist() if index < len(pixel_polygons) else []
        label = str(result.names.get(class_id, task)) if isinstance(result.names, dict) else task
        # The two single-class upstream checkpoints currently embed the generic
        # class name "item"; their checkpoint/task identity supplies the label.
        if label.lower() == "item" and task in {"vagina", "penis"}:
            label = task
        item = {
            "label": label,
            "task": task,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "bbox": {
                "x": round(x1), "y": round(y1),
                "width": round(x2 - x1), "height": round(y2 - y1),
            },
            "area_ratio": round(_polygon_area(normalized), 6),
            "_normalized_polygon": normalized,
        }
        if include_masks:
            item["polygon"] = [[round(x, 2), round(y, 2)] for x, y in pixels]
            item["polygon_normalized"] = [[round(x, 6), round(y, 6)] for x, y in normalized]
        detections.append(item)
    return detections


def analyze_image(image: Image.Image, include_masks: bool) -> dict[str, Any]:
    if len(models) != len(TASKS):
        raise RuntimeError(model_error or "Models are not loaded")
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    # Ultralytics interprets NumPy image inputs as OpenCV-style BGR. PIL stores
    # RGB, so reverse the channels; passing RGB here materially changes scores.
    image_array = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    with inference_lock:
        for task, model in models.items():
            results = model.predict(
                image_array, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS, device=DEVICE, verbose=False,
            )
            if results:
                predictions.extend(process_results(task, results[0], include_masks))

    predictions.sort(key=lambda item: item["confidence"], reverse=True)
    labels = sorted({item["label"] for item in predictions})
    max_confidence = max((item["confidence"] for item in predictions), default=0.0)
    # Rasterize normalized polygons to measure their union without double-counting overlap.
    coverage_mask = Image.new("1", (512, 512), 0)
    draw = ImageDraw.Draw(coverage_mask)
    for item in predictions:
        points = [(round(x * 511), round(y * 511)) for x, y in item.pop("_normalized_polygon")]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    coverage = float(np.asarray(coverage_mask, dtype=np.uint8).mean())
    return {
        "service": SERVICE_NAME,
        "status": "success",
        "nsfw": bool(predictions),
        "score": round(max_confidence, 4),
        "labels": labels,
        "coverage_ratio": round(coverage, 6),
        "predictions": predictions,
        "metadata": {
            "processing_time": round(time.perf_counter() - started, 3),
            "image": {"width": image.width, "height": image.height},
            "model_info": {
                "repository": MODEL_ID, "framework": "Ultralytics YOLO11 segmentation",
                "size": MODEL_SIZE, "image_size": IMAGE_SIZE,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            },
        },
    }


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])


@app.errorhandler(413)
def too_large(_: Any):
    return jsonify({"service": SERVICE_NAME, "status": "error", "error": {"message": "Image is too large"}}), 413


@app.get("/health")
def health():
    healthy = len(models) == len(TASKS)
    payload = {
        "schema_version": "health.v1", "service": SERVICE_NAME,
        "status": "healthy" if healthy else "unhealthy", "warnings": [], "dependencies": {},
        "model": {
            "status": "loaded" if healthy else "error", "name": MODEL_ID,
            "framework": "Ultralytics", "device": DEVICE,
            "checkpoints": {task: str(model_path(task)) for task in TASKS},
        },
        "endpoints": ["GET /health", "GET /classes", "GET,POST /analyze", "GET /v3/analyze"],
    }
    if model_error:
        payload["model"]["error"] = model_error
    return jsonify(payload), 200 if healthy else 503


@app.get("/classes")
def classes():
    return jsonify({"tasks": {
        "breast": ["breast", "areola", "nipple"],
        "vagina": ["vagina"],
        "penis": ["penis"],
    }})


@app.route("/analyze", methods=["GET", "POST"])
@app.route("/v3/analyze", methods=["GET", "POST"])
@app.route("/", methods=["POST"])
def analyze():
    started = time.perf_counter()
    try:
        include_masks = request.args.get("include_masks", "true").lower() not in {"0", "false", "no"}
        return jsonify(analyze_image(image_from_request(), include_masks))
    except ValueError as exc:
        code = 413 if "too large" in str(exc).lower() else 400
        return jsonify({
            "service": SERVICE_NAME, "status": "error", "predictions": [],
            "error": {"message": str(exc)},
            "metadata": {"processing_time": round(time.perf_counter() - started, 3)},
        }), code
    except Exception as exc:
        logger.exception("Analysis failed")
        return jsonify({
            "service": SERVICE_NAME, "status": "error", "predictions": [],
            "error": {"message": f"Segmentation failed: {exc}"},
            "metadata": {"processing_time": round(time.perf_counter() - started, 3)},
        }), 503 if len(models) != len(TASKS) else 500


if __name__ == "__main__":
    load_models()
    app.run(host="127.0.0.1" if PRIVATE else "0.0.0.0", port=PORT, threaded=True)
