#!/usr/bin/env python3
"""
SAM3 REST API - Text-prompted image segmentation.

Accepts an image and a list of nouns (typically from noun_consensus),
runs SAM3 once per noun, and returns bounding boxes and confidence
scores for all detected instances of each noun.
"""

import io
import os
import sys
import json
import base64
import logging
import time

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAM3_PATH = os.getenv('SAM3_PATH', '/home/sd/sam3')
SAM3_CHECKPOINT = os.getenv('SAM3_CHECKPOINT')  # optional; None = auto-download from HF
PORT = int(os.getenv('PORT', 9779))
PRIVATE = os.getenv('PRIVATE', 'true').lower() in ('true', '1', 'yes')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

sys.path.insert(0, SAM3_PATH)

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------

_model = None
_processor = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    global _model, _processor
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info("Loading SAM3 model on %s ...", _device)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        _model = build_sam3_image_model(
            device=_device,
            checkpoint_path=SAM3_CHECKPOINT,   # None → auto-download from HF
            load_from_HF=(SAM3_CHECKPOINT is None),
        )
        _processor = Sam3Processor(_model, confidence_threshold=CONFIDENCE_THRESHOLD)
        logger.info("SAM3 model loaded successfully")
        return True
    except Exception as e:
        logger.error("Failed to load SAM3 model: %s", e)
        return False


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _mask_to_rle(mask_tensor):
    """Convert a boolean [H, W] mask tensor to a simple RLE list [count0, count1, ...]."""
    flat = mask_tensor.cpu().numpy().astype(bool).flatten()
    rle = []
    if len(flat) == 0:
        return rle
    current = flat[0]
    count = 1
    for val in flat[1:]:
        if val == current:
            count += 1
        else:
            rle.append(int(count))
            current = val
            count = 1
    rle.append(int(count))
    return rle


def segment_nouns(image: Image.Image, nouns: list[str]) -> dict:
    """
    Run SAM3 text-prompted segmentation for each noun.

    Returns a dict keyed by noun with detected instances (boxes, scores,
    and optional RLE masks).
    """
    if not _processor:
        raise RuntimeError("SAM3 model not loaded")

    if image.mode != 'RGB':
        image = image.convert('RGB')

    results = {}

    with torch.inference_mode(), torch.autocast(_device, dtype=torch.bfloat16):
        state = _processor.set_image(image)

        for noun in nouns:
            noun = noun.strip().lower()
            if not noun:
                continue

            _processor.reset_all_prompts(state)
            state = _processor.set_text_prompt(prompt=noun, state=state)

            scores = state.get('scores')
            boxes = state.get('boxes')
            masks = state.get('masks')

            if scores is None or len(scores) == 0:
                results[noun] = {'instances': []}
                continue

            instances = []
            for i in range(len(scores)):
                box = boxes[i].cpu().tolist()  # [x0, y0, x1, y1] pixel coords
                score = float(scores[i].item())

                instance = {
                    'score': round(score, 4),
                    'bbox': {
                        'x': round(box[0]),
                        'y': round(box[1]),
                        'width': round(box[2] - box[0]),
                        'height': round(box[3] - box[1]),
                    },
                }

                # Include RLE mask if available
                if masks is not None:
                    mask_hw = masks[i, 0]  # [H, W] boolean
                    instance['mask_rle'] = _mask_to_rle(mask_hw)
                    instance['mask_shape'] = [mask_hw.shape[0], mask_hw.shape[1]]

                instances.append(instance)

            # Sort by score descending
            instances.sort(key=lambda x: x['score'], reverse=True)
            results[noun] = {'instances': instances}

    return results


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_image_from_request(req) -> Image.Image:
    """Extract a PIL image from a Flask request (file upload, URL, or base64)."""
    if req.content_type and 'multipart/form-data' in req.content_type:
        f = req.files.get('file')
        if not f:
            raise ValueError("No file in multipart request")
        return Image.open(io.BytesIO(f.read()))

    data = req.get_json(silent=True) or {}

    if 'image_url' in data:
        resp = requests.get(data['image_url'], timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    if 'image_b64' in data:
        raw = base64.b64decode(data['image_b64'])
        return Image.open(io.BytesIO(raw))

    if 'file' in data:
        path = data['file']
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        return Image.open(path)

    raise ValueError("Provide image via file upload, 'image_url', 'image_b64', or 'file' path")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, origins=['*'])


@app.route('/health', methods=['GET'])
def health():
    loaded = _processor is not None
    return jsonify({
        'status': 'healthy' if loaded else 'unhealthy',
        'model': 'SAM3',
        'device': _device,
        'model_loaded': loaded,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
    }), 200 if loaded else 503


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Segment objects described by text nouns.

    Accepts multipart/form-data with 'file' + 'nouns' (JSON array or
    comma-separated string), or JSON with image source + 'nouns'.

    Returns per-noun detected instances with bounding boxes and scores.
    """
    start = time.time()

    try:
        # --- resolve nouns ---
        nouns = None
        if request.content_type and 'multipart/form-data' in request.content_type:
            raw = request.form.get('nouns', '')
            try:
                nouns = json.loads(raw) if raw.startswith('[') else [n.strip() for n in raw.split(',')]
            except Exception:
                nouns = [n.strip() for n in raw.split(',')]
        else:
            data = request.get_json(silent=True) or {}
            nouns_val = data.get('nouns', [])
            if isinstance(nouns_val, str):
                nouns = [n.strip() for n in nouns_val.split(',')]
            else:
                nouns = list(nouns_val)

        nouns = [n for n in nouns if n]
        if not nouns:
            return jsonify({'error': "Provide 'nouns' as a JSON array or comma-separated string"}), 400

        # --- resolve image ---
        try:
            image = _load_image_from_request(request)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        # --- run SAM3 ---
        results = segment_nouns(image, nouns)

        processing_time = round(time.time() - start, 3)
        total_instances = sum(len(v['instances']) for v in results.values())

        logger.info(
            "SAM3: %d nouns, %d instances total, %.3fs",
            len(nouns), total_instances, processing_time
        )

        return jsonify({
            'service': 'sam3',
            'status': 'success',
            'nouns_queried': nouns,
            'results': results,
            'image_dimensions': {'width': image.width, 'height': image.height},
            'metadata': {
                'processing_time': processing_time,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'total_instances': total_instances,
            },
        })

    except Exception as e:
        logger.error("SAM3 analyze error: %s", e, exc_info=True)
        return jsonify({'error': str(e), 'processing_time': round(time.time() - start, 3)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not load_model():
        logger.error("SAM3 model failed to load — service will return 503")

    host = '127.0.0.1' if PRIVATE else '0.0.0.0'
    logger.info("Starting SAM3 service on %s:%d", host, PORT)
    app.run(host=host, port=PORT, debug=False, use_reloader=False, threaded=False)
