#!/usr/bin/env python3
"""
Florence-2 REST API Service - Multi-task vision analysis

Provides REST endpoints for image analysis using Microsoft Florence-2.
Supports captioning, object detection, OCR, phrase grounding, and segmentation.
"""

import json
import requests
import os
import sys
import logging
import random
import time
import nltk
from nltk.tokenize import MWETokenizer
from typing import List, Dict, Any, Optional
from io import BytesIO

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SERVICE_DIR, '..', 'shared')
sys.path.insert(0, _SERVICE_DIR)
sys.path.insert(0, _SHARED_DIR)

from dotenv import load_dotenv
load_dotenv()

TIMEOUT = float(os.getenv('TIMEOUT', '10.0'))
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'

import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from florence2_analyzer import FlorenceAnalyzer, VALID_TASKS, TEXT_REQUIRED_TASKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

PRIVATE = os.getenv('PRIVATE', 'false').lower() == 'true'
PORT = int(os.getenv('PORT', '7803'))
MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/Florence-2-large')
DEFAULT_TASK = os.getenv('DEFAULT_TASK', 'DENSE_REGION_CAPTION')

# Global analyzer - initialized once at startup
florence_analyzer = None

# Emoji infrastructure
emoji_mappings = {}
emoji_tokenizer = None

PRIORITY_OVERRIDES = {
    'glass': '🥛',
    'glasses': '👓',
    'wood': '🌲',
    'wooden': '🌲',
    'metal': '🔧',
    'metallic': '🔧',
}


# ---------------------------------------------------------------------------
# Emoji loading (same pattern as BLIP2)
# ---------------------------------------------------------------------------

def load_emoji_mappings():
    local_cache_path = os.path.join(_SERVICE_DIR, 'emoji_mappings.json')

    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
        try:
            logger.info(f"Florence-2: Loading emoji mappings from GitHub")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as cache_error:
                logger.warning(f"Florence-2: Failed to cache emoji mappings: {cache_error}")
            logger.info("Florence-2: Emoji mappings loaded from GitHub")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Florence-2: GitHub emoji load failed: {e}")
    else:
        logger.info("Florence-2: AUTO_UPDATE disabled, using local emoji cache")

    try:
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Florence-2: Emoji mappings loaded from local cache")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Florence-2: Failed to load emoji mappings: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from GitHub and local cache: {e}")
        raise Exception(f"Failed to load emoji mappings - no local cache and AUTO_UPDATE=False: {e}")


def load_mwe_mappings():
    local_cache_path = os.path.join(_SERVICE_DIR, 'mwe.txt')
    mwe_text = []

    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
        try:
            logger.info("Florence-2: Loading MWE mappings from GitHub")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            mwe_text = response.text.splitlines()
            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            except Exception as cache_error:
                logger.warning(f"Florence-2: Failed to cache MWE mappings: {cache_error}")
            logger.info("Florence-2: MWE mappings loaded from GitHub")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Florence-2: GitHub MWE load failed: {e}")
    else:
        logger.info("Florence-2: AUTO_UPDATE disabled, using local MWE cache")

    if not mwe_text:
        try:
            with open(local_cache_path, 'r', encoding='utf-8') as f:
                mwe_text = f.read().splitlines()
            logger.info("Florence-2: MWE mappings loaded from local cache")
        except FileNotFoundError as e:
            logger.error(f"Florence-2: Failed to load MWE mappings: {e}")
            if AUTO_UPDATE:
                raise Exception(f"Failed to load MWE mappings from GitHub and local cache: {e}")
            raise Exception(f"Failed to load MWE mappings - no local cache and AUTO_UPDATE=False: {e}")

    return [
        tuple(line.strip().replace('_', ' ').split())
        for line in mwe_text if line.strip()
    ]


def get_emoji_for_word(word: str) -> Optional[str]:
    if not word:
        return None
    word_clean = word.lower().strip()
    if word_clean in PRIORITY_OVERRIDES:
        return PRIORITY_OVERRIDES[word_clean]
    if word_clean in emoji_mappings:
        return emoji_mappings[word_clean]
    if word_clean.endswith('s') and len(word_clean) > 3:
        singular = word_clean[:-1]
        if singular in emoji_mappings:
            return emoji_mappings[singular]
    return None


def lookup_text_for_emojis(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"mappings": {}, "found_emojis": []}
    try:
        word_tokens = [
            token.strip('.,!?;:"()[]{}').lower()
            for token in text.split()
            if token.strip('.,!?;:"()[]{}')
        ]
        tokens = emoji_tokenizer.tokenize(word_tokens)
        mappings = {}
        found_emojis = []
        for token in tokens:
            emoji = get_emoji_for_word(token)
            if emoji:
                mappings[token] = emoji
                if emoji not in found_emojis:
                    found_emojis.append(emoji)
        return {"mappings": mappings, "found_emojis": found_emojis}
    except Exception as e:
        logger.error(f"Florence-2: Emoji lookup failed: {e}")
        return {"mappings": {}, "found_emojis": []}


def get_emojis_for_text(text: str):
    """Return (emojis[:3], word_mappings) for any text string."""
    if not text:
        return [], {}
    result = lookup_text_for_emojis(text)
    return result["found_emojis"][:3], result["mappings"]


def build_emoji_list(word_mappings: Dict[str, str]) -> list:
    return [{"emoji": e, "word": w} for w, e in word_mappings.items()]


# ---------------------------------------------------------------------------
# Shiny
# ---------------------------------------------------------------------------

def check_shiny():
    roll = random.randint(1, 2500)
    return roll == 1, roll


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------

# Caption tasks: raw is a plain string
CAPTION_TASKS = {"CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION"}

# Detection-style tasks: raw is {"bboxes": [...], "labels": [...]}
DETECTION_TASKS = {
    "OD",
    "DENSE_REGION_CAPTION",
    "CAPTION_TO_PHRASE_GROUNDING",
    "OPEN_VOCABULARY_DETECTION",
}

# OCR tasks
OCR_TASKS = {"OCR"}
OCR_REGION_TASKS = {"OCR_WITH_REGION"}

# Segmentation tasks: raw is {"polygons": [[...]], "labels": [...]}
SEGMENTATION_TASKS = {"REFERRING_EXPRESSION_SEGMENTATION"}


def normalize_output(task: str, raw: Any) -> List[Dict[str, Any]]:
    """
    Convert Florence-2 raw parsed output into Animal Farm prediction list.

    Caption tasks produce one prediction with a 'text' field.
    Detection tasks produce one prediction per detected region with 'label' and 'bbox'.
    OCR produces one prediction with 'text'.
    OCR_WITH_REGION produces one prediction per region with 'text' and 'quad_box'.
    Segmentation produces one prediction per region with 'label' and 'polygon'.
    """
    predictions = []

    if task in CAPTION_TASKS:
        text = str(raw).strip()
        _, word_mappings = get_emojis_for_text(text)
        prediction = {"task": task, "text": text}
        if word_mappings:
            prediction["emoji_mappings"] = build_emoji_list(word_mappings)
        is_shiny, roll = check_shiny()
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"Florence-2: Shiny caption! Roll: {roll}")
        predictions.append(prediction)

    elif task in DETECTION_TASKS:
        bboxes = raw.get("bboxes", [])
        labels = raw.get("labels", [])
        for bbox, label in zip(bboxes, labels):
            label_str = str(label).strip()
            _, word_mappings = get_emojis_for_text(label_str)
            # Florence-2 returns [x1, y1, x2, y2] absolute coords.
            # Convert to [x, y, width, height] as expected by Harmony.
            x1, y1, x2, y2 = bbox
            prediction = {
                "task": task,
                "label": label_str,
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            }
            if word_mappings:
                prediction["emoji_mappings"] = build_emoji_list(word_mappings)
            predictions.append(prediction)

        # CAPTION_TO_PHRASE_GROUNDING: drop component-word labels that are strict
        # subsets of a longer label in the same result (e.g. "code" when "qr code"
        # is also present).
        if task == "CAPTION_TO_PHRASE_GROUNDING" and len(predictions) > 1:
            all_words = [set(p["label"].lower().split()) for p in predictions]
            predictions = [
                p for i, p in enumerate(predictions)
                if not any(
                    all_words[i] < all_words[j]
                    for j in range(len(predictions)) if j != i
                )
            ]

    elif task in OCR_TASKS:
        text = str(raw).strip()
        _, word_mappings = get_emojis_for_text(text)
        prediction = {"task": task, "text": text}
        if word_mappings:
            prediction["emoji_mappings"] = build_emoji_list(word_mappings)
        predictions.append(prediction)

    elif task in OCR_REGION_TASKS:
        quad_boxes = raw.get("quad_boxes", [])
        labels = raw.get("labels", [])
        for quad_box, label in zip(quad_boxes, labels):
            label_str = str(label).strip()
            _, word_mappings = get_emojis_for_text(label_str)
            prediction = {
                "task": task,
                "text": label_str,
                "quad_box": [round(v, 2) for v in quad_box],
            }
            if word_mappings:
                prediction["emoji_mappings"] = build_emoji_list(word_mappings)
            predictions.append(prediction)

    elif task in SEGMENTATION_TASKS:
        polygons = raw.get("polygons", [])
        labels = raw.get("labels", [])
        for polygon_group, label in zip(polygons, labels):
            label_str = str(label).strip()
            # polygon_group is a list of polygon point lists; take the first
            polygon = polygon_group[0] if polygon_group else []
            prediction = {
                "task": task,
                "label": label_str,
                "polygon": [[round(polygon[i], 2), round(polygon[i+1], 2)]
                            for i in range(0, len(polygon) - 1, 2)],
            }
            predictions.append(prediction)

    else:
        # Unrecognized output shape — pass raw through
        predictions.append({"task": task, "raw": raw})

    return predictions


def create_response(task: str, predictions: list, processing_time: float) -> Dict[str, Any]:
    return {
        "service": "florence2",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "task": task,
            "model_info": {"model": MODEL_NAME, "framework": "Florence-2"},
        }
    }


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def download_image_from_url(url: str) -> Image.Image:
    headers = {'User-Agent': 'Florence-2 Vision Service'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    if len(response.content) > MAX_FILE_SIZE:
        raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB")
    image = Image.open(BytesIO(response.content))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def validate_image_file(file_path: str) -> Image.Image:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    image = Image.open(file_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def initialize_florence_analyzer() -> bool:
    global florence_analyzer
    try:
        logger.info("Initializing Florence-2 Analyzer...")
        florence_analyzer = FlorenceAnalyzer(model_name=MODEL_NAME)
        if not florence_analyzer.initialize():
            logger.error("Failed to initialize Florence-2 Analyzer")
            return False
        logger.info("Florence-2 Analyzer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Florence-2 Analyzer: {str(e)}")
        return False


# Load emoji infrastructure at module level (before Flask app, fail fast)
emoji_mappings = load_emoji_mappings()
mwe_mappings = load_mwe_mappings()
emoji_tokenizer = MWETokenizer(mwe_mappings, separator='_')

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("Florence-2 service: CORS enabled for direct browser communication")


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "status": "error"}), 413


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "status": "error"}), 400


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "status": "error"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    try:
        model_loaded = florence_analyzer is not None and florence_analyzer.model is not None
        device_str = str(florence_analyzer.device) if florence_analyzer else "unknown"
        return jsonify({
            "status": "healthy",
            "model_status": "loaded" if model_loaded else "not_loaded",
            "model": MODEL_NAME,
            "device": device_str,
            "default_task": DEFAULT_TASK,
            "valid_tasks": sorted(VALID_TASKS),
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"error": "Health check failed", "status": "error"}), 500


@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3():
    return analyze()


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint"""
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "florence2",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        # Determine task
        task = request.args.get('task', DEFAULT_TASK).upper()
        if task not in VALID_TASKS:
            return error_response(
                f"Unknown task '{task}'. Valid tasks: {sorted(VALID_TASKS)}"
            )

        # Text input (for tasks that require it)
        text_input = request.args.get('text') or None
        if task in TEXT_REQUIRED_TASKS and not text_input:
            return error_response(f"Task '{task}' requires the 'text' parameter")

        # Step 1: Get image
        image = None
        if request.method == 'POST' and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")
            file_data = uploaded_file.read()
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")
            image = Image.open(BytesIO(file_data)).convert('RGB')
        else:
            url = request.args.get('url')
            file = request.args.get('file')
            if not url and not file:
                return error_response("Must provide 'url' or 'file' parameter, or POST a file")
            if url and file:
                return error_response("Cannot provide both 'url' and 'file' parameters")
            if url:
                try:
                    image = download_image_from_url(url)
                except Exception as e:
                    return error_response(str(e))
            else:
                try:
                    image = validate_image_file(file)
                except Exception as e:
                    return error_response(str(e))

        # Step 2: Run inference
        result = florence_analyzer.analyze(image, task, text_input=text_input)
        if not result["success"]:
            return error_response(result["error"], 500)

        # Step 3: Normalize output
        predictions = normalize_output(task, result["raw"])

        # Step 4: Build response
        response = create_response(task, predictions, time.time() - start_time)
        return jsonify(response)

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


def _load_image_from_request() -> tuple:
    """
    Shared image loading for both single and batch endpoints.
    Returns (image, error_message). One of the two will be None.
    """
    if request.method == 'POST' and 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return None, "No file selected"
        if not is_allowed_file(uploaded_file.filename):
            return None, "File type not allowed"
        file_data = uploaded_file.read()
        if len(file_data) > MAX_FILE_SIZE:
            return None, "File too large"
        return Image.open(BytesIO(file_data)).convert('RGB'), None

    url = request.args.get('url')
    file = request.args.get('file')
    if not url and not file:
        return None, "Must provide 'url' or 'file' parameter, or POST a file"
    if url and file:
        return None, "Cannot provide both 'url' and 'file' parameters"
    try:
        if url:
            return download_image_from_url(url), None
        return validate_image_file(file), None
    except Exception as e:
        return None, str(e)


@app.route('/v3/analyze/batch', methods=['POST'])
@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Multi-task endpoint: encode image once, run N tasks, return all results.

    Request body (JSON):
        {
          "tasks": [
            {"task": "MORE_DETAILED_CAPTION"},
            {"task": "OD"},
            {"task": "CAPTION_TO_PHRASE_GROUNDING", "text": "a woman on grass"}
          ]
        }

    Image supplied via multipart file upload, ?url=, or ?file= (same as /analyze).
    """
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "florence2",
            "status": "error",
            "error": {"message": message},
            "metadata": {"total_processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        # Parse task list from JSON body
        body = request.get_json(silent=True) or {}
        tasks = body.get("tasks")
        if not tasks or not isinstance(tasks, list):
            return error_response("Request body must be JSON with a 'tasks' array")
        if len(tasks) == 0:
            return error_response("'tasks' array must not be empty")

        # Load image
        image, err = _load_image_from_request()
        if err:
            return error_response(err)

        # Run all tasks (pixel_values computed once inside analyze_batch)
        batch_results = florence_analyzer.analyze_batch(image, tasks)

        # Build per-task response entries
        results = {}
        succeeded = 0
        failed = 0
        for result in batch_results:
            task = result["task"]
            if result["success"]:
                predictions = normalize_output(task, result["raw"])
                # Use a unique key if the same task appears more than once
                key = task
                suffix = 2
                while key in results:
                    key = f"{task}_{suffix}"
                    suffix += 1
                results[key] = {
                    "predictions": predictions,
                    "processing_time": round(result.get("processing_time", 0), 3),
                }
                succeeded += 1
            else:
                key = task
                suffix = 2
                while key in results:
                    key = f"{task}_{suffix}"
                    suffix += 1
                results[key] = {"error": result["error"]}
                failed += 1

        if succeeded == 0:
            status = "error"
        elif failed > 0:
            status = "partial"
        else:
            status = "success"

        return jsonify({
            "service": "florence2",
            "status": status,
            "results": results,
            "metadata": {
                "total_processing_time": round(time.time() - start_time, 3),
                "task_count": len(batch_results),
                "succeeded": succeeded,
                "failed": failed,
                "model_info": {"model": MODEL_NAME, "framework": "Florence-2"},
            }
        })

    except Exception as e:
        logger.error(f"Batch analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info("Starting Florence-2 service...")

    if not initialize_florence_analyzer():
        logger.error("Failed to initialize Florence-2 analyzer. Exiting.")
        exit(1)

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    logger.info(f"Starting Florence-2 service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Default task: {DEFAULT_TASK}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
