import base64
import json
import logging
import os
import random
import sys
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.tokenize import MWETokenizer
from PIL import Image

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SERVICE_DIR)

load_dotenv(os.path.join(_SERVICE_DIR, ".env"))

from joycaption_analyzer import DEFAULT_MODEL, DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT, JoyCaptionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

PRIVATE_STR = os.getenv("PRIVATE")
PORT_STR = os.getenv("PORT")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

SERVICE_NAME = "joycaption"
PRIVATE = PRIVATE_STR.lower() in {"true", "1", "yes"}
PORT = int(PORT_STR)
AUTO_UPDATE = os.getenv("AUTO_UPDATE", "true").lower() == "true"
TIMEOUT = float(os.getenv("TIMEOUT", "15.0"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(32 * 1024 * 1024)))
MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "4000"))
RAW_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}

MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL)
DEVICE = os.getenv("DEVICE", "auto")
PRECISION = os.getenv("PRECISION", "8bit")
PROMPT = os.getenv("PROMPT", DEFAULT_PROMPT)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
TOP_P_STR = os.getenv("TOP_P", "0.9")
TOP_P = None if TOP_P_STR.lower() in {"none", "null", ""} else float(TOP_P_STR)
GREEDY = os.getenv("GREEDY", "false").lower() in {"true", "1", "yes"}

emoji_mappings: Dict[str, str] = {}
emoji_tokenizer: Optional[MWETokenizer] = None
joycaption_analyzer: Optional[JoyCaptionAnalyzer] = None

PRIORITY_OVERRIDES = {
    "glass": "🥛",
    "glasses": "👓",
    "wood": "🌲",
    "wooden": "🌲",
    "metal": "🔧",
    "metallic": "🔧",
}


def load_emoji_mappings() -> Dict[str, str]:
    local_cache_path = os.path.join(_SERVICE_DIR, "emoji_mappings.json")
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"

    if AUTO_UPDATE:
        try:
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            with open(local_cache_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            return data
        except Exception as exc:
            logger.warning("JoyCaption: failed to refresh emoji mappings: %s", exc)

    with open(local_cache_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_mwe_mappings() -> List[tuple]:
    local_cache_path = os.path.join(_SERVICE_DIR, "mwe.txt")
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
    lines: List[str] = []

    if AUTO_UPDATE:
        try:
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            lines = response.text.splitlines()
            with open(local_cache_path, "w", encoding="utf-8") as handle:
                handle.write(response.text)
        except Exception as exc:
            logger.warning("JoyCaption: failed to refresh MWE mappings: %s", exc)

    if not lines:
        with open(local_cache_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()

    return [tuple(line.strip().replace("_", " ").split()) for line in lines if line.strip()]


def get_emoji_for_word(word: str) -> Optional[str]:
    word_clean = word.lower().strip()
    if word_clean in PRIORITY_OVERRIDES:
        return PRIORITY_OVERRIDES[word_clean]
    if word_clean in emoji_mappings:
        return emoji_mappings[word_clean]
    if word_clean.endswith("s") and len(word_clean) > 3:
        return emoji_mappings.get(word_clean[:-1])
    return None


def check_shiny():
    roll = random.randint(1, 2500)
    return roll == 1, roll


def get_emojis_for_text(text: str) -> List[Dict[str, str]]:
    if not text or emoji_tokenizer is None:
        return []

    word_tokens = []
    for token in text.split():
        clean_token = token.strip(".,!?;:\"()[]{}'`")
        if clean_token:
            word_tokens.append(clean_token.lower())

    found = []
    seen = set()
    for token in emoji_tokenizer.tokenize(word_tokens):
        emoji = get_emoji_for_word(token)
        normalized = token.lower().replace(" ", "_")
        if emoji and normalized not in seen:
            seen.add(normalized)
            mapping = {"word": normalized, "emoji": emoji}
            is_shiny, shiny_roll = check_shiny()
            if is_shiny:
                mapping["shiny"] = True
                logger.info("Shiny JoyCaption emoji mapping detected for %s: roll=%s", normalized, shiny_roll)
            found.append(mapping)
    return found


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def request_value(name: str, default: Any = None) -> Any:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if name in payload:
            return payload[name]
    if name in request.form:
        return request.form.get(name)
    return request.args.get(name, default)


def is_raw_image_request() -> bool:
    return (request.content_type or "").split(";", 1)[0].strip().lower() in RAW_IMAGE_CONTENT_TYPES


def image_from_request() -> Image.Image:
    if request.method == "POST" and is_raw_image_request():
        data = request.get_data(cache=False)
        if not data:
            raise ValueError("No image body provided")
        if len(data) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB")
        return Image.open(BytesIO(data)).convert("RGB")

    if request.method == "POST" and "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            raise ValueError("No file selected")
        uploaded_file.seek(0, 2)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB")
        return Image.open(BytesIO(uploaded_file.read())).convert("RGB")

    image_b64 = request_value("image_base64")
    if image_b64:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        data = base64.b64decode(image_b64)
        if len(data) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB")
        return Image.open(BytesIO(data)).convert("RGB")

    url = request_value("url")
    file_path = request_value("file")
    if url and file_path:
        raise ValueError("Cannot provide both 'url' and 'file' parameters")
    if url:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Downloaded file too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB")
        return Image.open(BytesIO(response.content)).convert("RGB")
    if file_path:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        return Image.open(file_path).convert("RGB")

    raise ValueError("Must provide 'url', 'file', 'image_base64', or POST a multipart file")


def create_response(caption: str, prompt: str, processing_time: float, generation: Dict[str, Any]) -> Dict[str, Any]:
    if len(caption) > MAX_RESPONSE_LENGTH:
        caption = caption[:MAX_RESPONSE_LENGTH] + "... [truncated]"

    prediction: Dict[str, Any] = {"text": caption}
    emoji_list = get_emojis_for_text(caption)
    if emoji_list:
        prediction["emoji_mappings"] = emoji_list

    is_shiny, shiny_roll = check_shiny()
    if is_shiny:
        prediction["shiny"] = True
        logger.info("Shiny JoyCaption prediction detected: roll=%s", shiny_roll)

    return {
        "service": SERVICE_NAME,
        "status": "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "transformers/llava",
                "model": MODEL_ID,
                "device": joycaption_analyzer.device if joycaption_analyzer else DEVICE,
                "precision": PRECISION,
                "prompt": prompt,
                **generation,
            },
        },
    }


app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])


@app.route("/health", methods=["GET"])
def health_check():
    model_loaded = joycaption_analyzer is not None and joycaption_analyzer.model is not None
    status = "healthy" if model_loaded else "unhealthy"
    return jsonify({
        "status": status,
        "schema_version": "health.v1",
        "service": "JoyCaption Vision API",
        "model": {
            "name": MODEL_ID,
            "status": "loaded" if model_loaded else "not_loaded",
            "framework": "transformers/llava",
            "device": joycaption_analyzer.device if joycaption_analyzer else DEVICE,
            "precision": PRECISION,
        },
        "dependencies": {},
        "warnings": [] if model_loaded else ["JoyCaption model is not loaded"],
        "endpoints": [
            "GET /health - Health check",
            "GET,POST /v3/analyze - Unified endpoint (url, file, image_base64, or multipart file)",
            "GET,POST /analyze - Unified endpoint (url, file, image_base64, or multipart file)",
        ],
    }), 200 if model_loaded else 503


@app.route("/analyze", methods=["GET", "POST"])
@app.route("/v3/analyze", methods=["GET", "POST"])
def analyze():
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": SERVICE_NAME,
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)},
        }), status_code

    if joycaption_analyzer is None or joycaption_analyzer.model is None:
        return error_response("JoyCaption model is not loaded", 503)

    try:
        prompt = str(request_value("prompt", PROMPT))
        system_prompt = str(request_value("system_prompt", SYSTEM_PROMPT))
        max_new_tokens = int(request_value("max_new_tokens", MAX_NEW_TOKENS))
        temperature = float(request_value("temperature", TEMPERATURE))
        top_p_raw = request_value("top_p", TOP_P)
        top_p = None if str(top_p_raw).lower() in {"none", "null", ""} else float(top_p_raw)
        greedy = parse_bool(request_value("greedy", GREEDY))

        image = image_from_request()
        result = joycaption_analyzer.caption(
            image=image,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            greedy=greedy,
        )
        if not result["success"]:
            return error_response(f"Caption generation failed: {result['error']}", 500)

        return jsonify(create_response(
            result["caption"],
            prompt,
            time.time() - start_time,
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "greedy": greedy,
            },
        ))
    except ValueError as exc:
        return error_response(str(exc))
    except Exception as exc:
        logger.exception("JoyCaption analyze failed: %s", exc)
        return error_response(f"Internal error: {exc}", 500)


if __name__ == "__main__":
    logger.info("Starting JoyCaption service on port %s", PORT)
    logger.info("Model: %s", MODEL_ID)
    logger.info("Device: %s, precision: %s", DEVICE, PRECISION)

    emoji_mappings = load_emoji_mappings()
    emoji_tokenizer = MWETokenizer(load_mwe_mappings(), separator="_")

    joycaption_analyzer = JoyCaptionAnalyzer(
        model_id=MODEL_ID,
        precision=PRECISION,
        device=DEVICE,
        system_prompt=SYSTEM_PROMPT,
    )
    if not joycaption_analyzer.initialize():
        raise RuntimeError("Failed to initialize JoyCaption model")

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"
    app.run(host=host, port=PORT, debug=False, threaded=True)
