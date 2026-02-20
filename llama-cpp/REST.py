import base64
import json
import requests
import os
import time
import logging
import random
import nltk
from nltk.tokenize import MWETokenizer
from io import BytesIO
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE', 'true')
AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'

FOLDER = './uploads'
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')
LLAMA_SERVER_HOST = os.getenv('LLAMA_SERVER_HOST')
TEMPERATURE_STR = os.getenv('TEMPERATURE')
VISION_PROMPT = os.getenv('PROMPT')

if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not LLAMA_SERVER_HOST:
    raise ValueError("LLAMA_SERVER_HOST environment variable is required")
if not TEMPERATURE_STR:
    raise ValueError("TEMPERATURE environment variable is required")
if not VISION_PROMPT:
    raise ValueError("PROMPT environment variable is required")

PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
PORT = int(PORT_STR)
TEMPERATURE = float(TEMPERATURE_STR)
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
MAX_RESPONSE_LENGTH = 4000

os.makedirs(FOLDER, exist_ok=True)

emoji_mappings = {}
emoji_tokenizer = None


def load_emoji_mappings():
    """Load emoji mappings from GitHub, fall back to local cache"""
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
    local_cache_path = 'emoji_mappings.json'

    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 LLaMA.cpp: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            logger.info("✅ LLaMA.cpp: Loaded emoji mappings from GitHub and cached locally")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  LLaMA.cpp: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")

    try:
        with open(local_cache_path, 'r') as f:
            local_mappings = json.load(f)
            logger.info("✅ LLaMA.cpp: Successfully loaded emoji mappings from local cache")
            return local_mappings
    except Exception as local_error:
        logger.error(f"❌ LLaMA.cpp: Failed to load local emoji mappings: {local_error}")
        raise Exception(f"Both GitHub and local emoji mappings failed: Local cache={local_error}")


def load_mwe_mappings():
    """Load MWE mappings from GitHub, fall back to local cache, convert to tuples"""
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
    local_cache_path = 'mwe.txt'

    def process_mwe_lines(mwe_text):
        mwe_tuples = []
        for line in mwe_text:
            if line.strip():
                mwe_tuples.append(tuple(line.strip().replace('_', ' ').split()))
        return mwe_tuples

    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 LLaMA.cpp: Loading MWE mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            with open(local_cache_path, 'w') as f:
                f.write(response.text)
            mwe_tuples = process_mwe_lines(response.text.splitlines())
            logger.info("✅ LLaMA.cpp: Loaded MWE mappings from GitHub and cached locally")
            return mwe_tuples
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  LLaMA.cpp: Failed to load MWE mappings from GitHub ({e}), falling back to local cache")

    try:
        with open(local_cache_path, 'r') as f:
            local_mwe_text = f.read().splitlines()
            mwe_tuples = process_mwe_lines(local_mwe_text)
            logger.info(f"✅ LLaMA.cpp: Successfully loaded {len(mwe_tuples)} multi-word expressions from local cache")
            return mwe_tuples
    except Exception as local_error:
        logger.error(f"❌ LLaMA.cpp: Failed to load local MWE mappings: {local_error}")
        raise Exception(f"Both GitHub and local MWE mappings failed: Local cache={local_error}")


emoji_mappings = load_emoji_mappings()
mwe_mappings = load_mwe_mappings()
emoji_tokenizer = MWETokenizer(mwe_mappings, separator='_')

PRIORITY_OVERRIDES = {
    'glass': '🥛',
    'glasses': '👓',
    'wood': '🌲',
    'wooden': '🌲',
    'metal': '🔧',
    'metallic': '🔧',
}


def get_emoji_for_word(word: str) -> str:
    """Get emoji for a single word with morphological variations"""
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


def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    return roll == 1, roll


def lookup_text_for_emojis(text: str) -> Dict[str, Any]:
    """Look up emojis for text with tokenization"""
    if not text or not text.strip():
        return {"mappings": {}, "found_emojis": []}

    try:
        word_tokens = []
        for token in text.split():
            if token:
                clean_token = token.strip('.,!?;:"()[]{}\'`')
                if clean_token:
                    word_tokens.append(clean_token)

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
        logger.error(f"LLaMA.cpp: Failed to process emoji mappings: {e}")
        return {"mappings": {}, "found_emojis": []}


def get_emojis_for_text(text: str) -> List[Dict[str, str]]:
    """Extract emojis from sentence using local emoji service"""
    if not text or not text.strip():
        return []

    try:
        result = lookup_text_for_emojis(text)

        found_mappings = []
        seen_words = set()
        for word, emoji in result["mappings"].items():
            normalized_word = word.lower().replace(' ', '_')
            if normalized_word not in seen_words:
                seen_words.add(normalized_word)

                is_shiny, shiny_roll = check_shiny()
                mapping = {"word": normalized_word, "emoji": emoji}

                if is_shiny:
                    mapping["shiny"] = True
                    logger.info(f"✨ SHINY {normalized_word.upper()} EMOJI DETECTED! Roll: {shiny_roll} ✨")

                found_mappings.append(mapping)

        return found_mappings

    except Exception as e:
        logger.error(f"LLaMA.cpp: Failed to process emoji mappings: {e}")
        return []


def process_image_with_llamacpp(image: Image.Image, prompt: str = None, temperature: float = None) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns llama-server vision analysis data.
    Uses /v1/chat/completions with OAI image_url format — required for llama.cpp v8100+
    which uses the mtmd multimodal library. The legacy /completion + image_data approach
    does not correctly embed images in this version.
    """
    start_time = time.time()
    prompt = prompt or VISION_PROMPT
    temperature = temperature if temperature is not None else TEMPERATURE

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        jpg_buffer = BytesIO()
        image.save(jpg_buffer, 'JPEG', quality=95)
        image_b64 = base64.b64encode(jpg_buffer.getvalue()).decode('utf-8')
        jpg_buffer.close()

        payload = {
            "model": "llava-llama3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 75,
            "temperature": temperature,
        }

        response = requests.post(
            f"{LLAMA_SERVER_HOST}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        output = result["choices"][0]["message"]["content"].strip()

        if len(output) > MAX_RESPONSE_LENGTH:
            output = output[:MAX_RESPONSE_LENGTH] + "... [truncated]"

        processing_time = round(time.time() - start_time, 3)
        emoji_list = get_emojis_for_text(output)

        return {
            "success": True,
            "data": {
                "response": output,
                "prompt": prompt,
                "response_length": len(output),
                "emoji_mappings": emoji_list,
                "type": "vision",
                "temperature": temperature,
            },
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = round(time.time() - start_time, 3)
        return {
            "success": False,
            "error": f"Image analysis failed: {str(e)}",
            "processing_time": processing_time,
        }


def create_response(data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized response with emoji mappings"""
    text_response = data.get('response', '')
    emoji_list = data.get('emoji_mappings', [])

    is_shiny, shiny_roll = check_shiny()

    prediction = {"text": text_response}

    if emoji_list:
        prediction["emoji_mappings"] = emoji_list

    if is_shiny:
        prediction["shiny"] = True
        logger.info(f"✨ SHINY LLAMA.CPP VISION ANALYSIS! Roll: {shiny_roll} ✨")

    return {
        "service": "llama.cpp",
        "status": "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "llama.cpp",
                "server": LLAMA_SERVER_HOST,
                "prompt": data.get('prompt', VISION_PROMPT),
            }
        }
    }


app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint — verifies llama-server is reachable"""
    try:
        response = requests.get(f"{LLAMA_SERVER_HOST}/health", timeout=5)
        server_status = response.json() if response.ok else {"status": "error"}
        status = "healthy" if response.ok else "unhealthy"
    except Exception as e:
        server_status = {"status": f"error: {str(e)}"}
        status = "unhealthy"

        return jsonify({
            "status": status,
            "reason": f"llama-server unreachable: {str(e)}",
            "service": "llama.cpp Vision API",
        }), 503

    return jsonify({
        "status": status,
        "service": "llama.cpp Vision API",
        "llama_server": {
            "host": LLAMA_SERVER_HOST,
            "status": server_status.get("status"),
        },
        "endpoints": [
            "GET /health - Health check",
            "GET,POST /v3/analyze - Unified endpoint (url or file)",
            "GET,POST /analyze - Unified endpoint (url or file)",
        ]
    })


@app.route('/analyze', methods=['GET', 'POST'])
@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint"""
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "llama.cpp",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        prompt = request.args.get('prompt') or request.form.get('prompt') or VISION_PROMPT
        temperature_param = request.args.get('temperature') or request.form.get('temperature')
        temperature = float(temperature_param) if temperature_param else TEMPERATURE

        # Get image into memory from any source
        if request.method == 'POST' and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")

            uploaded_file.seek(0, 2)
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)

            if file_size > MAX_FILE_SIZE:
                return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")

            try:
                image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process uploaded image: {str(e)}", 500)

        else:
            url = request.args.get('url')
            file_path = request.args.get('file')

            if not url and not file_path:
                return error_response("Must provide 'url' or 'file' parameter, or POST a file")

            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                try:
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    if len(r.content) > MAX_FILE_SIZE:
                        return error_response("Downloaded file too large")
                    image = Image.open(BytesIO(r.content)).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)

        processing_result = process_image_with_llamacpp(image, prompt, temperature)

        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)

        return jsonify(create_response(
            processing_result["data"],
            processing_result["processing_time"]
        ))

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info("Starting llama.cpp vision service...")
    logger.info(f"Port: {PORT}")
    logger.info(f"llama-server: {LLAMA_SERVER_HOST}")
    logger.info(f"Private mode: {PRIVATE}")

    try:
        r = requests.get(f"{LLAMA_SERVER_HOST}/health", timeout=5)
        if r.ok:
            logger.info(f"✅ llama-server reachable at {LLAMA_SERVER_HOST}")
        else:
            raise ValueError(f"Unexpected status {r.status_code}")
    except Exception as e:
        raise RuntimeError(f"llama-server not reachable at {LLAMA_SERVER_HOST}: {e}")

    host = "0.0.0.0" if not PRIVATE else "127.0.0.1"
    app.run(host=host, port=PORT, debug=False, threaded=True)
