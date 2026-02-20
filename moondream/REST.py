import json
import requests
import os
import sys
import logging
import random
import nltk
from nltk.tokenize import MWETokenizer
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Optional configuration with defaults
TIMEOUT = float(os.getenv('TIMEOUT', '10.0'))
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'

import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from moondream_analyzer import MoondreamAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')

# Validate required configuration - fail fast
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

# Convert types
PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)

# Optional configuration with defaults
MODEL_ID = os.getenv('MODEL_ID', 'vikhyatk/moondream2')
MODEL_REVISION = os.getenv('MODEL_REVISION') or None
CAPTION_LENGTH = os.getenv('CAPTION_LENGTH', 'normal')

if CAPTION_LENGTH not in ('short', 'normal', 'long'):
    raise ValueError("CAPTION_LENGTH must be 'short', 'normal', or 'long'")

# Load emoji mappings from GitHub raw files with local caching
emoji_mappings = {}
emoji_tokenizer = None

# Global Moondream analyzer - initialize once at startup
moondream_analyzer = None

# Priority overrides for critical ambiguities
PRIORITY_OVERRIDES = {
    'glass': '🥛',
    'glasses': '👓',
    'wood': '🌲',
    'wooden': '🌲',
    'metal': '🔧',
    'metallic': '🔧',
}


def load_emoji_mappings():
    """Load emoji mappings from GitHub raw files with local caching"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')

    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
        try:
            logger.info(f"Moondream: Loading fresh emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()

            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Moondream: Cached emoji mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"Moondream: Failed to cache emoji mappings: {cache_error}")

            logger.info("Moondream: Successfully loaded emoji mappings from GitHub")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Moondream: Failed to load emoji mappings from GitHub: {e}")
            logger.info("Moondream: Falling back to local cache due to GitHub failure")
    else:
        logger.info("Moondream: AUTO_UPDATE disabled, using local cache only")

    try:
        logger.info(f"Moondream: Loading emoji mappings from local cache: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Moondream: Successfully loaded emoji mappings from local cache")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Moondream: Failed to load local emoji mappings from {local_cache_path}: {e}")
        if AUTO_UPDATE:
            raise Exception(f"Failed to load emoji mappings from both GitHub and local cache: {e}")
        else:
            raise Exception(
                f"Failed to load emoji mappings - AUTO_UPDATE disabled and no local cache available. "
                f"Set AUTO_UPDATE=True or provide emoji_mappings.json in moondream directory: {e}"
            )


def load_mwe_mappings():
    """Load MWE mappings from GitHub raw files with local caching and convert to tuples"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'mwe.txt')
    mwe_text = []

    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
        try:
            logger.info(f"Moondream: Loading fresh MWE mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            mwe_text = response.text.splitlines()

            try:
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Moondream: Cached MWE mappings to {local_cache_path}")
            except Exception as cache_error:
                logger.warning(f"Moondream: Failed to cache MWE mappings: {cache_error}")

            logger.info("Moondream: Successfully loaded MWE mappings from GitHub")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Moondream: Failed to load MWE mappings from GitHub: {e}")
            logger.info("Moondream: Falling back to local cache due to GitHub failure")
    else:
        logger.info("Moondream: AUTO_UPDATE disabled, using local cache only")

    if not mwe_text:
        try:
            logger.info(f"Moondream: Loading MWE mappings from local cache: {local_cache_path}")
            with open(local_cache_path, 'r', encoding='utf-8') as f:
                mwe_text = f.read().splitlines()
            logger.info("Moondream: Successfully loaded MWE mappings from local cache")
        except FileNotFoundError as e:
            logger.error(f"Moondream: Failed to load local MWE mappings from {local_cache_path}: {e}")
            if AUTO_UPDATE:
                raise Exception(f"Failed to load MWE mappings from both GitHub and local cache: {e}")
            else:
                raise Exception(
                    f"Failed to load MWE mappings - AUTO_UPDATE disabled and no local cache available. "
                    f"Set AUTO_UPDATE=True or provide mwe.txt in moondream directory: {e}"
                )

    mwe_tuples = []
    for line in mwe_text:
        if line.strip():
            mwe_tuples.append(tuple(line.strip().replace('_', ' ').split()))

    return mwe_tuples


# Load emoji mappings on startup
emoji_mappings = load_emoji_mappings()
mwe_mappings = load_mwe_mappings()

# Initialize MWE tokenizer
emoji_tokenizer = MWETokenizer(mwe_mappings, separator='_')


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
    is_shiny = roll == 1
    return is_shiny, roll


def lookup_text_for_emojis(text: str) -> Dict[str, Any]:
    """Look up emojis for text with tokenization"""
    if not text or not text.strip():
        return {"mappings": {}, "found_emojis": []}

    try:
        word_tokens = []
        for token in text.split():
            token = token.strip('.,!?;:"()[]{}\'`')
            if token:
                word_tokens.append(token)

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
        logger.error(f"Moondream: Failed to process emoji mappings: {e}")
        return {"mappings": {}, "found_emojis": []}


logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")


def initialize_moondream_analyzer() -> bool:
    """Initialize Moondream analyzer once at startup - fail fast"""
    global moondream_analyzer
    try:
        logger.info("Initializing Moondream Analyzer...")

        moondream_analyzer = MoondreamAnalyzer(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            caption_length=CAPTION_LENGTH,
        )

        if not moondream_analyzer.initialize():
            logger.error("Failed to initialize Moondream Analyzer")
            return False

        logger.info("Moondream Analyzer initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing Moondream Analyzer: {e}")
        return False


def get_emojis_and_mappings_for_caption(caption: str) -> tuple[List[str], Dict[str, str]]:
    """Extract emojis and word mappings from caption using local emoji service"""
    if not caption:
        return [], {}

    try:
        result = lookup_text_for_emojis(caption)
        emojis = result["found_emojis"]
        mappings = result["mappings"]
        return emojis[:3], mappings
    except Exception as e:
        logger.error(f"Moondream: Failed to process emoji mappings: {e}")

    return [], {}


def process_image_for_caption(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns caption data
    Core business logic, separated from HTTP concerns
    """
    try:
        result = moondream_analyzer.analyze_caption_from_array(image)

        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error', 'Caption generation failed')
            }

        caption = result.get('caption', '')
        emojis, word_mappings = get_emojis_and_mappings_for_caption(caption)

        return {
            "success": True,
            "caption": caption,
            "emojis": emojis,
            "word_mappings": word_mappings
        }

    except Exception as e:
        logger.error(f"Error processing image for caption: {e}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}"
        }


def create_moondream_response(caption: str, word_mappings: Dict[str, str], processing_time: float) -> Dict[str, Any]:
    """Create standardized Moondream response with metadata"""
    is_shiny, shiny_roll = check_shiny()

    mapped_emojis = []
    for word, emoji in word_mappings.items():
        mapped_emojis.append({
            "word": word,
            "emoji": emoji
        })

    prediction = {
        "text": caption,
        "emoji_mappings": mapped_emojis
    }

    if is_shiny:
        prediction["shiny"] = True
        logger.info(f"SHINY CAPTION GENERATED! Roll: {shiny_roll}")

    return {
        "service": "moondream",
        "status": "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {"framework": "Moondream2 (vikhyatk/moondream2)"}
        }
    }


def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image"""
    try:
        headers = {'User-Agent': 'Moondream Caption Generation Service'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")

        from io import BytesIO
        image = Image.open(BytesIO(response.content))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not is_allowed_file(file_path):
        raise ValueError("File type not allowed")

    try:
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
print("Moondream service: CORS enabled for direct browser communication")


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
    """Health check endpoint"""
    try:
        model_status = "loaded" if moondream_analyzer and moondream_analyzer.model else "not_loaded"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "device": device_str
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"error": "Health check failed", "status": "error"}), 500


@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3():
    """V3 analyze endpoint"""
    return analyze()


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    import time
    from io import BytesIO
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "moondream",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        image = None

        # Step 1: Get image into memory from any source
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
                return error_response("Must provide either url or file parameter, or POST a file")

            if url and file:
                return error_response("Cannot provide both url and file parameters")

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

        # Step 2: Process the image
        processing_result = process_image_for_caption(image)

        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)

        # Step 4: Create response
        response = create_moondream_response(
            processing_result["caption"],
            processing_result["word_mappings"],
            time.time() - start_time
        )

        return jsonify(response)

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info("Starting Moondream service...")

    model_loaded = initialize_moondream_analyzer()

    if not model_loaded:
        logger.error("Failed to initialize Moondream analyzer. Service cannot function.")
        exit(1)

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"Starting Moondream service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model: {MODEL_ID}@{MODEL_REVISION or 'LATEST_REVISION'}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
