import os
import sys
import logging
import time
import base64
import requests
from io import BytesIO
from urllib.parse import urlparse
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from ben2_analyzer import BEN2Analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB — background removal benefits from larger images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Required configuration — fail fast
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')
BEN2_MODEL_PATH = os.getenv('BEN2_MODEL_PATH')
BEN2_CODE_DIR = os.getenv('BEN2_CODE_DIR')

if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not BEN2_MODEL_PATH:
    raise ValueError("BEN2_MODEL_PATH environment variable is required")
if not BEN2_CODE_DIR:
    raise ValueError("BEN2_CODE_DIR environment variable is required")

PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)

# Optional configuration
REFINE_FOREGROUND = os.getenv('REFINE_FOREGROUND', 'false').lower() == 'true'

# Validate paths exist at startup
if not os.path.exists(BEN2_MODEL_PATH):
    raise FileNotFoundError(f"BEN2_MODEL_PATH does not exist: {BEN2_MODEL_PATH}")
if not os.path.exists(BEN2_CODE_DIR):
    raise FileNotFoundError(f"BEN2_CODE_DIR does not exist: {BEN2_CODE_DIR}")

# Global analyzer — initialized once at startup
ben2_analyzer = None


def initialize_analyzer() -> bool:
    global ben2_analyzer
    try:
        logger.info("BEN2: Initializing analyzer...")
        ben2_analyzer = BEN2Analyzer(
            model_path=BEN2_MODEL_PATH,
            code_dir=BEN2_CODE_DIR,
            refine_foreground=REFINE_FOREGROUND,
        )
        if not ben2_analyzer.initialize():
            logger.error("BEN2: Analyzer initialization returned False")
            return False
        logger.info("BEN2: Analyzer ready")
        return True
    except Exception as e:
        logger.error(f"BEN2: Error during initialization: {e}")
        return False


def image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 PNG string"""
    buf = BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_image_from_url(url: str) -> Image.Image:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL format")

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    if not response.headers.get('content-type', '').startswith('image/'):
        raise ValueError("URL does not point to an image")

    if len(response.content) > MAX_FILE_SIZE:
        raise ValueError(f"Downloaded image too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")

    return Image.open(BytesIO(response.content)).convert('RGB')


def validate_image_file(file_path: str) -> Image.Image:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not is_allowed_file(file_path):
        raise ValueError("File type not allowed")
    return Image.open(file_path).convert('RGB')


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
logger.info("BEN2 service: CORS enabled")


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
    model_status = "loaded" if ben2_analyzer and ben2_analyzer.model else "not_loaded"
    device_str = str(ben2_analyzer.device) if ben2_analyzer and ben2_analyzer.device else "unknown"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "device": device_str,
        "refine_foreground": REFINE_FOREGROUND,
    })


@app.route('/analyze', methods=['GET', 'POST'])
@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint — accepts file upload, url param, or file path param"""
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "ben2",
            "status": "error",
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

        # Step 2: Run background removal
        result = ben2_analyzer.remove_background(image)

        if not result['success']:
            return error_response(result['error'], 500)

        # Step 3: Encode outputs and return
        return jsonify({
            "service": "ben2",
            "status": "success",
            "mask": image_to_base64(result['mask']),
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "width": result['width'],
                "height": result['height'],
                "model_info": {
                    "model": "BEN2_Base",
                    "refine_foreground": REFINE_FOREGROUND,
                }
            }
        })

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"BEN2: Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info("Starting BEN2 service...")

    if not initialize_analyzer():
        logger.error("BEN2: Failed to initialize analyzer. Service cannot function.")
        exit(1)

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"BEN2: Starting on {host}:{PORT}")
    logger.info(f"BEN2: Private mode: {PRIVATE}")
    logger.info(f"BEN2: Model: {BEN2_MODEL_PATH}")
    logger.info(f"BEN2: Code dir: {BEN2_CODE_DIR}")
    logger.info(f"BEN2: Refine foreground: {REFINE_FOREGROUND}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
