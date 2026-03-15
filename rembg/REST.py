import os
import logging
import time
import base64
import requests
from io import BytesIO
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Required configuration — fail fast
PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')
BACKEND = os.getenv('BACKEND')

if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not BACKEND:
    raise ValueError("BACKEND environment variable is required")
if BACKEND not in ('rembg', 'ben2'):
    raise ValueError(f"BACKEND must be 'rembg' or 'ben2', got: {BACKEND!r}")

PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)

# Backend-specific config — validate only what the chosen backend needs
if BACKEND == 'rembg':
    MODEL_NAME = os.getenv('MODEL_NAME')
    if not MODEL_NAME:
        raise ValueError("MODEL_NAME environment variable is required when BACKEND=rembg")
else:
    BEN2_MODEL_PATH = os.getenv('BEN2_MODEL_PATH')
    BEN2_CODE_DIR = os.getenv('BEN2_CODE_DIR')
    if not BEN2_MODEL_PATH:
        raise ValueError("BEN2_MODEL_PATH environment variable is required when BACKEND=ben2")
    if not BEN2_CODE_DIR:
        raise ValueError("BEN2_CODE_DIR environment variable is required when BACKEND=ben2")
    if not os.path.exists(BEN2_MODEL_PATH):
        raise FileNotFoundError(f"BEN2_MODEL_PATH does not exist: {BEN2_MODEL_PATH}")
    if not os.path.exists(BEN2_CODE_DIR):
        raise FileNotFoundError(f"BEN2_CODE_DIR does not exist: {BEN2_CODE_DIR}")
    REFINE_FOREGROUND = os.getenv('REFINE_FOREGROUND', 'false').lower() == 'true'

# Global analyzer — initialized once at startup
analyzer = None


def initialize_analyzer() -> bool:
    global analyzer
    try:
        if BACKEND == 'rembg':
            from rembg_analyzer import RembgAnalyzer
            logger.info(f"rembg: Initializing rembg backend with model '{MODEL_NAME}'...")
            analyzer = RembgAnalyzer(model_name=MODEL_NAME)
        else:
            from ben2_analyzer import BEN2Analyzer
            logger.info("rembg: Initializing ben2 backend...")
            analyzer = BEN2Analyzer(
                model_path=BEN2_MODEL_PATH,
                code_dir=BEN2_CODE_DIR,
                refine_foreground=REFINE_FOREGROUND,
            )

        if not analyzer.initialize():
            logger.error(f"rembg: {BACKEND} analyzer initialization returned False")
            return False
        logger.info(f"rembg: {BACKEND} analyzer ready")
        return True
    except Exception as e:
        logger.error(f"rembg: Error during initialization: {e}")
        return False


def image_to_base64(image: Image.Image) -> str:
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
logger.info("rembg service: CORS enabled")


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
    if BACKEND == 'rembg':
        backend_status = "loaded" if analyzer and analyzer.session else "not_loaded"
        device = "cpu"
        model = MODEL_NAME
    else:
        backend_status = "loaded" if analyzer and analyzer.model else "not_loaded"
        device = str(analyzer.device) if analyzer and analyzer.device else "unknown"
        model = "BEN2_Base"
    return jsonify({
        "status": "healthy",
        "backend": BACKEND,
        "backend_status": backend_status,
        "device": device,
        "model": model,
    })


@app.route('/analyze', methods=['GET', 'POST'])
@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint — accepts file upload, url param, or file path param"""
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "rembg",
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
        result = analyzer.remove_background(image)

        if not result['success']:
            return error_response(result['error'], 500)

        # Step 3: Encode and return
        if BACKEND == 'rembg':
            model_info = {"model": MODEL_NAME, "device": "cpu"}
        else:
            model_info = {"model": "BEN2_Base", "refine_foreground": REFINE_FOREGROUND}

        return jsonify({
            "service": "rembg",
            "status": "success",
            "mask": image_to_base64(result['mask']),
            "metadata": {
                "processing_time": round(time.time() - start_time, 3),
                "width": result['width'],
                "height": result['height'],
                "model_info": model_info,
            }
        })

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"rembg: Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info("Starting rembg service...")

    if not initialize_analyzer():
        logger.error("rembg: Failed to initialize. Service cannot function.")
        exit(1)

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"rembg: Starting on {host}:{PORT}")
    logger.info(f"rembg: Private mode: {PRIVATE}")
    logger.info(f"rembg: Backend: {BACKEND}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
