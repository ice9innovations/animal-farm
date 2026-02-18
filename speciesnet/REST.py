import os
import logging
import tempfile
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from speciesnet import SpeciesNet, DEFAULT_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB - camera trap images can be large
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp'}

PRIVATE_STR = os.getenv('PRIVATE')
PORT_STR = os.getenv('PORT')
MODEL_NAME = os.getenv('MODEL', DEFAULT_MODEL)

# Validate critical configuration
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not PORT_STR:
    raise ValueError("PORT environment variable is required")

PRIVATE = PRIVATE_STR.lower() == 'true'
PORT = int(PORT_STR)

# Global SpeciesNet model - initialize once at startup
speciesnet_model = None


def initialize_speciesnet() -> bool:
    """Initialize SpeciesNet model once at startup - fail fast."""
    global speciesnet_model
    try:
        logger.info(f"Initializing SpeciesNet model: {MODEL_NAME}")
        speciesnet_model = SpeciesNet(MODEL_NAME)
        logger.info("SpeciesNet model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize SpeciesNet: {e}")
        return False


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@contextmanager
def temp_image_file(data: bytes, ext: str):
    """Write bytes to a temp file, yield its path, delete on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def run_prediction(
    filepath: str,
    country: Optional[str] = None,
    admin1_region: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    """Run SpeciesNet prediction on a single image (path or URL)."""
    instance = {"filepath": filepath}
    if country:
        instance["country"] = country
    if admin1_region:
        instance["admin1_region"] = admin1_region
    if latitude is not None:
        instance["latitude"] = latitude
    if longitude is not None:
        instance["longitude"] = longitude

    result = speciesnet_model.predict(
        instances_dict={"instances": [instance]},
        run_mode="single_thread",
        progress_bars=False,
    )

    if result and result.get("predictions"):
        return result["predictions"][0]
    return {"filepath": filepath, "failures": ["UNKNOWN"]}


def create_response(prediction: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
    """Create standardized API response."""
    has_failures = bool(prediction.get("failures"))
    return {
        "service": "speciesnet",
        "status": "error" if has_failures else "success",
        "predictions": [prediction],
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "model": MODEL_NAME,
                "framework": "SpeciesNet (Google Camera Trap AI)",
            },
        },
    }


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
logger.info("SpeciesNet service: CORS enabled")


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
    """Health check endpoint."""
    model_status = "loaded" if speciesnet_model else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "model": MODEL_NAME,
    })


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - accepts file upload, URL, or local file path.

    Optional geo parameters (query string or form fields):
        country       - ISO 3166-1 alpha-3 country code (e.g. 'USA', 'AUS')
        admin1_region - ISO 3166-2 region code (e.g. 'CA' for California)
        latitude      - float
        longitude     - float
    """
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "speciesnet",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)},
        }), status_code

    try:
        # Collect geo parameters from query string or form data
        country = request.args.get('country') or request.form.get('country')
        admin1_region = request.args.get('admin1_region') or request.form.get('admin1_region')

        latitude = None
        longitude = None
        lat_str = request.args.get('latitude') or request.form.get('latitude')
        lon_str = request.args.get('longitude') or request.form.get('longitude')
        if lat_str:
            try:
                latitude = float(lat_str)
            except ValueError:
                return error_response("Invalid latitude value")
        if lon_str:
            try:
                longitude = float(lon_str)
            except ValueError:
                return error_response("Invalid longitude value")

        geo = dict(country=country, admin1_region=admin1_region,
                   latitude=latitude, longitude=longitude)

        # --- File upload (multipart POST) ---
        if request.method == 'POST' and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            if not is_allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")

            file_data = uploaded_file.read()
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")

            ext = '.' + uploaded_file.filename.rsplit('.', 1)[1].lower()
            with temp_image_file(file_data, ext) as filepath:
                prediction = run_prediction(filepath, **geo)

        # --- URL or local file path (GET or POST query params) ---
        else:
            url = request.args.get('url')
            file = request.args.get('file')

            if not url and not file:
                return error_response(
                    "Must provide a 'url' or 'file' query parameter, or POST a file"
                )
            if url and file:
                return error_response("Cannot provide both 'url' and 'file' parameters")

            if url:
                # SpeciesNet's load_rgb_image() handles HTTP/HTTPS natively - no temp file needed
                if not (url.startswith('http://') or url.startswith('https://')):
                    return error_response("URL must start with http:// or https://")
                prediction = run_prediction(url, **geo)
            else:
                if not os.path.exists(file):
                    return error_response(f"File not found: {file}")
                if not is_allowed_file(file):
                    return error_response("File type not allowed")
                prediction = run_prediction(file, **geo)

        return jsonify(create_response(prediction, time.time() - start_time))

    except Exception as e:
        logger.error(f"Analyze error: {e}", exc_info=True)
        return error_response(f"Internal error: {str(e)}", 500)


if __name__ == '__main__':
    logger.info("Starting SpeciesNet service...")

    model_loaded = initialize_speciesnet()

    if not model_loaded:
        logger.error("Failed to initialize SpeciesNet model. Service cannot function.")
        exit(1)

    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"Starting SpeciesNet service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Model: {MODEL_NAME}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
