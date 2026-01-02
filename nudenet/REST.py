import os
import json
import requests
import time
import random
import logging
from typing import Dict, Any, List
from io import BytesIO

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Load environment variables FIRST
load_dotenv()

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
DETECTION_THRESHOLD_STR = os.getenv('DETECTION_THRESHOLD')

# Step 2: Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")
if not DETECTION_THRESHOLD_STR:
    raise ValueError("DETECTION_THRESHOLD environment variable is required")

# Step 3: Convert to appropriate types after validation
PORT = int(PORT_STR)
PRIVATE = PRIVATE_STR.lower() in ['true', '1', 'yes']
DETECTION_THRESHOLD = float(DETECTION_THRESHOLD_STR)

MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import NudeNet after environment setup
try:
    from nudenet import NudeDetector
    NUDENET_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import NudeNet: {e}")
    logger.error("Install with: pip install nudenet")
    NUDENET_AVAILABLE = False

# Load emoji mappings from local JSON file
def load_emoji_mappings() -> Dict[str, str]:
    """Load emoji mappings from local JSON file"""
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')

    try:
        logger.info(f"Loading emoji mappings from: {local_cache_path}")
        with open(local_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("✅ Successfully loaded emoji mappings")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ Failed to load emoji mappings from {local_cache_path}: {e}")
        raise Exception(f"Failed to load emoji mappings: {e}")

# Load emoji mappings on startup
emoji_mappings = load_emoji_mappings()

# Global NudeNet detector - initialize once at startup
nude_detector = None

def initialize_detector() -> bool:
    """Initialize NudeNet detector once at startup - fail fast"""
    global nude_detector

    if not NUDENET_AVAILABLE:
        logger.error("❌ NudeNet library not available")
        return False

    try:
        logger.info("Initializing NudeNet Detector...")
        nude_detector = NudeDetector()
        logger.info("✅ NudeNet Detector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing NudeNet Detector: {str(e)}")
        return False

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def get_emoji_for_category(category: str) -> str:
    """Get emoji for a detection category"""
    return emoji_mappings.get(category, "")

def process_image_for_detection(image: Image.Image) -> Dict[str, Any]:
    """
    Main processing function - takes PIL Image, returns NudeNet detection data
    This is the core business logic, separated from HTTP concerns
    """
    global nude_detector
    start_time = time.time()

    if nude_detector is None:
        return {
            "success": False,
            "error": "NudeNet detector not initialized",
            "processing_time": round(time.time() - start_time, 3)
        }

    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # NudeNet can work with PIL Images directly via detect() method
        # It expects the image to be saved temporarily or passed as path
        # For in-memory processing, we need to save to a BytesIO buffer
        # However, NudeDetector.detect() expects a file path
        # Let's check if there's a better way...

        # Actually, NudeDetector has a detect() method that accepts file paths
        # For in-memory, we'll need to convert PIL Image to the format it expects
        # The library internally uses cv2, so we might need a temporary approach

        # Convert PIL Image to format NudeNet expects
        import numpy as np
        import tempfile

        # Save to temporary file for NudeNet processing
        # This is necessary because NudeNet's detect() expects a file path
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            image.save(temp_path, 'JPEG')

        try:
            # Detect using NudeNet
            detections = nude_detector.detect(temp_path)

            # Filter by confidence threshold
            filtered_detections = [
                det for det in detections
                if det['score'] * 100 >= DETECTION_THRESHOLD
            ]

            processing_time = round(time.time() - start_time, 3)

            return {
                "success": True,
                "detections": filtered_detections,
                "processing_time": processing_time
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"NudeNet detection failed: {e}")
        return {
            "success": False,
            "error": f"Detection failed: {str(e)}",
            "processing_time": round(time.time() - start_time, 3)
        }

def create_nudenet_response(detections: List[Dict], processing_time: float) -> Dict[str, Any]:
    """Create standardized NudeNet response with metadata"""
    predictions = []

    for detection in detections:
        category = detection['class']
        confidence = round(detection['score'], 3)
        bbox = detection['box']  # [x1, y1, x2, y2]
        emoji = get_emoji_for_category(category)

        # Check for shiny on each detection
        is_shiny, shiny_roll = check_shiny()

        prediction = {
            "label": category,
            "confidence": confidence,
            "bbox": bbox,
            "emoji": emoji
        }

        # Add shiny flag for rare detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY DETECTION! Category: {category}, Roll: {shiny_roll} ✨")

        predictions.append(prediction)

    return {
        "service": "nudenet",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {
                "framework": "NudeNet+"
            }
        }
    }

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image"""
    try:
        headers = {'User-Agent': 'NudeNet Detection Service'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")

        # Return PIL Image directly from bytes
        image = Image.open(BytesIO(response.content))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        image = Image.open(file_path)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for direct browser access
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
logger.info("NudeNet service: CORS enabled for direct browser communication")

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
        model_status = "loaded" if nude_detector is not None else "not_loaded"
        return jsonify({
            "status": "healthy",
            "service": "NudeNet Detection",
            "model": {
                "status": model_status,
                "framework": "NudeNet+",
                "threshold": DETECTION_THRESHOLD
            },
            "endpoints": [
                "GET /health - Health check",
                "GET,POST /analyze - Unified endpoint (URL/file/upload)",
                "GET /v3/analyze - V3 compatibility",
                "GET /v2/analyze - V2 compatibility (deprecated)",
                "GET /v2/analyze_file - V2 compatibility (deprecated)"
            ]
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"error": "Health check failed", "status": "error"}), 500

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()

    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "nudenet",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code

    try:
        image = None

        # Step 1: Get image into memory from any source
        if request.method == 'POST' and 'file' in request.files:
            # Handle file upload - direct to memory
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")

            # Read directly into memory
            file_data = uploaded_file.read()
            if len(file_data) > MAX_FILE_SIZE:
                return error_response("File too large")

            image = Image.open(BytesIO(file_data)).convert('RGB')

        else:
            # Handle URL or file parameter
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

            else:  # file parameter
                try:
                    image = validate_image_file(file)
                except Exception as e:
                    return error_response(str(e))

        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_detection(image)

        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)

        # Step 4: Create response
        response = create_nudenet_response(
            processing_result["detections"],
            processing_result["processing_time"]
        )

        return jsonify(response)

    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        logger.error(f"Analyze API error: {e}")
        return error_response(f"Internal error: {str(e)}", 500)

@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - calls new analyze function directly"""
    return analyze()

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to new analyze format"""
    file_path = request.args.get('file_path')

    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        with app.test_request_context('/analyze'):
            return analyze()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to new analyze format"""
    image_url = request.args.get('image_url')

    if image_url:
        # Parameter translation: image_url -> url
        new_args = {'url': image_url}
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # Let new analyze handle validation errors
        with app.test_request_context('/analyze'):
            return analyze()

if __name__ == '__main__':
    # Initialize model
    logger.info("Starting NudeNet service...")

    model_loaded = initialize_detector()

    if not model_loaded:
        logger.error("Failed to initialize NudeNet detector. Service cannot function.")
        exit(1)

    # Determine host based on private mode
    host = "127.0.0.1" if PRIVATE else "0.0.0.0"

    logger.info(f"Starting NudeNet service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE}")
    logger.info(f"Detection threshold: {DETECTION_THRESHOLD}%")
    logger.info(f"Model loaded: {model_loaded}")

    app.run(
        host=host,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )
