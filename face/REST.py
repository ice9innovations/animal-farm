#!/usr/bin/env python3
"""
Face Detection Service using MediaPipe
Provides dedicated face detection and facial analysis capabilities
Replaces the outdated and biased Haar Cascade/SSD models with Google's MediaPipe framework
"""

import os
import io
import time
import logging
import tempfile
import uuid
import json
import random
import threading
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import mediapipe as mp
import requests
from datetime import datetime
from urllib.parse import urlparse
from face_analyzer import FaceAnalyzer

# Load environment variables
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
TIMEOUT_STR = os.getenv('TIMEOUT')
AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE')
PRIVATE_STR = os.getenv('PRIVATE')

# Step 2: Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not TIMEOUT_STR:
    raise ValueError("TIMEOUT environment variable is required")
if not AUTO_UPDATE_STR:
    raise ValueError("AUTO_UPDATE environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Step 3: Convert to appropriate types after validation
PORT = int(PORT_STR)
TIMEOUT = float(TIMEOUT_STR)
AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'
PRIVATE_MODE = PRIVATE_STR.lower() == 'true'

# Global emoji mappings - loaded from API on startup
emoji_mappings = {}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# GPU configuration
USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'

# Global face analyzer - initialize once at startup
face_analyzer = None

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB

# MediaPipe model configuration
FACE_MIN_DETECTION_CONFIDENCE = 0.2
CONFIDENCE_DECIMAL_PLACES = 3
LANDMARK_DECIMAL_PLACES = 3  # Precision for landmark coordinates and visibility

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_face_analyzer():
    """Initialize face analyzer once at startup"""
    global face_analyzer
    try:
        if USE_GPU:
            logger.info("Initializing MediaPipe Face Analyzer with GPU acceleration...")
        else:
            logger.info("Initializing MediaPipe Face Analyzer with CPU...")

        face_analyzer = FaceAnalyzer(
            min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            model_selection=1,  # 1 for full range detection (better for diverse faces)
            use_gpu=USE_GPU
        )

        gpu_status = "GPU" if USE_GPU else "CPU"
        logger.info(f"✅ Face Analyzer initialized successfully ({gpu_status})")
        return True

    except Exception as e:
        logger.error(f"❌ Error initializing Face Analyzer: {str(e)}")
        return False

def load_emoji_mappings():
    """Load emoji mappings from GitHub raw files with local caching"""
    global emoji_mappings
    local_cache_path = os.path.join(os.path.dirname(__file__), 'emoji_mappings.json')
    
    # Try GitHub raw file first if AUTO_UPDATE is enabled
    if AUTO_UPDATE:
        github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
        
        try:
            logger.info(f"Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=TIMEOUT)
            response.raise_for_status()
            emoji_mappings = response.json()
            
            # Cache the successful download locally
            try:
                with open(local_cache_path, 'w') as f:
                    json.dump(emoji_mappings, f, indent=2)
                logger.info(f"✅ Loaded {len(emoji_mappings)} emoji mappings from GitHub and cached locally")
            except Exception as cache_error:
                logger.warning(f"⚠️ Failed to cache emoji mappings locally: {cache_error}")
                logger.info(f"✅ Loaded {len(emoji_mappings)} emoji mappings from GitHub (no local cache)")
            
            return
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load emoji mappings from GitHub: {e}")
            logger.info("Attempting to use local cache...")
    
    # Fall back to local cached file
    try:
        if os.path.exists(local_cache_path):
            with open(local_cache_path, 'r') as f:
                emoji_mappings = json.load(f)
            logger.info(f"✅ Loaded {len(emoji_mappings)} emoji mappings from local cache")
        else:
            logger.warning("⚠️ No local emoji mappings cache found")
            emoji_mappings = {}
    except Exception as e:
        logger.error(f"❌ Failed to load emoji mappings from local cache: {e}")
        emoji_mappings = {}

def get_emoji(word):
    """Get emoji for a given word"""
    return emoji_mappings.get(word.lower(), "")

def check_shiny():
    """Check if this detection should be shiny (1/2500 chance)"""
    roll = random.randint(1, 2500)
    is_shiny = roll == 1
    return is_shiny, roll

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image"""
    try:
        headers = {'User-Agent': 'MediaPipe Face Analysis Service'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")

        # Return PIL Image directly from bytes
        image = Image.open(io.BytesIO(response.content))

        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")



def process_image_for_faces(image: Image.Image) -> dict:
    """
    Main processing function - takes PIL Image, returns face detection data
    This is the core business logic, separated from HTTP concerns
    Uses pure in-memory processing
    """
    start_time = time.time()

    if not face_analyzer:
        return {
            "success": False,
            "error": "Face analyzer not loaded"
        }

    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Perform face detection analysis using FaceAnalyzer
        face_data = face_analyzer.analyze_faces_from_image(image)

        processing_time = time.time() - start_time

        return {
            "success": True,
            "data": face_data,
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"Face processing error: {str(e)}")
        return {
            "success": False,
            "error": f"Face detection failed: {str(e)}",
            "processing_time": time.time() - start_time
        }

def create_face_response(data: dict, processing_time: float) -> dict:
    """Create standardized face detection response"""
    faces = data.get('faces', [])
    
    # Build V3 predictions
    predictions = []
    for face in faces:
        is_shiny, shiny_roll = check_shiny()
        
        prediction = {
            "label": "face",
            "emoji": get_emoji("face"),
            "confidence": round(float(face['confidence']), CONFIDENCE_DECIMAL_PLACES),
            "bbox": face['bbox'],
            "keypoints": face.get('keypoints', {})
        }
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY FACE DETECTED! Roll: {shiny_roll} ✨")
        
        predictions.append(prediction)
    
    return {
        "service": "face",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {"framework": "MediaPipe"}
        }
    }



@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global face_analyzer
    try:
        # Check if analyzer is initialized
        face_status = 'ready' if face_analyzer is not None else 'not_initialized'

        all_ready = face_analyzer is not None
        
        return jsonify({
            'status': 'healthy' if all_ready else 'degraded',
            'service': 'face',
            'capabilities': ['face_detection', 'facial_keypoints'],
            'models': {
                'face_detection': {
                    'status': face_status,
                    'version': mp.__version__,
                    'model': 'MediaPipe Face Detection (Full Range)',
                    'fairness': 'Tested across demographics',
                    'keypoints': 6,
                    'gpu_enabled': USE_GPU
                }
            },
            'endpoints': [
                "GET /health - Health check",
                "GET /v3/analyze?url=<image_url> - Analyze face detection from URL", 
                "GET /v3/analyze?file=<file_path> - Analyze face detection from file",
                "POST /v3/analyze - Analyze face detection from uploaded file",
                "GET /v2/analyze?image_url=<image_url> - V2 compatibility (deprecated)",
                "GET /v2/analyze_file?file_path=<file_path> - V2 compatibility (deprecated)"
            ],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

def build_error_response(error_message, start_time, status_code):
    """Build standardized error response for V3 API"""
    return jsonify({
        "service": "face",
        "status": "error",
        "predictions": [],
        "error": {"message": error_message},
        "metadata": {
            "processing_time": round(time.time() - start_time, 3),
            "model_info": {"framework": "MediaPipe"}
        }
    }), status_code


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    start_time = time.time()
    
    def error_response(message: str, status_code: int = 400):
        return jsonify({
            "service": "face",
            "status": "error",
            "predictions": [],
            "error": {"message": message},
            "metadata": {"processing_time": round(time.time() - start_time, 3)}
        }), status_code
    
    try:
        # Step 1: Get image into memory from any source
        if request.method == 'POST' and 'file' in request.files:
            # Handle file upload
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return error_response("No file selected")
            
            # Validate file size
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB")
            
            # Validate file type
            if not allowed_file(uploaded_file.filename):
                return error_response("File type not allowed")
            
            try:
                from io import BytesIO
                file_data = uploaded_file.read()
                image = Image.open(BytesIO(file_data)).convert('RGB')
            except Exception as e:
                return error_response(f"Failed to process uploaded image: {str(e)}", 500)
        
        else:
            # Handle URL or file parameter
            url = request.args.get('url')
            file_path = request.args.get('file')
            
            if not url and not file_path:
                return error_response("Must provide either 'url' or 'file' parameter, or POST a file")
            
            if url and file_path:
                return error_response("Cannot provide both 'url' and 'file' parameters")
            
            if url:
                # Download from URL directly into memory
                try:
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        return error_response("Invalid URL format")
                    
                    headers = {'User-Agent': 'MediaPipe Face Analysis Service'}
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    if len(response.content) > MAX_FILE_SIZE:
                        return error_response("Downloaded file too large")
                    
                    from io import BytesIO
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    
                except Exception as e:
                    return error_response(f"Failed to download/process image: {str(e)}")
            else:  # file_path
                # Load file directly into memory
                if not os.path.exists(file_path):
                    return error_response(f"File not found: {file_path}")
                
                if not allowed_file(file_path):
                    return error_response("File type not allowed")
                
                try:
                    image = Image.open(file_path).convert('RGB')
                except Exception as e:
                    return error_response(f"Failed to load image file: {str(e)}", 500)
        
        # Step 2: Process the image (unified processing path)
        processing_result = process_image_for_faces(image)
        
        # Step 3: Handle processing result
        if not processing_result["success"]:
            return error_response(processing_result["error"], 500)
        
        # Step 4: Create response
        response = create_face_response(
            processing_result["data"],
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

@app.route('/v3/analyze_old', methods=['GET', 'POST'])
def analyze_v3():
    """Unified V3 API endpoint for both URL and file path analysis"""
    start_time = time.time()
    
    try:
        # Handle POST file upload
        if request.method == 'POST':
            # Check for file upload
            if 'file' not in request.files:
                return build_error_response("No file provided in POST request", start_time, 400)
            
            file = request.files['file']
            if file.filename == '':
                return build_error_response("No file selected", start_time, 400)
            
            # Validate file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return build_error_response(f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB", start_time, 400)
            
            # Process uploaded file
            try:
                # Save to temporary file for processing
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    file.save(temp_path)
                
                # Use the new unified analyze function
                try:
                    image = Image.open(temp_path).convert('RGB')
                    result = process_image_for_faces(image)
                    if not result["success"]:
                        return build_error_response(result["error"], start_time, 500)
                    return jsonify(create_face_response(result["data"], result["processing_time"]))
                except Exception as e:
                    return build_error_response(f"Failed to process image: {str(e)}", start_time, 500)
                
            except Exception as e:
                return build_error_response(f"Failed to process uploaded image: {str(e)}", start_time, 500)
            finally:
                # Clean up temporary file
                try:
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Could not clean up temp file: {e}")
        
        # Handle GET requests
        # Get parameters from query string
        url = request.args.get('url')
        file_path = request.args.get('file')
        
        # Validate input - exactly one parameter required
        if not url and not file_path:
            return build_error_response("Must provide either 'url' or 'file' parameter", start_time, 400)
        
        if url and file_path:
            return build_error_response("Cannot provide both 'url' and 'file' parameters - choose one", start_time, 400)
        
        # Handle URL input
        if url:
            try:
                image = download_image_from_url(url)
                result = process_image_for_faces(image)
                if not result["success"]:
                    return build_error_response(result["error"], start_time, 500)
                return jsonify(create_face_response(result["data"], result["processing_time"]))
            except Exception as e:
                return build_error_response(f"Failed to process URL: {str(e)}", start_time, 500)

        # Handle file path input
        elif file_path:
            # Validate file path
            if not os.path.exists(file_path):
                return build_error_response(f"File not found: {file_path}", start_time, 404)

            if not allowed_file(file_path):
                return build_error_response("File type not allowed", start_time, 400)

            try:
                image = Image.open(file_path).convert('RGB')
                result = process_image_for_faces(image)
                if not result["success"]:
                    return build_error_response(result["error"], start_time, 500)
                return jsonify(create_face_response(result["data"], result["processing_time"]))
            except Exception as e:
                return build_error_response(f"Failed to process file: {str(e)}", start_time, 500)
        
    except Exception as e:
        logger.error(f"Error in V3 analysis: {str(e)}")
        return build_error_response(str(e), start_time, 500)

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
        # Parameter translation
        new_args = request.args.copy().to_dict()
        new_args['url'] = image_url
        if 'image_url' in new_args:
            del new_args['image_url']
        
        # Call new analyze with translated parameters
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # Let new analyze handle validation errors
        with app.test_request_context('/analyze'):
            return analyze()

if __name__ == '__main__':
    # Initialize face analyzer at startup - fail fast if it doesn't work
    logger.info("Initializing Face Analysis Service...")
    if not initialize_face_analyzer():
        logger.error("Failed to initialize Face Analyzer. Service cannot function.")
        exit(1)
    
    # Load emoji mappings on startup
    load_emoji_mappings()
    
    # Determine host based on private mode (like other services)
    host = "127.0.0.1" if PRIVATE_MODE else "0.0.0.0"
    
    logger.info(f"Starting Face Detection Service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE_MODE}")
    logger.info("Using MediaPipe framework for dedicated face detection and facial keypoints")
    logger.info("V3 API available at /v3/analyze endpoint with V2 backward compatibility")
    app.run(host=host, port=PORT, debug=False)
