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
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import requests
from datetime import datetime
from urllib.parse import urlparse

# Load environment variables
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
API_HOST = os.getenv('API_HOST')
API_PORT_STR = os.getenv('API_PORT')
API_TIMEOUT_STR = os.getenv('API_TIMEOUT')
PRIVATE_STR = os.getenv('PRIVATE')

# Step 2: Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not API_HOST:
    raise ValueError("API_HOST environment variable is required")
if not API_PORT_STR:
    raise ValueError("API_PORT environment variable is required")
if not API_TIMEOUT_STR:
    raise ValueError("API_TIMEOUT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Step 3: Convert to appropriate types after validation
PORT = int(PORT_STR)
API_PORT = int(API_PORT_STR)
API_TIMEOUT = float(API_TIMEOUT_STR)
PRIVATE_MODE = PRIVATE_STR.lower() == 'true'

# Global emoji mappings - loaded from API on startup
emoji_mappings = {}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global MediaPipe models - initialize once at startup
face_detection_model = None

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

def initialize_models():
    """Initialize MediaPipe face detection model at startup"""
    global face_detection_model
    try:
        logger.info("Initializing MediaPipe Face Detection model...")
        
        # Face Detection
        face_detection_model = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection (better for diverse faces)
            min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE
        )
        logger.info("✅ MediaPipe Face Detection model initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing MediaPipe Face Detection model: {str(e)}")
        return False

def load_emoji_mappings():
    """Load emoji mappings from central API"""
    global emoji_mappings
    try:
        emoji_url = f"http://{API_HOST}:{API_PORT}/emoji_mappings.json"
        response = requests.get(emoji_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        emoji_mappings = response.json()
        logger.info(f"✅ Loaded {len(emoji_mappings)} emoji mappings from {emoji_url}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load emoji mappings: {e}. Using empty mappings.")
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

def download_image(url):
    """Download image from URL and return as bytes"""
    try:
        headers = {'User-Agent': 'MediaPipe Face Analysis Service'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if len(response.content) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")
        
        return io.BytesIO(response.content)
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")

def convert_to_jpg(image_bytes):
    """Convert any image format to JPG for consistent processing"""
    try:
        image = Image.open(image_bytes)
        
        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPG
        jpg_bytes = io.BytesIO()
        image.save(jpg_bytes, format='JPEG', quality=85)
        jpg_bytes.seek(0)
        
        return jpg_bytes
        
    except Exception as e:
        raise Exception(f"Failed to convert image: {str(e)}")

def detect_faces_mediapipe(image_path):
    """Detect faces using MediaPipe"""
    global face_detection_model
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Could not load image from {image_path}")
        
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_detection_model.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to pixel coordinates
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Get confidence
                confidence = detection.score[0] if detection.score else 0.0
                
                # Get key points if available
                keypoints = {}
                if detection.location_data.relative_keypoints:
                    keypoints = {
                        'right_eye': [
                            int(detection.location_data.relative_keypoints[0].x * width),
                            int(detection.location_data.relative_keypoints[0].y * height)
                        ],
                        'left_eye': [
                            int(detection.location_data.relative_keypoints[1].x * width),
                            int(detection.location_data.relative_keypoints[1].y * height)
                        ],
                        'nose_tip': [
                            int(detection.location_data.relative_keypoints[2].x * width),
                            int(detection.location_data.relative_keypoints[2].y * height)
                        ],
                        'mouth_center': [
                            int(detection.location_data.relative_keypoints[3].x * width),
                            int(detection.location_data.relative_keypoints[3].y * height)
                        ],
                        'right_ear_tragion': [
                            int(detection.location_data.relative_keypoints[4].x * width),
                            int(detection.location_data.relative_keypoints[4].y * height)
                        ],
                        'left_ear_tragion': [
                            int(detection.location_data.relative_keypoints[5].x * width),
                            int(detection.location_data.relative_keypoints[5].y * height)
                        ]
                    }
                
                faces.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'keypoints': keypoints,
                    'method': 'mediapipe'
                })
        
        return faces, {'width': width, 'height': height}
        
    except Exception as e:
        logger.error(f"MediaPipe face detection error: {str(e)}")
        return [], {'width': 0, 'height': 0}


def process_image(image_source, is_url=False, is_file_path=False):
    """Process image and perform face detection analysis - returns raw data"""
    start_time = time.time()
    temp_path = None
    
    try:
        # Handle image source
        if is_url:
            image_bytes = download_image(image_source)
        elif is_file_path:
            # Direct file path - use directly without temporary conversion
            faces, dimensions = detect_faces_mediapipe(image_source)
            processing_time = time.time() - start_time
            return {
                'faces': faces,
                'dimensions': dimensions,
                'processing_time': processing_time,
                'error': None
            }
        else:
            image_bytes = image_source
        
        # Convert to JPG for consistent processing
        jpg_bytes = convert_to_jpg(image_bytes)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            jpg_bytes.seek(0)
            tmp_file.write(jpg_bytes.read())
            temp_path = tmp_file.name
        
        # Perform face detection analysis
        faces, dimensions = detect_faces_mediapipe(temp_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'faces': faces,
            'dimensions': dimensions,
            'processing_time': processing_time,
            'error': None
        }
    
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        processing_time = time.time() - start_time
        return {
            'faces': [],
            'dimensions': {'width': 0, 'height': 0},
            'processing_time': processing_time,
            'error': str(e)
        }
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not clean up temp file {temp_path}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global face_detection_model
    try:
        # Check if model is initialized
        face_status = 'ready' if face_detection_model is not None else 'not_initialized'
        
        all_ready = face_detection_model is not None
        
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
                    'keypoints': 6
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

def build_v3_response(raw_data):
    """Build standardized V3 response from raw face detection data"""
    # Handle errors
    if raw_data['error']:
        return jsonify({
            "service": "face",
            "status": "error",
            "predictions": [],
            "metadata": {
                "processing_time": round(raw_data['processing_time'], 3),
                "model_info": {"framework": "MediaPipe"}
            },
            "error": {"message": str(raw_data['error'])}
        }), 500
    
    # Build V3 predictions
    predictions = []
    
    # Add face predictions
    for face in raw_data['faces']:
        is_shiny, shiny_roll = check_shiny()
        
        prediction = {
            "label": "face",
            "emoji": get_emoji("face"),
            "confidence": round(float(face['confidence']), CONFIDENCE_DECIMAL_PLACES),
            "bbox": face['bbox'],
            "properties": {
                "keypoints": face.get('keypoints', {}),
                "method": face['method']
            }
        }
        
        # Add shiny flag only for shiny detections
        if is_shiny:
            prediction["shiny"] = True
            logger.info(f"✨ SHINY FACE DETECTED! Roll: {shiny_roll} ✨")
        
        predictions.append(prediction)
    
    return jsonify({
        "service": "face",
        "status": "success",
        "predictions": predictions,
        "metadata": {
            "processing_time": round(raw_data['processing_time'], 3),
            "model_info": {"framework": "MediaPipe"}
        }
    })

@app.route('/v3/analyze', methods=['GET', 'POST'])
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
                
                # Process with face detection
                raw_data = process_image(temp_path, is_file_path=True)
                return build_v3_response(raw_data)
                
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
            raw_data = process_image(url, is_url=True)
        
        # Handle file path input
        elif file_path:
            # Validate file path
            if not os.path.exists(file_path):
                return build_error_response(f"File not found: {file_path}", start_time, 404)
            
            if not allowed_file(file_path):
                return build_error_response("File type not allowed", start_time, 400)
            
            raw_data = process_image(file_path, is_file_path=True)
        
        # Use shared response builder
        return build_v3_response(raw_data)
        
    except Exception as e:
        logger.error(f"Error in V3 analysis: {str(e)}")
        return build_error_response(str(e), start_time, 500)

@app.route('/v2/analyze_file', methods=['GET'])
def analyze_file_v2_compat():
    """V2 file compatibility - translate parameters to V3 format"""
    file_path = request.args.get('file_path')
    
    if file_path:
        new_args = {'file': file_path}
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

@app.route('/v2/analyze', methods=['GET'])
def analyze_v2_compat():
    """V2 compatibility - translate parameters to V3 format"""
    image_url = request.args.get('image_url')
    
    if image_url:
        # Parameter translation
        new_args = request.args.copy().to_dict()
        new_args['url'] = image_url
        del new_args['image_url']
        
        # Call V3 with translated parameters
        with app.test_request_context('/v3/analyze', query_string=new_args):
            return analyze_v3()
    else:
        # Let V3 handle validation errors
        with app.test_request_context('/v3/analyze'):
            return analyze_v3()

if __name__ == '__main__':
    # Initialize MediaPipe face detection model at startup
    logger.info("Initializing MediaPipe Face Detection model...")
    if not initialize_models():
        logger.error("Failed to initialize MediaPipe Face Detection model. Service will not function properly.")
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
