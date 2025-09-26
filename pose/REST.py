#!/usr/bin/env python3
"""
Standalone Pose Estimation Service using MediaPipe
Provides dedicated human pose analysis with enhanced classification and features
Extracted from face service for independent scaling and specialized pose capabilities
"""

import os
import io
import time
import logging
import uuid
import json
import random
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import mediapipe as mp
import requests
from datetime import datetime
from urllib.parse import urlparse
from pose_analyzer import PoseAnalyzer

# Load environment variables
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Step 1: Load as strings (no fallbacks)
PORT_STR = os.getenv('PORT')
PRIVATE_STR = os.getenv('PRIVATE')
AUTO_UPDATE_STR = os.getenv('AUTO_UPDATE', 'true')

# Step 2: Validate critical environment variables
if not PORT_STR:
    raise ValueError("PORT environment variable is required")
if not PRIVATE_STR:
    raise ValueError("PRIVATE environment variable is required")

# Step 3: Convert to appropriate types after validation
PORT = int(PORT_STR)
PRIVATE_MODE = PRIVATE_STR.lower() == 'true'
AUTO_UPDATE = AUTO_UPDATE_STR.lower() == 'true'

# Pose analysis settings from environment (with defaults)
POSE_MIN_DETECTION_CONFIDENCE = float(os.getenv('POSE_MIN_DETECTION_CONFIDENCE', '0.5'))
POSE_MIN_TRACKING_CONFIDENCE = float(os.getenv('POSE_MIN_TRACKING_CONFIDENCE', '0.5'))
POSE_MODEL_COMPLEXITY = int(os.getenv('POSE_MODEL_COMPLEXITY', '2'))
ENABLE_SEGMENTATION = os.getenv('ENABLE_SEGMENTATION', 'true').lower() == 'true'
USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'

# Global emoji mappings - loaded from API on startup
emoji_mappings = {}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global pose analyzer - initialize once at startup
pose_analyzer = None

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(8 * 1024 * 1024)))  # 8MB

CONFIDENCE_DECIMAL_PLACES = 3

# Pose classifications with enhanced types
POSE_CLASSIFICATIONS = {
    "standing": {"emoji": "🧍", "description": "Person standing upright"},
    "sitting": {"emoji": "🪑", "description": "Person sitting down"},
    "lying": {"emoji": "🛌", "description": "Person lying down"},
    "walking": {"emoji": "🚶", "description": "Person in walking motion"},
    "running": {"emoji": "🏃", "description": "Person in running motion"},
    "dancing": {"emoji": "💃", "description": "Person in dance pose"},
    "exercising": {"emoji": "🏋️", "description": "Person exercising"},
    "waving": {"emoji": "👋", "description": "Person waving"},
    "pointing": {"emoji": "👉", "description": "Person pointing"},
    "unknown": {"emoji": "🧑", "description": "Pose not classified"}
}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_pose_analyzer():
    """Initialize pose analyzer once at startup"""
    global pose_analyzer
    try:
        if USE_GPU:
            logger.info("Initializing MediaPipe Pose Analyzer with GPU acceleration...")
        else:
            logger.info("Initializing MediaPipe Pose Analyzer with CPU...")

        pose_analyzer = PoseAnalyzer(
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
            model_complexity=POSE_MODEL_COMPLEXITY,
            enable_segmentation=ENABLE_SEGMENTATION,
            use_gpu=USE_GPU
        )

        gpu_status = "GPU" if USE_GPU else "CPU"
        logger.info(f"✅ Pose Analyzer initialized successfully ({gpu_status})")
        return True

    except Exception as e:
        logger.error(f"❌ Error initializing Pose Analyzer: {str(e)}")
        return False

def load_emoji_mappings():
    """Load emoji mappings from GitHub, fall back to local cache"""
    global emoji_mappings
    
    github_url = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/emoji_mappings.json"
    local_cache_path = 'emoji_mappings.json'
    
    if AUTO_UPDATE:
        try:
            logger.info(f"🔄 Pose: Loading emoji mappings from GitHub: {github_url}")
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            
            # Save to local cache (preserve emoji characters)
            with open(local_cache_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            
            emoji_mappings = response.json()
            logger.info(f"✅ Pose: Loaded emoji mappings from GitHub and cached locally ({len(emoji_mappings)} entries)")
            return
        except Exception as e:
            logger.warning(f"⚠️ Pose: Failed to load emoji mappings from GitHub ({e}), falling back to local cache")
    
    # Fall back to local cache
    try:
        with open(local_cache_path, 'r') as f:
            emoji_mappings = json.load(f)
        logger.info(f"✅ Pose: Successfully loaded emoji mappings from local cache ({len(emoji_mappings)} entries)")
    except Exception as local_error:
        logger.warning(f"⚠️ Pose: Failed to load local emoji mappings: {local_error}. Using built-in pose emojis.")
        emoji_mappings = {}

def get_emoji(pose_type):
    """Get emoji for pose type, with fallback to built-in classifications"""
    # Try API emoji mappings first
    emoji = emoji_mappings.get(pose_type.lower(), "")
    if emoji:
        return emoji
    
    # Fallback to built-in pose classifications
    return POSE_CLASSIFICATIONS.get(pose_type, {}).get("emoji", "🧑")

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
        headers = {'User-Agent': 'MediaPipe Pose Analysis Service'}
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

def validate_image_file(file_path: str) -> Image.Image:
    """Validate and load image file as PIL Image"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not allowed_file(file_path):
        raise ValueError("File type not allowed")
    
    try:
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def process_image_for_pose(image: Image.Image) -> dict:
    """Main processing function - takes PIL Image, returns pose data
    This is the core business logic, separated from HTTP concerns"""
    try:
        # Convert PIL Image to numpy array for MediaPipe (RGB format)
        image_array = np.array(image)
        
        # Perform pose analysis using MediaPipe
        pose_data = pose_analyzer.analyze_pose_from_array(image_array)
        
        return {
            'success': True,
            'data': pose_data,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Pose processing error: {str(e)}")
        return {
            'success': False,
            'data': {
                'predictions': [],
                'persons_detected': 0,
                'image_dimensions': {'width': image.width, 'height': image.height}
            },
            'error': str(e)
        }

def create_pose_response(pose_data: dict, processing_time: float) -> dict:
    """Create standardized pose response with metadata"""
    enhanced_predictions = []
    
    for prediction in pose_data.get('predictions', []):
        is_shiny, shiny_roll = check_shiny()
        
        enhanced_prediction = {
            "landmarks": prediction['landmarks'],
            "pose_analysis": prediction['pose_analysis']
        }
        
        # Add shiny flag for rare detections
        if is_shiny:
            enhanced_prediction["shiny"] = True
            logger.info(f"✨ SHINY POSE LANDMARKS DETECTED! Roll: {shiny_roll} ✨")
        
        enhanced_predictions.append(enhanced_prediction)
    
    return {
        "service": "pose",
        "status": "success",
        "predictions": enhanced_predictions,
        "metadata": {
            "processing_time": round(processing_time, 3),
            "model_info": {"framework": "MediaPipe Pose"}
        }
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with fail-fast validation"""
    global pose_analyzer
    try:
        # Check if pose analyzer is initialized and functional
        if pose_analyzer is None:
            return jsonify({
                'status': 'unhealthy',
                'service': 'pose',
                'error': 'Pose analyzer not initialized'
            }), 503
        
        # Test actual functionality with a small test array
        try:
            test_array = np.zeros((100, 100, 3), dtype=np.uint8)
            pose_analyzer.analyze_pose_from_array(test_array)
            pose_status = 'ready'
        except Exception as e:
            logger.error(f"Health check failed: pose analyzer non-functional: {e}")
            return jsonify({
                'status': 'unhealthy', 
                'service': 'pose',
                'error': f'Pose analyzer non-functional: {str(e)}'
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'service': 'pose',
            'capabilities': ['pose_estimation', 'pose_classification', 'body_segmentation', 'joint_analysis'],
            'models': {
                'pose_estimation': {
                    'status': pose_status,
                    'version': mp.__version__,
                    'model': 'MediaPipe Pose',
                    'landmarks': 33,
                    'complexity': POSE_MODEL_COMPLEXITY,
                    'gpu_enabled': USE_GPU
                }
            },
            'supported_poses': list(POSE_CLASSIFICATIONS.keys()),
            'endpoints': [
                "GET /health - Health check",
                "GET /analyze?url=<image_url> - Analyze pose from URL", 
                "GET /analyze?file=<file_path> - Analyze pose from file",
                "POST /analyze - Analyze pose from uploaded file",
                "GET /v3/analyze?url=<image_url> - V3 compatibility (redirects to /analyze)",
                "GET /v2/analyze?image_url=<image_url> - V2 compatibility (redirects to /analyze)"
            ],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'pose',
            'error': str(e)
        }), 500

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Unified analyze endpoint - orchestrates input handling and processing"""
    import time
    from io import BytesIO
    start_time = time.time()
    
    try:
        # Step 1: Get image data (URL/file/POST)
        if request.method == 'POST':
            # Handle POST file upload
            if 'file' not in request.files:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file provided in POST request"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "No file selected"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            # Validate file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)     # Seek back to beginning
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"File too large. Maximum size: {MAX_FILE_SIZE//1024//1024}MB"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            # Load image from POST upload
            try:
                image = Image.open(file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": f"Failed to load uploaded image: {str(e)}"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
        
        else:
            # Handle GET requests
            url = request.args.get('url')
            file_path = request.args.get('file')
            
            # Validate input - exactly one parameter required
            if not url and not file_path:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "Must provide either 'url' or 'file' parameter"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            if url and file_path:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": "Cannot provide both 'url' and 'file' parameters - choose one"},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
            
            # Load image from URL or file
            try:
                if url:
                    image = download_image_from_url(url)
                elif file_path:
                    image = validate_image_file(file_path)
            except Exception as e:
                return jsonify({
                    "service": "pose",
                    "status": "error",
                    "predictions": [],
                    "error": {"message": str(e)},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 3),
                        "model_info": {"framework": "MediaPipe Pose"}
                    }
                }), 400
        
        # Step 2: Call processing function
        result = process_image_for_pose(image)
        processing_time = time.time() - start_time
        
        # Step 3: Handle processing result
        if not result['success']:
            return jsonify({
                "service": "pose",
                "status": "error",
                "predictions": [],
                "error": {"message": result['error']},
                "metadata": {
                    "processing_time": round(processing_time, 3),
                    "model_info": {"framework": "MediaPipe Pose"}
                }
            }), 500
        
        # Step 4: Create response
        return jsonify(create_pose_response(result['data'], processing_time))
        
    except Exception as e:
        logger.error(f"Error in pose analysis: {str(e)}")
        return jsonify({
            'service': 'pose',
            'status': 'error',
            'predictions': [],
            'metadata': {
                'processing_time': round(time.time() - start_time, 3),
                'model_info': {'framework': 'MediaPipe Pose'}
            },
            'error': {'message': str(e)}
        }), 500

# V3 compatibility route
@app.route('/v3/analyze', methods=['GET', 'POST'])
def analyze_v3_compat():
    """V3 compatibility - redirect to new analyze endpoint"""
    return analyze()

# V2 compatibility routes
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
        del new_args['image_url']
        
        # Call analyze with translated parameters
        with app.test_request_context('/analyze', query_string=new_args):
            return analyze()
    else:
        # Let analyze handle validation errors
        with app.test_request_context('/analyze'):
            return analyze()

if __name__ == '__main__':
    # Initialize pose analyzer at startup - fail fast if it doesn't work
    logger.info("Initializing Pose Analysis Service...")
    if not initialize_pose_analyzer():
        logger.error("Failed to initialize Pose Analyzer. Service cannot function.")
        exit(1)
    
    # Load emoji mappings on startup
    load_emoji_mappings()
    
    # Determine host based on private mode (like other services)
    host = "127.0.0.1" if PRIVATE_MODE else "0.0.0.0"
    
    logger.info(f"Starting Pose Analysis Service on {host}:{PORT}")
    logger.info(f"Private mode: {PRIVATE_MODE}")
    logger.info(f"Model complexity: {POSE_MODEL_COMPLEXITY}, Segmentation: {ENABLE_SEGMENTATION}")
    logger.info("Dedicated pose estimation service with enhanced pose classification")
    logger.info("Unified /analyze endpoint with V2/V3 backward compatibility")
    app.run(host=host, port=PORT, debug=False)